# Semantic Mapping

This guide explains how to extend GSO-SLAM with **open-vocabulary semantic understanding** by storing and rendering per-Gaussian feature vectors learned from OpenCLIP, DINO v2, or other vision foundation models.

This approach follows the pattern established by [LangSplat](https://github.com/minghanqin/LangSplat), [LERF](https://github.com/kerrj/lerf), and [Feature3DGS](https://github.com/ShijieZhou-UCLA/feature-3dgs).

---

## Concept

Each Gaussian stores a **semantic feature vector** in addition to its colour SH coefficients. At render time, these features are alpha-composited identically to RGB, producing a **dense semantic feature map** at any viewpoint. An open-vocabulary text query can then be answered as a cosine-similarity search over the rendered feature map.

```
Per-Gaussian storage:
  xyz_         [N, 3]         3D position
  features_dc_ [N, 1, 3]     colour SH
  opacity_     [N, 1]         transparency
  scaling_     [N, 3]         ellipsoid axes
  rotation_    [N, 4]         orientation
+ features_semantic_ [N, D]  ← NEW: semantic feature vector
```

---

## Step 1 — Extend `GaussianModel`

### `include/gaussian_model.h`

Add a new tensor field and expose it via the Adam optimiser:

```cpp
// In class GaussianModel — public section:
torch::Tensor features_semantic_;   ///< [N, D_feat] semantic feature vectors

// Expose to optimiser (in trainingSetup() in gaussian_model.cpp):
// Add an entry for features_semantic_ with its own learning rate
```

Extend the `GAUSSIAN_MODEL_INIT_TENSORS` macro:

```cpp
#define GAUSSIAN_MODEL_INIT_TENSORS(device_type)                                                    \
    /* ... existing tensors ... */                                                                    \
    this->features_semantic_ = torch::empty(0, torch::TensorOptions().device(device_type));
```

Add to `createFromPcd()` and `increasePcd()` to initialise feature vectors to zero or from a pre-extracted feature map:

```cpp
void GaussianModel::createFromPcdWithFeatures(
    std::vector<float> points,
    std::vector<float> colors,
    std::vector<float> semantic_features,  // flat [N * D_feat]
    int D_feat,
    float spatial_lr_scale)
{
    // ... existing initialisation ...
    auto feat_tensor = torch::tensor(semantic_features)
        .reshape({(long)points.size()/3, D_feat})
        .to(device_type_);
    features_semantic_ = feat_tensor.clone().requires_grad_(true);
}
```

### `src/GS/gaussian_model.cpp` — `trainingSetup`

Add the semantic feature tensor to the Adam parameter groups:

```cpp
std::vector<torch::optim::OptimizerParamGroup> param_groups;
// ... existing groups ...
param_groups.push_back({
    {features_semantic_},
    std::make_unique<torch::optim::AdamOptions>(semantic_feature_lr)
});
optimizer_ = std::make_shared<torch::optim::Adam>(param_groups, ...);
```

---

## Step 2 — Extend the CUDA Rasterizer

### `cuda_rasterizer/rasterizer.h`

Add a `features` input and `rendered_features` output:

```cpp
std::tuple<int, torch::Tensor, /* ... existing ... */, torch::Tensor>
forward(/* ... existing args ... */,
        const torch::Tensor& semantic_features,   // [N, D_feat]
        int D_feat);
```

### `cuda_rasterizer/forward.cu` — render kernel

In the tile-compositing loop, alpha-composite the semantic features alongside colour:

```c
// After existing colour accumulation:
for (int d = 0; d < D_FEAT; d++) {
    float feat_val = point_features[collected_id[j] * D_FEAT + d];
    feat_accum[d] += alpha * T_current * feat_val;
}
```

Output tensor `rendered_features` of shape `[D_feat, H, W]`.

### `cuda_rasterizer/backward.cu`

Add gradient computation for `semantic_features` analogous to the existing `dL_dcolors` gradient.

---

## Step 3 — Feature Extraction at Keyframe Creation

### Extract features with OpenCLIP

```python
import open_clip
import torch
import numpy as np
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.cuda().eval()

def extract_clip_features(image_path: str) -> np.ndarray:
    """
    Extract dense CLIP features from an image.

    Uses a sliding-window approach (MaskCLIP-style) to produce
    per-pixel feature vectors.

    Args:
        image_path: Path to the input RGB image.

    Returns:
        np.ndarray of shape [H, W, 512] (float32).
    """
    img = Image.open(image_path).convert("RGB")
    H, W = img.height, img.width

    # For dense features, tile the image and aggregate
    # This is a simplified version — use MaskCLIP for production
    tiles = []
    stride = 32
    patch_size = 224
    for y in range(0, H - patch_size, stride):
        for x in range(0, W - patch_size, stride):
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            tiles.append((x, y, preprocess(patch)))

    feature_map = np.zeros((H, W, 512), dtype=np.float32)
    count_map   = np.zeros((H, W),      dtype=np.float32)

    for x, y, tile_tensor in tiles:
        with torch.no_grad():
            feat = model.encode_image(
                tile_tensor.unsqueeze(0).cuda()).cpu().numpy()[0]  # [512]
        feat /= np.linalg.norm(feat) + 1e-8
        feature_map[y:y+patch_size, x:x+patch_size] += feat
        count_map[y:y+patch_size,   x:x+patch_size] += 1.0

    feature_map /= np.maximum(count_map[:, :, None], 1.0)
    return feature_map  # [H, W, 512]
```

### Assign features to new Gaussians in `GaussianMapper`

In `handleNewKeyframe()` (inside `gaussian_mapper.cpp`), after extracting sparse points:

```cpp
// Python call via pybind11 / subprocess, or pre-computed offline
// Assume semantic_features is a [num_points, D_feat] float tensor
for (size_t i = 0; i < sparse_points.size(); i++) {
    // Project point to image → look up feature in feature map
    Eigen::Vector2f uv = projectPoint(sparse_points[i], K, Tcw);
    int px = std::clamp((int)uv.x(), 0, feat_map_W - 1);
    int py = std::clamp((int)uv.y(), 0, feat_map_H - 1);
    auto feat = feature_map_tensor[py][px];  // [D_feat]
    initial_semantic_features.push_back(feat);
}
```

---

## Step 4 — Semantic Distillation Loss

Add to `GaussianTrainer::trainingOnce()`:

```cpp
// Render semantic feature map
auto [rendered_feat, ...] = GaussianRenderer::render(
    viewpoint_cam, image_height, image_width,
    gaussians, pipe, bg_color, override_color);
// rendered_feat: [D_feat, H, W]

// Target: pre-computed OpenCLIP feature map for this keyframe
auto target_feat = viewpoint_cam->semantic_feature_map_;  // [D_feat, H, W]

// Distillation loss (cosine similarity or L2)
auto L_sem = torch::nn::functional::mse_loss(rendered_feat, target_feat);

// Total loss
auto loss = L_l1 + lambda_dssim * (1.0f - ssim_val) + lambda_semantic * L_sem;
loss.backward();
```

Typical `lambda_semantic` value: `0.1` — tune based on scene.

---

## Step 5 — Open-vocabulary Query at Inference

```python
import open_clip
import torch

# Encode text query
model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

def query_semantic_map(rendered_features: torch.Tensor, text: str) -> torch.Tensor:
    """
    Compute cosine-similarity relevance map between rendered features and a text query.

    Args:
        rendered_features: [D_feat, H, W] float32 CUDA tensor of rendered feature map.
        text: Open-vocabulary text query, e.g. "a yellow chair".

    Returns:
        [H, W] float32 tensor with relevance score in [0, 1] at each pixel.
    """
    with torch.no_grad():
        tokens = tokenizer([text]).cuda()
        text_feat = model.encode_text(tokens)  # [1, 512]
        text_feat /= text_feat.norm(dim=-1, keepdim=True)

    D, H, W = rendered_features.shape
    feat_flat = rendered_features.permute(1, 2, 0).reshape(-1, D)  # [H*W, D]
    feat_flat = feat_flat / (feat_flat.norm(dim=-1, keepdim=True) + 1e-8)

    similarity = (feat_flat @ text_feat.T).squeeze(-1)  # [H*W]
    return similarity.reshape(H, W).clamp(0.0, 1.0)
```

---

## Feature Encoder Comparison

| Encoder | Feature dim | Language grounding | Spatial resolution | Notes |
|---|---|---|---|---|
| **OpenCLIP ViT-B/32** (LAION-2B) | 512 | ✅ Strong | Patch-level (~14×14 tiles) | Best for open-vocabulary queries |
| **DINO v2 (ViT-L/14)** | 1024 | ❌ None alone | Pixel-level via interpolation | Excellent spatial features; pair with CLIP for language |
| **SAM + CLIP** (Grounded-SAM) | 256 + 512 | ✅ Object-level | Instance masks | Ideal for object-level semantic mapping |
| **LSeg / OpenSeg** | 512 | ✅ Categorical | Pixel-level | Needs a fixed label vocabulary upfront |
| **PCA-compressed** (3-d) | 3 | None | — | For visualisation as pseudo-colour in the ImGui viewer |

---

## Memory Budget

| Gaussians | Feature dim | Precision | GPU Memory |
|---|---|---|---|
| 500k | 512 (CLIP full) | float32 | 1.0 GB |
| 500k | 64 (PCA) | float32 | 0.125 GB |
| 500k | 3 (vis only) | float32 | 6 MB |

**Recommendation:** Use PCA to compress to 64 dimensions before storing in Gaussians. Apply the full 512-d CLIP encoder offline at keyframe creation, then PCA-project the result:

```python
from sklearn.decomposition import IncrementalPCA
pca = IncrementalPCA(n_components=64)
features_compressed = pca.fit_transform(features_512d)
```

Store the PCA matrix in the PLY file or as a companion `.npy` file.

---

## Visualisation in the ImGui Viewer

Extend `ImGuiViewer` to render the first 3 PCA components as pseudo-colour:

1. Add a `bool show_semantic_` flag to `ImGuiViewer`
2. When enabled, call `GaussianRenderer::render()` with `override_color_` set to the PCA-3 features of each Gaussian
3. Display the result in the main panel alongside the regular RGB view
