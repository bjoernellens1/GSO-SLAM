# C++ API Reference

The C++ API is documented with Doxygen comments throughout all public headers in `include/`.

## Generating the API Docs

### Prerequisites

```bash
sudo apt-get install doxygen graphviz
```

### Generate HTML docs

```bash
cd /path/to/GSO-SLAM
doxygen Doxyfile
# Output: docs/api/html/index.html
```

Open `docs/api/html/index.html` in a browser.

### Generate PDF docs

```bash
cd docs/api/latex
make
# Output: refman.pdf
```

---

## Key Classes

### `GaussianMapper`

Central orchestrator. Manages the 3DGS training loop, keyframe ingestion from DSO, and render APIs.

→ See `include/gaussian_mapper.h`

### `GaussianModel`

Stores all per-Gaussian learnable parameters as LibTorch tensors. Exposes densification, pruning, PLY I/O, and optimiser management.

→ See `include/gaussian_model.h`

### `GaussianScene`

Keyframe and camera database. Computes NeRF++ scene normalisation.

→ See `include/gaussian_scene.h`

### `GaussianKeyframe`

A single training viewpoint: pose, intrinsics, RGB image tensor, sparse depth, and transform tensors for the rasterizer.

→ See `include/gaussian_keyframe.h`

### `GaussianRenderer`

Static render function. Assembles projection matrices and calls the rasterizer.

→ See `include/gaussian_renderer.h`

### `GaussianRasterizer` / `GaussianRasterizerFunction`

PyTorch autograd-compatible wrapper over the CUDA tile rasterizer.

→ See `include/gaussian_rasterizer.h`

### `GaussianTrainer`

Stateless trainer — `trainingOnce()` performs one iteration of the training loop.

→ See `include/gaussian_trainer.h`

### Parameter Classes

| Class | Purpose |
|---|---|
| `GaussianModelParams` | Model paths, SH degree, background |
| `GaussianOptimizationParams` | Learning rates, densification schedule |
| `GaussianPipelineParams` | Pre-computation flags |

→ See `include/gaussian_parameters.h`

---

## Namespaces

| Namespace | File | Contents |
|---|---|---|
| `general_utils` | `include/general_utils.h` | `inverse_sigmoid`, `build_rotation` |
| `graphics_utils` | `include/graphics_utils.h` | `fov2focal`, `focal2fov`, `roundToIntegerMultipleOf16` |
| `loss_utils` | `include/loss_utils.h` | `l1_loss`, `ssim`, `psnr`, `psnr_gaussian_splatting` |

---

## Type Aliases

Defined in `include/types.h`:

| Alias | Underlying type | Description |
|---|---|---|
| `point2D_idx_t` | `uint32_t` | Index of a 2D keypoint |
| `point3D_id_t` | `uint64_t` | Global ID of a 3D map point |
| `camera_id_t` | `uint32_t` | Camera model identifier |
