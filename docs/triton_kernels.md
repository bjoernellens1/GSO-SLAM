# Triton-compiled Kernels

[OpenAI Triton](https://github.com/openai/triton) is a Python-based GPU programming language that compiles to PTX (NVIDIA) or HSA/GCN (AMD) via LLVM. This page analyses which GSO-SLAM CUDA kernels are suitable for Triton replacement and how to implement them.

---

## Feasibility Summary

| Kernel | File | Triton suitability | Notes |
|---|---|---|---|
| `computeColorFromSH` | `forward.cu` | ✅ Excellent | Pure element-wise math, no shared memory |
| `preprocessCUDA` | `forward.cu` | ✅ Good | Per-Gaussian projection + 2D covariance — embarrassingly parallel |
| `FORWARD::render` (alpha compositing) | `forward.cu` | ⚠️ Partial | Tile-level shared memory + warp-scan; Triton supports but complex |
| `BACKWARD::render` | `backward.cu` | ⚠️ Partial | Same tile structure; gradients are complex |
| CUB radix sort (`duplicateWithKeys`) | `rasterizer_impl.cu` | ❌ Not applicable | Triton has no sorting primitives; must keep in CUDA / use `torch.sort` |
| KNN (`spatial.cu`) | `thirdparty/simple-knn/` | ⚠️ Possible | Can be replaced with `torch_cluster.knn` or a Triton CUDA kernel |
| `operatePoints` | `src/GS/operate_points.cu` | ✅ Good | Simple per-point transformations |

---

## Key Insight: Triton and CUDA Coexistence

Triton kernels return `torch.Tensor` objects. The existing C++ rasterizer accepts `torch::Tensor` inputs. This means:

- Triton handles the **pure compute kernels** (SH evaluation, 3D→2D projection)
- The C++ rasterizer handles the **sorting and tile-compositing** that requires CUB
- The two can coexist in the same pipeline

---

## Example: SH→RGB Triton Kernel

The `computeColorFromSH` kernel in `forward.cu` evaluates spherical harmonics for each Gaussian. It is a natural Triton candidate:

```python
import triton
import triton.language as tl
import torch

# Spherical harmonic constants
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199

@triton.jit
def sh_to_rgb_kernel(
    means_ptr,   # [N, 3] float32 — Gaussian centres
    shs_ptr,     # [N, max_coeffs, 3] float32 — SH coefficients
    campos_ptr,  # [3] float32 — camera position
    out_ptr,     # [N, 3] float32 — output RGB colours
    N: tl.constexpr,
    max_coeffs: tl.constexpr,
    sh_degree: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load Gaussian centre
    x = tl.load(means_ptr + offsets * 3 + 0, mask=mask)
    y = tl.load(means_ptr + offsets * 3 + 1, mask=mask)
    z = tl.load(means_ptr + offsets * 3 + 2, mask=mask)

    # Load camera position (broadcast)
    cx = tl.load(campos_ptr + 0)
    cy = tl.load(campos_ptr + 1)
    cz = tl.load(campos_ptr + 2)

    # Compute normalised view direction
    dx = x - cx; dy = y - cy; dz = z - cz
    inv_len = 1.0 / tl.sqrt(dx*dx + dy*dy + dz*dz + 1e-8)
    dx *= inv_len; dy *= inv_len; dz *= inv_len

    # Zero-order SH (DC term)
    sh0r = tl.load(shs_ptr + offsets * max_coeffs * 3 + 0, mask=mask)
    sh0g = tl.load(shs_ptr + offsets * max_coeffs * 3 + 1, mask=mask)
    sh0b = tl.load(shs_ptr + offsets * max_coeffs * 3 + 2, mask=mask)
    r = SH_C0 * sh0r
    g = SH_C0 * sh0g
    b = SH_C0 * sh0b

    # First-order SH (if sh_degree >= 1)
    if sh_degree >= 1:
        sh1_yr = tl.load(shs_ptr + offsets * max_coeffs * 3 + 3, mask=mask)
        # ... (remaining 8 first-order terms)
        r = r - SH_C1 * dy * sh1_yr  # + ...
        # similarly for g, b

    # Clamp to valid range and store
    r = tl.maximum(r + 0.5, 0.0)
    g = tl.maximum(g + 0.5, 0.0)
    b = tl.maximum(b + 0.5, 0.0)

    tl.store(out_ptr + offsets * 3 + 0, r, mask=mask)
    tl.store(out_ptr + offsets * 3 + 1, g, mask=mask)
    tl.store(out_ptr + offsets * 3 + 2, b, mask=mask)


def sh_to_rgb(means: torch.Tensor, shs: torch.Tensor,
              campos: torch.Tensor, sh_degree: int) -> torch.Tensor:
    """
    Evaluate spherical harmonics to get per-Gaussian RGB colours.

    Args:
        means:    [N, 3] float32 CUDA tensor of Gaussian centres.
        shs:      [N, max_coeffs, 3] float32 CUDA tensor of SH coefficients.
        campos:   [3] float32 CUDA tensor of camera position.
        sh_degree: Maximum SH degree (0–3).

    Returns:
        [N, 3] float32 CUDA tensor of clamped RGB colours.
    """
    N = means.shape[0]
    max_coeffs = shs.shape[1]
    out = torch.empty(N, 3, device=means.device, dtype=torch.float32)
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    sh_to_rgb_kernel[grid](
        means, shs, campos, out,
        N, max_coeffs, sh_degree, BLOCK_SIZE
    )
    return out
```

---

## Example: Gaussian Projection (preprocessCUDA)

The projection kernel computes 2D screen positions and 2D covariances:

```python
@triton.jit
def preprocess_gaussians_kernel(
    means3d_ptr,      # [N, 3]
    scales_ptr,       # [N, 3]
    rotations_ptr,    # [N, 4] quaternion
    viewmatrix_ptr,   # [4, 4]
    projmatrix_ptr,   # [4, 4]
    means2d_ptr,      # [N, 2] output
    cov2d_ptr,        # [N, 3] output (upper triangle of 2x2)
    depths_ptr,       # [N] output
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    # ... project xyz through viewmatrix + projmatrix
    # ... compute 3D covariance from scale + rotation quaternion
    # ... project 3D covariance to 2D via Jacobian of perspective projection
```

---

## Integration with C++ Rasterizer

The Triton kernels produce `torch.Tensor` outputs. Pass these to the existing `RasterizeGaussiansCUDA` C++ function as `colors_precomp` (pre-computed RGB) with the `prefiltered=false` flag, bypassing the CUDA SH kernel:

```python
# Python orchestration layer
colors = sh_to_rgb(means3d, shs, campos, sh_degree)    # Triton kernel
# Pass colors as colors_precomp to rasterizer
rendered = rasterizer(means3d=means3d,
                      colors_precomp=colors,           # skip CUDA SH
                      ...)
```

---

## ROCm Benefit

Triton has experimental ROCm support via `triton-rocm` (since Triton 2.1). Implementing the algorithmic kernels (SH, projection) in Triton simultaneously solves the ROCm porting problem for those kernels — a `triton.jit`-decorated function compiles to HSA/GCN bytecode on AMD GPUs without modification.

```bash
pip install triton  # NVIDIA PTX on CUDA, HSA on ROCm
```

---

## When NOT to Use Triton

- **CUB radix sort** — Triton provides no sorting API; keep `cub::DeviceRadixSort` in CUDA
- **Tile-level shared memory compositing** — The alpha-compositing loop in `FORWARD::render` uses hand-tuned warp-level operations and a shared-memory load queue that are difficult to replicate in Triton without performance loss
- **simple-knn** — Use `torch_scatter` or `torch_cluster.knn` from Python instead of porting the CUDA KNN kernel
