# ROCm Porting Guide

This guide documents how to port GSO-SLAM from NVIDIA CUDA to AMD ROCm, enabling execution on AMD Instinct / Radeon Pro GPUs.

**Difficulty:** Challenging — 2–4 weeks of focused work.  
**Main obstacle:** OpenCV CUDA module has no ROCm build.

---

## 1. LibTorch / PyTorch — No Code Changes Needed

PyTorch provides first-class ROCm support since version 2.x via HIP (Heterogeneous-Compute Interface for Portability). The `torch::Tensor`, `torch::optim::Adam`, and `c10::cuda::*` APIs transparently dispatch to HIP on ROCm.

```bash
# Install PyTorch with ROCm 5.7 support
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/rocm5.7
```

After this, `torch::kCUDA` selects the AMD GPU and all tensor operations run via HIP kernels — **no changes to GaussianModel, GaussianTrainer, or loss_utils are needed**.

The only header change is in `gaussian_model.h`:

```cpp
// Before:
#include <c10/cuda/CUDACachingAllocator.h>
// After (ROCm PyTorch ships this header):
#include <c10/hip/HIPCachingAllocator.h>
```

---

## 2. CUDA Kernels — Automated Hipification

All `.cu` files in `cuda_rasterizer/` and `src/GS/` must be converted from CUDA to HIP.

### Automated conversion with `hipify-clang`

```bash
# Install hipify tools
sudo apt-get install rocm-hip-sdk

# Hipify all CUDA source files
hipify-clang cuda_rasterizer/rasterizer_impl.cu \
             cuda_rasterizer/forward.cu \
             cuda_rasterizer/backward.cu \
             src/GS/rasterize_points.cu \
             src/GS/operate_points.cu \
             src/GS/stereo_vision.cu \
             thirdparty/simple-knn/simple_knn.cu \
             thirdparty/simple-knn/spatial.cu \
    --cuda-path=/usr/local/cuda \
    -o hip/
```

`hipify-clang` performs these replacements automatically:

| CUDA | HIP equivalent |
|---|---|
| `cudaXxx` | `hipXxx` |
| `cuda_runtime.h` | `hip/hip_runtime.h` |
| `device_launch_parameters.h` | `hip/hip_runtime.h` |
| `cooperative_groups` | `hip/hip_cooperative_groups.h` |

### Manual changes after hipification

**`rasterizer_impl.cu` / `rasterizer_impl_hip.cu`:**

```cpp
// Replace CUB headers with hipCUB equivalents
// Before:
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
// After:
#include <hipcub/hipcub.hpp>

// All cub:: calls become hipcub::
// e.g. cub::DeviceRadixSort::SortPairs → hipcub::DeviceRadixSort::SortPairs
```

**GLM define in `.cu` files:**

```cpp
// Before:
#define GLM_FORCE_CUDA
// After:
#define GLM_FORCE_HIP
```

---

## 3. CMakeLists.txt Changes

Replace the CUDA project setup with HIP:

```cmake
# Before:
PROJECT(${PROJECT_NAME} LANGUAGES CXX CUDA C)
find_package(CUDA REQUIRED)

set_target_properties(cuda_rasterizer PROPERTIES CUDA_ARCHITECTURES "75;86")

# After:
PROJECT(${PROJECT_NAME} LANGUAGES CXX HIP C)
find_package(HIP REQUIRED)
find_package(hipCUB REQUIRED)

set_target_properties(cuda_rasterizer PROPERTIES
    HIP_ARCHITECTURES "gfx906;gfx908;gfx1030;gfx1100")
# gfx906 = MI50/MI60, gfx908 = MI100, gfx1030 = RX 6800, gfx1100 = RX 7900
```

Rename source files to `.hip` extension or use `set_source_files_properties`:

```cmake
set_source_files_properties(
    cuda_rasterizer/rasterizer_impl.cu
    cuda_rasterizer/forward.cu
    cuda_rasterizer/backward.cu
    PROPERTIES LANGUAGE HIP
)
```

---

## 4. OpenCV CUDA Module — The Main Obstacle

`gaussian_mapper.h` uses three OpenCV CUDA modules:

```cpp
#include <opencv2/cudaimgproc.hpp>   // GPU image processing
#include <opencv2/cudastereo.hpp>    // cv::cuda::StereoSGM
#include <opencv2/cudawarping.hpp>   // GPU image warping
```

**These modules do not have a ROCm build.** Options:

### Option A: Fall back to CPU (simplest)

Guard the CUDA stereo path with a preprocessor flag:

```cpp
#ifdef USE_ROCM
    // CPU SGM fallback
    cv::StereoSGBM::create(stereo_min_disparity_, stereo_num_disparity_)->compute(
        left_gray, right_gray, disparity);
#else
    stereo_cv_sgm_->compute(left_gpu, right_gpu, disparity_gpu);
#endif
```

### Option B: Replace with HIP-native disparity solver

Use [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) or implement SGM as a HIP kernel. RAFT-Stereo has a TensorRT path that can be ported to ROCm via MIGraphX.

### Option C: Use CPU OpenCV + ROCm for everything else

Run OpenCV operations on CPU (they are not in the hot path for monocular mode). Only the stereo sensor mode requires `cv::cuda::StereoSGM`. Monocular mode is unaffected.

---

## 5. simple-knn

The `thirdparty/simple-knn/` CUDA kernels hipify cleanly since they only use basic CUDA primitives. After hipification, set:

```cmake
set_source_files_properties(
    thirdparty/simple-knn/simple_knn.cu
    thirdparty/simple-knn/spatial.cu
    PROPERTIES LANGUAGE HIP
)
```

---

## 6. Summary Checklist

```
[ ] Install ROCm 5.7+ and hip-sdk
[ ] pip install torch==2.2.2 (ROCm wheel)
[ ] Run hipify-clang on all .cu files
[ ] Replace cub/ headers with hipcub/
[ ] Replace GLM_FORCE_CUDA with GLM_FORCE_HIP
[ ] Update c10/cuda/ includes to c10/hip/
[ ] Update CMakeLists.txt (LANGUAGES HIP, HIP_ARCHITECTURES)
[ ] Guard or replace cv::cuda:: calls for OpenCV CUDA module
[ ] Test on target AMD GPU (gfx1030 / gfx1100 recommended)
```

---

## 7. Known Limitations After Porting

| Feature | Status after ROCm port |
|---|---|
| Monocular SLAM | Fully functional |
| Stereo SLAM | Requires CPU fallback or custom HIP SGM |
| Interactive ImGui viewer | Functional (OpenGL/GLFW is GPU-agnostic) |
| Rerun visualisation | Functional (CPU-side) |
| PSNR/SSIM evaluation scripts | Functional (PyTorch ROCm) |
