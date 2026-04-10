# Dependencies

## System Libraries (apt)

Install all required system libraries at once:

```bash
sudo apt-get update && sudo apt-get install -y \
    libsuitesparse-dev libeigen3-dev libboost-all-dev \
    libglm-dev libjsoncpp-dev libvtk9-dev libpcl-dev \
    libopencv-dev libglfw3-dev libglew-dev libzip-dev libflann-dev
```

| Library | Version | Role |
|---|---|---|
| **libsuitesparse** | any recent | Sparse linear algebra (CHOLMOD, CXSparse) used by DSO's Hessian back-end for efficient Schur complement solve |
| **libeigen3** | ≥ 3.3 | Dense linear algebra (matrices, quaternions, Lie-group helpers) throughout all components |
| **libboost** | ≥ 1.65 | `boost::thread` for DSO's multi-threading; `boost::system` |
| **libglm** | any | GLSL-style math types (vec3, mat4) used in CUDA kernels and OpenGL viewer |
| **libjsoncpp** | any | Config serialisation and keyframe metadata JSON export |
| **libvtk9** | 9.x | Required as a transitive dependency of PCL |
| **libpcl** | ≥ 1.12 | Point cloud utilities for inactive geometry densification |
| **libopencv** | ≥ 4.5 | Image I/O, distortion correction, and `cv::cuda::StereoSGM` for stereo depth |
| **libglfw3** | ≥ 3.3 | GLFW windowing system for the ImGui/OpenGL visualiser |
| **libglew** | any | OpenGL extension wrangling |
| **libzip** | any | *Optional* — load datasets packaged as ZIP archives |
| **libflann** | any | Fast approximate nearest-neighbour (used through PCL) |

---

## GPU / CUDA

| Component | Version | Notes |
|---|---|---|
| **CUDA Toolkit** | **11.8** | Required for all GPU kernels. Tested version; 12.x may also work |
| **CUDA driver** | compatible with toolkit | Must match or exceed toolkit version |
| **GPU architecture** | sm_75, sm_86 | Hard-coded in `CMakeLists.txt` as `CUDA_ARCHITECTURES "75;86"` (Turing/Ampere). Adjust to match your GPU |

Install CUDA 11.8: [https://developer.nvidia.com/cuda-11-8-0-download-archive](https://developer.nvidia.com/cuda-11-8-0-download-archive)

---

## PyTorch / LibTorch

| Library | Version | Notes |
|---|---|---|
| **PyTorch** | **2.2.2** (CUDA 11.8) | LibTorch headers and libraries are found via `find_package(Torch)` |
| `torch::Tensor` | — | All GaussianModel parameters, forward/backward passes |
| `torch::optim::Adam` | — | Per-parameter Adam optimiser for Gaussian training |
| `c10/cuda/CUDACachingAllocator` | — | GPU memory management inside GaussianModel |

Install PyTorch:

```bash
pip install torch==2.2.2 torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then point CMake to it:

```bash
cmake -DTorch_DIR=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")/Torch ..
```

---

## Optional Libraries

| Library | CMake flag | Feature enabled |
|---|---|---|
| **Pangolin 0.2** | auto-detected | Legacy DSO 3D viewer (`dso_dataset` executable) |
| **OpenCV** | auto-detected | Image display + image read/write; falls back to dummy stubs if absent |
| **LibZip** | auto-detected | Loading zipped dataset archives |

---

## Fetched at Configure Time

| Library | Source | Role |
|---|---|---|
| **Rerun C++ SDK** | `FetchContent` from GitHub releases | Runtime visualisation of trajectories and map; see [rerun.io](https://rerun.io) |

The Rerun SDK is automatically downloaded during `cmake ..` via CMake's `FetchContent`. Requires internet access at configure time or a pre-cached copy.

---

## Bundled Third-party (in-tree)

These are committed directly to the repository under `thirdparty/`:

| Library | Path | License | Purpose |
|---|---|---|---|
| **Sophus** | `thirdparty/Sophus/` | MIT | SE3/SO3/Sim3 Lie-group algebra for pose representation |
| **simple-knn** | `thirdparty/simple-knn/` | MIT | CUDA KNN for initial Gaussian scale estimation from point clouds |
| **tinyply** | `thirdparty/tinyply/` | Public domain | Binary and ASCII PLY file reader/writer |
| **COLMAP endian utils** | `thirdparty/colmap/utils/endian.h` | BSD-3 | Endianness helpers for PLY I/O |

---

## Viewer Dependencies (bundled)

| Library | Path | License | Purpose |
|---|---|---|---|
| **Dear ImGui** | `viewer/imgui/` | MIT | Immediate-mode GPU UI for the interactive viewer |
| **ImGui GLFW back-end** | `viewer/imgui/imgui_impl_glfw.*` | MIT | GLFW integration for Dear ImGui |
| **ImGui OpenGL3 back-end** | `viewer/imgui/imgui_impl_opengl3.*` | MIT | OpenGL 3.x renderer for Dear ImGui |

---

## Python (evaluation scripts only)

The evaluation scripts in `experiments_bash/scripts/` require:

```bash
pip install torch torchvision torchmetrics opencv-python numpy matplotlib
```

| Package | Use |
|---|---|
| `torch`, `torchvision` | PSNR/SSIM computation |
| `torchmetrics` | `LearnedPerceptualImagePatchSimilarity` (LPIPS) |
| `opencv-python` | Image loading and comparison |
| `numpy` | ATE trajectory alignment (Horn's method) |
| `matplotlib` | PSNR curve plotting (`eval_psnr.py`) |

---

## Version Matrix

The following combination is tested and known to work:

| Component | Version |
|---|---|
| Ubuntu | 22.04 LTS |
| CUDA | 11.8 |
| PyTorch | 2.2.2 |
| OpenCV | 4.5.x |
| Eigen | 3.4.x |
| PCL | 1.12.x |
| CMake | ≥ 3.22 |
| GCC | ≥ 10 (C++17 required) |
