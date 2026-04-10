# System Architecture

## Overview

GSO-SLAM is structured as a multi-threaded C++ application with CUDA GPU kernels.
The two primary subsystems run on separate threads and communicate through shared data structures protected by mutexes.

```
main_dso_pangolin.cpp
  ├─ Thread 1 — dso::FullSystem (DSO tracking + windowed BA)
  │    ├─ CoarseTracker          photometric frame-to-frame alignment
  │    ├─ CoarseInitializer      monocular scale bootstrap
  │    ├─ FullSystemOptimize     sliding-window Hessian bundle adjustment
  │    └─ EnergyFunctional       Schur complement + marginalisation
  │
  ├─ Thread 2 — GaussianMapper (3DGS map management + training)
  │    ├─ GaussianModel          per-Gaussian parameter tensors
  │    ├─ GaussianScene          keyframe database + scene extent
  │    ├─ GaussianTrainer        single optimisation iteration
  │    ├─ GaussianRenderer       calls rasterizer, returns rendered tensors
  │    └─ GaussianRasterizer     C++ wrapper over cuda_rasterizer CUDA kernels
  │
  ├─ cuda_rasterizer/            CUDA GPU kernels
  │    ├─ rasterizer_impl.cu     tile-based alpha compositing, depth-sorted radix sort (CUB)
  │    ├─ forward.cu             SH→RGB, 3D→2D projection, 3D covariance
  │    └─ backward.cu            autodiff-compatible gradient back-propagation kernels
  │
  └─ ImGuiViewer (optional Thread 3)
       OpenGL/GLFW interactive visualiser driven by Dear ImGui
```

---

## Bidirectional Data Flow

```
┌─────────────────────┐   keyframe pose + sparse points   ┌──────────────────────┐
│   dso::FullSystem   │ ─────────────────────────────────> │   GaussianMapper     │
│   (DSO tracking)    │                                    │   (3DGS map)         │
│                     │ <───────────────────────────────── │                      │
└─────────────────────┘   rendered depth (back to DSO)     └──────────────────────┘
```

### Forward direction (DSO → GaussianMapper)

1. `dso::FullSystem` processes a new image frame via `addActiveFrame()`.
2. When a keyframe is selected, its pose (`SE3f`), sparse depth measurements, and the RGB image are pushed into `GaussianMapper` via `updateGSKeyFramesFromDSO()`.
3. The mapper creates or updates a `GaussianKeyframe` and triggers geometry densification via `increasePcdByKeyframeInactiveGeoDensify()`, which back-projects DSO's sparse 3D points into new Gaussian primitives.
4. On loop closure or local BA, `scaledTransformVisiblePointsOfKeyframe()` rigidly warps the affected Gaussian primitives to match the corrected poses.

### Backward direction (GaussianMapper → DSO)

1. `GaussianMapper::renderDepthFromPose()` renders a full dense depth map from any given SE3 pose using the differentiable rasterizer.
2. This depth is fed back into DSO's `ImmaturePoint` depth initialisation, enabling the tracker to bootstrap depth estimates from the dense Gaussian map rather than relying solely on photometric triangulation.
3. `renderFromPose()` also produces RGB frames used by the ImGui viewer as a side-by-side comparison with the live camera image.

---

## Component Descriptions

### `dso::FullSystem`

The DSO tracking and mapping back-end. Maintains a sliding window of **active frames** and **active points**. Each active point has a host frame and a set of residual observations in other frames. The windowed bundle adjustment minimises the photometric energy functional via Gauss-Newton with Schur complement elimination.

Key files: `src/FullSystem/`, `src/OptimizationBackend/`

### `GaussianMapper`

The central orchestrator of the 3DGS subsystem. Responsible for:

- Receiving keyframes from DSO
- Managing the training loop (`trainForOneIteration`)
- Controlling densification, pruning, and opacity reset schedules
- Exposing render APIs used by the viewer and DSO depth feedback
- Saving and loading PLY snapshots

Key file: `include/gaussian_mapper.h`, `src/GS/gaussian_mapper.cpp`

### `GaussianModel`

Stores all per-Gaussian learnable parameters as LibTorch tensors on the GPU:

| Tensor | Shape | Description |
|---|---|---|
| `xyz_` | `[N, 3]` | 3D Gaussian centres |
| `features_dc_` | `[N, 1, 3]` | DC (zero-order) spherical harmonic coefficients |
| `features_rest_` | `[N, K, 3]` | Higher-order SH coefficients (K depends on sh_degree) |
| `scaling_` | `[N, 3]` | Log-scale values (activated with `exp`) |
| `rotation_` | `[N, 4]` | Unit quaternions (activated with `normalise`) |
| `opacity_` | `[N, 1]` | Logit-opacity values (activated with `sigmoid`) |

All parameters are managed by a `torch::optim::Adam` optimiser with per-parameter learning rates and exponential decay for position.

### `GaussianScene`

The keyframe and camera database. Maps `fid` (frame ID) to `GaussianKeyframe` instances and `camera_id_t` to `Camera` intrinsics. Computes the NeRF++ normalisation radius used to set the scene extent for densification.

### `GaussianRenderer` + `GaussianRasterizer`

Thin C++ wrappers over the CUDA rasterizer. `GaussianRenderer::render()` assembles the camera projection matrices, calls `GaussianRasterizer::forward()`, and returns a 10-tuple of tensors: rendered colour, depth, alpha, transmittance, etc.

### `cuda_rasterizer`

Tile-based alpha-compositing rasterizer implemented as CUDA kernels. Steps:

1. **Preprocess** (`preprocessCUDA`): project 3D Gaussians to 2D, compute 2D covariances, SH→RGB, compute depth + tile overlap keys.
2. **Sort**: CUB `DeviceRadixSort` sorts Gaussians by `(tile_id, depth)` key — back-to-front ordering within each tile.
3. **Render** (`FORWARD::render`): each CUDA thread block handles one 16×16 tile; Gaussians are composited in sorted order using the standard alpha-compositing equation.
4. **Backward** (`BACKWARD::render`): gradient kernels for autograd through the rasterizer, enabling end-to-end optimisation.

### `ImGuiViewer`

Interactive OpenGL/GLFW visualiser using Dear ImGui. Displays:

- Live DSO tracking image (left panel)
- Gaussian-rendered view (right panel, with free-view navigation)
- Training control sliders (learning rates, densification parameters)
- Sparse map-point cloud overlay

Must run on the main thread (OpenGL constraint).

---

## Configuration System

All runtime parameters are read from YAML files in `cfg/gaussian_mapper/`. Parameters are organised into sections:

| Section | Controls |
|---|---|
| `SLAM.*` | Image dimensions, distortion coefficients, dataset type |
| `Model.*` | SH degree, resolution, background colour |
| `Camera.*` | Near/far clip planes |
| `Mapper.*` | Keyframe management, densification, loop closure |
| `GausPyramid.*` | Multi-resolution training schedule |
| `Pipeline.*` | Whether to pre-compute SH / covariance on CPU |
| `Record.*` | Image logging intervals |
| `Optimization.*` | Learning rates, densification thresholds |
| `GaussianViewer.*` | Window size, rendering scale |

---

## Thread Safety

The mapper exposes three mutexes:

- `mutex_status_` — guards `stopped_`, `initial_mapped_`, iteration counter
- `mutex_settings_` — guards all variable training parameters (updated live from ImGui)
- `mutex_render_` — guards the Gaussian model from concurrent read/write during rendering

DSO's `FullSystem` uses its own internal locking. The two systems communicate through a lock-free queue of keyframe tuples.

---

## Code Lineage

| Component | Original Source | License |
|---|---|---|
| `src/FullSystem/`, `src/OptimizationBackend/` | DSO (TU Munich, Jakob Engel) | GPL-3.0 |
| `cuda_rasterizer/` | 3DGS (Inria, Kerbl et al. 2023) | Non-commercial research |
| `include/gaussian_*.h`, `src/GS/gaussian_*.cpp` | Photo-SLAM (Li et al., SYSU/HKUST) | GPL-3.0 |
| `thirdparty/simple-knn/` | 3DGS auxiliary | MIT |
| `thirdparty/Sophus/` | Sophus | MIT |
| `thirdparty/tinyply/` | tinyply | Public domain |
| `viewer/imgui/` | Dear ImGui | MIT |
