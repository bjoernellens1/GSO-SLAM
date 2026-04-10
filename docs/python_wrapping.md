# Python Wrapping

GSO-SLAM is written in C++ with LibTorch. Exposing it to Python via **pybind11** is feasible with moderate effort and enables interactive use from Jupyter notebooks, ROS2 Python nodes, and downstream ML pipelines.

---

## Recommended Approach: pybind11 over `GaussianMapper`

### Why pybind11?

- LibTorch ships with pybind11 headers and provides first-class `torch::Tensor ↔ torch.Tensor` conversion automatically via `torch/csrc/utils/pybind.h`.
- The `GaussianMapper` public API (`renderFromPose`, `renderDepthFromPose`, `loadPly`, `run`, `signalStop`) maps naturally to Python.
- No manual memory management is needed for tensors — they share reference-counted storage across C++ and Python.

---

## Step-by-step Implementation

### 1. Add pybind11 to CMakeLists.txt

```cmake
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(gso_slam python/gso_slam.cpp)
target_link_libraries(gso_slam PRIVATE
    gaussian_mapper
    gaussian_viewer
    dso
    ${TORCH_LIBRARIES}
    pybind11::module
)
```

### 2. Create `python/gso_slam.cpp`

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>   // torch::Tensor <-> torch.Tensor

#include "include/gaussian_mapper.h"

namespace py = pybind11;

PYBIND11_MODULE(gso_slam, m) {
    m.doc() = "GSO-SLAM Python bindings";

    // Expose sensor type enum
    py::enum_<SystemSensorType>(m, "SensorType")
        .value("MONOCULAR", SystemSensorType::MONOCULAR)
        .value("STEREO",    SystemSensorType::STEREO)
        .value("RGBD",      SystemSensorType::RGBD)
        .export_values();

    // Expose GaussianMapper
    py::class_<GaussianMapper, std::shared_ptr<GaussianMapper>>(m, "GaussianMapper")
        .def(py::init<std::shared_ptr<dso::FullSystem>,
                      std::filesystem::path,
                      std::filesystem::path,
                      int,
                      torch::DeviceType>(),
             py::arg("slam"),
             py::arg("config_path"),
             py::arg("result_dir") = "",
             py::arg("seed")       = 0,
             py::arg("device")     = torch::kCUDA)
        .def("run",           &GaussianMapper::run)
        .def("signal_stop",   &GaussianMapper::signalStop,
             py::arg("stop") = true)
        .def("is_stopped",    &GaussianMapper::isStopped)
        .def("get_iteration", &GaussianMapper::getIteration)
        .def("load_ply",      &GaussianMapper::loadPly,
             py::arg("ply_path"), py::arg("camera_path") = "")
        // Returns cv::Mat — convert to numpy via cv_bridge or manually
        .def("render_from_pose",
             [](GaussianMapper& self,
                py::array_t<float> pose_4x4,
                int width, int height) {
                 // Convert numpy 4x4 float array to Sophus::SE3f
                 auto buf = pose_4x4.unchecked<2>();
                 Eigen::Matrix4f mat;
                 for (int r = 0; r < 4; ++r)
                     for (int c = 0; c < 4; ++c)
                         mat(r,c) = buf(r,c);
                 Sophus::SE3f Tcw(mat.topLeftCorner<3,3>(),
                                  mat.topRightCorner<3,1>());
                 cv::Mat img = self.renderFromPose(Tcw, width, height);
                 // Return as numpy array
                 return py::array_t<uint8_t>(
                     {img.rows, img.cols, img.channels()},
                     img.data);
             },
             py::arg("pose_4x4"), py::arg("width"), py::arg("height"))
        .def("render_depth_from_pose",
             [](GaussianMapper& self,
                py::array_t<double> pose_4x4,
                int width, int height) -> std::tuple<torch::Tensor, torch::Tensor> {
                 auto buf = pose_4x4.unchecked<2>();
                 Eigen::Matrix4d mat;
                 for (int r = 0; r < 4; ++r)
                     for (int c = 0; c < 4; ++c)
                         mat(r,c) = buf(r,c);
                 Sophus::SE3d Tcw(mat.topLeftCorner<3,3>(),
                                  mat.topRightCorner<3,1>());
                 return self.renderDepthFromPose(Tcw, width, height);
             },
             py::arg("pose_4x4"), py::arg("width"), py::arg("height"));
}
```

### 3. Build and install

```bash
mkdir build && cd build
cmake ..
make -j$(nproc) gso_slam
# The .so lands in build/lib/gso_slam.cpython-3xx-linux-gnu.so
```

### 4. Usage from Python

```python
import sys
sys.path.insert(0, "build/lib")
import gso_slam
import numpy as np

mapper = gso_slam.GaussianMapper(
    slam=None,                          # None = offline / COLMAP mode
    config_path="cfg/gaussian_mapper/Monocular/Replica/office0.yaml",
    result_dir="/tmp/gso_result"
)
mapper.load_ply("results/output.ply")

# Render from a custom pose (camera-to-world SE3 as 4x4 numpy float32)
pose = np.eye(4, dtype=np.float32)
rgb = mapper.render_from_pose(pose, width=1200, height=680)  # numpy [H,W,3] uint8

depth_tensor, alpha_tensor = mapper.render_depth_from_pose(
    pose.astype(np.float64), width=1200, height=680
)  # torch.Tensor [H,W]
```

---

## Alternative: ctypes / cffi (lightweight)

For simple integration without pybind11 overhead:

```cpp
// python/gso_slam_capi.cpp
extern "C" {
    void* gso_mapper_create(const char* config_path, const char* result_dir);
    void  gso_mapper_destroy(void* handle);
    void  gso_mapper_load_ply(void* handle, const char* ply_path);
    int   gso_mapper_get_iteration(void* handle);
}
```

Then load from Python:
```python
import ctypes
lib = ctypes.CDLL("build/lib/libgso_slam_capi.so")
handle = lib.gso_mapper_create(b"cfg/...", b"/tmp/result")
```

---

## Caveats

| Issue | Workaround |
|---|---|
| DSO uses global mutable state (`setting_*` globals in `settings.cpp`) | Not safe for multiple Python instances in the same process; use subprocess isolation |
| `ImGuiViewer` requires the **main thread** for OpenGL | Call viewer from the main thread only; run mapper in a background thread |
| `cv::cuda::StereoSGM` requires a valid CUDA context | Ensure CUDA is initialised before constructing `GaussianMapper` |
| `torch::Tensor` returned to Python holds a GPU allocation | Call `.cpu()` before pickling or passing across processes |
