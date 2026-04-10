# GSO-SLAM Documentation

**GSO-SLAM: Bidirectionally Coupled Gaussian Splatting and Direct Visual Odometry**

*Jiung Yeon · Seongbo Ha · Hyeonwoo Yu — IEEE Robotics and Automation Letters, 2026*

[📄 Paper](https://arxiv.org/pdf/2602.11714) | [🎬 Video](https://www.youtube.com/watch?v=io5RmjNzkik&t) | [💾 GitHub](https://github.com/bjoernellens1/GSO-SLAM)

---

## What is GSO-SLAM?

GSO-SLAM is a research SLAM system that **bidirectionally couples** two established subsystems:

| Subsystem | Role |
|---|---|
| **DSO** (Direct Sparse Odometry) | Photometric, keypoint-free direct visual odometry — tracks camera poses via sparse photometric bundle adjustment |
| **3D Gaussian Splatting (3DGS)** | Differentiable, real-time neural rendering representation — maintains the dense, photorealistic map |

The key innovation is the **bidirectional information flow**: DSO continuously feeds new keyframe poses and sparse depth into the Gaussian map; the Gaussian map's rendered depth corrects and initialises DSO's immature tracked points. This tight coupling improves both tracking and mapping quality compared to prior loosely-coupled methods (MonoGS, Photo-SLAM).

---

## Quick Start

### Build

```bash
sudo apt-get install -y libsuitesparse-dev libeigen3-dev libboost-all-dev \
    libglm-dev libjsoncpp-dev libvtk9-dev libpcl-dev \
    libopencv-dev libglfw3-dev libglew-dev libzip-dev libflann-dev

mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Run on Replica

```bash
cd dataset && bash download_replica.sh && bash preprocess.sh
cd ../experiments_bash && bash replica.sh
```

### Run on TUM-RGBD

```bash
cd dataset && bash download_tum.sh
cd ../experiments_bash && bash tum.sh
```

---

## Documentation Sections

| Section | Description |
|---|---|
| [Architecture](architecture.md) | System design, component interactions, data flows |
| [Dependencies](dependencies.md) | All required and optional libraries with version notes |
| [Python Wrapping](python_wrapping.md) | How to expose the C++ API to Python via pybind11 |
| [ROCm Port](rocm_port.md) | Step-by-step AMD GPU porting guide |
| [Triton Kernels](triton_kernels.md) | Feasibility analysis for OpenAI Triton-compiled GPU kernels |
| [ROS2 Integration](ros2_integration.md) | How to wrap GSO-SLAM as a ROS2 node |
| [Semantic Mapping](semantic_mapping.md) | Extending the map with OpenCLIP / DINO semantic features |
| [API Reference](api/index.md) | Auto-generated C++ API reference (Doxygen) |

---

## Citation

```bibtex
@article{yeon2026gsoslam,
  title  = {GSO-SLAM: Bidirectionally Coupled Gaussian Splatting and Direct Visual Odometry},
  author = {Yeon, Jiung and Ha, Seongbo and Yu, Hyeonwoo},
  journal = {IEEE Robotics and Automation Letters},
  year   = {2026}
}
```
