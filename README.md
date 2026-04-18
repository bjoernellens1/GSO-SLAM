<div align=center>

# GSO-SLAM: Bidirectionally Coupled Gaussian Splatting and Direct Visual Odometry

[Jiung Yeon*](https://humdrum-balance-b8f.notion.site/Jiung-Yeon-6754922a22814c9a95af88801a96fb4b), [Seongbo Ha*](https://riboha.github.io), [Hyeonwoo Yu](https://bogus2000.github.io/)

(* Equal Contribution)

<h3 align="center"> IEEE Robotics and Automation Letters, 2026 </h3>

[Paper](https://arxiv.org/pdf/2602.11714) | [Video](https://www.youtube.com/watch?v=io5RmjNzkik&t) | [📚 Documentation](docs/index.md)

</div>

> **Documentation:** Full system documentation — including architecture, dependency map, Python wrapping, ROCm porting, Triton kernel analysis, ROS2 integration, and semantic mapping extension — is available in the [`docs/`](docs/) directory and can be built as a browseable site with [MkDocs](#documentation).

## Environments
```bash
sudo apt-get update && sudo apt-get install -y \
    libsuitesparse-dev libeigen3-dev libboost-all-dev \
    libglm-dev libjsoncpp-dev libvtk9-dev libpcl-dev \
    libopencv-dev libglfw3-dev libglew-dev libzip-dev libflann-dev
```
Install [CUDA](https://developer.nvidia.com/cuda-11-8-0-download-archive) and [PyTorch](https://pytorch.org/get-started/locally/) respectively.
We used CUDA Version: 11.8 and PyTorch Version: 2.2.2


## Datasets
-Download
```bash
cd dataset
bash download_replica.sh
bash download_tum.sh
```
-Preprocess
```bash
bash preprocess.sh
```

## Run
### Build
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```
### Replica
```bash
cd experiments_bash
bash replica.sh
```
### TUM-RGBD
```bash
cd experiments_bash
bash tum.sh
```
If use_gaussian_viewer is set as 1, gaussian viewer will appear.

## Evaluation
```bash
cd experiments_bash
bash replica_eval_rendering.sh
bash replica_eval_depth.sh
```

## Container Quickstart
Published GHCR images can be exercised with the Compose files in the repo:

```bash
docker compose -f compose.cuda.yml run --rm smoke
docker compose -f compose.cuda.yml run --rm preprocess-datasets
docker compose -f compose.cuda.yml run --rm -e GSO_REPLICA_SCENE=room0 replica-ate
```

For AMD GPUs, use `compose.rocm.yml`. Those services pass `/dev/kfd` and `/dev/dri` through for ROCm and are compatible with Podman's standard device mapping approach. Full details are in [`docs/container_quickstart.md`](docs/container_quickstart.md).

For a GUI run with persisted output under Podman, build the local ROCm image first:

```bash
podman build --target builder -f docker/Dockerfile.rocm -t gso-slam:rocm .
```

Then allow local X11 access and run one of the following commands. Each writes to `docker/results/...` on the host instead of a container-local `/tmp` path.

TUM GUI example:

```bash
xhost +local:$(whoami)
podman run --rm -it \
  --network=host \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri:/dev/dri \
  --group-add video \
  --group-add render \
  --security-opt seccomp=unconfined \
  --cap-add=SYS_PTRACE \
  --ipc=host \
  -e DISPLAY="$DISPLAY" \
  -e XAUTHORITY=/tmp/.Xauthority \
  -e GSO_ROOT_DIR=/workspace/GSO-SLAM \
  -e GSO_DATASET_ROOT=/datasets \
  -e GSO_RESULTS_ROOT=/results \
  -v "$XAUTHORITY:/tmp/.Xauthority:ro" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PWD/dataset:/datasets:ro" \
  -v "$PWD/docker/results:/results" \
  gso-slam:rocm \
  bash -lc "stdbuf -oL -eL /workspace/GSO-SLAM/build/bin/dso_dataset \
    files=/datasets/TUM/rgbd_dataset_freiburg1_desk/rgb \
    calib=/datasets/TUM/rgbd_dataset_freiburg1_desk/camera.txt \
    dataassociation=/datasets/TUM/rgbd_dataset_freiburg1_desk/rgb.txt \
    preset=0 mode=1 nogui=1 use_gaussian_viewer=1 \
    cfg_yaml=/workspace/GSO-SLAM/cfg/gaussian_mapper/Monocular/TUM/tum_freiburg1_desk.yaml \
    save_dir=/results/tum_gui_test"
```

Replica GUI example:

```bash
xhost +local:$(whoami)
podman run --rm -it \
  --network=host \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri:/dev/dri \
  --group-add video \
  --group-add render \
  --security-opt seccomp=unconfined \
  --cap-add=SYS_PTRACE \
  --ipc=host \
  -e DISPLAY="$DISPLAY" \
  -e XAUTHORITY=/tmp/.Xauthority \
  -e GSO_ROOT_DIR=/workspace/GSO-SLAM \
  -e GSO_DATASET_ROOT=/datasets \
  -e GSO_RESULTS_ROOT=/results \
  -v "$XAUTHORITY:/tmp/.Xauthority:ro" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PWD/dataset:/datasets:ro" \
  -v "$PWD/docker/results:/results" \
  gso-slam:rocm \
  bash -lc "stdbuf -oL -eL /workspace/GSO-SLAM/build/bin/dso_dataset \
    files=/datasets/Replica/room0/results \
    calib=/datasets/Replica/room0/camera.txt \
    preset=0 mode=1 nogui=1 use_gaussian_viewer=1 which_dataset=replica \
    cfg_yaml=/workspace/GSO-SLAM/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
    save_dir=/results/replica_room0_gui"
```

## Documentation

Full documentation is in the [`docs/`](docs/) directory.

| Section | Description |
|---|---|
| [Architecture](docs/architecture.md) | System design, thread model, data flows |
| [Dependencies](docs/dependencies.md) | All libraries with versions |
| [Python Wrapping](docs/python_wrapping.md) | pybind11 bindings |
| [ROCm Port](docs/rocm_port.md) | AMD GPU porting guide |
| [Triton Kernels](docs/triton_kernels.md) | GPU kernel feasibility |
| [ROS2 Integration](docs/ros2_integration.md) | ROS2 node implementation |
| [Semantic Mapping](docs/semantic_mapping.md) | OpenCLIP / DINO extension |
| [API Reference](docs/api/index.md) | C++ API (Doxygen) |

### Build the MkDocs site

```bash
pip install mkdocs mkdocs-material
mkdocs serve        # live preview at http://127.0.0.1:8000
mkdocs build        # static HTML in site/
```

### Build the C++ API reference (Doxygen)

```bash
sudo apt-get install doxygen graphviz
doxygen Doxyfile    # HTML at docs/api/html/index.html
```
