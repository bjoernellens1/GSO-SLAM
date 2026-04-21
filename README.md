<div align=center>

# GSO-SLAM: Bidirectionally Coupled Gaussian Splatting and Direct Visual Odometry

[Jiung Yeon*](https://humdrum-balance-b8f.notion.site/Jiung-Yeon-6754922a22814c9a95af88801a96fb4b), [Seongbo Ha*](https://riboha.github.io), [Hyeonwoo Yu](https://bogus2000.github.io/)

(* Equal Contribution)

<h3 align="center"> IEEE Robotics and Automation Letters, 2026 </h3>

[Paper](https://arxiv.org/pdf/2602.11714) | [Video](https://www.youtube.com/watch?v=io5RmjNzkik&t)

</div>

## Environments
For a JupyterLab or other headless notebook server, build with GUI support disabled and a single CUDA architecture to keep compile memory well below the server limit.

If you are setting up this workspace from scratch, the missing C++ dependencies can be installed into the current conda environment with:
```bash
mamba install -y -c conda-forge opencv jsoncpp boost-cpp eigen glm libzip suitesparse
```

```bash
sudo apt-get update && sudo apt-get install -y \
    libsuitesparse-dev libeigen3-dev libboost-all-dev \
    libglm-dev libjsoncpp-dev libopencv-dev libzip-dev
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
cmake -DGSO_ENABLE_GUI=OFF -DGSO_ENABLE_RERUN=OFF -DGSO_CUDA_ARCHITECTURES=80 ..
cmake --build . -j1
```
The headless build is the default in this workspace. If CMake cannot find the conda packages automatically, point `CMAKE_PREFIX_PATH` at `${CONDA_PREFIX}` before configuring.
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
`use_gaussian_viewer=1` is ignored in the headless build.
For TUM data, the loader accepts the raw `rgb.txt` list used by the existing scripts, or a 4-column rgb/depth association file if you want depth paths available. For RGB-D processing, set `SLAM.sensor_type: rgbd` in the scene config and pass the association file.

## Evaluation
```bash
cd experiments_bash
bash replica_eval_rendering.sh
bash replica_eval_depth.sh
```
