# Container Quickstart

GSO-SLAM now ships Compose examples for the published GHCR images:

- `ghcr.io/bjoernellens1/gso-slam-cuda:cuda-latest`
- `ghcr.io/bjoernellens1/gso-slam-rocm:rocm-latest`

The Compose files are:

- `compose.cuda.yml`
- `compose.rocm.yml`

Both mount the local `dataset/` directory at `/datasets` and write outputs to `docker/results/`.

## What Works Today

The published images are pushed from the Docker `builder` stage. They are suitable for:

- image smoke checks
- dataset preprocessing
- post-run evaluation on mounted result folders

Example commands:

```bash
docker compose -f compose.cuda.yml run --rm smoke
docker compose -f compose.cuda.yml run --rm preprocess-datasets
docker compose -f compose.cuda.yml run --rm -e GSO_REPLICA_SCENE=room0 replica-ate
docker compose -f compose.cuda.yml run --rm -e GSO_REPLICA_SCENE=room0 replica-rendering
docker compose -f compose.cuda.yml run --rm -e GSO_TUM_SCENE=freiburg1_desk tum-ate
```

ROCm uses the same service names:

```bash
docker compose -f compose.rocm.yml run --rm smoke
docker compose -f compose.rocm.yml run --rm -e GSO_REPLICA_SCENE=office0 replica-depth
```

## Running the Mapping Example

To run the full SLAM and Gaussian Mapping system on a dataset, use the `dso_dataset` binary. The following examples assume you have downloaded and preprocessed the datasets into the `dataset/` folder.

### TUM RGB-D Example (Monocular)

```bash
# Using the ROCm Compose service
docker compose -f compose.rocm.yml run --rm smoke \
  bash -lc "stdbuf -oL -eL ./build/bin/dso_dataset \
    files=/datasets/TUM/rgbd_dataset_freiburg1_desk/rgb \
    calib=/datasets/TUM/rgbd_dataset_freiburg1_desk/camera.txt \
    dataassociation=/datasets/TUM/rgbd_dataset_freiburg1_desk/rgb.txt \
    preset=0 mode=1 nogui=1 \
    cfg_yaml=/workspace/GSO-SLAM/cfg/gaussian_mapper/Monocular/TUM/tum_freiburg1_desk.yaml \
    save_dir=/results/tum_desk_run"
```

### Replica Example (Monocular)

```bash
docker compose -f compose.rocm.yml run --rm smoke \
  bash -lc "stdbuf -oL -eL ./build/bin/dso_dataset \
    files=/datasets/Replica/room0/results \
    preset=0 mode=1 nogui=1 which_dataset=replica \
    cfg_yaml=/workspace/GSO-SLAM/cfg/gaussian_mapper/Monocular/Replica/replica_room0.yaml \
    save_dir=/results/replica_room0_run"
```

### Run All Monocular Presets

The `cfg/gaussian_mapper/Monocular/` directory currently contains these presets:

- Replica: `replica_mono.yaml`, `replica_mono_0.yaml`, `replica_mono_500.yaml`, `room0.yaml`, `room1.yaml`, `room2.yaml`, `office0.yaml`, `office1.yaml`, `office2.yaml`, `office3.yaml`, `office4.yaml`
- TUM: `tum_mono.yaml`, `tum_freiburg1_desk.yaml`, `tum_freiburg2_xyz.yaml`, `tum_freiburg3_long_office_household.yaml`

A simple way to run them all with Podman is to define a helper once and then call it for each YAML:

```bash
run_monocular_cfg() {
  local files_path="$1"
  local calib_path="$2"
  local data_assoc_arg="$3"
  local extra_args="$4"
  local cfg_path="$5"
  local save_path="$6"

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
      files=${files_path} \
      calib=${calib_path} \
      ${data_assoc_arg} \
      preset=0 mode=1 nogui=1 use_gaussian_viewer=1 \
      ${extra_args} \
      cfg_yaml=/workspace/GSO-SLAM/${cfg_path} \
      save_dir=/results/${save_path}"
}
```

Replica configs:

```bash
run_monocular_cfg /datasets/Replica/room0/results \
  /datasets/Replica/room0/camera.txt \
  "" \
  "which_dataset=replica" \
  cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
  replica_room0_run

run_monocular_cfg /datasets/Replica/room0/results \
  /datasets/Replica/room0/camera.txt \
  "" \
  "which_dataset=replica" \
  cfg/gaussian_mapper/Monocular/Replica/replica_mono_0.yaml \
  replica_room0_mono_0_run

run_monocular_cfg /datasets/Replica/room0/results \
  /datasets/Replica/room0/camera.txt \
  "" \
  "which_dataset=replica" \
  cfg/gaussian_mapper/Monocular/Replica/replica_mono_500.yaml \
  replica_room0_mono_500_run

run_monocular_cfg /datasets/Replica/room0/results \
  /datasets/Replica/room0/camera.txt \
  "" \
  "which_dataset=replica" \
  cfg/gaussian_mapper/Monocular/Replica/room0.yaml \
  replica_room0_scene_run

run_monocular_cfg /datasets/Replica/room1/results \
  /datasets/Replica/room1/camera.txt \
  "" \
  "which_dataset=replica" \
  cfg/gaussian_mapper/Monocular/Replica/room1.yaml \
  replica_room1_scene_run

run_monocular_cfg /datasets/Replica/room2/results \
  /datasets/Replica/room2/camera.txt \
  "" \
  "which_dataset=replica" \
  cfg/gaussian_mapper/Monocular/Replica/room2.yaml \
  replica_room2_scene_run

run_monocular_cfg /datasets/Replica/office0/results \
  /datasets/Replica/office0/camera.txt \
  "" \
  "which_dataset=replica" \
  cfg/gaussian_mapper/Monocular/Replica/office0.yaml \
  replica_office0_scene_run

run_monocular_cfg /datasets/Replica/office1/results \
  /datasets/Replica/office1/camera.txt \
  "" \
  "which_dataset=replica" \
  cfg/gaussian_mapper/Monocular/Replica/office1.yaml \
  replica_office1_scene_run

run_monocular_cfg /datasets/Replica/office2/results \
  /datasets/Replica/office2/camera.txt \
  "" \
  "which_dataset=replica" \
  cfg/gaussian_mapper/Monocular/Replica/office2.yaml \
  replica_office2_scene_run

run_monocular_cfg /datasets/Replica/office3/results \
  /datasets/Replica/office3/camera.txt \
  "" \
  "which_dataset=replica" \
  cfg/gaussian_mapper/Monocular/Replica/office3.yaml \
  replica_office3_scene_run

run_monocular_cfg /datasets/Replica/office4/results \
  /datasets/Replica/office4/camera.txt \
  "" \
  "which_dataset=replica" \
  cfg/gaussian_mapper/Monocular/Replica/office4.yaml \
  replica_office4_scene_run
```

TUM configs:

```bash
run_monocular_cfg /datasets/TUM/rgbd_dataset_freiburg1_desk/rgb \
  /datasets/TUM/rgbd_dataset_freiburg1_desk/camera.txt \
  "dataassociation=/datasets/TUM/rgbd_dataset_freiburg1_desk/rgb.txt" \
  "" \
  cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml \
  tum_freiburg1_desk_mono_run

run_monocular_cfg /datasets/TUM/rgbd_dataset_freiburg1_desk/rgb \
  /datasets/TUM/rgbd_dataset_freiburg1_desk/camera.txt \
  "dataassociation=/datasets/TUM/rgbd_dataset_freiburg1_desk/rgb.txt" \
  "" \
  cfg/gaussian_mapper/Monocular/TUM/tum_freiburg1_desk.yaml \
  tum_freiburg1_desk_scene_run

run_monocular_cfg /datasets/TUM/rgbd_dataset_freiburg2_xyz/rgb \
  /datasets/TUM/rgbd_dataset_freiburg2_xyz/camera.txt \
  "dataassociation=/datasets/TUM/rgbd_dataset_freiburg2_xyz/rgb.txt" \
  "" \
  cfg/gaussian_mapper/Monocular/TUM/tum_freiburg2_xyz.yaml \
  tum_freiburg2_xyz_scene_run

run_monocular_cfg /datasets/TUM/rgbd_dataset_freiburg3_long_office_household/rgb \
  /datasets/TUM/rgbd_dataset_freiburg3_long_office_household/camera.txt \
  "dataassociation=/datasets/TUM/rgbd_dataset_freiburg3_long_office_household/rgb.txt" \
  "" \
  cfg/gaussian_mapper/Monocular/TUM/tum_freiburg3_long_office_household.yaml \
  tum_freiburg3_long_office_household_scene_run
```

| Argument | Description |
| :--- | :--- |
| `files` | Path to image directory |
| `calib` | Camera calibration file |
| `preset` | 0: Default, 1: Fast, 2: High Quality |
| `mode` | 0: Indirect, 1: Direct (DSO), 2: Tracking only |
| `nogui` | 1: Headless, 0: Enable Pangolin GUI |
| `cfg_yaml` | Path to the Gaussian Mapper configuration |

## Running with GUI

The ROCm image is built without the legacy Pangolin viewer. Use the ImGui/OpenGL viewer instead by enabling `use_gaussian_viewer=1`. The GUI path should also save to a mounted results directory, not `/tmp` inside the container, so the output survives `--rm`.

1.  **Allow local connections** (on the host):
    ```bash
    xhost +local:root
    ```
2.  **Build the local ROCm builder image**:
    ```bash
    docker build --target builder -f docker/Dockerfile.rocm -t gso-slam:rocm .
    ```
3.  **Run the helper script**:
    ```bash
    bash docker/run_rocm_gui_tum.sh
    ```

The script writes results to `docker/results/tum_gui_test/` on the host. Override the image or output directory with:

```bash
GSO_ROCM_IMAGE=gso-slam:rocm GSO_GUI_RUN_NAME=my_run bash docker/run_rocm_gui_tum.sh
```

**Manual equivalent:** if you want to run the executable directly, keep `save_dir` under `/results/...` so the files land in the mounted host directory.

```bash
docker run --rm -it \
  --network=host \
  --device /dev/kfd:/dev/kfd --device /dev/dri:/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video --group-add render \
  --cap-add=SYS_PTRACE --ipc=host \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=/tmp/.Xauthority \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PWD/dataset:/datasets" \
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

**Note:** `nogui=1` disables the legacy Pangolin path; the Gaussian viewer still opens when `use_gaussian_viewer=1` is set.

## Dataset Layout
...

The Compose wrappers assume the repository dataset layout:

- `dataset/data/Replica/...` and `dataset/data/TUM/...` for raw downloads
- `dataset/Replica/...` and `dataset/TUM/...` for preprocessed runtime data

Run `preprocess-datasets` after downloading data with the existing repo scripts.

## Important Limitation

`dso_dataset` is built without Pangolin in the ROCm image, so the evaluation and runtime path is the ImGui viewer (`use_gaussian_viewer=1`). Run `smoke` first to confirm the image contains `dso_dataset`.

## Podman / ROCm Notes

`compose.rocm.yml` passes `/dev/kfd` and `/dev/dri` through with `devices:`, adds the `video` and `render` groups, and disables the default seccomp profile with `security_opt: [seccomp=unconfined]`. That combination is often required for ROCm containers under Podman, and it can also matter under Docker when device discovery works but HIP execution still faults.

### Strix Point / APU Troubleshooting

If you encounter "Memory critical error by agent node-0" or other HSA initialization faults on AMD APUs (like Ryzen AI Max / Strix Point), ensure the following are set in your environment or Compose file:

- `HSA_ENABLE_SDMA: 0` (Bypass faulty SDMA engine in containerized memory mapping)
- `HSA_XNACK: 1`
- `HSA_OVERRIDE_GFX_VERSION: 11.0.0` (Spoof gfx1151 to a standard gfx11 architecture for Torch compatibility)
- `cap_add: [SYS_PTRACE]`
- `ipc: host`

Equivalent manual Podman invocation for troubleshooting:

```bash
podman run --rm -it \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri:/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --group-add render \
  --cap-add=SYS_PTRACE \
  --ipc=host \
  -e HSA_ENABLE_SDMA=0 \
  -e HSA_XNACK=1 \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  ghcr.io/bjoernellens1/gso-slam-rocm:rocm-latest \
  bash -lc "stdbuf -oL -eL ./build/bin/dso_dataset ..."
```

On Fedora or other SELinux-enabled hosts, add relabelled bind mounts if needed (`:Z`). If your environment uses the AMD Container Runtime Toolkit with CDI, adapt the ROCm service to your CDI device names instead of raw device paths.
