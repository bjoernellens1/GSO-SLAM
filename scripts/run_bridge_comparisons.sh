#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bin_path="${GSO_DSO_BIN:-$repo_root/build-gfx1151/bin/dso_dataset}"
result_root="$repo_root/docker/results"
cfg_root="$repo_root/docker/results/bridge_compare_cfgs"
log_root="$repo_root/docker/results/bridge_compare_logs"

if [[ ! -x "$bin_path" ]]; then
  echo "Missing binary: $bin_path" >&2
  exit 1
fi

mkdir -p "$cfg_root" "$log_root"

make_cfg() {
  local base_cfg="$1"
  local out_cfg="$2"
  local bridge_on="$3"
  local transfer_known="$4"
  local transfer_immature="$5"
  local replay_initial="$6"
  local densify_new="$7"
  local flush_shutdown="$8"

  cp "$base_cfg" "$out_cfg"
  sed -i \
    -e "s/^Mapper.dso_bridge_inactive_geo_densify:.*/Mapper.dso_bridge_inactive_geo_densify: ${bridge_on}/" \
    -e "s/^Mapper.dso_bridge_transfer_known_depth:.*/Mapper.dso_bridge_transfer_known_depth: ${transfer_known}/" \
    -e "s/^Mapper.dso_bridge_transfer_immature_points:.*/Mapper.dso_bridge_transfer_immature_points: ${transfer_immature}/" \
    -e "s/^Mapper.dso_bridge_replay_initial_keyframes:.*/Mapper.dso_bridge_replay_initial_keyframes: ${replay_initial}/" \
    -e "s/^Mapper.dso_bridge_densify_new_keyframes:.*/Mapper.dso_bridge_densify_new_keyframes: ${densify_new}/" \
    -e "s/^Mapper.flush_inactive_geo_cache_on_shutdown:.*/Mapper.flush_inactive_geo_cache_on_shutdown: ${flush_shutdown}/" \
    "$out_cfg"
}

run_case() {
  local name="$1"
  shift
  local save_dir="$result_root/$name"
  local log_file="$log_root/$name.log"
  mkdir -p "$save_dir"

  echo "Starting $name"
  podman run --rm \
    --userns=keep-id \
    --network=host \
    --device /dev/kfd:/dev/kfd \
    --device /dev/dri:/dev/dri \
    --group-add video \
    --group-add render \
    --security-opt seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    --ipc=host \
    -e GSO_ROOT_DIR=/workspace/GSO-SLAM \
    -e GSO_DATASET_ROOT=/datasets \
    -e GSO_RESULTS_ROOT=/results \
    -v "$repo_root:/workspace/GSO-SLAM:Z" \
    -v "$repo_root/dataset:/datasets:ro" \
    -v "$result_root:/results" \
    gso-slam:rocm \
    bash -lc "stdbuf -oL -eL '$bin_path' $* save_dir=/results/$name" \
    >"$log_file" 2>&1
}

base_replica="$repo_root/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml"
base_tum="$repo_root/cfg/gaussian_mapper/Monocular/TUM/tum_freiburg1_desk.yaml"

make_cfg "$base_replica" "$cfg_root/replica_known_depth.yaml" 1 1 0 0 1 0
make_cfg "$base_replica" "$cfg_root/replica_immature.yaml" 1 1 1 0 1 0
make_cfg "$base_tum" "$cfg_root/tum_known_depth.yaml" 1 1 0 0 1 0
make_cfg "$base_tum" "$cfg_root/tum_immature.yaml" 1 1 1 0 1 0

run_case replica_baseline \
  files=/datasets/Replica/office1/results \
  calib=/datasets/Replica/office1/camera.txt \
  preset=0 mode=1 nogui=1 which_dataset=replica \
  cfg_yaml=/workspace/GSO-SLAM/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml

run_case replica_known_depth \
  files=/datasets/Replica/office1/results \
  calib=/datasets/Replica/office1/camera.txt \
  preset=0 mode=1 nogui=1 which_dataset=replica \
  cfg_yaml=/workspace/GSO-SLAM/docker/results/bridge_compare_cfgs/replica_known_depth.yaml

run_case replica_immature \
  files=/datasets/Replica/office1/results \
  calib=/datasets/Replica/office1/camera.txt \
  preset=0 mode=1 nogui=1 which_dataset=replica \
  cfg_yaml=/workspace/GSO-SLAM/docker/results/bridge_compare_cfgs/replica_immature.yaml

run_case tum_baseline \
  files=/datasets/TUM/rgbd_dataset_freiburg1_desk/rgb \
  calib=/datasets/TUM/rgbd_dataset_freiburg1_desk/camera.txt \
  dataassociation=/datasets/TUM/rgbd_dataset_freiburg1_desk/rgb.txt \
  preset=0 mode=1 nogui=1 \
  cfg_yaml=/workspace/GSO-SLAM/cfg/gaussian_mapper/Monocular/TUM/tum_freiburg1_desk.yaml

run_case tum_known_depth \
  files=/datasets/TUM/rgbd_dataset_freiburg1_desk/rgb \
  calib=/datasets/TUM/rgbd_dataset_freiburg1_desk/camera.txt \
  dataassociation=/datasets/TUM/rgbd_dataset_freiburg1_desk/rgb.txt \
  preset=0 mode=1 nogui=1 \
  cfg_yaml=/workspace/GSO-SLAM/docker/results/bridge_compare_cfgs/tum_known_depth.yaml

run_case tum_immature \
  files=/datasets/TUM/rgbd_dataset_freiburg1_desk/rgb \
  calib=/datasets/TUM/rgbd_dataset_freiburg1_desk/camera.txt \
  dataassociation=/datasets/TUM/rgbd_dataset_freiburg1_desk/rgb.txt \
  preset=0 mode=1 nogui=1 \
  cfg_yaml=/workspace/GSO-SLAM/docker/results/bridge_compare_cfgs/tum_immature.yaml
