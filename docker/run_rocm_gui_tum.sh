#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${GSO_ROCM_IMAGE:-gso-slam:rocm}"
DATASET_ROOT="${GSO_DATASET_ROOT:-${ROOT_DIR}/dataset}"
RESULTS_ROOT="${GSO_RESULTS_ROOT:-${ROOT_DIR}/docker/results}"
RUN_NAME="${GSO_GUI_RUN_NAME:-tum_gui_test}"
SCENE="${GSO_TUM_SCENE:-freiburg1_desk}"
DISPLAY_VALUE="${DISPLAY:-}"

if [[ -z "${DISPLAY_VALUE}" ]]; then
    echo "DISPLAY is required for the GUI run."
    exit 1
fi

if [[ ! -d "${DATASET_ROOT}/TUM/rgbd_dataset_${SCENE}" ]]; then
    echo "missing dataset directory: ${DATASET_ROOT}/TUM/rgbd_dataset_${SCENE}"
    exit 1
fi

mkdir -p "${RESULTS_ROOT}/${RUN_NAME}"

XAUTH_FILE="$(mktemp /tmp/gso-slam-xauth.XXXXXX)"
cleanup() {
    rm -f "${XAUTH_FILE}"
}
trap cleanup EXIT

if [[ -n "${XAUTHORITY:-}" && -f "${XAUTHORITY}" ]]; then
    cp "${XAUTHORITY}" "${XAUTH_FILE}"
else
    xauth nlist "${DISPLAY_VALUE}" | sed -e 's/^..../ffff/' | xauth -f "${XAUTH_FILE}" nmerge -
fi

docker run --rm -it \
    --network=host \
    --device /dev/kfd:/dev/kfd \
    --device /dev/dri:/dev/dri \
    --group-add video \
    --group-add render \
    --security-opt seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    --ipc=host \
    -e DISPLAY="${DISPLAY_VALUE}" \
    -e XAUTHORITY=/tmp/.Xauthority \
    -e GSO_ROOT_DIR=/workspace/GSO-SLAM \
    -e GSO_DATASET_ROOT=/datasets \
    -e GSO_RESULTS_ROOT=/results \
    -v "${XAUTH_FILE}:/tmp/.Xauthority:ro" \
    -v "${DATASET_ROOT}:/datasets:ro" \
    -v "${RESULTS_ROOT}:/results" \
    "${IMAGE}" \
    bash -lc "stdbuf -oL -eL /workspace/GSO-SLAM/build/bin/dso_dataset \
        files=/datasets/TUM/rgbd_dataset_${SCENE}/rgb \
        calib=/datasets/TUM/rgbd_dataset_${SCENE}/camera.txt \
        dataassociation=/datasets/TUM/rgbd_dataset_${SCENE}/rgb.txt \
        preset=0 mode=1 nogui=1 use_gaussian_viewer=1 \
        cfg_yaml=/workspace/GSO-SLAM/cfg/gaussian_mapper/Monocular/TUM/tum_${SCENE}.yaml \
        save_dir=/results/${RUN_NAME}"
