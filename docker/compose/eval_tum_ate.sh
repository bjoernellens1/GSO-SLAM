#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${GSO_ROOT_DIR:-/workspace/GSO-SLAM}"
DATASET_ROOT="${GSO_DATASET_ROOT:-/datasets}"
RESULTS_ROOT="${GSO_RESULTS_ROOT:-/results}"
SCENE="${GSO_TUM_SCENE:-freiburg1_desk}"

GT_PATH="${DATASET_ROOT}/TUM/rgbd_dataset_${SCENE}"
EST_PATH="${RESULTS_ROOT}/tum_${SCENE}"
ASSOC_FILE="${GT_PATH}/rgb.txt"

[[ -d "${GT_PATH}" ]] || { echo "missing ${GT_PATH}"; exit 1; }
[[ -d "${EST_PATH}/pose" ]] || { echo "missing ${EST_PATH}/pose"; exit 1; }
[[ -f "${ASSOC_FILE}" ]] || { echo "missing ${ASSOC_FILE}"; exit 1; }

cd "${ROOT_DIR}/experiments_bash"
python3 -W ignore scripts/evaluate_ate_scale_tum.py "${GT_PATH}" "${EST_PATH}" "${ASSOC_FILE}"
