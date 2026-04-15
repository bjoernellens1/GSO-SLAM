#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${GSO_ROOT_DIR:-/workspace/GSO-SLAM}"
DATASET_ROOT="${GSO_DATASET_ROOT:-/datasets}"
RESULTS_ROOT="${GSO_RESULTS_ROOT:-/results}"
SCENE="${GSO_REPLICA_SCENE:-room0}"

GT_FILE="${DATASET_ROOT}/Replica/${SCENE}/traj.txt"
EST_PATH="${RESULTS_ROOT}/replica_${SCENE}"

[[ -f "${GT_FILE}" ]] || { echo "missing ${GT_FILE}"; exit 1; }
[[ -d "${EST_PATH}/pose" ]] || { echo "missing ${EST_PATH}/pose"; exit 1; }

cd "${ROOT_DIR}/experiments_bash"
python3 -W ignore scripts/evaluate_ate_scale_replica.py "${GT_FILE}" "${EST_PATH}"
