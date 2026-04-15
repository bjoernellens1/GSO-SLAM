#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${GSO_ROOT_DIR:-/workspace/GSO-SLAM}"
DATASET_ROOT="${GSO_DATASET_ROOT:-/datasets}"
RESULTS_ROOT="${GSO_RESULTS_ROOT:-/results}"
SCENE="${GSO_REPLICA_SCENE:-room0}"

GT_DIR="${DATASET_ROOT}/Replica/${SCENE}"
EST_DIR="${RESULTS_ROOT}/replica_${SCENE}"
GT_TRAJ="${GT_DIR}/traj.txt"

[[ -f "${GT_TRAJ}" ]] || { echo "missing ${GT_TRAJ}"; exit 1; }
[[ -d "${EST_DIR}" ]] || { echo "missing ${EST_DIR}"; exit 1; }

cd "${ROOT_DIR}/experiments_bash"
python3 -W ignore scripts/eval_depth.py "${GT_TRAJ}" "${EST_DIR}" "${GT_DIR}"
