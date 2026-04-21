#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TUM_DIR="${REPO_DIR}/dataset/TUM"
RESULT_DIR="${SCRIPT_DIR}/results/test"
OUTPUT_TXT="${RESULT_DIR}/tum_results.txt"
DATASETS="${DATASETS:-freiburg1_desk freiburg2_xyz freiburg3_long_office_household}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${RESULT_DIR}"

echo "--- TUM Dataset Evaluation Start ---" > "${OUTPUT_TXT}"

for dataset_name in ${DATASETS}
do
    DATASET_DIR="${TUM_DIR}/rgbd_dataset_${dataset_name}"
    SAVE_PATH="${RESULT_DIR}/tum_${dataset_name}"
    
    mkdir -p "${SAVE_PATH}"

    echo "========================================"
    echo "Processing: TUM ${dataset_name}"
    echo "TUM ${dataset_name}" >> "${OUTPUT_TXT}"

    "${REPO_DIR}/build/bin/dso_dataset" \
        files="${DATASET_DIR}/rgb" \
        calib="${DATASET_DIR}/camera.txt" \
        dataassociation="${DATASET_DIR}/rgb.txt" \
        which_dataset=tum_rgbd \
        preset=0 \
        mode=1 \
        quiet=1 \
        nogui=1 \
        cfg_yaml="${REPO_DIR}/cfg/gaussian_mapper/Monocular/TUM/tum_${dataset_name}.yaml" \
        save_dir="${SAVE_PATH}" \
        use_gaussian_viewer=0 \
        ${EXTRA_ARGS}

    echo "Evaluating: ${dataset_name}"
    python3 -W ignore "${SCRIPT_DIR}/scripts/evaluate_ate_scale_tum.py" \
        "${DATASET_DIR}" \
        "${SAVE_PATH}" \
        "${DATASET_DIR}/rgb.txt" >> "${OUTPUT_TXT}"

    echo "Done ${dataset_name}"
    echo "--------------------------" >> "${OUTPUT_TXT}"
done

echo "All TUM experiments finished. Results saved in ${OUTPUT_TXT}"
