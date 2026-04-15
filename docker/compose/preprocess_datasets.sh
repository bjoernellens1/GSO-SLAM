#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${GSO_DATASET_ROOT:-/datasets}"

mkdir -p "${DATASET_ROOT}/Replica" "${DATASET_ROOT}/TUM"

echo "== Preprocess datasets =="
echo "dataset_root: ${DATASET_ROOT}"

REPLICA_CAM_SRC="${DATASET_ROOT}/data/Replica/camera.txt"
if [[ -f "${REPLICA_CAM_SRC}" ]]; then
    for scene in room0 room1 room2 office0 office1 office2 office3 office4; do
        mkdir -p "${DATASET_ROOT}/Replica/${scene}"
        cp "${REPLICA_CAM_SRC}" "${DATASET_ROOT}/Replica/${scene}/camera.txt"
    done
    echo "replica camera.txt propagated"
else
    echo "warning: missing ${REPLICA_CAM_SRC}"
fi

for name in freiburg1_desk freiburg2_xyz freiburg3_long_office_household; do
    src="${DATASET_ROOT}/data/TUM/rgbd_dataset_${name}"
    dst="${DATASET_ROOT}/TUM/rgbd_dataset_${name}"
    mkdir -p "${dst}"

    if [[ -f "${src}/camera.txt" ]]; then
        cp "${src}/camera.txt" "${dst}/camera.txt"
    fi

    assoc="tum_${name}.txt"
    if [[ -f "${src}/${assoc}" ]]; then
        cp "${src}/${assoc}" "${dst}/${assoc}"
    fi
done

echo "preprocess complete"
