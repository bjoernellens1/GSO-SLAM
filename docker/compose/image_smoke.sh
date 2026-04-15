#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${GSO_ROOT_DIR:-/workspace/GSO-SLAM}"

echo "== GSO-SLAM image smoke check =="
echo "root: ${ROOT_DIR}"
echo "python: $(python3 --version)"
echo "cmake: $(cmake --version | head -n 1)"

if [[ -d "${ROOT_DIR}/build/bin" ]]; then
    echo "build/bin:"
    find "${ROOT_DIR}/build/bin" -maxdepth 1 -type f | sort || true
else
    echo "build/bin missing"
fi

if [[ -x "${ROOT_DIR}/build/bin/dso_dataset" ]]; then
    echo "dso_dataset: present"
else
    echo "dso_dataset: missing"
    echo "note: the published images are built from the builder stage; dso_dataset is only produced when Pangolin is available at image build time."
fi

python3 - <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"cuda_available: {torch.cuda.is_available()}")
PY
