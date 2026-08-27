#!/usr/bin/env bash
set -euo pipefail

SUGAR_ROOT="${SUGAR_ROOT:-/home/linzz/Desktop/SuGaR}"
SUGAR_PYTHON="${SUGAR_PYTHON:-/home/linzz/miniconda3/envs/sugar/bin/python}"
SUGAR_PIP="${SUGAR_PIP:-/home/linzz/miniconda3/envs/sugar/bin/pip}"
RASTERIZER_SOURCE="${SUGAR_ROOT}/gaussian_splatting/submodules/diff-gaussian-rasterization"
CUDA_ARCH="${TORCH_CUDA_ARCH_LIST:-12.0}"

if [[ ! -x "${SUGAR_PYTHON}" || ! -x "${SUGAR_PIP}" ]]; then
  echo "[ERROR] Missing sugar Python environment." >&2
  exit 2
fi
if [[ ! -f "${RASTERIZER_SOURCE}/setup.py" ]]; then
  echo "[ERROR] Missing SuGaR rasterizer source: ${RASTERIZER_SOURCE}" >&2
  exit 3
fi

"${SUGAR_PIP}" uninstall -y diff-gaussian-rasterization || true
cd "${RASTERIZER_SOURCE}"
/usr/bin/env \
  TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}" \
  MAX_JOBS="${MAX_JOBS:-8}" \
  "${SUGAR_PIP}" install --no-build-isolation -e .

"${SUGAR_PYTHON}" -B -c '
import inspect
from pathlib import Path
import diff_gaussian_rasterization as module

parameters = inspect.signature(module.GaussianRasterizer.forward).parameters
print("rasterizer:", Path(module.__file__).resolve())
print("forward:", inspect.signature(module.GaussianRasterizer.forward))
assert "mask" not in parameters
'
