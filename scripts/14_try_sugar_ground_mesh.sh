#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUGAR_ROOT="${SUGAR_ROOT:-/home/linzz/Desktop/SuGaR}"
SUGAR_PYTHON="${SUGAR_PYTHON:-/home/linzz/miniconda3/envs/sugar/bin/python}"
ADAPTER_ROOT="${ADAPTER_ROOT:-${PROJECT_ROOT}/data/scenes/roman_tomato_02/09_sugar_ground}"
MODE="${1:---check}"

SCENE_DIR="${ADAPTER_ROOT}/scene"
CHECKPOINT_DIR="${ADAPTER_ROOT}/checkpoint"
OUTPUT_DIR="${SUGAR_OUTPUT_DIR:-${ADAPTER_ROOT}/mesh_output}"
POINT_CLOUD="${CHECKPOINT_DIR}/point_cloud/iteration_7000/point_cloud.ply"
CAMERAS_JSON="${CHECKPOINT_DIR}/cameras.json"
SURFACE_LEVEL="${SUGAR_SURFACE_LEVEL:-0.3}"
DECIMATION_TARGET="${SUGAR_DECIMATION_TARGET:-20000}"
OPACITY_THRESHOLD="${SUGAR_OPACITY_THRESHOLD:-0.01}"
POISSON_DEPTH="${SUGAR_POISSON_DEPTH:-10}"
VERTICES_DENSITY_QUANTILE="${SUGAR_VERTICES_DENSITY_QUANTILE:-0.1}"
PROJECT_MESH_ON_SURFACE_POINTS="${SUGAR_PROJECT_MESH_ON_SURFACE_POINTS:-False}"
BBOX_MIN="${SUGAR_BBOX_MIN:-(-1,-1,-0.25)}"
BBOX_MAX="${SUGAR_BBOX_MAX:-(1,1,0.25)}"

for required in \
  "${SUGAR_ROOT}/extract_mesh.py" \
  "${SCENE_DIR}/images" \
  "${POINT_CLOUD}" \
  "${CAMERAS_JSON}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[ERROR] Missing required path: ${required}" >&2
    exit 2
  fi
done

if [[ ! -x "${SUGAR_PYTHON}" ]]; then
  echo "[ERROR] SuGaR Python environment is missing: ${SUGAR_PYTHON}" >&2
  echo "Create it from ${SUGAR_ROOT}/environment.yml and install the CUDA extensions first." >&2
  exit 3
fi

"${SUGAR_PYTHON}" -c '
import inspect
import json
import sys
from pathlib import Path

import diff_gaussian_rasterization as rasterizer
import open3d
import plyfile
import pytorch3d
import torch

cameras = json.load(open(sys.argv[1]))
sugar_root = Path(sys.argv[2]).resolve()
rasterizer_path = Path(rasterizer.__file__).resolve()
forward_parameters = inspect.signature(rasterizer.GaussianRasterizer.forward).parameters

print("cameras:", len(cameras))
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("rasterizer:", rasterizer_path)

assert torch.cuda.is_available(), "CUDA is required by SuGaR"
assert sugar_root in rasterizer_path.parents, "Loaded rasterizer is not the SuGaR copy"
assert "mask" not in forward_parameters, "Loaded rasterizer requires the SA3D mask API"
' "${CAMERAS_JSON}" "${SUGAR_ROOT}"

if [[ "${MODE}" == "--check" ]]; then
  echo "[OK] SuGaR repository, adapter data, Python dependencies and CUDA are ready."
  exit 0
fi

if [[ "${MODE}" != "--run" && "${MODE}" != "--run-centers" ]]; then
  echo "Usage: $0 [--check|--run-centers|--run]" >&2
  exit 4
fi

USE_CENTERS=False
if [[ "${MODE}" == "--run-centers" ]]; then
  USE_CENTERS=True
fi

mkdir -p "${OUTPUT_DIR}"
cd "${SUGAR_ROOT}"
"${SUGAR_PYTHON}" extract_mesh.py \
  --scene_path "${SCENE_DIR}" \
  --checkpoint_path "${CHECKPOINT_DIR}/" \
  --iteration_to_load 7000 \
  --mesh_output_dir "${OUTPUT_DIR}" \
  --surface_level "${SURFACE_LEVEL}" \
  --decimation_target "${DECIMATION_TARGET}" \
  --bboxmin "${BBOX_MIN}" \
  --bboxmax "${BBOX_MAX}" \
  --center_bbox False \
  --skip_background True \
  --eval False \
  --use_vanilla_3dgs True \
  --use_centers_to_extract_mesh "${USE_CENTERS}" \
  --project_mesh_on_surface_points "${PROJECT_MESH_ON_SURFACE_POINTS}" \
  --opacity_threshold "${OPACITY_THRESHOLD}" \
  --poisson_depth "${POISSON_DEPTH}" \
  --vertices_density_quantile "${VERTICES_DENSITY_QUANTILE}"
