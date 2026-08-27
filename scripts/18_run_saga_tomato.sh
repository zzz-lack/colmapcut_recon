#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/linzz/Desktop/colmapcut_recon"
SAGA_ROOT="$PROJECT_ROOT/third_party/SegAnyGAussians"
SAGA_PYTHON="/home/linzz/miniconda3/envs/gaussian_splatting_sa3d/bin/python"
SAGA_SCENE="$PROJECT_ROOT/data/scenes/roman_tomato_02/07_saga"
SAM_CHECKPOINT="/home/linzz/Desktop/SegmentAnythingin3D/third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth"

mode="${1:-smoke}"
shift || true

case "$mode" in
  prepare)
    "$SAGA_PYTHON" "$PROJECT_ROOT/scripts/18_prepare_saga_scene.py" "$@"
    ;;
  smoke)
    cd "$SAGA_ROOT"
    "$SAGA_PYTHON" -c "import diff_gaussian_rasterization, diff_gaussian_rasterization_depth, diff_gaussian_rasterization_contrastive_f, segment_anything; print('SAGA runtime OK')"
    ;;
  masks)
    cd "$SAGA_ROOT"
    "$SAGA_PYTHON" extract_segment_everything_masks.py \
      --image_root "$SAGA_SCENE" \
      --sam_checkpoint_path "$SAM_CHECKPOINT" \
      --downsample 4 \
      "$@"
    ;;
  scales)
    cd "$SAGA_ROOT"
    "$SAGA_PYTHON" get_scale.py \
      --model_path "$SAGA_SCENE" \
      --source_path "$SAGA_SCENE" \
      --image_root "$SAGA_SCENE" \
      --iteration 30000 \
      --resolution 4 \
      "$@"
    ;;
  train)
    cd "$SAGA_ROOT"
    "$SAGA_PYTHON" train_contrastive_feature.py \
      --model_path "$SAGA_SCENE" \
      --source_path "$SAGA_SCENE" \
      --iteration 30000 \
      --iterations 10000 \
      --save_iterations 10000 \
      --resolution 4 \
      --num_sampled_rays 512 \
      --smooth_K 1 \
      --feature_dim 32 \
      --target contrastive_feature \
      "$@"
    ;;
  *)
    echo "Usage: $0 {prepare|smoke|masks|scales|train} [extra arguments]" >&2
    exit 2
    ;;
esac
