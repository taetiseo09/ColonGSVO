#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${C3VD_DATA_ROOT:-}"
SCENE="${C3VD_SCENE:-sigmoid_t3_a}"
GPU_ID="${CUDA_DEVICE:-0}"
START_CAM=0
END_CAM=300
SPLIT_RATE=4
IMAGE_DIR="color"
EXPNAME="c3vd_inverse_depth_demo"
PYTHON_BIN="${PYTHON_BIN:-python}"
INTENSITY_THRESHOLD=0.5
EVAL_ONLY=0
SKIP_EVAL=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Run the C3VD inverse-depth demo.

Usage:
  ./train_c3vd_depth.sh --data-root PATH [options]

Required:
  --data-root PATH       Directory containing C3VD scene folders.

Options:
  --scene NAME          Scene folder (default: sigmoid_t3_a).
  --gpu ID              CUDA device id (default: 0).
  --start N             First frame, inclusive (default: 0).
  --end N               Last frame, exclusive (default: 300; 0 means all).
  --split N             Use the last frame in each N-frame group for testing
                        (default: 4).
  --image-dir NAME      RGB directory inside the scene (default: color).
  --expname NAME        Output experiment name.
  --python PATH         Python executable (default: python).
  --intensity FLOAT     Intensity mask threshold (default: 0.5).
  --eval-only           Reuse an existing checkpoint and only evaluate.
  --skip-eval           Train without pose/NVS evaluation.
  --dry-run             Print commands without running them.
  -h, --help            Show this help.

Environment alternatives:
  C3VD_DATA_ROOT, C3VD_SCENE, CUDA_DEVICE, PYTHON_BIN
EOF
}

while (($#)); do
    case "$1" in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --scene) SCENE="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --start) START_CAM="$2"; shift 2 ;;
        --end) END_CAM="$2"; shift 2 ;;
        --split) SPLIT_RATE="$2"; shift 2 ;;
        --image-dir) IMAGE_DIR="$2"; shift 2 ;;
        --expname) EXPNAME="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --intensity) INTENSITY_THRESHOLD="$2"; shift 2 ;;
        --eval-only) EVAL_ONLY=1; shift ;;
        --skip-eval) SKIP_EVAL=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$DATA_ROOT" ]]; then
    echo "Error: --data-root or C3VD_DATA_ROOT is required." >&2
    exit 2
fi
if ! [[ "$START_CAM" =~ ^[0-9]+$ && "$END_CAM" =~ ^[0-9]+$ && "$SPLIT_RATE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --start/--end must be non-negative integers and --split must be positive." >&2
    exit 2
fi
if ((END_CAM != 0 && END_CAM <= START_CAM)); then
    echo "Error: --end must be greater than --start, or 0 to use all frames." >&2
    exit 2
fi

SOURCE_PATH="${DATA_ROOT%/}/${SCENE}"
OUTPUT_PATH="output/${EXPNAME}/${SCENE}_${START_CAM}_${END_CAM}"
MODEL_PATH="${OUTPUT_PATH}/chkpnt/ep00_init.pth"

if [[ ! -d "$SOURCE_PATH/$IMAGE_DIR" || ! -f "$SOURCE_PATH/pose.txt" ]]; then
    echo "Error: expected $SOURCE_PATH/$IMAGE_DIR and $SOURCE_PATH/pose.txt" >&2
    exit 1
fi
if [[ "$DRY_RUN" -eq 0 ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

run_command() {
    printf '+ CUDA_VISIBLE_DEVICES=%q' "$GPU_ID"
    printf ' %q' "$@"
    printf '\n'
    if [[ "$DRY_RUN" -eq 0 ]]; then
        CUDA_VISIBLE_DEVICES="$GPU_ID" "$@"
    fi
}

common_args=(
    --source "$SOURCE_PATH"
    --data_type tanks
    --images "$IMAGE_DIR"
    --depth_model_type dpt
    --start_cam "$START_CAM"
    --end_cam "$END_CAM"
    --train_test_split "$SPLIT_RATE"
    --inverse_depth
    --intensity_threshold "$INTENSITY_THRESHOLD"
)

echo "Scene:       $SCENE"
echo "Data:        $SOURCE_PATH"
echo "Frame range: [$START_CAM, $END_CAM)"
echo "Output:      $OUTPUT_PATH"

if [[ "$EVAL_ONLY" -eq 0 ]]; then
    run_command "$PYTHON_BIN" run_cf3dgs.py \
        "${common_args[@]}" \
        --mode train \
        --use_flow_loss \
        --expname "$EXPNAME"
fi

if [[ "$SKIP_EVAL" -eq 1 ]]; then
    exit 0
fi
if [[ "$DRY_RUN" -eq 0 && ! -f "$MODEL_PATH" ]]; then
    echo "Error: checkpoint not found: $MODEL_PATH" >&2
    exit 1
fi

run_command "$PYTHON_BIN" run_cf3dgs.py \
    "${common_args[@]}" \
    --mode eval_pose \
    --model_path "$MODEL_PATH"

run_command "$PYTHON_BIN" run_cf3dgs.py \
    "${common_args[@]}" \
    --mode eval_nvs \
    --model_path "$MODEL_PATH"
