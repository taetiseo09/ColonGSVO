#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SOURCE_PATH=""
MODEL_PATH=""
GPU_ID="${CUDA_DEVICE:-0}"
START_CAM=0
END_CAM=300
SPLIT_RATE=8
IMAGE_DIR="color"
VIDEO_FPS=10
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
    cat <<'EOF'
Evaluate only RGB reconstruction quality and synchronized rendering FPS.

Usage:
  ./evaluate_c3vd.sh --source PATH --model-path PATH [options]

Required:
  --source PATH          C3VD scene directory containing pose.txt and RGB frames.
  --model-path PATH      Trained ep00_init.pth checkpoint.

Options:
  --gpu ID               CUDA device id (default: 0).
  --start N              First frame, inclusive (default: 0).
  --end N                Last frame, exclusive (default: 300).
  --split N              Test split interval (default: 8).
  --image-dir NAME       RGB directory inside the scene (default: color).
  --video-fps N          Output MP4 FPS (default: 10).
  --python PATH          Python executable (default: python).
  -h, --help             Show this help.

Outputs are written beside the checkpoint under test/:
  frames/, reconstruction.mp4, report.json, report.md
EOF
}

while (($#)); do
    case "$1" in
        --source) SOURCE_PATH="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --start) START_CAM="$2"; shift 2 ;;
        --end) END_CAM="$2"; shift 2 ;;
        --split) SPLIT_RATE="$2"; shift 2 ;;
        --image-dir) IMAGE_DIR="$2"; shift 2 ;;
        --video-fps) VIDEO_FPS="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$SOURCE_PATH" || -z "$MODEL_PATH" ]]; then
    echo "Error: --source and --model-path are required." >&2
    exit 2
fi
if [[ ! -d "$SOURCE_PATH/$IMAGE_DIR" || ! -f "$SOURCE_PATH/pose.txt" ]]; then
    echo "Error: expected $SOURCE_PATH/$IMAGE_DIR and $SOURCE_PATH/pose.txt" >&2
    exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: checkpoint not found: $MODEL_PATH" >&2
    exit 1
fi
if ! [[ "$START_CAM" =~ ^[0-9]+$ && "$END_CAM" =~ ^[0-9]+$ && "$SPLIT_RATE" =~ ^[1-9][0-9]*$ && "$VIDEO_FPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: frame values must be non-negative; split/video FPS must be positive." >&2
    exit 2
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" run_cf3dgs.py \
    --source "$SOURCE_PATH" \
    --mode eval_reconstruction \
    --data_type tanks \
    --images "$IMAGE_DIR" \
    --model_path "$MODEL_PATH" \
    --start_cam "$START_CAM" \
    --end_cam "$END_CAM" \
    --train_test_split "$SPLIT_RATE" \
    --video_fps "$VIDEO_FPS"
