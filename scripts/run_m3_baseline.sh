#!/usr/bin/env bash
# Fixed five-video baseline; prepare private/config.yaml before running.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/oGMemory"
export UV_NO_SYNC=1 CUDA_VISIBLE_DEVICES=2
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export SAMPLE_SIZE=5 SEED=42 NO_AUDIO=false ASR_VAD_MODE=refresh
export CONFIG="$ROOT/experiments/m3_baseline_20261031/private/config.yaml"
export RUN_ROOT="$ROOT/runs/m3_baseline_20261031"
export ANNOTATIONS=/data/m3_data/annotations/robot.json
export VIDEOS_ROOT=/data/m3_data/videos
bash scripts/run_m3_random10.sh \
  --account-id m3-baseline-20261031 \
  --set voice_asr.params.device=cuda \
  --set speaker_diarization.params.device=cuda \
  --set speaker_embedding.params.device=cuda "$@"
