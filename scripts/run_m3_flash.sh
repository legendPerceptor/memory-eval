#!/usr/bin/env bash
# Invoke from a shell with ~/.bashrc loaded (it requires interactive bash).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ID="${EXPERIMENT_ID:-m3_minimax_flash_20261031}"
EXPERIMENT_ACCOUNT="${EXPERIMENT_ACCOUNT:-m3-flash-20261031}"
EPISODIC_WORKERS="${EPISODIC_WORKERS:-1}"
: "${MINIMAX_API_KEY:?Source ~/.bashrc and export MINIMAX_API_KEY first}"
cd "$ROOT/oGMemory"
export MINIMAX_API_KEY
export OGMEM_MINIMAX_MODEL=MiniMax-M3.1-Flash
export UV_NO_SYNC=1 CUDA_VISIBLE_DEVICES=2
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export SAMPLE_SIZE=5 SEED=42 NO_AUDIO=false ASR_VAD_MODE=refresh
export CONFIG="$ROOT/experiments/$EXPERIMENT_ID/private/config.yaml"
export RUN_ROOT="$ROOT/runs/$EXPERIMENT_ID"
export ANNOTATIONS=/data/m3_data/annotations/robot.json VIDEOS_ROOT=/data/m3_data/videos
export M3_USAGE_LOG="$ROOT/experiments/$EXPERIMENT_ID/model-usage.jsonl"
export PYTHONPATH="$ROOT/scripts/instrumentation:$PWD${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -c 'from providers.llm.openai_llm import OpenAILLM; assert getattr(OpenAILLM._create_completion, "_m3_usage_audit", False)'
bash scripts/run_m3_random10.sh \
  --account-id "$EXPERIMENT_ACCOUNT" \
  --set "episodic_memory.params.workers=$EPISODIC_WORKERS" \
  --set voice_asr.params.device=cuda \
  --set speaker_diarization.params.device=cuda \
  --set speaker_embedding.params.device=cuda:0 "$@"
