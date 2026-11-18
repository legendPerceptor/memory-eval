#!/usr/bin/env bash
# ============================================================================
# run_full_bench.sh — parallel FULL M3-Bench run on THIS host (t4-omni)
# ============================================================================
# Runbook analogue: run_maas.sh (§4.2) + launch pattern (§5).
# The RUNBOOK baseline flag set is preserved verbatim so scores stay
# comparable with the colleague's 0.514 full-100 reference:
#   - all 100 robot videos (empty M3_VIDEOS = no --videos filter)
#   - 6 stages disabled (speaker stack, face/voice clustering, consolidator)
#   - extract_frames/episodic max_frames=4, no_audio=true (ASR STILL runs)
#   - qa-top-k 8, transcript_top_k=20, max_iterations=6, resume=true
#   - live per-video judge with deepseek-v4.1-flash-1017
#
# Machine adaptations vs runbook:
#   * no m3runner container → run the repo venv directly on the host
#   * pgvector container m3p4-pg-20261113 @127.0.0.1:31877 (fresh DB per run)
#   * embedder BAAI/bge-m3 @127.0.0.1:8001 ; cv @127.0.0.1:18081
#   * MaaS endpoint (4 req/s) — parallelism follows runbook (2 video workers)
#
# All knobs are overridable via environment. This script BLOCKS; to run it
# detached use full_run_loop.sh (which wraps it in a resume loop + nohup).
#
# Usage (foreground):
#   RUN=full100-ds M3_VIDEOS="" ./run_full_bench.sh
#   RUN=smoke1 VIDEOS="living_room_22" QA_TIMEOUT=300 ./run_full_bench.sh
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../oGMemory" && pwd)"        # driver REPO_ROOT
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

# ---- run identity / topology knobs (mirror runbook §5) ----
RUN="${RUN:-full100-ds}"                            # work root name AND db suffix
VIDEOS="${VIDEOS:-${M3_VIDEOS:-}}"                  # comma list; empty = all 100
ACCOUNT="${M3_ACCOUNT:-m3bench}"
ASR_GPUS="${ASR_GPUS:-0,3}"                          # physical GPUs for GPU ASR (GPU0/GPU3 idle)
ASR_DEVICE="${ASR_DEVICE:-cuda}"                    # faster-whisper device
ASR_COMPUTE_TYPE="${ASR_COMPUTE_TYPE:-float16}"     # float16 on T4 (no bf16)
# Default video-workers = number of ASR GPUs so each concurrent video owns a card.
_n_gpu=$(awk -F, '{print NF}' <<<"$ASR_GPUS")
VIDEO_WORKERS="${VIDEO_WORKERS:-$_n_gpu}"           # 1 worker per ASR GPU
STAGE_WORKERS="${OGMEM_M3_MAX_WORKERS:-4}"          # per-stage workers
QA_TIMEOUT="${M3_QA_TIMEOUT:-300}"
JUDGE="${M3_JUDGE:-1}"
CONFIG="${M3_CONFIG:-$HERE/ogmemory.m3-ds.yaml}"

# ---- data / artifacts ----
ANNOTATIONS="${ANNOTATIONS:-/data/m3_data/annotations/robot.json}"
VIDEOS_ROOT="${VIDEOS_ROOT:-/data/m3_data/videos}"
WORK_ROOT="${M3_WORK_ROOT:-/data/m3_runs_yuanjian/$RUN}"
PIPELINE="${PIPELINE:-config/m3/pipelines/video_episodic_full.yaml}"

# ---- postgres (pgvector container) ----
PG_CONTAINER="${PG_CONTAINER:-m3p4-pg-20261113}"
PG_HOST="${PG_HOST:-127.0.0.1}"
PG_PORT="${PG_PORT:-31877}"
PG_USER="${PG_USER:-ogmem}"
PG_PASSWORD="${PG_PASSWORD:-$(cat "$HERE/../runs/.m3p4_pgpass_20261113" 2>/dev/null)}"
DB="${DB:-ogmem_$RUN}"

# ---- models / endpoints (Huawei MaaS + local services) ----
LLM_PROVIDER="${LLM_PROVIDER:-openai}"
LLM_BASE_URL="${LLM_BASE_URL:-https://api.modelarts-maas.com/v2}"
LLM_API_KEY="${LLM_API_KEY:-${HUAWEI_API_KEY:?set HUAWEI_API_KEY env var}}"
LLM_MODEL="${LLM_MODEL:-deepseek-v4.1-flash-1017}"
M3_EMBED_API_KEY="${M3_EMBED_API_KEY:-abc-123}"

DSN="postgresql://${PG_USER}:${PG_PASSWORD}@${PG_HOST}:${PG_PORT}/${DB}"

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" >&2; }

# ---- 0. sanity checks ----
[[ -x "$PYTHON" ]] || { log "python not found: $PYTHON"; exit 1; }
[[ -f "$CONFIG" ]] || { log "config not found: $CONFIG"; exit 1; }
[[ -f "$ANNOTATIONS" ]] || { log "annotations not found: $ANNOTATIONS"; exit 1; }
command -v docker >/dev/null || { log "docker not found"; exit 1; }
docker ps --format '{{.Names}}' | grep -qx "$PG_CONTAINER" || { log "postgres container '$PG_CONTAINER' not running"; exit 1; }
curl -sf -H "Authorization: Bearer $M3_EMBED_API_KEY" "http://127.0.0.1:8001/health" >/dev/null || { log "embedder :8001 unhealthy"; exit 1; }
curl -sf "http://127.0.0.1:18081/health" >/dev/null || { log "cv :18081 unhealthy"; exit 1; }

# ---- 1. fresh DB (deterministic node ids → must be a clean DB per run) ----
if docker exec "$PG_CONTAINER" psql -U "$PG_USER" -d postgres -tAc \
      "SELECT 1 FROM pg_database WHERE datname='$DB'" | grep -qx 1; then
  log "database '$DB' already exists — reusing (resume). Pass DB=<new> for a clean run."
else
  log "creating database '$DB'"
  docker exec "$PG_CONTAINER" psql -U "$PG_USER" -d postgres -c "CREATE DATABASE \"$DB\" OWNER $PG_USER" >/dev/null
fi
docker exec "$PG_CONTAINER" psql -U "$PG_USER" -d "$DB" -c "CREATE EXTENSION IF NOT EXISTS vector" >/dev/null
log "pgvector extension ensured on '$DB'"

# ---- 2. export runtime env (consumed by m3/config.py ${VAR} refs) ----
export LLM_PROVIDER LLM_BASE_URL LLM_API_KEY LLM_MODEL
export M3_EMBED_API_KEY
export M3_STORAGE_DSN="$DSN"
export M3_VECTOR_DSN="$DSN"
export OGMEM_GRAPH_CONNECTION_STRING="$DSN"
export OGMEM_M3_MAX_WORKERS="$STAGE_WORKERS"
export OGMEM_M3_FACE_WORKERS="${OGMEM_M3_FACE_WORKERS:-$STAGE_WORKERS}"
export OGMEM_GRAPH_POOL_SIZE="${OGMEM_GRAPH_POOL_SIZE:-8}"
export OGMEM_VECTOR_POOL_SIZE="${OGMEM_VECTOR_POOL_SIZE:-8}"
export OGMEM_EMBED_MAX_CONCURRENCY="${OGMEM_EMBED_MAX_CONCURRENCY:-4}"
export PYTHONUNBUFFERED=1

# faster-whisper (CTranslate2) needs cuBLAS/cuDNN from the venv's nvidia
# wheels; they ship inside site-packages but are NOT on the default loader
# path. Without this the GPU ASR subprocess dies with
# "libcublas.so.12 is not found".
NV_LIB="$(cd "$REPO_ROOT" && pwd)/.venv/lib/python3.12/site-packages/nvidia"
if [[ -d "$NV_LIB" ]]; then
  export LD_LIBRARY_PATH="$NV_LIB/cublas/lib:$NV_LIB/cudnn/lib:$NV_LIB/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
  log "LD_LIBRARY_PATH augmented with venv nvidia libs ($NV_LIB)"
fi
mkdir -p "$WORK_ROOT"

# ---- 3. compose the driver argv (baseline flag set, §4.2) ----
VIDS_ARG=()
[[ -n "$VIDEOS" ]] && VIDS_ARG=(--videos "$VIDEOS")
JUDGE_ARG=()
[[ "$JUDGE" = "1" ]] && JUDGE_ARG=(--judge)

log "RUN=$RUN db=$DB work_root=$WORK_ROOT videos=${VIDEOS:-<ALL 100>} video_workers=$VIDEO_WORKERS asr_gpus=$ASR_GPUS asr_device=$ASR_DEVICE/$ASR_COMPUTE_TYPE stage_workers=$STAGE_WORKERS config=$CONFIG"

cd "$REPO_ROOT"
exec "$PYTHON" scripts/run_m3_bench_subset.py \
  "${VIDS_ARG[@]}" \
  --annotations "$ANNOTATIONS" \
  --videos-root "$VIDEOS_ROOT" \
  --work-dir-root "$WORK_ROOT" \
  --config "$CONFIG" \
  --pipeline "$PIPELINE" \
  --graph-store sql \
  --connection-string "$DSN" \
  --account-id "$ACCOUNT" \
  --seconds 0 \
  --disable-stage speaker_diarization \
  --disable-stage speaker_embedding \
  --disable-stage voice_segmentation \
  --disable-stage voice_clustering \
  --disable-stage face_clustering \
  --disable-stage semantic_memory_consolidator \
  --set extract_frames.params.max_frames=4 \
  --set episodic_memory.params.max_frames=4 \
  --set episodic_memory.params.no_audio=true \
  --set voice_asr.params.device="$ASR_DEVICE" \
  --set voice_asr.params.compute_type="$ASR_COMPUTE_TYPE" \
  --qa-top-k 8 \
  --set qa.params.transcript_top_k=20 \
  --qa-timeout "$QA_TIMEOUT" \
  --set qa.params.resume=true \
  --video-workers "$VIDEO_WORKERS" \
  --asr-gpus "$ASR_GPUS" \
  "${JUDGE_ARG[@]}" \
  --python "$PYTHON"
