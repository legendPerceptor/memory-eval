#!/usr/bin/env bash
# ============================================================================
# full_run_loop.sh — resume-safe wrapper for a full M3-Bench run (runbook §5.1)
# ============================================================================
# Loops run_full_bench.sh until check_complete.py reports every video done.
# Each pass is resume-safe: completed videos/stages/questions are skipped by
# the driver's smart-resume probe layer. Transient failures (LLM timeouts,
# episodic single-clip aborts) heal on the next pass.
#
# Env knobs are forwarded to run_full_bench.sh (RUN, VIDEOS, VIDEO_WORKERS,
# STAGE_WORKERS, QA_TIMEOUT, M3_CONFIG, ...). Loop-only knobs:
#   PASS_SLEEP   seconds between resume passes (default 120)
#   MAX_PASSES   hard cap on passes (default 100; 0 = unlimited)
#
# Detached usage (recommended for a ~24h full-100 run):
#   RUN=full100-ds nohup ./full_run_loop.sh >/data/m3_runs_yuanjian/full100-ds-loop.log 2>&1 &
# Foreground (smoke test):
#   RUN=smoke1 VIDEOS=living_room_22 ./full_run_loop.sh
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${RUN:-full100-ds}"
PASS_SLEEP="${PASS_SLEEP:-120}"
MAX_PASSES="${MAX_PASSES:-100}"
WORK_ROOT="${M3_WORK_ROOT:-/data/m3_runs_yuanjian/$RUN}"
ANNOTATIONS="${ANNOTATIONS:-/data/m3_data/annotations/robot.json}"
PYTHON="${PYTHON:-$HERE/../oGMemory/.venv/bin/python}"
CONFIG="${M3_CONFIG:-$HERE/ogmemory.m3-ds.yaml}"

mkdir -p "$WORK_ROOT"
LOG="$WORK_ROOT/loop.log"

# --- self-heal: make sure local model services are up before each pass. ---
# The TEI embedding server has a known framework-level crash
# (Rust panic in core/src/queue.rs -> Exit 139) that can kill the
# container mid-run. With restart:unless-stopped it *should* come back on
# its own, but we also `docker start` it defensively so the resume loop
# never spins on an "embedder :8001 unhealthy" preflight failure.
ensure_services() {
  local name
  for name in "${EMBED_CONTAINER:-t4-omni-embeddings-1}" \
              "${CV_CONTAINER:-t4-omni-cv-gpu-1}"; do
    if docker inspect "$name" >/dev/null 2>&1; then
      if [[ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != "true" ]]; then
        echo "--- [self-heal] $name not running, starting it $(date -u +%FT%TZ) ---" | tee -a "$LOG"
        docker start "$name" >>"$LOG" 2>&1 || true
      fi
    fi
  done
}

PASS=0
while :; do
  PASS=$((PASS+1))
  echo "=== pass $PASS $(date -u +%FT%TZ)  config=$CONFIG qa_timeout=${M3_QA_TIMEOUT:-300} ===" | tee -a "$LOG"
  ensure_services
  RUN="$RUN" M3_CONFIG="$CONFIG" "$HERE/run_full_bench.sh" >>"$LOG" 2>&1
  rc=$?
  echo "--- pass $PASS exit=$rc $(date -u +%FT%TZ) ---" | tee -a "$LOG"

  if "$PYTHON" "$HERE/check_complete.py" "$ANNOTATIONS" "$WORK_ROOT" >>"$LOG" 2>&1; then
    echo "=== ALL COMPLETE after $PASS pass(es) $(date -u +%FT%TZ) ===" | tee -a "$LOG"
    exit 0
  fi

  if [[ "$MAX_PASSES" != "0" && "$PASS" -ge "$MAX_PASSES" ]]; then
    echo "=== hit MAX_PASSES=$MAX_PASSES, still incomplete — exiting $(date -u +%FT%TZ) ===" | tee -a "$LOG"
    exit 1
  fi

  echo "--- sleeping ${PASS_SLEEP}s before resume pass ---" | tee -a "$LOG"
  sleep "$PASS_SLEEP"
done
