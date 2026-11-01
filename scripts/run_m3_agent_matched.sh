#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/experiments/m3_agent_matched_20261101/exit_code.txt" ]]; then
  echo "Experiment already finished; use a new experiment directory, not an overwrite." >&2
  exit 2
fi
: "${MINIMAX_API_KEY:?Source ~/.bashrc first}"
export M3_EVAL_ROOT="$ROOT" OGMEM_MINIMAX_MODEL=MiniMax-M3.1-Flash PYTHONHASHSEED=0
export PYTHONPATH="$ROOT/scripts/instrumentation:$ROOT/runs/m3_agent_matched_20261101/_source/oGMemory"
export M3_USAGE_LOG="$ROOT/experiments/m3_agent_matched_20261101/model-usage.jsonl"
exec 9>"$ROOT/runs/m3_agent_matched_20261101/driver.lock"
flock -n 9 || { echo "Another matched driver is running" >&2; exit 2; }
exec "$ROOT/oGMemory/.venv/bin/python" "$ROOT/runs/m3_agent_matched_20261101/_adapter/driver.py"
