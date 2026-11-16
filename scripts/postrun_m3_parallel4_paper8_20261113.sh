#!/usr/bin/env bash
set -u
cd /home/yuanjian/Development/memory-eval
export M3_EVAL_ROOT=/home/yuanjian/Development/memory-eval
PY=oGMemory/.venv/bin/python
OUT=experiments/m3_parallel4_paper8_20261113/private/postrun.out
{
  echo "=== postrun start $(date -Is) ==="
  echo "--- QA budget sweep (waits for baseline 'finished') ---"
  $PY scripts/run_m3_parallel4_paper8_qa_sweep_20261113.py; echo "sweep_exit=$?"
  echo "--- build pareto plot ---"
  $PY scripts/plot_m3_parallel4_paper8_qa_pareto_20261113.py > experiments/m3_parallel4_paper8_20261113/private/plot.out 2>&1; echo "plot_exit=$?"
  echo "--- build report ---"
  $PY scripts/report_m3_parallel4_paper8_20261113.py; echo "report_exit=$?"
  echo "=== postrun done $(date -Is) ==="
} >> "$OUT" 2>&1
