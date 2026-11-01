#!/usr/bin/env python3
"""Wait for all registered arms, then run the single common judge and renderer."""
import datetime
import json
import os
from pathlib import Path
import time

import coordinate_context_experiments as runtime

ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / "experiments/m3_memory_design_20261102"
DEADLINE = "2026-11-02T10:03:00+08:00"


def main():
    runtime.EXP = STUDY / "judging"
    (runtime.EXP / "private").mkdir(parents=True, exist_ok=True, mode=0o700)
    runtime.DEADLINE = datetime.datetime.fromisoformat(DEADLINE).timestamp()
    if not os.environ.get("MINIMAX_API_KEY"):
        raise RuntimeError("Source interactive ~/.bashrc first")
    followup = STUDY / "followup/completion.json"
    while time.time() < runtime.DEADLINE - 3600:
        state = json.loads(followup.read_text()) if followup.exists() else {"status": "not_started"}
        if state["status"] == "finished":
            break
        runtime.status(status="waiting_followup", followup_status=state["status"])
        if state["status"] in ("needs_review", "driver_failed", "deadline_stopped"):
            return
        time.sleep(60)
    else:
        runtime.status(status="insufficient_judging_time")
        return
    while time.time() < runtime.DEADLINE - 3600:
        quota = runtime.quota()
        if quota and quota["interval_remaining_percent"] >= 15:
            break
        until = quota["window_end_unix"] + 60 if quota else time.time() + 120
        if not runtime.wait_until(until, "joint_judging"):
            return
    else:
        runtime.status(status="insufficient_judging_quota")
        return
    name = "m3_memory_design_joint_judge_20261102"
    python = str(ROOT / "oGMemory/.venv/bin/python")
    env = {**os.environ, "M3_EVAL_ROOT": str(ROOT),
           "M3_EXPERIMENT_DEADLINE": DEADLINE,
           "PYTHONPATH": str(ROOT / "runs/m3_source_fact_index_v2_20261102/_instrumentation"),
           "M3_USAGE_LOG": str(ROOT / "experiments" / name / "model-usage.jsonl")}
    for key in ("M3_AUDIT_VIDEO", "M3_AUDIT_QUESTION_INDEX"):
        env.pop(key, None)
    code = runtime.run_command(
        [python, str(ROOT / "scripts/judge_memory_design.py")],
        runtime.EXP / "private/judge.log", env, "joint_judge")
    judge_status = ROOT / "experiments" / name / "status.json"
    state = json.loads(judge_status.read_text()) if judge_status.exists() else {}
    if code or state.get("status") != "finished":
        runtime.status(status="needs_judge_review", exit_code=code, judge_status=state)
        return
    code = runtime.run_command(
        [python, str(ROOT / "scripts/render_memory_design.py")],
        runtime.EXP / "private/renderer.log", env, "render_comparison")
    runtime.status(status="ready_for_analysis" if code == 0 else "renderer_failed", exit_code=code)


if __name__ == "__main__":
    main()
