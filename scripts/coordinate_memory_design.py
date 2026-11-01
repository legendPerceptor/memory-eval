#!/usr/bin/env python3
"""Serialize prepared memory-design arms with quota and deadline protection."""
import datetime
import json
import os
from pathlib import Path
import time

import coordinate_context_experiments as runtime

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments/m3_memory_design_20261102"
DEADLINE = "2026-11-02T10:03:00+08:00"
JOBS = [
    "m3_facts_identity_incremental_20261102",
    "m3_facts_only_20261102",
    "m3_facts_identity_20261102",
]
MINIMUM_QUOTA = 40
MINIMUM_QUOTA_BY_JOB = {}


def main():
    runtime.EXP = EXP
    runtime.DEADLINE = datetime.datetime.fromisoformat(DEADLINE).timestamp()
    env = {**os.environ, "M3_EVAL_ROOT": str(ROOT), "M3_EXPERIMENT_DEADLINE": DEADLINE}
    if not env.get("MINIMAX_API_KEY"):
        raise RuntimeError("Load interactive ~/.bashrc without printing credentials")
    runtime.save(EXP / "execution_schedule.json", {
        "jobs": JOBS, "deadline": DEADLINE, "minimum_interval_remaining_percent": MINIMUM_QUOTA,
        "per_job_minimum_interval_remaining_percent": MINIMUM_QUOTA_BY_JOB,
        "note": "First job already launched; wait for it, then serialize remaining prepared arms.",
    })
    for name in JOBS:
        path = ROOT / "experiments" / name / "status.json"
        while True:
            if time.time() > runtime.DEADLINE - 1800:
                runtime.status(status="deadline_stopped", pending=name)
                return
            state = json.loads(path.read_text())
            if state["status"] == "finished":
                break
            if state["status"] == "running":
                runtime.status(status="waiting_existing_job", job=name)
                time.sleep(30)
                continue
            if state["status"] != "prepared":
                runtime.status(status="needs_review", job=name, worker_status=state)
                return
            quota = runtime.quota()
            if not quota:
                runtime.status(status="quota_unavailable", job=name)
                time.sleep(120)
                continue
            if quota["interval_remaining_percent"] < MINIMUM_QUOTA_BY_JOB.get(name, MINIMUM_QUOTA):
                if not runtime.wait_until(quota["window_end_unix"] + 60, name):
                    return
                continue
            code = runtime.run_command([
                str(ROOT / "oGMemory/.venv/bin/python"),
                str(ROOT / "runs" / name / "runner.py"), "--name", name, "--driver",
            ], EXP / "private" / f"{name}-driver.log", env, name)
            if code:
                runtime.status(status="driver_failed", job=name, exit_code=code)
                return
    runtime.status(status="finished", jobs=JOBS)


if __name__ == "__main__":
    main()
