#!/usr/bin/env python3
"""Follow-up queue waits for the initial queue; never overlaps model experiments."""
import datetime
import json
import os
from pathlib import Path
import time

import coordinate_memory_design as queue

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments/m3_memory_design_20261102"


def optional_quota_threshold(method, quota, decisions):
    """Repeat may use measured predecessor consumption, never its score."""
    defaults = 30 if method.startswith("strict_incremental") else 40
    repeat_rules = {"source_index_scene_repeat": ("source_index_scene", 17),
                    "strict_incremental_repeat": ("strict_incremental", 12)}
    if method in repeat_rules and quota and decisions:
        primary, reserve = repeat_rules[method]
        previous = decisions[-1]
        before = previous.get("quota_before")
        if (previous.get("method") == primary and before
                and before["window_end_unix"] == quota["window_end_unix"]
                and before["interval_remaining_percent"] >= quota["interval_remaining_percent"]):
            consumed = before["interval_remaining_percent"] - quota["interval_remaining_percent"]
            return max(30, consumed + reserve)
    return defaults


def main():
    followup = EXP / "followup"
    (followup / "private").mkdir(parents=True, exist_ok=True, mode=0o700)
    deadline = datetime.datetime.fromisoformat(queue.DEADLINE).timestamp()
    while time.time() < deadline - 1800:
        status = json.loads((EXP / "coordinator_status.json").read_text())
        if status["status"] == "finished":
            break
        (followup / "waiting.json").write_text(json.dumps({
            "status": "waiting_initial_queue", "observed_at_unix": time.time(),
            "initial_status": status["status"]}, indent=2) + "\n")
        if status["status"] in ("needs_review", "driver_failed", "deadline_stopped"):
            return
        time.sleep(30)
    else:
        return
    queue.EXP = followup
    queue.JOBS = [
        "m3_source_fact_index_v2_20261102",
        "m3_source_fact_index_v2_repeat_20261102",
        "m3_balanced_contemporary_20261102",
    ]
    queue.MINIMUM_QUOTA = 30
    queue.MINIMUM_QUOTA_BY_JOB = {"m3_balanced_contemporary_20261102": 40}
    queue.main()
    core = json.loads((followup / "coordinator_status.json").read_text())
    if core["status"] != "finished":
        queue.runtime.save(followup / "completion.json", {"status": "needs_review", "core": core})
        return
    optional = [
        ("strict_incremental", "m3_strict_incremental_20261102"),
        ("strict_incremental_repeat", "m3_strict_incremental_repeat_20261102"),
    ]
    decisions = []
    for method, name in optional:
        if method.endswith("_repeat") and (not decisions or decisions[-1]["status"] != "finished"):
            decisions.append({"method": method, "run": name, "status": "not_executed_without_primary"})
            break
        remaining = deadline - time.time()
        quota = queue.runtime.quota()
        minimum = optional_quota_threshold(method, quota, decisions)
        if remaining < 7200 or not quota or quota["interval_remaining_percent"] < minimum:
            decisions.append({"method": method, "run": name, "status": "not_executed_budget",
                              "remaining_seconds": remaining, "quota": quota, "minimum_quota": minimum})
            continue  # Never select optional trials based on their scores.
        env = {**os.environ, "M3_EVAL_ROOT": str(ROOT), "M3_EXPERIMENT_DEADLINE": queue.DEADLINE}
        code = queue.runtime.run_command([
            str(ROOT / "oGMemory/.venv/bin/python"),
            str(ROOT / "runs" / name / "runner.py"), "--name", name, "--driver",
        ], followup / "private" / f"{name}-driver.log", env, name)
        state = json.loads((ROOT / "experiments" / name / "status.json").read_text())
        decisions.append({"method": method, "run": name, "status": state["status"], "exit_code": code,
                          "quota_before": quota, "minimum_quota": minimum})
        if code or state["status"] != "finished":
            break
        path = EXP / "methods.json"
        methods = json.loads(path.read_text())
        methods[method] = name
        queue.runtime.save(path, methods)
    queue.runtime.save(followup / "optional_decisions.json", {
        "minimum_quota_percent": 30, "repeat_rule": "max(30, preceding E observed consumption + 12), same window only",
        "minimum_seconds_remaining": 7200, "decisions": decisions,
        "quality_selection": False})
    queue.runtime.save(followup / "completion.json", {
        "status": "finished", "core_jobs": queue.JOBS, "optional_decisions": decisions})


if __name__ == "__main__":
    main()
