#!/usr/bin/env python3
"""Follow-up queue waits for the initial queue; never overlaps model experiments."""
import datetime
import json
from pathlib import Path
import time

import coordinate_memory_design as queue

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments/m3_memory_design_20261102"


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
        "m3_source_fact_index_20261102",
        "m3_source_fact_index_repeat_20261102",
        "m3_balanced_contemporary_20261102",
    ]
    queue.main()


if __name__ == "__main__":
    main()
