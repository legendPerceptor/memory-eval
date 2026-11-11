#!/usr/bin/env python3
"""Blind-score a completed video without modifying the running experiment status."""

import importlib.util
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "efficiency_driver", ROOT / "scripts/run_m3_qa_efficiency_20261109.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def main():
    policy, video = sys.argv[1:]
    name = os.environ.get("M3_ACCURACY_EXPERIMENT", "m3_accuracy_first_20261109")
    driver.RUN = ROOT / "runs" / name
    parent = ROOT / "experiments" / name
    driver.EXP = parent / f"pilot-{policy}-{video}"
    driver.EXP.mkdir(parents=True, exist_ok=True)
    (driver.EXP / "private").mkdir(exist_ok=True, mode=0o700)
    config = driver.EXP / "private" / f"moderate-{video}.yaml"
    shutil.copyfile(parent / "private" / f"{policy}-{video}.yaml", config)
    config.chmod(0o600)
    status = {"status": "running", "completed": [], "stages": {}}
    driver.judge([video], ["baseline", "m3_agent", policy], status, "pilot")
    rows = driver.rows(driver.EXP / "judgments-pilot.jsonl")
    summary = {}
    for arm in ("baseline", "m3_agent", policy):
        selected = [row for row in rows if row["arm"] == arm]
        summary[arm] = {
            "score": sum(row["judge_score"] for row in selected) / len(selected),
            "questions": len(selected),
            "failures": sum(bool(row.get("error")) or not row.get("predicted_answer")
                            for row in selected),
        }
    driver.save(driver.EXP / "summary.json", summary)


if __name__ == "__main__":
    main()
