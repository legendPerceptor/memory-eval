#!/usr/bin/env python3
"""Blind-judge adaptive and baseline answers together using frozen inputs."""

import importlib.util
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "four_way", ROOT / "scripts/judge_m3_four_way_20261109.py"
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)

comparison.NAME = "m3_adaptive_four_way_20261109"
comparison.RUN = ROOT / "runs" / comparison.NAME
comparison.EXP = ROOT / "experiments" / comparison.NAME
comparison.SOURCES = {
    "baseline": ROOT / "runs/m3_compact_replay_20261109",
    "baseline_repeat": ROOT / "runs/m3_baseline_repeat_20261109/baseline_repeat",
    "m3_agent": ROOT / "experiments/m3_dev_1031_ten_retry_20261108/frozen_baseline/m3_qa",
    "adaptive_incremental": ROOT / "runs/m3_adaptive_incremental_20261109/adaptive_incremental",
}
comparison.driver.RUN = comparison.RUN
comparison.driver.EXP = comparison.EXP
comparison.driver.BASE_RUN = comparison.RUN / "baseline"

if sys.argv[1:] == ["prepare"]:
    comparison.prepare()
    shutil.copyfile(Path(__file__), comparison.RUN / "comparison_launcher.py")
elif sys.argv[1:] == ["run"]:
    comparison.run()
else:
    raise ValueError("Use prepare or run")
