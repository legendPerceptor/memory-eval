#!/usr/bin/env python3
"""Independently repeat the adaptive incremental five-video experiment."""

import importlib.util
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "accuracy_driver", ROOT / "scripts/run_m3_accuracy_first_20261109.py"
)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
experiment.NAME = "m3_adaptive_repeat_20261109"
experiment.RUN = ROOT / "runs" / experiment.NAME
experiment.EXP = ROOT / "experiments" / experiment.NAME
experiment.POLICIES = {"adaptive_incremental": {"later_incremental_tokens": 2000}}
experiment.driver.RUN = experiment.RUN
experiment.driver.EXP = experiment.EXP

if __name__ == "__main__":
    experiment.main()
    if sys.argv[1:] == ["prepare"]:
        shutil.copyfile(Path(__file__), experiment.RUN / "comparison_launcher.py")
