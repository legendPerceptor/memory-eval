#!/usr/bin/env python3
"""Evaluate full initial evidence with smaller subsequent increments."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "accuracy_driver", ROOT / "scripts/run_m3_accuracy_first_20261109.py"
)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
experiment.NAME = "m3_adaptive_incremental_20261109"
experiment.RUN = ROOT / "runs" / experiment.NAME
experiment.EXP = ROOT / "experiments" / experiment.NAME
experiment.POLICIES = {"adaptive_incremental": {"later_incremental_tokens": 2000}}
experiment.driver.RUN = experiment.RUN
experiment.driver.EXP = experiment.EXP

if __name__ == "__main__":
    experiment.main()
