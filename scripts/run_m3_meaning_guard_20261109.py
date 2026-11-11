#!/usr/bin/env python3
"""Evaluate ambiguity-aware prompting with a smaller evidence increment."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "accuracy_driver", ROOT / "scripts/run_m3_accuracy_first_20261109.py"
)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
experiment.NAME = "m3_meaning_guard_20261109"
experiment.RUN = ROOT / "runs" / experiment.NAME
experiment.EXP = ROOT / "experiments" / experiment.NAME
experiment.POLICIES = {"meaning_guard": {"incremental_tokens": 2500, "ambiguity_guard": True}}
experiment.driver.RUN = experiment.RUN
experiment.driver.EXP = experiment.EXP

if __name__ == "__main__":
    experiment.main()
