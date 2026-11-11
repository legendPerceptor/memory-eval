#!/usr/bin/env python3
"""Evaluate a prompt-only early-answer policy on frozen five-video memory."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "accuracy_driver", ROOT / "scripts/run_m3_accuracy_first_20261109.py"
)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
experiment.NAME = "m3_early_answer_20261109"
experiment.RUN = ROOT / "runs" / experiment.NAME
experiment.EXP = ROOT / "experiments" / experiment.NAME
experiment.POLICIES = {"early_answer": {"early_answer": True}}
experiment.driver.RUN = experiment.RUN
experiment.driver.EXP = experiment.EXP

if __name__ == "__main__":
    experiment.main()
