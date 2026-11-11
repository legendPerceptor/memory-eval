#!/usr/bin/env python3
"""Repeat the unmodified QA policy to measure run-to-run variability."""

import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "accuracy_driver", ROOT / "scripts/run_m3_accuracy_first_20261109.py"
)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
experiment.RUN = ROOT / "runs/m3_baseline_repeat_20261109"
experiment.EXP = ROOT / "experiments/m3_baseline_repeat_20261109"
experiment.POLICIES = {"baseline_repeat": {}}
experiment.VIDEOS = ["kitchen_01", "bedroom_04", "bedroom_05", "gym_03", "study_23"]
experiment.driver.RUN = experiment.RUN
experiment.driver.EXP = experiment.EXP

if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        experiment.prepare()
        manifest_path = experiment.EXP / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for name in ("control_adapter.py", "compact_person.py"):
            source = experiment.BASE_RUN / "_source/m3" / name
            target = experiment.RUN / "_source/m3" / name
            shutil.copyfile(source, target)
            manifest["code_sha256"][name] = experiment.driver.digest(target)
        experiment.driver.save(manifest_path, manifest)
    else:
        experiment.main()
