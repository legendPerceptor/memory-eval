#!/usr/bin/env python3
"""Run the lossless columnar evidence comparison on the frozen five videos."""

import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "accuracy_driver", ROOT / "scripts/run_m3_accuracy_first_20261109.py"
)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
experiment.RUN = ROOT / "runs/m3_columnar_20261109"
experiment.EXP = ROOT / "experiments/m3_columnar_20261109"
experiment.POLICIES = {"columnar": {"evidence_format": "columnar"}}
experiment.driver.RUN = experiment.RUN
experiment.driver.EXP = experiment.EXP


if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        experiment.prepare()
        source = ROOT / "oGMemory/m3/context_budget.py"
        target = experiment.RUN / "_source/m3/context_budget.py"
        shutil.copyfile(source, target)
        manifest_path = experiment.EXP / "manifest.json"
        import json
        manifest = json.loads(manifest_path.read_text())
        manifest["code_sha256"]["context_budget.py"] = experiment.driver.digest(target)
        experiment.driver.save(manifest_path, manifest)
    else:
        experiment.main()
