#!/usr/bin/env python3
"""Restart the ten-video oGMemory run with its audited parallel source."""

import importlib.util
import os
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(os.environ.get("M3_EVAL_ROOT", Path.cwd())).resolve()
os.environ["M3_EVAL_ROOT"] = str(ROOT)
NAME = "m3_parallel_8_restart_20261103"
COMMIT = "eceb96ff2958e72b373a7c3ea280d46827981351"
BASE = ROOT / "runs/m3_parallel_ogonly_20261103/driver.py"
spec = importlib.util.spec_from_file_location("parallel_base", BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.NAME = NAME
base.RUN = ROOT / "runs" / NAME
base.EXP = ROOT / "experiments" / NAME
base.SOURCE = base.RUN / "_source/oGMemory"
base.CONFIG = base.EXP / "private/config.yaml"
base.ACCOUNT = "m3-parallel-8-restart-20261103"
base.DB = "ogmem_m3_parallel_8_restart_20261103"
base.__file__ = __file__
for key in ("NAME", "RUN", "EXP", "SOURCE", "CONFIG", "ACCOUNT", "DB"):
    setattr(base.h, key, getattr(base, key))


def archive(repo, target):
    target.mkdir(parents=True)
    archive_path = target.parent / (target.name + ".tar")
    with archive_path.open("wb") as output:
        subprocess.run(["git", "-C", str(repo), "archive", COMMIT], stdout=output, check=True)
    with tarfile.open(archive_path) as bundle:
        bundle.extractall(target, filter="data")
    return {"commit": COMMIT, "archive_sha256": base.h.digest(archive_path)}


base.h.archive = archive

if __name__ == "__main__":
    raise SystemExit(base.main())
