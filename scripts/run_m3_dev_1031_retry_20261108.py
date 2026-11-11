#!/usr/bin/env python3
"""Isolated repeat of the frozen m3_dev_1031 ten-video comparison."""

import importlib.util
import json
from pathlib import Path
import sys

import yaml

ROOT = Path("/home/yuanjian/Development/memory-eval")
NAME = "m3_dev_1031_ten_retry_20261108"
spec = importlib.util.spec_from_file_location(
    "comparison", ROOT / "scripts/run_m3_dev_1031_ten_compare.py"
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
base = comparison.base
values = {
    "NAME": NAME,
    "RUN": ROOT / "runs" / NAME,
    "EXP": ROOT / "experiments" / NAME,
    "ACCOUNT": "m3-dev-1031-ten-retry-20261108",
    "DB": "ogmem_m3_dev_1031_ten_retry_20261108",
}
values["SOURCE"] = values["RUN"] / "_source/oGMemory"
values["CONFIG"] = values["EXP"] / "private/config.yaml"
for key, value in values.items():
    setattr(base, key, value)
    setattr(base.h, key, value)
base.__file__ = str(Path(__file__).resolve())


def configure():
    config = yaml.safe_load(base.CONFIG.read_text())
    config["llm"]["max_tokens"] = 16384
    for role in ("episodic_llm", "qa_llm"):
        config["m3"][role]["max_tokens"] = 16384
    base.CONFIG.write_text(yaml.safe_dump(config, allow_unicode=True))
    (base.EXP / "config.redacted.yaml").write_text(
        yaml.safe_dump(base.h.redact(config), allow_unicode=True)
    )
    manifest_path = base.EXP / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config_sha256"] = base.h.digest(base.CONFIG)
    manifest["driver_sha256"] = base.h.digest(base.RUN / "driver.py")
    manifest["memory_and_qa_max_tokens"] = 16384
    manifest["version_label"] = "m3_dev_1031"
    manifest["comparison_caveat"] = (
        "The token limit, code, concurrency and execution time differ from the "
        "parallel control; this is not a pure parallelism ablation."
    )
    base.h.save(manifest_path, manifest)


def prepare():
    comparison.prepare()
    configure()


base.prepare = prepare
if __name__ == "__main__":
    if sys.argv[1:] == ["configure"]:
        configure()
    else:
        sys.exit(base.main())
