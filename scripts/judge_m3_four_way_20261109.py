#!/usr/bin/env python3
"""Freeze and jointly blind-judge four completed five-video QA runs."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
NAME = "m3_four_way_20261109"
RUN = ROOT / "runs" / NAME
EXP = ROOT / "experiments" / NAME
VIDEOS = ["bedroom_04", "kitchen_01", "bedroom_05", "gym_03", "study_23"]
SOURCES = {
    "baseline": ROOT / "runs/m3_compact_replay_20261109",
    "baseline_repeat": ROOT / "runs/m3_baseline_repeat_20261109/baseline_repeat",
    "m3_agent": ROOT / "experiments/m3_dev_1031_ten_retry_20261108/frozen_baseline/m3_qa",
    "incremental_2500": ROOT / "runs/m3_accuracy_first_20261109/incremental_2500",
}
DEADLINE = time.mktime(time.strptime("2026-11-09 23:00:00", "%Y-%m-%d %H:%M:%S"))
spec = importlib.util.spec_from_file_location(
    "efficiency_driver", ROOT / "scripts/run_m3_qa_efficiency_20261109.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
driver.RUN = RUN
driver.EXP = EXP
driver.BASE_RUN = RUN / "baseline"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    if RUN.exists() or EXP.exists():
        raise FileExistsError(NAME)
    RUN.mkdir(parents=True)
    (EXP / "private").mkdir(parents=True, mode=0o700)
    shutil.copytree(ROOT / "runs/m3_accuracy_first_20261109/_source", RUN / "_source",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    config = EXP / "private" / f"moderate-{VIDEOS[0]}.yaml"
    shutil.copyfile(ROOT / "experiments/m3_accuracy_first_20261109/private"
                    / f"person_only-{VIDEOS[0]}.yaml", config)
    config.chmod(0o600)
    hashes = {}
    for video in VIDEOS:
        reference = None
        for arm, source in SOURCES.items():
            original = source / video / "m3_qa_results.jsonl"
            actual = driver.latest(original)
            fields = {key: (row.get("question"), row.get("expected_answer"))
                      for key, row in actual.items()}
            if reference is None:
                reference = fields
            elif fields != reference:
                raise ValueError(f"Question mismatch: {arm}/{video}")
            target = RUN / arm / video / "m3_qa_results.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            hashes[f"{arm}/{video}"] = digest(target)
    shutil.copyfile(Path(__file__), RUN / "driver.py")
    driver.save(EXP / "manifest.json", {"inputs_sha256": hashes, "videos": VIDEOS,
                                        "arms": list(SOURCES), "deadline": "2026-11-09 23:00 CST"})
    driver.save(EXP / "status.json", {"status": "prepared", "completed": [], "stages": {}})


def run():
    manifest = json.loads((EXP / "manifest.json").read_text())
    for name, expected in manifest["inputs_sha256"].items():
        if digest(RUN / name / "m3_qa_results.jsonl") != expected:
            raise ValueError(f"Frozen answers changed: {name}")
    status = json.loads((EXP / "status.json").read_text())
    try:
        if time.time() >= DEADLINE:
            status.update(status="deadline_reached", current=None)
            return
        driver.judge(VIDEOS, list(SOURCES), status, "four_way")
        rows = driver.rows(EXP / "judgments-four_way.jsonl")
        summary = {}
        for arm in SOURCES:
            values = [row["judge_score"] for row in rows if row["arm"] == arm]
            summary[arm] = {
                "score": sum(values) / len(values),
                "questions": len(values),
                "by_video": {
                    video: sum(row["judge_score"] for row in rows
                               if row["arm"] == arm and row["video_id"] == video)
                    / sum(1 for row in rows if row["arm"] == arm and row["video_id"] == video)
                    for video in VIDEOS
                },
            }
        driver.save(EXP / "summary.json", summary)
        status.update(status="completed", current=None)
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        driver.save(EXP / "status.json", status)


if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        prepare()
    elif sys.argv[1:] == ["run"]:
        run()
    else:
        raise ValueError("Use prepare or run")
