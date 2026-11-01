#!/usr/bin/env python3
"""Verify frozen source trees and (when available) every joint-judge input."""
import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent.parent


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def verify_joint(root, grades):
    run = root / "runs/m3_context_joint_judge_20261101"
    manifest = json.loads((root / "experiments/m3_context_joint_judge_20261101/manifest.json").read_text())
    tasks = rows(run / "inputs.jsonl")
    keyed = {(r["method"], r["question_id"]): r for r in tasks}
    assert len(keyed) == len(tasks) == manifest["tasks"]
    assert len(grades) == len(tasks)
    assert len({(r["method"], r["question_id"]) for r in grades}) == len(grades)
    checked = 0
    for files in manifest["source_prediction_hashes"].values():
        for relative, expected in files.items():
            assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
            checked += 1
    for grade in grades:
        task = keyed[(grade["method"], grade["question_id"])]
        assert all(grade.get(k) == value for k, value in task.items()), (grade["method"], grade["question_id"])
        assert grade["judge_score"] in (0., .5, 1.)
    return {"tasks_verified": len(tasks), "prediction_files_verified": checked,
            "all_predictions_and_task_fields_unchanged": True,
            "judge_errors": sum(bool(r["judge_error"]) for r in grades)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-final", action="store_true")
    args = parser.parse_args()
    from judge_context_comparison import ARMS

    archives = {}
    verified = {}
    for name in ARMS.values():
        manifest = json.loads((ROOT / "experiments" / name / "manifest.json").read_text())
        revision = manifest["source_commit"]
        if revision not in archives:
            archive = subprocess.check_output(["git", "-C", str(ROOT / "oGMemory"),
                                               "archive", revision])
            expected = {}
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
                for member in tar:
                    if member.isfile():
                        expected[member.name] = ("file", hashlib.sha256(tar.extractfile(member).read()).hexdigest())
                    elif member.issym():
                        expected[member.name] = ("symlink", member.linkname)
            archives[revision] = (hashlib.sha256(archive).hexdigest(), expected)
        digest, expected = archives[revision]
        assert digest == manifest["source_archive_sha256"], name
        for relative, (kind, value) in expected.items():
            path = ROOT / "runs" / name / "_source" / relative
            actual = os.readlink(path) if kind == "symlink" else hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual == value, (name, relative)
        run = ROOT / "runs" / name
        verified[name] = {
            "source_commit": revision, "tracked_source_entries_verified": len(expected),
            "matches_recorded_archive_and_git_revision": True,
            "runner_sha256_observed_now": hashlib.sha256((run / "runner.py").read_bytes()).hexdigest(),
            "instrumentation_sha256_observed_now": hashlib.sha256(
                (run / "_instrumentation/sitecustomize.py").read_bytes()).hexdigest(),
        }
    joint = {}
    path = ROOT / "runs/m3_context_joint_judge_20261101/grades_recovered.jsonl"
    if not path.exists():
        path = path.with_name("grades.jsonl")
    if path.exists():
        joint = verify_joint(ROOT, rows(path))
    if args.require_final:
        assert joint and not joint["judge_errors"], "Final joint judging not verified"
        for name in ARMS.values():
            metrics = json.loads((ROOT / "experiments" / name / "metrics.json").read_text())
            assert metrics["complete"]
            for field in ("input_limit_exceedances", "usage_pairing_errors",
                          "tokenizer_hash_mismatches", "temporal_scope_mismatches",
                          "model_setting_deviations"):
                assert not metrics[field], (name, field)
    result = {
        "observed_at": datetime.datetime.now().astimezone().isoformat(),
        "source_runs": verified, "joint_judging": joint,
        "note": "Core frozen files are checked against Git archives. Outer runner/instrumentation hashes are observations now, not retrospective proof of historical immutability.",
    }
    name = "final_artifact_verification.json" if args.require_final else "source_artifact_verification.json"
    (ROOT / "experiments/m3_context_study_20261101" / name).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"runs_verified": len(verified), "joint_judging": joint}, indent=2))


if __name__ == "__main__":
    main()
