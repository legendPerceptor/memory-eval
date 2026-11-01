#!/usr/bin/env python3
"""Verify frozen sources, benchmark annotations, index pins and common grading."""
import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

from verify_context_artifacts import verify_annotations

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments/m3_memory_design_20261102"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(name, manifest, cache):
    revision = manifest["source_commit"]
    if revision not in cache:
        archive = subprocess.check_output(["git", "-C", str(ROOT / "oGMemory"), "archive", revision])
        entries = {}
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            for member in tar:
                if member.isfile():
                    entries[member.name] = ("file", hashlib.sha256(tar.extractfile(member).read()).hexdigest())
                elif member.issym():
                    entries[member.name] = ("symlink", member.linkname)
        cache[revision] = hashlib.sha256(archive).hexdigest(), entries
    digest, entries = cache[revision]
    assert digest == manifest["source_archive_sha256"], name
    for relative, (kind, expected) in entries.items():
        path = ROOT / "runs" / name / "_source" / relative
        actual = os.readlink(path) if kind == "symlink" else sha(path)
        assert actual == expected, (name, relative)
    return {"source_commit": revision, "source_entries_verified": len(entries),
            "archive_matches_git": True,
            "runner_sha256_observed_now": sha(ROOT / "runs" / name / "runner.py"),
            "instrumentation_sha256_observed_now": sha(
                ROOT / "runs" / name / "_instrumentation/sitecustomize.py")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-final", action="store_true")
    args = parser.parse_args()
    cohort = json.loads((ROOT / "experiments/m3_flash_medium_20261031/sample_manifest.json").read_text())
    annotation = verify_annotations(Path("/data/m3_data/annotations/robot.json"),
                                    cohort["annotations_sha256"])
    cache, sources, repetitions = {}, {}, {}
    for path in sorted((ROOT / "experiments").glob("m3_*20261102/manifest.json")):
        manifest = json.loads(path.read_text())
        if "source_archive_sha256" not in manifest:
            continue
        name = path.parent.name
        sources[name] = verify_source(name, manifest, cache)
        if manifest.get("attempt_relationship") == "independent_replication":
            origin = json.loads((ROOT / "experiments" / manifest["source_from"] / "manifest.json").read_text())
            for key in ("source_archive_sha256", "qa_context", "cohort", "memory_runs", "account",
                        "model", "reasoning_effort", "thinking", "qa_output_tokens",
                        "max_iterations", "timeout_seconds", "temporal_policy"):
                assert origin[key] == manifest[key], (name, key)
            repetitions[name] = {"source": manifest["source_from"], "model_source_and_config_equal": True}
    index = ROOT / "runs/m3_source_facts_20261102"
    receipt = json.loads((OUT / "fact_index_receipt.json").read_text())
    assert sha(index / "manifest.json") == receipt["manifest_sha256"]
    assert sha(index / "vectors.npy") == receipt["vectors_sha256"]
    reference = {}
    path = ROOT / "experiments/m3_reference_repeat_20261102/manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        for relative, expected in manifest["input_hashes"].items():
            assert sha(ROOT / relative) == expected, relative
        run = ROOT / "runs/m3_reference_repeat_20261102"
        assert sha(run / "_source/reference_text.py") == manifest["controller_sha256"]
        assert sha(run / "runner.py") == manifest["runner_sha256"]
        reference = {"inputs_verified": len(manifest["input_hashes"]), "all_unchanged": True}
    joint = {}
    if args.require_final:
        from judge_memory_design import load_inputs, NAME, rows
        from render_memory_design import verify
        tasks, hashes = load_inputs()
        manifest = json.loads((ROOT / "experiments" / NAME / "manifest.json").read_text())
        grades = rows(ROOT / "runs" / NAME / "grades.jsonl")
        verify(tasks, grades, hashes, manifest)
        frozen_tasks = rows(ROOT / "runs" / NAME / "inputs.jsonl")
        assert tasks == frozen_tasks, "Common judge task ordering or fields changed"
        assert len(grades) == manifest["tasks"]
        joint = {"tasks_verified": len(tasks), "prediction_files_verified": sum(map(len, hashes.values())),
                 "all_predictions_and_tasks_unchanged": True, "judge_errors": 0}
        initial = json.loads((OUT / "input_audit_initial.json").read_text())
        final = json.loads((OUT / "input_audit_final.json").read_text())
        assert initial["databases"] == final["databases"], "Database rows changed"
    result = {
        "observed_at": datetime.datetime.now().astimezone().isoformat(), "source_runs": sources,
        "independent_repetitions": repetitions, "reference_repeat": reference,
        "annotations": annotation, "joint_judging": joint,
        "index_manifest_sha256": receipt["manifest_sha256"], "index_vectors_sha256": receipt["vectors_sha256"],
        "note": "Core archives verified against Git; runner/instrumentation hashes without original pins are current observations only.",
    }
    name = "final_artifact_verification.json" if args.require_final else "source_artifact_verification.json"
    (OUT / name).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"source_runs": len(sources), "reference_repeat": reference, "joint": joint}, indent=2))


if __name__ == "__main__":
    main()
