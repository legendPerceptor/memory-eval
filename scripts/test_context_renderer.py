"""Synthetic rendering fixtures live only in pytest tmp directories, never runs/."""
import hashlib
import json
import csv

import pytest

import render_context_comparison as render


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def prepare(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "ROOT", tmp_path)
    (tmp_path / "latex").mkdir()
    out = tmp_path / "experiments/m3_context_study_20261101"
    out.mkdir(parents=True)
    sizes = dict(zip(render.VIDEOS, [12, 11, 11, 14, 17]))
    inputs, grades, hashes = [], [], {}
    usage = {"logical_calls": 65, "prompt_tokens": 10000, "peak_prompt_tokens": 1000}
    for method in render.LABELS:
        files = {}
        for video in render.VIDEOS:
            tasks = [{
                "method": method, "video_id": video, "question_id": f"{video}_{i}",
                "question_index": i, "question": "Where is the cup?",
                "expected_answer": "desk", "predicted_answer": f"{method}: desk",
                "error": "", "finish_reason": "done", "type": [],
            } for i in range(sizes[video])]
            path = tmp_path / "fixtures" / method / f"{video}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r) + "\n" for r in tasks))
            files[str(path.relative_to(tmp_path))] = hashlib.sha256(path.read_bytes()).hexdigest()
            inputs.extend(tasks)
            score = 1. if method == "finalization_repeat" else .5
            grades.extend({**r, "judge_score": score, "judge_error": False,
                           "judge_reasoning": "synthetic unit fixture"} for r in tasks)
        hashes[method] = files
        if method in render.ARMS:
            name = render.ARMS[method]
            write_json(tmp_path / "experiments" / name / "metrics.json", {
                "complete": True, "historical_scope_audit_available": True,
                "usage_pairing_errors": [], "input_limit_exceedances": [],
                "temporal_scope_mismatches": [], "model_setting_deviations": [],
                "tokenizer_hash_mismatches": [], "counter_exceedances": [],
                "mean_score_all_65": score, "qa_usage": usage,
                "finish_reasons": {"done": 65}, "finalization_reasons": {},
                "packing_seconds": {}, "question_wall_seconds": {"max": 10.},
                "source_commit": "synthetic",
            })
            manifest = {
                "source_archive_sha256": "synthetic", "qa_context": {},
                "cohort": render.VIDEOS, "memory_runs": {},
                "model": "synthetic", "reasoning_effort": "medium", "thinking": "adaptive",
                "qa_output_tokens": 8192, "max_iterations": 6,
                "timeout_seconds": 180, "temporal_policy": "current_clip",
            }
            if method == "finalization_repeat":
                manifest["independent_replication_of"] = render.ARMS["finalization"]
            write_json(tmp_path / "experiments" / name / "manifest.json", manifest)
    write_json(tmp_path / "experiments/m3_end_to_end_comparison_20261101/metrics.json", {
        "methods": {name: {"mean_score": .5, "qa_usage": {**usage, "prompt_tokens": 100000}}
                    for name in ["ogmemory_recovered", "m3_agent_reference_text"]}})
    joint = tmp_path / "runs/m3_context_joint_judge_20261101"
    joint.mkdir(parents=True)
    (joint / "inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inputs))
    (joint / "grades.jsonl").write_text("".join(json.dumps(r) + "\n" for r in grades))
    write_json(tmp_path / "experiments/m3_context_joint_judge_20261101/manifest.json",
               {"tasks": len(inputs), "source_prediction_hashes": hashes})
    return out, joint


def test_full_seven_condition_report_and_repeat_mean(tmp_path, monkeypatch):
    output, _ = prepare(tmp_path, monkeypatch)
    render.main()
    result = json.loads((output / "comparison.json").read_text())
    assert len(result["methods"]) == 7
    assert result["replication"]["scores"] == [.5, 1.]
    assert result["replication"]["mean_of_two_run_scores"] == .75
    assert not result["replication"]["best_of_selection"]
    tex = (tmp_path / "latex/context_optimization_results.tex").read_text()
    assert "CONTEXT_REPORT_STATUS: final" in tex
    assert "455" in tex and "B独立重复" in tex
    with (output / "joint_questions.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 455
    assert {r["method"] for r in records} == set(render.LABELS)


def test_unresolved_judge_error_cannot_publish_final_report(tmp_path, monkeypatch):
    output, joint = prepare(tmp_path, monkeypatch)
    grades = [json.loads(s) for s in (joint / "grades.jsonl").read_text().splitlines()]
    grades[0]["judge_error"] = True
    (joint / "grades.jsonl").write_text("".join(json.dumps(r) + "\n" for r in grades))
    with pytest.raises(AssertionError, match="incomplete"):
        render.main()
    assert not (output / "comparison.json").exists()
    assert not (tmp_path / "latex/context_optimization_results.tex").exists()
