from copy import deepcopy
import json

import pytest

from judge_memory_design import load_inputs, VIDEOS
from render_memory_design import verify


def fixture(tmp_path):
    methods = {"balanced_prior": "old", "facts": "new"}
    for name in methods.values():
        exp = tmp_path / "experiments" / name
        exp.mkdir(parents=True)
        (exp / "status.json").write_text('{"status":"finished"}')
        for video, count in zip(VIDEOS, (12, 11, 11, 14, 17)):
            path = tmp_path / "runs" / name / video
            path.mkdir(parents=True)
            records = [{
                "video_id": video, "question_id": f"{video}_{i}",
                "question_index": i, "question": f"Where is item {i}?",
                "expected_answer": "shelf", "predicted_answer": "shelf", "error": None,
                "finish_reason": "done", "type": "location",
                "temporal_scope": {"policy": "current_clip", "resolved_before_clip": 3,
                                   "effective_before_sec": 120.167},
            } for i in range(count)]
            (path / "m3_qa_results.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in records))
    return methods


def test_common_judge_inputs_are_complete_deterministic_and_shuffled(tmp_path):
    methods = fixture(tmp_path)
    tasks, hashes = load_inputs(tmp_path, methods)
    assert len(tasks) == 130 and len(hashes) == 2
    assert tasks == load_inputs(tmp_path, methods)[0]
    assert len({t["method"] for t in tasks[:10]}) == 2


@pytest.mark.parametrize("change", ["missing", "duplicate", "scope", "answer", "unfinished"])
def test_common_judge_rejects_cohort_or_scope_changes(tmp_path, change):
    methods = fixture(tmp_path)
    path = tmp_path / "runs/new/study_05/m3_qa_results.jsonl"
    records = [json.loads(s) for s in path.read_text().splitlines()]
    if change == "missing":
        records.pop()
    elif change == "duplicate":
        records[-1] = records[0]
    elif change == "scope":
        records[0]["temporal_scope"]["effective_before_sec"] = 999
    elif change == "answer":
        records[0]["expected_answer"] = "different gold"
    elif change == "unfinished":
        (tmp_path / "experiments/new/status.json").write_text('{"status":"running"}')
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError):
        load_inputs(tmp_path, methods)


def test_failed_qa_is_retained_not_removed(tmp_path):
    methods = fixture(tmp_path)
    path = tmp_path / "runs/new/study_05/m3_qa_results.jsonl"
    records = [json.loads(s) for s in path.read_text().splitlines()]
    records[0].update(error="protocol failure", predicted_answer=None)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    tasks, _ = load_inputs(tmp_path, methods)
    assert len(tasks) == 130 and sum(bool(t["error"]) for t in tasks) == 1


@pytest.mark.parametrize("change", ["missing", "duplicate", "predicted", "judge_error", "hash", "score"])
def test_final_verifier_rejects_incomplete_or_changed_judging(tmp_path, change):
    methods = fixture(tmp_path)
    tasks, hashes = load_inputs(tmp_path, methods)
    grades = [{**t, "judge_score": 1., "judge_error": False} for t in tasks]
    manifest = {"source_prediction_hashes": deepcopy(hashes)}
    assert len(verify(tasks, grades, hashes, manifest)) == 130
    if change == "missing":
        grades.pop()
    elif change == "duplicate":
        grades[-1] = grades[0]
    elif change == "predicted":
        grades[0]["predicted_answer"] = "changed"
    elif change == "judge_error":
        grades[0]["judge_error"] = True
    elif change == "hash":
        manifest["source_prediction_hashes"] = {}
    elif change == "score":
        grades[0]["judge_score"] = .7
    with pytest.raises(ValueError):
        verify(tasks, grades, hashes, manifest)
