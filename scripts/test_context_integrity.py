import hashlib
import json

import pytest

from verify_context_artifacts import verify_joint, verify_annotations


def fixture(tmp_path):
    run = tmp_path / "runs/m3_context_joint_judge_20261101"
    exp = tmp_path / "experiments/m3_context_joint_judge_20261101"
    run.mkdir(parents=True)
    exp.mkdir(parents=True)
    prediction = tmp_path / "prediction.jsonl"
    prediction.write_text('{"predicted_answer":"desk"}\n')
    task = {"method": "a", "question_id": "q", "predicted_answer": "desk",
            "question": "Where?", "expected_answer": "desk", "error": ""}
    (run / "inputs.jsonl").write_text(json.dumps(task) + "\n")
    (exp / "manifest.json").write_text(json.dumps({
        "tasks": 1, "source_prediction_hashes": {"a": {
            "prediction.jsonl": hashlib.sha256(prediction.read_bytes()).hexdigest()}}}))
    return [{**task, "judge_score": 1., "judge_error": False}]


def test_verifies_actual_prediction_files_and_frozen_task_fields(tmp_path):
    grades = fixture(tmp_path)
    result = verify_joint(tmp_path, grades)
    assert result["tasks_verified"] == 1
    assert result["prediction_files_verified"] == 1
    assert result["all_predictions_and_task_fields_unchanged"]


def test_changed_answer_in_grade_is_rejected(tmp_path):
    grades = fixture(tmp_path)
    grades[0]["predicted_answer"] = "shelf"
    with pytest.raises(AssertionError):
        verify_joint(tmp_path, grades)


def test_changed_source_file_is_rejected(tmp_path):
    grades = fixture(tmp_path)
    (tmp_path / "prediction.jsonl").write_text('{"predicted_answer":"shelf"}\n')
    with pytest.raises(AssertionError):
        verify_joint(tmp_path, grades)


def test_annotation_audit_uses_original_canonical_json_not_raw_format(tmp_path):
    path = tmp_path / "annotations.json"
    data = {"z": ["问答", 1], "a": {"value": True}}
    expected = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    path.write_text(json.dumps(data, indent=4, ensure_ascii=True))
    result = verify_annotations(path, expected)
    assert result["raw_file_sha256"] != expected
    assert result["matches_original_sampling_manifest"]
    data["z"].reverse()  # List order is part of the pinned annotation content.
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        verify_annotations(path, expected)
