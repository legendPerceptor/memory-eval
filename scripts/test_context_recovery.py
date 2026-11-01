import json

import pytest

import run_context_experiment as runner


def test_only_successful_matching_answers_and_grades_are_carried(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "COHORT", ["v", "not_started"])
    origin, destination = tmp_path / "old", tmp_path / "new"
    (origin / "v/judge").mkdir(parents=True)
    destination.mkdir()
    good = {"video_id": "v", "question_id": "q1", "question_index": 0,
            "predicted_answer": "desk", "error": "", "finish_reason": "done"}
    second = {**good, "question_id": "q2", "question_index": 1}
    failed = {**good, "question_id": "q3", "question_index": 2,
              "predicted_answer": "", "error": "Token Plan (2056)", "finish_reason": ""}
    qa_path = origin / "v/m3_qa_results.jsonl"
    qa_path.write_text("".join(json.dumps(r) + "\n" for r in [good, second, failed]))
    original_bytes = qa_path.read_bytes()
    grades = [{**good, "judge_error": False, "judge_score": 1},
              {**second, "judge_error": True, "judge_score": 0},
              {**failed, "judge_error": False, "judge_score": 0}]
    (origin / "v/judge/m3_eval_results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in grades))
    receipt = runner.seed_recovery(destination, origin, "old")
    assert len(receipt["reused_answers"]) == 2
    assert len(receipt["reused_grades"]) == 1
    assert qa_path.read_bytes() == original_bytes
    assert [json.loads(s) for s in (destination / "v/m3_qa_results.jsonl").read_text().splitlines()] == [good, second]
    assert [json.loads(s) for s in (destination / "v/judge/m3_eval_results.jsonl").read_text().splitlines()] == grades[:1]
    assert (destination / "not_started/m3_qa_results.jsonl").read_text() == ""


def test_stale_grade_cannot_be_reused_for_another_prediction(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "COHORT", ["v"])
    origin, destination = tmp_path / "old", tmp_path / "new"
    (origin / "v/judge").mkdir(parents=True)
    destination.mkdir()
    row = {"video_id": "v", "question_id": "q", "question_index": 0,
           "predicted_answer": "desk", "error": "", "finish_reason": "done"}
    (origin / "v/m3_qa_results.jsonl").write_text(json.dumps(row) + "\n")
    (origin / "v/judge/m3_eval_results.jsonl").write_text(
        json.dumps({**row, "predicted_answer": "shelf", "judge_error": False}) + "\n")
    with pytest.raises(AssertionError):
        runner.seed_recovery(destination, origin, "old")


def test_algorithmic_failures_are_not_silently_selected_for_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "COHORT", ["v"])
    origin, destination = tmp_path / "old", tmp_path / "new"
    (origin / "v").mkdir(parents=True)
    destination.mkdir()
    (origin / "v/m3_qa_results.jsonl").write_text(json.dumps({
        "question_id": "q", "predicted_answer": "", "error": "loop timeout",
        "finish_reason": "timeout", "question_index": 0}) + "\n")
    with pytest.raises(ValueError, match="limited to quota"):
        runner.seed_recovery(destination, origin, "old")
