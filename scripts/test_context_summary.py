import json
from pathlib import Path

import pytest

import summarize_context_experiment as summary


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def fixture(tmp_path, monkeypatch, *, judge_error=False, actual=100):
    monkeypatch.setattr(summary, "ROOT", tmp_path)
    name = "test"
    exp, run = tmp_path / "experiments" / name, tmp_path / "runs" / name
    exp.mkdir(parents=True)
    (exp / "manifest.json").write_text(json.dumps({"source_commit": "test"}))
    qa = dict(video_id="study_05", question_id="q", question_index=0, question="where?",
              expected_answer="desk", predicted_answer="desk", error="", finish_reason="done",
              search_history=[], temporal_scope={"policy": "current_clip"},
              search_stats={"wall_seconds": 3, "search_requests": 0,
                            "context_calls": [{"estimated_input_tokens": 120,
                                               "input_limit": 130, "packing_seconds": .1}]})
    write_rows(run / "study_05/m3_qa_results.jsonl", [qa])
    judge = {**qa, "judge_score": 0 if judge_error else 1,
             "judge_error": judge_error, "judge_reasoning": "API error" if judge_error else "yes"}
    write_rows(run / "study_05/judge/m3_eval_results.jsonl", [judge])
    call = dict(process="runner.py", qa_video_id="study_05", qa_question_index="0",
                started_at_unix=10, seconds=2,
                usage={"prompt_tokens": actual, "completion_tokens": 5})
    write_rows(exp / "model-usage.jsonl", [call])
    return name, exp, run


def test_partial_report_preserves_full_denominator(tmp_path, monkeypatch):
    name, exp, _ = fixture(tmp_path, monkeypatch)
    result = summary.summarize(name)
    assert result["mean_score_all_65"] == 1 / 65
    assert result["mean_score_observed_judged"] == 1
    assert not result["complete"]
    assert result["qa_usage"]["prompt_tokens"] == 100
    assert result["input_limit_exceedances"] == []
    assert result["usage_pairing_errors"] == []
    assert (exp / "questions.csv").exists()


def test_judge_failures_and_underestimation_are_not_hidden(tmp_path, monkeypatch):
    name, _, _ = fixture(tmp_path, monkeypatch, judge_error=True, actual=140)
    result = summary.summarize(name)
    assert result["judge_errors"] == 1
    assert result["mean_score_all_65"] == 0
    assert not result["complete"]
    assert result["counter_exceedances"] == [["q", 0, 140, 120]]
    assert result["input_limit_exceedances"] == [["q", 0, 140, 130]]


def test_refuses_grade_for_different_prediction(tmp_path, monkeypatch):
    name, _, run = fixture(tmp_path, monkeypatch)
    path = run / "study_05/judge/m3_eval_results.jsonl"
    grade = summary.rows(path)[0]
    grade["predicted_answer"] = "another"
    write_rows(path, [grade])
    with pytest.raises(AssertionError):
        summary.summarize(name)


def test_partial_line_is_ignored_but_malformed_complete_line_fails(tmp_path):
    path = tmp_path / "rows"
    path.write_text('{"a": 1}\n{"a":')
    assert summary.rows(path) == [{"a": 1}]
    path.write_text('{"a": 1}\n{"a":\n')
    with pytest.raises(json.JSONDecodeError):
        summary.rows(path)


def test_independent_repeat_does_not_count_reference_run_as_failed_overhead(tmp_path, monkeypatch):
    name, exp, _ = fixture(tmp_path, monkeypatch)
    (exp / "manifest.json").write_text(json.dumps({
        "source_commit": "test", "source_from": "reference",
        "attempt_relationship": "independent_replication",
        "independent_replication_of": "reference",
    }))
    result = summary.summarize(name)
    assert result["qa_usage"]["logical_calls"] == 1
    assert result["prior_attempts"] == {}
    assert result["independent_replication_of"] == "reference"
