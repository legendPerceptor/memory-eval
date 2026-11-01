import json

import pytest

from analyze_context_cache import observed_usage
from render_context_comparison import clustered_interval, replication_summary


def test_cached_and_unknown_usage_are_separate(tmp_path):
    path = tmp_path / "calls.jsonl"
    calls = [
        {"process": "qa", "images": 0,
         "usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 60}}},
        {"process": "qa", "images": 0, "status": "error"},
        {"process": "memory", "images": 16,
         "usage": {"prompt_tokens": 1000, "prompt_tokens_details": {"cached_tokens": 0}}},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in calls))
    result = observed_usage([path], "qa")
    assert result["logical_calls"] == 2
    assert result["unknown_usage_calls"] == 1
    assert result["prompt_tokens"] == 100
    assert result["uncached_prompt_tokens"] == 40
    assert result["cache_fraction"] == .6


@pytest.mark.parametrize("cached", [None, 101, -1])
def test_missing_or_invalid_cache_counts_are_not_silently_costed(tmp_path, cached):
    path = tmp_path / "calls.jsonl"
    path.write_text(json.dumps({
        "process": "qa", "images": 0,
        "usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": cached}}}) + "\n")
    with pytest.raises(ValueError):
        observed_usage([path], "qa")


def test_cluster_bootstrap_is_zero_for_identical_scores():
    result = clustered_interval([0] * 5, [12, 11, 11, 14, 17])
    assert result["samples"] == 3125
    assert result["lower"] == result["upper"] == 0


def test_cluster_bootstrap_preserves_constant_per_question_difference():
    sizes = [12, 11, 11, 14, 17]
    result = clustered_interval([n * .5 for n in sizes], sizes)
    assert result["lower"] == result["upper"] == .5
    with pytest.raises(ValueError):
        clustered_interval([1], [0])


def test_repeat_is_not_best_of_and_identical_answer_grading_drift_is_visible():
    grades = [
        {"method": "finalization", "question_id": "q1", "predicted_answer": "desk", "judge_score": 1.},
        {"method": "finalization", "question_id": "q2", "predicted_answer": "left", "judge_score": 0.},
        {"method": "finalization_repeat", "question_id": "q1", "predicted_answer": "desk", "judge_score": .5},
        {"method": "finalization_repeat", "question_id": "q2", "predicted_answer": "right", "judge_score": 1.},
    ]
    result = replication_summary(grades)
    assert result["scores"] == [.5, .75]
    assert result["mean_of_two_run_scores"] == .625
    assert result["identical_predictions_with_different_judge_scores"] == ["q1"]
    assert not result["best_of_selection"]
