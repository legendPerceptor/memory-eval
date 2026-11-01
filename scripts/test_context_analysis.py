import json

import pytest

from analyze_context_cache import observed_usage
from render_context_comparison import clustered_interval


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
