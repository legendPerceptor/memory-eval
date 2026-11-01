from summarize_context_costs import summarize


def test_failed_calls_remain_unknown_not_zero_cost():
    result = summarize([
        {"status": "returned", "seconds": 2,
         "usage": {"prompt_tokens": 100, "completion_tokens": 5}},
        {"status": "error", "seconds": 3},
    ])
    assert result["recorded_logical_calls"] == 2
    assert result["unknown_usage_calls"] == 1
    assert result["received_total_tokens"] == 105
    assert result["recorded_api_seconds_sum"] == 5
    assert result["statuses"] == {"returned": 1, "error": 1}
