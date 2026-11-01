from copy import deepcopy

import pytest

from recover_context_joint_judges import recover_records


def tasks():
    return [
        {"method": "a", "question_id": f"q{i}", "question": "Where?",
         "expected_answer": "desk", "predicted_answer": "desk", "error": ""}
        for i in range(3)
    ]


def grade(task, *, error=False, score=.5):
    return {**task, "judge_score": score, "judge_error": error, "judge_reasoning": "previous",
            "judge_seconds": 1.}


def test_only_failed_and_missing_are_called_and_successes_are_identical():
    source = tasks()
    existing = [grade(source[0]), grade(source[1], error=True, score=0.)]
    before = deepcopy(existing)
    called, checkpoints = [], []

    def judge(task):
        called.append(task["question_id"])
        return 1., "recovered", False

    result, audit = recover_records(source, existing, judge,
                                    lambda records, attempts: checkpoints.append(deepcopy(records)))
    assert called == ["q1", "q2"]
    assert existing == before and result[0] == before[0]
    assert audit["preserved_successful_grades"] == 1
    assert audit["remaining_missing"] == audit["remaining_judge_errors"] == 0
    assert checkpoints[0] == existing


def test_quota_stops_and_a_later_pass_does_not_regrade_previous_successes():
    source = tasks()
    result, audit = recover_records(
        source, [grade(source[0])],
        lambda task: (0., "已达到 Token Plan 用量上限 (2056)", True),
        lambda *args: None)
    assert audit["stopped"] and audit["remaining_missing"] == 1
    called = []
    recovered, _ = recover_records(
        source, result, lambda t: (called.append(t["question_id"]) or .5, "ok", False),
        lambda *args: None)
    assert called == ["q1", "q2"]
    assert recovered[0] == result[0]


def test_changed_prediction_is_rejected_before_judge_call():
    source = tasks()
    changed = grade(source[0])
    changed["predicted_answer"] = "shelf"
    with pytest.raises(AssertionError):
        recover_records(source, [changed], lambda _: pytest.fail("no call"), lambda *args: None)


def test_deadline_stop_preserves_every_existing_success_even_later_in_order():
    source = tasks()
    existing = [grade(source[2])]
    result, audit = recover_records(
        source, existing, lambda _: pytest.fail("no call"), lambda *args: None,
        should_stop=lambda: True)
    assert result == existing
    assert audit["remaining_missing"] == 2 and audit["stopped"]


def test_invalid_qa_gets_zero_without_judge_request():
    source = tasks()[:1]
    source[0].update(predicted_answer="", error="protocol failure")
    result, audit = recover_records(source, [], lambda _: pytest.fail("no call"), lambda *args: None)
    assert result[0]["judge_score"] == 0 and not result[0]["judge_error"]
    assert not audit["remaining_missing"]
