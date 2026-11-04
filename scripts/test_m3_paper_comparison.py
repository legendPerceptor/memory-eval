import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "paper_comparison", Path(__file__).with_name("build_m3_paper_comparison.py")
)
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)


def row(video, index, score, error=""):
    return {
        "video_id": video,
        "question_id": str(index),
        "judge_score": score,
        "judge_error": False,
        "error": error,
        "predicted_answer": "" if error else "answer",
    }


def test_question_weighting_and_failure_denominator():
    records = [row("a", i, 1) for i in range(2)] + [row("b", i, 0) for i in range(8)]
    records[-1] = row("b", 7, 0, "timeout")
    result = paper.metrics(records)
    assert result["mean_score"] == 0.2  # Not the 0.5 mean of two video means.
    assert result["questions"] == 10 and result["valid_answers"] == 9


def test_invalid_grade_or_duplicate_cannot_enter_paper():
    with pytest.raises(ValueError):
        paper.metrics([row("a", 0, 1, "timeout")])
    with pytest.raises(ValueError):
        paper.metrics([row("a", 0, 0.7)])
    with pytest.raises(ValueError):
        paper.metrics([row("a", 0, 1), row("a", 0, 1)])
    with pytest.raises(ValueError):
        paper.metrics([{**row("a", 0, 1), "judge_error": True}])


def test_frozen_final_judge_and_resource_reconciliation():
    data = paper.build_data()
    a, b = data["methods"]["ogmemory"], data["methods"]["m3_agent"]
    assert data["questions_per_method"] == 97 and len(data["cohort"]) == 8
    assert a["mean_score"] == pytest.approx(55.5 / 97)
    assert b["mean_score"] == pytest.approx(42.5 / 97)
    assert (a["valid_answers"], b["valid_answers"]) == (95, 97)
    assert a["usage"]["qa"]["prompt_tokens"] == 4039490
    assert b["usage"]["qa"]["prompt_tokens"] == 733810
    assert b["usage"]["memory_plus_qa"]["unknown_usage_calls"] == 1
    for method in (a, b):
        assert sum(method["stage_share"].values()) == pytest.approx(1)
        assert sum(
            method["time_seconds"][key] for key in paper.CATEGORIES
        ) == pytest.approx(method["time_seconds"]["allocated_total"])
    assert (
        a["time_seconds"]["shared_perception"] == b["time_seconds"]["shared_perception"]
    )


def test_table_two_uses_one_decimal_percent_and_percentage_point_difference(
    tmp_path, monkeypatch
):
    data = json.loads(
        (
            paper.ROOT / "experiments/m3_paper_paired8_20261103/comparison.json"
        ).read_text()
    )
    monkeypatch.setattr(paper, "ROOT", tmp_path)
    (tmp_path / "latex").mkdir()
    paper.latex(data)
    text = (tmp_path / "latex/paired8_results.tex").read_text()
    table = text.split(r"\end{table}", 1)[0]
    assert r"题目加权合计 & 97 & 57.2\% & 43.8\% & +13.4\\" in table
    assert r"\code{study_23} & 14 & 67.9\% & 32.1\% & +35.7\\" in table
    assert r"\code{kitchen_23} & 12 & 54.2\% & 62.5\% & -8.3\\" in table
    assert "分差（百分点）" in table
    assert "0.5722" not in table
