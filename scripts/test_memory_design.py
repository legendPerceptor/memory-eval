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


def test_optional_repeat_quota_rule_uses_consumption_not_quality():
    from continue_memory_design import optional_quota_threshold
    previous = [{"method": "source_index_scene", "status": "finished", "quota_before": {
        "window_end_unix": 100, "interval_remaining_percent": 56}}]
    current = {"window_end_unix": 100, "interval_remaining_percent": 37}
    assert optional_quota_threshold("source_index_scene_repeat", current, previous) == 36
    previous[0]["score"] = 0  # Even a low score does not cancel a repeat.
    assert optional_quota_threshold("source_index_scene_repeat", current, previous) == 36
    assert optional_quota_threshold("incremental_only", current, previous) == 40
    assert optional_quota_threshold("source_index_scene_repeat", {**current, "window_end_unix": 200}, previous) == 40
    assert optional_quota_threshold("source_index_scene_repeat", None, previous) == 40
    previous[0]["method"] = "strict_incremental"
    assert optional_quota_threshold("strict_incremental", current, previous) == 30
    assert optional_quota_threshold("strict_incremental_repeat", current, previous) == 31


def test_replay_embedding_cache_preserves_order_and_does_not_duplicate_queries():
    from replay_source_fact_retrieval import CachedEmbedding
    class Embedder:
        def __init__(self):
            self.calls = []
        def embed_texts(self, texts):
            self.calls.append(texts)
            return [[len(t), 1.] for t in texts]
    inner = Embedder()
    cached = CachedEmbedding(inner)
    assert cached.embed_texts(["one", "two", "one"]) == [[3, 1.], [3, 1.], [3, 1.]]
    assert cached.embed_texts(["two", "three"]) == [[3, 1.], [5, 1.]]
    assert inner.calls == [["one", "two"], ["three"]]
    assert cached.new_queries == 3


def test_replay_embedding_cache_rejects_missing_vectors():
    from types import SimpleNamespace
    from replay_source_fact_retrieval import CachedEmbedding
    with pytest.raises(ValueError):
        CachedEmbedding(SimpleNamespace(embed_texts=lambda texts: [])).embed_texts(["one"])


def test_tradeoff_plot_refuses_incomplete_or_unknown_usage():
    from plot_memory_design import validate
    row = {"questions": 65, "mean_score": .5,
           "qa_usage": {"prompt_tokens": 100, "uncached_prompt_tokens": 40, "unknown_usage_calls": 0}}
    data = {"joint_judge_tasks": 65, "methods": {"facts": row}}
    assert validate(data) == data["methods"]
    with pytest.raises(ValueError):
        validate({**data, "joint_judge_tasks": 64})
    row["qa_usage"]["unknown_usage_calls"] = 1
    with pytest.raises(ValueError):
        validate(data)


def test_paired_contrast_averages_repeats_instead_of_best_of_selection():
    from render_memory_design import paired_contrast
    grades = [
        {"method": method, "video_id": f"v{i}", "question_id": f"q{i}",
         "judge_score": score, "judge_error": False}
        for method, scores in {"a": [1., 0.], "a_repeat": [0., 1.], "b": [0., 0.]}.items()
        for i, score in enumerate(scores)]
    result = paired_contrast(grades, ["a", "a_repeat"], ["b"])
    assert result["questions"] == 2
    assert result["mean_paired_delta"] == .5  # Not 1.0 from cherry-picking.
    assert result["video_cluster_interval"]["lower"] == .5
    assert result["video_cluster_interval"]["upper"] == .5
    with pytest.raises(ValueError):
        paired_contrast(grades[:-1], ["a", "a_repeat"], ["b"])
