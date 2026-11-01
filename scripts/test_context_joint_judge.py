import json

import pytest

import judge_context_comparison as judge


def prepare(tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "ROOT", tmp_path)
    counts = dict(zip(judge.VIDEOS, [12, 11, 11, 14, 17]))
    for method in ["historical_ogmemory", "m3_reference_text", *judge.ARMS]:
        for video in judge.VIDEOS:
            if method == "historical_ogmemory":
                source = "m3_study18_recovery_20261101" if video == "study_18" else "m3_flash_medium_20261031"
            elif method == "m3_reference_text":
                source = "m3_agent_reference_text_20261101"
            else:
                source = judge.ARMS[method]
                status = tmp_path / "experiments" / source / "status.json"
                status.parent.mkdir(parents=True, exist_ok=True)
                status.write_text('{"status": "finished"}')
            path = tmp_path / "runs" / source / video / "m3_qa_results.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            records = [
                {"video_id": video, "question_id": f"{video}_{i}", "question_index": i,
                 "question": f"Question {i}?", "expected_answer": "truth",
                 "predicted_answer": f"{method} answer", "error": "",
                 "finish_reason": "done", "type": [],
                 "temporal_scope": {"policy": "current_clip", "resolved_before_clip": i,
                                    "effective_before_sec": i * 30}}
                for i in range(counts[video])
            ]
            path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_shuffled_order_is_complete_deterministic_and_interleaved(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    tasks, hashes = judge.inputs()
    second, same_hashes = judge.inputs()
    assert tasks == second and hashes == same_hashes
    assert len(tasks) == 325
    assert len({r["method"] for r in tasks[:10]}) > 2
    assert len({(r["method"], r["question_id"]) for r in tasks}) == 325


def test_rejects_changed_temporal_scope(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    path = tmp_path / "runs" / judge.ARMS["compact12k"] / "study_05/m3_qa_results.jsonl"
    records = judge.rows(path)
    records[0]["temporal_scope"]["effective_before_sec"] = 999
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(AssertionError):
        judge.inputs()


def test_refuses_incomplete_arm(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    path = tmp_path / "experiments" / judge.ARMS["budget_only"] / "status.json"
    path.write_text('{"status": "running"}')
    with pytest.raises(RuntimeError, match="not finished"):
        judge.inputs()
