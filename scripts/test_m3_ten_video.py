import ast
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

import run_m3_ten_video as study


def test_cohort_excludes_original_and_is_stable():
    old = ["study_05", "gym_03", "bedroom_04", "study_18", "kitchen_20"]
    manifest = json.loads((study.ROOT / "experiments/m3_flash_medium_20261031/sample_manifest.json").read_text())
    got = study.select_cohort(manifest["candidate_ids"], old)
    assert got == ["study_09", "kitchen_01", "bedroom_05", "study_23", "kitchen_23"]
    assert not set(got) & set(old)
    assert study.select_cohort(reversed(manifest["candidate_ids"]), old) == got
    assert len(set(got + old)) == 10


def test_invalid_cohort_rejected():
    with pytest.raises(ValueError):
        study.select_cohort(["a"] * 10, ["a"] * 5)


def test_private_settings_are_redacted():
    assert study.redact({"m3": {"api_key": "secret"}, "storage": {"connection_string": "private"}}) == {
        "m3": {"api_key": "<REDACTED>"}, "storage": {"connection_string": "<REDACTED>"}}


def test_qa_uses_frozen_source_and_instrumentation(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "RUN", tmp_path / "run")
    monkeypatch.setattr(study, "SOURCE", tmp_path / "run/_source/oGMemory")
    env = study.environment("og_qa", "study_09")
    assert str(study.SOURCE) in env["PYTHONPATH"]
    assert "_instrumentation" in env["PYTHONPATH"]
    assert env["M3_MATCHED_ACCOUNT"] == study.ACCOUNT
    assert env["M3_USAGE_LOG"].endswith("usage-og_qa-study_09.jsonl")
    assert env["OGMEM_MINIMAX_MODEL"] == "MiniMax-M3.1-Flash"


def test_adapter_uses_same_uniform_selector_for_base_images():
    path = study.ROOT / "comparisons/m3_agent_matched/runner.py"
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_clips")
    # Static regression: the previously unbounded frames expression must not
    # return as we increase perception to 30 frames.
    code = ast.unparse(fn)
    assert "frame_indices = _evenly_spaced_indices(len(frames), 16)" in code
    assert "_image_part_from_node(frames[i], 'auto') for i in frame_indices" in code
    assert "'base_frame_node_ids'" in code


def test_invalid_answers_keep_fixed_denominator(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "EXP", tmp_path)
    manifest = {"methods": ["ogmemory_strict_incremental", "m3_reference_text"],
                "selected_video_ids": ["old", "new"], "original_video_ids": ["old"],
                "new_video_ids": ["new"], "question_counts": {"old": 1, "new": 1},
                "total_questions": 2}
    study.save(tmp_path / "manifest.json", manifest)
    grades = [{"method": m, "video_id": v, "question_id": v, "judge_score": 0 if v == "old" else 1,
               "error": "upstream_failure" if v == "old" else "", "predicted_answer": "" if v == "old" else "answer",
               "judge_error": False} for m in manifest["methods"] for v in ("old", "new")]
    (tmp_path / "joint_questions.jsonl").write_text("".join(json.dumps(g) + "\n" for g in grades))
    result = study.summarize()
    for m in result["methods"].values():
        assert m["mean_score"] == .5 and m["valid_answers"] == 1 and m["questions"] == 2
    grades[-1]["judge_error"] = True
    (tmp_path / "joint_questions.jsonl").write_text("".join(json.dumps(g) + "\n" for g in grades))
    assert study.summarize()["methods"]["m3_reference_text"]["mean_score"] is None


def test_quota_safety_escapes_per_clip_exception_handlers():
    gate = study.QuotaGate.__new__(study.QuotaGate)
    gate.lock = threading.Lock()
    gate.checked = time.time()
    gate.deadline = time.time() + 100
    gate.observation = {"weekly_remaining_percent": 4, "interval_remaining_percent": 100}
    with pytest.raises(SystemExit) as error:
        gate.wait()
    assert error.value.code == 74


def test_quota_unknown_fails_closed_without_calls():
    gate = study.QuotaGate.__new__(study.QuotaGate)
    gate.lock = threading.Lock()
    gate.checked = 0
    gate.deadline = time.time() + 100
    gate.observation = None
    def unavailable():
        raise ValueError("unavailable")
    gate.api = SimpleNamespace(read_quota_observation=unavailable)
    with pytest.raises(SystemExit) as error:
        gate.wait()
    assert error.value.code == 76
