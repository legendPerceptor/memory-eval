import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "parallel_recovery_tests", Path(__file__).with_name("recover_m3_parallel_ogonly.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_only_failed_video_disables_perception_and_forces_failed_clips():
    original = [
        "--set",
        "episodic_memory.params.workers=8",
        "--max-frames",
        "30",
        "--disable-stage",
        "qa",
        "--episodic-max-frames",
        "16",
    ]
    plan = {"reuse_stages": ["extract_frames", "detect_faces", "face_clustering"]}
    recovered = module.pipeline_args(original, "study_09", plan)
    assert "episodic_memory.params.workers=2" in recovered
    assert "episodic_memory.params.retry_failed=true" in recovered
    assert "episodic_memory.params.retry_base_seconds=30" in recovered
    assert "episodic_memory.params.retry_max_seconds=300" in recovered
    assert "episodic_memory.params.failure_policy=stop" in recovered
    assert recovered.count("--disable-stage") == 4
    fresh = module.pipeline_args(original, "kitchen_01", plan)
    assert fresh.count("--disable-stage") == 1
    assert "episodic_memory.params.retry_failed=true" not in fresh
    assert "episodic_memory.params.workers=8" in original


def test_successful_memory_is_preserved_except_operational_receipts():
    row = {
        "video_id": "study_09",
        "clip_id": 1,
        "description": "unchanged",
        "resumed": False,
        "generation_retry_counts": {"transient": 1},
    }
    assert module.verify_preserved([row], [{**row, "resumed": True}]) == 1
    with pytest.raises(ValueError):
        module.verify_preserved([row], [{**row, "description": "regenerated"}])
    with pytest.raises(ValueError):
        module.verify_preserved([row], [])
    with pytest.raises(ValueError):
        module.verify_preserved([row], [row, row])


def test_model_gate_rejects_successful_or_unscoped_clips():
    module.check_memory_scope(
        {"input_video_id": "study_09", "input_clip_id": "25"}, [25, 27]
    )
    for scope in [
        {},
        {"input_video_id": "study_09", "input_clip_id": 1},
        {"input_video_id": "other", "input_clip_id": 25},
    ]:
        with pytest.raises(RuntimeError):
            module.check_memory_scope(scope, [25, 27])


def test_receipt_keeps_original_failure_cost_without_double_counting():
    old = {"seconds": 5500, "started_at_unix": 1, "exit_code": 2, "status": "failed"}
    new = {
        "seconds": 100,
        "started_at_unix": 6000,
        "exit_code": 0,
        "status": "completed",
    }
    combined = module.combined_receipt(old, new)
    assert combined["seconds"] == 5600
    assert combined["failed_attempts"] == 1 and combined["attempts"][0] == old
    another = module.combined_receipt(combined, new)
    assert another["seconds"] == 5700 and len(another["attempts"]) == 3
