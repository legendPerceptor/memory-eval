#!/usr/bin/env python3
"""Repeat the frozen M3 reference-text QA, only in a confirmed idle quota slot."""
import argparse
from collections import Counter
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(os.environ.get("M3_EVAL_ROOT", Path(__file__).resolve().parent.parent))
STUDY = ROOT / "experiments/m3_memory_design_20261102"
NAME = "m3_reference_repeat_20261102"
ORIGIN = ROOT / "runs/m3_agent_matched_20261101"
VIDEOS = ["study_05", "gym_03", "bedroom_04", "study_18", "kitchen_20"]
DEADLINE = datetime.datetime.fromisoformat("2026-11-02T10:03:00+08:00").timestamp()


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def fingerprint():
    files = []
    for video in VIDEOS:
        files.extend(sorted((ORIGIN / video / "m3_memories").glob("*.json")))
        files.extend(ORIGIN / video / name for name in (
            "embedding_cache.json", "identity_updates.json", "clip_inputs.json", "memory_summary.json"))
    files.extend(sorted((ORIGIN / "_adapter").glob("*.py")))
    files.extend(sorted((ORIGIN / "_source/m3-agent").rglob("*.py")))
    files.extend(sorted((ORIGIN / "_source/m3-agent/configs").glob("*.json")))
    files.extend(sorted((ORIGIN / "_source/oGMemory").rglob("*.py")))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def worker(slot_end):
    exp, run = ROOT / "experiments" / NAME, ROOT / "runs" / NAME
    sys.path.insert(0, str(ORIGIN / "_source/oGMemory"))
    spec = importlib.util.spec_from_file_location("frozen_reference_repeat", run / "_source/reference_text.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.EXP, mod.OUTPUT, mod.CONFIG = exp, run, exp / "private/config.yaml"
    before = json.loads((exp / "manifest.json").read_text())["input_hashes"]
    if fingerprint() != before:
        raise ValueError("Frozen M3 inputs changed before repetition")
    original_prefix, original_llm = mod.build_prefix, mod.build_qa_llm
    state = {"video": None, "position": -1, "indices": [], "quota_stop": False, "question_start": None}
    wall_times, receipts = [], []

    def finish_timer():
        if state["question_start"] is not None:
            wall_times.append(time.monotonic() - state["question_start"])
            state["question_start"] = None

    def prefix(*args, **kwargs):
        finish_timer()
        if state["quota_stop"] or time.time() >= slot_end - 300:
            raise SystemExit(75 if state["quota_stop"] else 74)
        state["position"] += 1
        state["question_start"] = time.monotonic()
        os.environ["M3_AUDIT_VIDEO"] = state["video"]
        os.environ["M3_AUDIT_QUESTION_INDEX"] = str(state["indices"][state["position"]])
        return original_prefix(*args, **kwargs)

    def llm_factory(*args, **kwargs):
        llm = original_llm(*args, **kwargs)
        if getattr(getattr(llm, "_client", None), "api_key", None) != os.environ.get("MINIMAX_API_KEY"):
            raise ValueError("Reference config does not use the observed quota account; no request sent")
        original_call = llm.complete_with_tools_messages
        def guarded(*a, **kw):
            if time.time() >= slot_end:
                raise SystemExit(74)
            try:
                return original_call(*a, **kw)
            except Exception as error:
                text = str(error)
                if "已达到 Token Plan 用量上限" in text or ("Token Plan" in text and "(2056)" in text):
                    state["quota_stop"] = True
                raise
        llm.complete_with_tools_messages = guarded
        return llm

    mod.build_prefix, mod.build_qa_llm = prefix, llm_factory
    try:
        for video in VIDEOS:
            state.update(video=video, position=-1, question_start=None)
            state["indices"] = [r["question_index"] for r in rows(
                ROOT / "runs/m3_agent_reference_text_20261101" / video / "m3_qa_results.jsonl")]
            save(exp / "status.json", {"status": "running", "video": video, "videos": receipts})
            receipts.append(mod.run_video(video))
            finish_timer()
            if state["quota_stop"]:
                raise SystemExit(75)
    except BaseException as error:
        save(exp / "status.json", {"status": "stopped" if isinstance(error, SystemExit) else "failed",
                                   "error_type": type(error).__name__, "videos": receipts,
                                   "exit_code": error.code if isinstance(error, SystemExit) else 1})
        raise
    if fingerprint() != before:
        raise ValueError("M3 source memory or code mutated during repetition")
    answers = [r for video in VIDEOS for r in rows(run / video / "m3_qa_results.jsonl")]
    calls = [r for r in rows(exp / "model-usage.jsonl") if r.get("qa_video_id")]
    known = [r for r in calls if r.get("usage")]
    assert len(answers) == len({(r["video_id"], r["question_id"]) for r in answers}) == 65
    deviations = [i for i, c in enumerate(calls) if (
        c.get("requested_model") != "MiniMax-M3.1-Flash"
        or (c.get("status") == "returned" and c.get("reported_model") != "MiniMax-M3.1-Flash")
        or c.get("max_tokens") != 8192 or c.get("reasoning_effort") != "medium"
        or c.get("thinking") != {"type": "adaptive"} or c.get("tools")
        or c.get("temperature") is not None or c.get("endpoint_host") != "api.minimax.cn")]
    assert not deviations, deviations
    metrics = {
        "name": NAME, "qa_records": len(answers), "judge_records": 0,
        "initial_judging": "Omitted; all fixed answers will enter the common blinded judge.",
        "valid_answers": sum(bool(r["predicted_answer"]) and not r["error"] for r in answers),
        "finish_reasons": dict(Counter(r["finish_reason"] for r in answers)),
        "question_wall_seconds": {"count": len(wall_times), "mean": statistics.mean(wall_times),
                                  "max": max(wall_times), "median": statistics.median(wall_times)},
        "qa_usage": {
            "logical_calls": len(calls), "unknown_usage_calls": len(calls) - len(known),
            "prompt_tokens": sum(c["usage"]["prompt_tokens"] for c in known),
            "completion_tokens": sum(c["usage"]["completion_tokens"] for c in known),
            "peak_prompt_tokens": max((c["usage"]["prompt_tokens"] for c in known), default=0),
            "cached_prompt_tokens": sum(
                (c["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0 for c in known),
            "unknown_cache_usage_calls": sum(
                (c["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") is None for c in known),
            "model_seconds_sum": sum(c["seconds"] for c in calls),
        },
        "model_setting_deviations": deviations, "frozen_inputs_unchanged": True,
    }
    save(exp / "metrics.json", metrics)
    save(exp / "status.json", {"status": "finished", "videos": receipts,
                               "valid_answers": metrics["valid_answers"]})


def wait_and_launch():
    import coordinate_context_experiments as quota_api
    quota_api.EXP = STUDY
    cutoff = datetime.datetime.fromisoformat("2026-11-02T04:00:00+08:00").timestamp()
    observation = STUDY / "reference_repeat_schedule.json"
    while time.time() < cutoff:
        path = STUDY / "followup/coordinator_status.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        if state.get("status") == "scheduled_wait":
            next_start = datetime.datetime.fromisoformat(state["next_start"]).timestamp()
            if next_start - time.time() >= 3600:
                quota = quota_api.quota()
                if quota and quota["interval_remaining_percent"] >= 10:
                    slot_end = min(DEADLINE - 300, next_start - 300)
                    break
        save(observation, {"status": "waiting_idle_slot", "observed_at_unix": time.time(),
                           "main_queue_status": state.get("status")})
        time.sleep(30)
    else:
        save(observation, {"status": "not_executed_no_idle_slot", "quality_selection": False})
        return
    exp, run = ROOT / "experiments" / NAME, ROOT / "runs" / NAME
    exp.mkdir()
    (exp / "private").mkdir(mode=0o700)
    (run / "_source").mkdir(parents=True)
    original = ROOT / "runs/m3_agent_reference_text_20261101/_source/reference_text.py"
    shutil.copyfile(original, run / "_source/reference_text.py")
    (run / "runner.py").write_bytes(Path(__file__).read_bytes())
    config = exp / "private/config.yaml"
    shutil.copyfile(ROOT / "experiments/m3_agent_matched_20261101/private/config.yaml", config)
    config.chmod(0o600)
    instrument = run / "_instrumentation"
    instrument.mkdir()
    shutil.copyfile(ROOT / "scripts/instrumentation/sitecustomize.py", instrument / "sitecustomize.py")
    save(exp / "manifest.json", {
        "name": NAME, "independent_replication_of": "m3_agent_reference_text_20261101",
        "controller_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_hashes": fingerprint(), "cohort": VIDEOS, "total_questions": 65,
        "model": "MiniMax-M3.1-Flash", "reasoning_effort": "medium", "thinking": "adaptive",
        "qa_output_tokens": 8192, "max_iterations": 6, "timeout_seconds": 180,
        "temporal_policy": "current_clip", "source_memory_regenerated": False,
        "slot_end": datetime.datetime.fromtimestamp(slot_end).astimezone().isoformat(),
        "note": "Unchanged frozen reference controller; wrappers only set audit context and outer quota/deadline guards.",
    })
    save(observation, {"status": "running", "slot_end_unix": slot_end, "quota_at_start": quota,
                       "overlaps_main_model_jobs": False, "quality_selection": False})
    env = {**os.environ, "M3_EVAL_ROOT": str(ROOT), "OGMEM_MINIMAX_MODEL": "MiniMax-M3.1-Flash",
           "M3_USAGE_LOG": str(exp / "model-usage.jsonl"),
           "PYTHONPATH": f"{instrument}:{ORIGIN / '_source/oGMemory'}"}
    with (exp / "private/worker.log").open("x") as log:
        try:
            result = subprocess.run([
                str(ROOT / "oGMemory/.venv/bin/python"), str(run / "runner.py"),
                "--worker", "--slot-end", str(slot_end)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=max(1, slot_end - time.time()))
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 74
    state = json.loads((exp / "status.json").read_text()) if (exp / "status.json").exists() else {}
    if code == 0 and state.get("status") == "finished":
        # Optional oGMemory runs cannot start until the mandatory post-reset
        # jobs complete, so this pre-reset publication precedes their updates.
        path = STUDY / "methods.json"
        methods = json.loads(path.read_text())
        methods["m3_reference_repeat"] = NAME
        save(path, methods)
    save(observation, {"status": "finished" if code == 0 else "failed_or_interrupted",
                       "exit_code": code, "worker_status": state,
                       "included_in_common_judge": code == 0 and state.get("status") == "finished"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--slot-end", type=float)
    args = parser.parse_args()
    if args.worker:
        worker(args.slot_end)
    else:
        wait_and_launch()
