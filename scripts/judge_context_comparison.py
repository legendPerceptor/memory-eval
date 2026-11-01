#!/usr/bin/env python3
"""One preregistered, shuffled judge pass over immutable answers from six arms."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(os.environ.get("M3_EVAL_ROOT", Path(__file__).resolve().parent.parent))
NAME = "m3_context_joint_judge_20261101"
VIDEOS = ["study_05", "gym_03", "bedroom_04", "study_18", "kitchen_20"]
ARMS = {
    "budget_only": "m3_context_budget_recovered_20261101",
    "finalization": "m3_context_finalization_recovered_20261101",
    "compact12k": "m3_context_compact12k_guarded_20261101",
    "grouped12k": "m3_context_grouped12k_20261101",
}
DEADLINE = datetime.datetime.fromisoformat("2026-11-01T23:30:00+08:00").timestamp()


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def inputs():
    data, hashes = {}, {}
    for method in ["historical_ogmemory", "m3_reference_text", *ARMS]:
        paths = []
        for video in VIDEOS:
            if method == "historical_ogmemory":
                source = "m3_study18_recovery_20261101" if video == "study_18" else "m3_flash_medium_20261031"
            elif method == "m3_reference_text":
                source = "m3_agent_reference_text_20261101"
            else:
                source = ARMS[method]
                status = json.loads((ROOT / "experiments" / source / "status.json").read_text())
                if status["status"] != "finished":
                    raise RuntimeError(f"{source} is not finished")
            paths.append(ROOT / "runs" / source / video / "m3_qa_results.jsonl")
        answers = [r for path in paths for r in rows(path)]
        keyed = {(r["video_id"], r["question_id"]): r for r in answers}
        assert len(answers) == len(keyed) == 65, method
        data[method] = keyed
        hashes[method] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in paths}
    expected = data["historical_ogmemory"]
    for method, answers in data.items():
        assert set(answers) == set(expected), method
        for key, r in answers.items():
            baseline = expected[key]
            assert r["question"] == baseline["question"]
            assert r["expected_answer"] == baseline["expected_answer"]
            for field in ("policy", "resolved_before_clip", "effective_before_sec"):
                assert r["temporal_scope"].get(field) == baseline["temporal_scope"].get(field), (method, key, field)
    tasks = [
        {"method": method, **{k: r.get(k) for k in (
            "video_id", "question_id", "question_index", "question", "expected_answer",
            "predicted_answer", "error", "finish_reason", "type")}}
        for method, answers in data.items() for r in answers.values()
    ]
    random.Random(42).shuffle(tasks)
    return tasks, hashes


def main():
    tasks, hashes = inputs()
    exp, run = ROOT / "experiments" / NAME, ROOT / "runs" / NAME
    exp.mkdir()
    run.mkdir()
    (exp / "private").mkdir(mode=0o700)
    (run / "runner.py").write_bytes(Path(__file__).read_bytes())
    source = ROOT / "runs/m3_context_compact12k_guarded_20261101/_source"
    sys.path.insert(0, str(source))
    from core.operation_deadline import operation_deadline
    from m3.eval import build_judge_llm, judge_answer

    config = ROOT / "experiments/m3_context_compact12k_guarded_20261101/private/study_05.yaml"
    llm = build_judge_llm(config_path=str(config))
    save(exp / "manifest.json", {
        "seed": 42, "tasks": len(tasks), "source_prediction_hashes": hashes,
        "evaluation_source_sha256": hashlib.sha256((source / "m3/eval.py").read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": "MiniMax-M3.1-Flash", "reasoning_effort": "medium",
        "max_completion_tokens": 2048, "thinking": "adaptive",
        "cooperative_request_deadline_seconds": 120,
        "ordering": "all (method, question) pairs shuffled; method not sent to judge",
        "retry_policy": "no selective success regrading; any API-error recovery saved separately",
        "deadline": "2026-11-01T23:30:00+08:00",
    })
    with (run / "inputs.jsonl").open("x") as stream:
        stream.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in tasks)
    completed, errors = 0, 0
    with (run / "grades.jsonl").open("x") as stream:
        for task in tasks:
            remaining = DEADLINE - time.time() - 180
            if remaining <= 0:
                save(exp / "status.json", {"status": "deadline_stopped", "completed": completed,
                                           "judge_errors": errors})
                return
            started = time.monotonic()
            if task["error"] or not task["predicted_answer"]:
                score, reason, error = 0., "QA did not produce a valid answer; no judge call.", False
            else:
                with operation_deadline(min(120., remaining)):
                    score, reason, error = judge_answer(
                        judge_llm=llm, question=task["question"],
                        expected=task["expected_answer"], predicted=task["predicted_answer"])
            result = {**task, "judge_score": score, "judge_reasoning": reason,
                      "judge_error": error, "judge_seconds": time.monotonic() - started}
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            completed += 1
            errors += bool(error)
            if error and ("已达到 Token Plan 用量上限" in reason
                          or ("Token Plan" in reason and "(2056)" in reason)):
                save(exp / "status.json", {"status": "stopped_provider_quota",
                                           "completed": completed, "judge_errors": errors})
                return
            save(exp / "status.json", {"status": "running", "completed": completed,
                                       "judge_errors": errors})
    save(exp / "status.json", {"status": "finished" if not errors else "finished_with_judge_errors",
                               "completed": completed, "judge_errors": errors})


if __name__ == "__main__":
    main()
