#!/usr/bin/env python3
"""One blinded interleaved judge pass over immutable memory-design predictions."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(os.environ.get("M3_EVAL_ROOT", Path(__file__).resolve().parent.parent))
STUDY = ROOT / "experiments/m3_memory_design_20261102"
NAME = "m3_memory_design_joint_judge_20261102"
VIDEOS = ["study_05", "gym_03", "bedroom_04", "study_18", "kitchen_20"]
DEADLINE_ISO = "2026-11-02T10:03:00+08:00"
DEADLINE = datetime.datetime.fromisoformat(DEADLINE_ISO).timestamp()
REFERENCES = {"balanced_prior", "balanced_prior_repeat", "m3_reference"}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def verify_usage_bootstrap(expected_log):
    """Python silently tolerates sitecustomize failures; model calls must not."""
    from providers.llm.openai_llm import OpenAILLM
    installed = OpenAILLM._create_completion
    module = sys.modules.get("sitecustomize")
    actual = getattr(installed, "__globals__", {}).get("path")
    if (module is None or not getattr(installed, "_m3_usage_audit", False)
            or not actual or Path(actual).resolve() != Path(expected_log).resolve()):
        raise RuntimeError("Usage instrumentation is not active at startup; refusing judge model calls")
    return True


def load_inputs(root=ROOT, methods=None):
    if methods is None:
        methods = json.loads((root / "experiments/m3_memory_design_20261102/methods.json").read_text())
    data, hashes = {}, {}
    for method, name in methods.items():
        if method not in REFERENCES:
            state = json.loads((root / "experiments" / name / "status.json").read_text())
            if state["status"] != "finished":
                raise ValueError(f"{name} is not finished")
        paths = [root / "runs" / name / video / "m3_qa_results.jsonl" for video in VIDEOS]
        records = [row for path in paths for row in rows(path)]
        keyed = {(row["video_id"], row["question_id"]): row for row in records}
        if len(records) != 65 or len(keyed) != 65:
            raise ValueError(f"{name} must have exactly 65 unique predictions")
        data[method] = keyed
        hashes[method] = {str(path.relative_to(root)): digest(path) for path in paths}
    reference = data["balanced_prior"]
    tasks = []
    for method, keyed in data.items():
        if set(keyed) != set(reference):
            raise ValueError("Question cohort mismatch")
        for key, record in keyed.items():
            base = reference[key]
            for field in ("question", "expected_answer", "type"):
                if record[field] != base[field]:
                    raise ValueError(f"Changed task field {field}")
            for field in ("policy", "resolved_before_clip", "effective_before_sec"):
                if record["temporal_scope"].get(field) != base["temporal_scope"].get(field):
                    raise ValueError(f"Temporal scope mismatch: {key}")
            tasks.append({"method": method, **{k: record.get(k) for k in (
                "video_id", "question_id", "question_index", "question", "expected_answer",
                "predicted_answer", "error", "finish_reason", "type")}})
    random.Random(1102).shuffle(tasks)
    return tasks, hashes


def main():
    tasks, hashes = load_inputs()
    exp, run = ROOT / "experiments" / NAME, ROOT / "runs" / NAME
    exp.mkdir()
    run.mkdir()
    (exp / "private").mkdir(mode=0o700)
    (run / "runner.py").write_bytes(Path(__file__).read_bytes())
    source = ROOT / "runs/m3_source_fact_index_v2_20261102/_source"
    sys.path.insert(0, str(source))
    from core.operation_deadline import operation_deadline
    from m3.eval import build_judge_llm, judge_answer

    verify_usage_bootstrap(exp / "model-usage.jsonl")
    config = ROOT / "experiments/m3_source_fact_index_v2_20261102/private/study_05.yaml"
    llm = build_judge_llm(config_path=str(config))
    save(exp / "manifest.json", {
        "seed": 1102, "tasks": len(tasks), "source_prediction_hashes": hashes,
        "evaluation_source_sha256": digest(source / "m3/eval.py"),
        "runner_sha256": digest(Path(__file__)),
        "methods": json.loads((STUDY / "methods.json").read_text()),
        "model": "MiniMax-M3.1-Flash", "reasoning_effort": "medium",
        "thinking": "adaptive", "max_completion_tokens": 2048,
        "cooperative_request_deadline_seconds": 120,
        "ordering": "all method/question pairs shuffled; method not sent to judge",
        "retry_policy": "No success regrading or answer retries. API recovery, if needed, separate.",
        "deadline": DEADLINE_ISO,
        "startup_usage_logging_verified": True,
    })
    with (run / "inputs.jsonl").open("x") as stream:
        stream.writelines(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks)
    completed, errors = 0, 0
    with (run / "grades.jsonl").open("x") as stream:
        for task in tasks:
            remaining = DEADLINE - time.time() - 300
            if remaining <= 0:
                save(exp / "status.json", {"status": "deadline_stopped", "completed": completed,
                                           "judge_errors": errors})
                return
            started = time.monotonic()
            if task["error"] or not task["predicted_answer"]:
                score, reason, error = 0., "No valid QA answer; retained at zero.", False
            else:
                with operation_deadline(min(120., remaining)):
                    score, reason, error = judge_answer(
                        judge_llm=llm, question=task["question"],
                        expected=task["expected_answer"], predicted=task["predicted_answer"])
            row = {**task, "judge_score": score, "judge_reasoning": reason,
                   "judge_error": error, "judge_seconds": time.monotonic() - started}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
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
