#!/usr/bin/env python3
"""Paired QA pilot on frozen paper-eight memories; no reference answers in QA."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "scripts/run_m3_paper8_fresh_20261110.py").is_file())
os.environ["M3_EVAL_ROOT"] = str(ROOT)
NAME = os.environ.get("M3_ACCURACY_EXPERIMENT", "m3_paper8_accuracy_pilot_20261110")
LATER_TOKENS = int(os.environ.get("M3_ACCURACY_LATER_TOKENS", "2000"))
AMBIGUITY_GUARD = os.environ.get("M3_ACCURACY_AMBIGUITY_GUARD", "1") == "1"
RETRY_FAILED = os.environ.get("M3_ACCURACY_RETRY_FAILED", "0") == "1"
EXP = ROOT / "experiments" / NAME
RUN = ROOT / "runs" / NAME
FRESH = ROOT / "runs/m3_paper8_fresh_20261110"
FRESH_EXP = ROOT / "experiments/m3_paper8_fresh_20261110"
SOURCE = RUN / "_source/oGMemory"
VIDEOS = tuple(os.environ.get(
    "M3_ACCURACY_VIDEOS", "study_23,kitchen_01,bedroom_05").split(","))
if not VIDEOS or len(VIDEOS) != len(set(VIDEOS)) or any(
    video not in {"study_09", "kitchen_01", "bedroom_05", "study_23",
                  "kitchen_23", "study_05", "gym_03", "bedroom_04"}
    for video in VIDEOS
):
    raise ValueError("Videos must be unique members of the frozen eight-video cohort")
PYTHON = ROOT / "oGMemory/.venv/bin/python"
ACCOUNT = "m3-paper8-fresh-20261110"


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare():
    if RUN.exists() or EXP.exists():
        raise FileExistsError("Refusing to overwrite an experiment")
    RUN.mkdir(parents=True)
    (EXP / "private").mkdir(parents=True, mode=0o700)
    shutil.copytree(FRESH / "_source", RUN / "_source",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    baseline = {}
    for video in VIDEOS:
        source = FRESH / "og_qa" / video / "m3_qa_results.jsonl"
        target = EXP / "baseline" / video / source.name
        target.parent.mkdir(parents=True)
        shutil.copyfile(source, target)
        baseline[video] = digest(target)
        if RETRY_FAILED:
            candidate = RUN / "candidate" / video / source.name
            candidate.parent.mkdir(parents=True)
            shutil.copyfile(source, candidate)
        config = FRESH_EXP / "private" / f"{video}.yaml"
        shutil.copyfile(config, EXP / "private" / f"{video}.yaml")
        (EXP / "private" / f"{video}.yaml").chmod(0o600)
    shutil.copyfile(Path(__file__), RUN / "driver.py")
    save(EXP / "manifest.json", {
        "videos": VIDEOS, "baseline_sha256": baseline,
        "safety_stop_at_unix": time.time() + 24 * 3600,
        "source_control_sha256": digest(SOURCE / "m3/control_adapter.py"),
        "configuration_sha256": {video: digest(EXP / "private" / f"{video}.yaml")
                                 for video in VIDEOS},
        "change": {"ambiguity_guard": AMBIGUITY_GUARD, "later_incremental_tokens": LATER_TOKENS,
                   "retry_failed": RETRY_FAILED},
        "design": "All questions per video; frozen memory and first baseline answers; "
                  "randomized blinded rejudging; failures counted as zero. Retry policy "
                  "copies all first answers and reruns only incomplete or failed questions once.",
        "limitations": "Selected videos, one stochastic run, no causal "
                       "conclusion for memory construction or person grounding.",
    })
    save(EXP / "status.json", {"status": "prepared", "completed": []})


def run_qa(video):
    output = RUN / "candidate" / video / "m3_qa_results.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OGMEM_M3_COMPACT_PERSON="1",
               OGMEM_M3_LATER_INCREMENTAL_TOKENS=str(LATER_TOKENS),
               OGMEM_M3_MAX_WORKERS="1", OGMEM_EMBED_MAX_CONCURRENCY="1",
               PYTHONHASHSEED="0", PYTHONUNBUFFERED="1",
               PYTHONPATH=f"{FRESH / '_instrumentation'}:{SOURCE}",
               M3_USAGE_LOG=str(EXP / f"usage-candidate-{video}.jsonl"),
               M3_AUDIT_VIDEO=video)
    for key in ("OGMEM_M3_EARLY_ANSWER", "OGMEM_M3_PERSON_ON_DEMAND"):
        env.pop(key, None)
    if AMBIGUITY_GUARD:
        env["OGMEM_M3_AMBIGUITY_GUARD"] = "1"
    else:
        env.pop("OGMEM_M3_AMBIGUITY_GUARD", None)
    config = EXP / "private" / f"{video}.yaml"
    command = [str(PYTHON), str(RUN / "driver.py"), "worker", video]
    started = time.monotonic()
    with (EXP / "private" / f"{video}.log").open("a") as log:
        result = subprocess.run(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode not in (0, 2) or not output.exists():
        raise RuntimeError(f"{video} worker exit {result.returncode}")
    actual = {row["question_id"]: row for row in rows(output)}
    reference = {row["question_id"]: row for row in rows(EXP / "baseline" / video / output.name)}
    if actual.keys() != reference.keys() or any(
        actual[key].get(field) != reference[key].get(field)
        for key in actual for field in ("question", "expected_answer")
    ):
        raise ValueError(f"{video}: question coverage mismatch")
    return {"seconds": time.monotonic() - started, "questions": len(actual),
            "failures": sum(bool(row.get("error")) or not row.get("predicted_answer")
                            for row in actual.values())}


def worker(video):
    sys.path.insert(0, str(SOURCE))
    spec = importlib.util.spec_from_file_location("paper_run", ROOT / "scripts/run_m3_paper8_fresh_20261110.py")
    paper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paper)
    paper.base.h.EXP = EXP
    gate = paper.base.h.QuotaGate()
    paper.base.h.instrument(gate)
    from m3.config import load_m3_runtime_config
    from m3.control_adapter import OGMemoryControlAdapter
    from m3.qa import main_react_qa
    config = EXP / "private" / f"{video}.yaml"
    os.environ["OGMEM_GRAPH_CONNECTION_STRING"] = load_m3_runtime_config(str(config)).graph_connection_string
    original = OGMemoryControlAdapter.run

    def guarded(self, ctx, task):
        gate.wait()
        return original(self, ctx, task)

    OGMemoryControlAdapter.run = guarded
    output = RUN / "candidate" / video / "m3_qa_results.jsonl"
    args = ["--config", str(config), "--annotations", "/data/m3_data/annotations/robot.json",
            "--video", video, "--graph-store", "sql", "--account-id", ACCOUNT,
            "--output-jsonl", str(output), "--temperature", "0", "--max-iterations", "6",
            "--timeout-seconds", "180", "--top-k", "8", "--candidate-k", "20",
            "--transcript-top-k", "20", "--transcript-fusion", "rrf", "--context-k", "8",
            "--include-neighbors", "1", "--temporal-policy", "current_clip",
            "--person-retrieval", "--memory-coverage-json", str(FRESH / video / "memory_coverage.json")]
    if output.exists():
        args.append("--resume")
    return main_react_qa(args)


def judge():
    sys.path.insert(0, str(SOURCE))
    from core.operation_deadline import operation_deadline
    from m3.eval import build_judge_llm, judge_answer
    spec = importlib.util.spec_from_file_location("paper_run", ROOT / "scripts/run_m3_paper8_fresh_20261110.py")
    paper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paper)
    paper.base.h.EXP = EXP
    gate = paper.base.h.QuotaGate()
    paper.base.h.instrument(gate)
    tasks = []
    for video in VIDEOS:
        for arm in ("baseline", "candidate"):
            folder = EXP / "baseline" if arm == "baseline" else RUN / "candidate"
            latest = {row["question_id"]: row for row in
                      rows(folder / video / "m3_qa_results.jsonl")}
            for row in latest.values():
                tasks.append({key: row.get(key) for key in
                              ("video_id", "question_id", "question", "expected_answer",
                               "predicted_answer", "error")} | {"arm": arm})
    random.Random(20261110).shuffle(tasks)
    frozen = EXP / "judge_inputs.json"
    if frozen.exists() and json.loads(frozen.read_text()) != tasks:
        raise ValueError("Judge inputs changed")
    save(frozen, tasks)
    output = EXP / "judgments.jsonl"
    previous = rows(output) if output.exists() else []
    done = {(r["arm"], r["video_id"], r["question_id"]) for r in previous}
    judged = {(r["video_id"], r["question_id"], r["predicted_answer"], r["error"]):
              (r["judge_score"], r["judge_reasoning"]) for r in previous}
    llm = build_judge_llm(config_path=str(EXP / "private" / f"{VIDEOS[0]}.yaml"))
    with output.open("a") as stream:
        for task in tasks:
            key = (task["arm"], task["video_id"], task["question_id"])
            if key in done:
                continue
            answer_key = (task["video_id"], task["question_id"],
                          task["predicted_answer"], task["error"])
            score, reason, error = 0.0, "Missing or failed answer", False
            if answer_key in judged:
                score, reason = judged[answer_key]
            elif not task["error"] and task["predicted_answer"]:
                for attempt in range(3):
                    gate.wait()
                    with operation_deadline(120):
                        score, reason, error = judge_answer(
                            judge_llm=llm, question=task["question"],
                            expected=task["expected_answer"], predicted=task["predicted_answer"])
                    if not error:
                        break
                    time.sleep(30 * (attempt + 1))
                if error:
                    raise RuntimeError(f"Judge failed: {key}")
            judged[answer_key] = (score, reason)
            stream.write(json.dumps({**task, "judge_score": score, "judge_reasoning": reason},
                                    ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def run():
    status = json.loads((EXP / "status.json").read_text())
    status.pop("error", None)
    manifest = json.loads((EXP / "manifest.json").read_text())
    if manifest["change"] != {"ambiguity_guard": AMBIGUITY_GUARD,
                              "later_incremental_tokens": LATER_TOKENS,
                              "retry_failed": RETRY_FAILED}:
        raise ValueError("Policy differs from frozen manifest")
    for video in VIDEOS:
        if digest(EXP / "baseline" / video / "m3_qa_results.jsonl") != manifest["baseline_sha256"][video]:
            raise ValueError(f"Baseline changed: {video}")
        if digest(EXP / "private" / f"{video}.yaml") != manifest["configuration_sha256"][video]:
            raise ValueError(f"Configuration changed: {video}")
    if digest(SOURCE / "m3/control_adapter.py") != manifest["source_control_sha256"]:
        raise ValueError("Frozen controller changed")
    try:
        for video in VIDEOS:
            if video in status["completed"]:
                continue
            status.update(status="running", current=video)
            save(EXP / "status.json", status)
            status.setdefault("stages", {})[video] = run_qa(video)
            status["completed"].append(video)
            save(EXP / "status.json", status)
        status.update(status="judging", current=None)
        save(EXP / "status.json", status)
        env = dict(os.environ, PYTHONPATH=f"{FRESH / '_instrumentation'}:{SOURCE}",
                   M3_USAGE_LOG=str(EXP / "usage-judge.jsonl"), PYTHONUNBUFFERED="1")
        with (EXP / "private" / "judge.log").open("a") as log:
            result = subprocess.run([str(PYTHON), str(RUN / "driver.py"), "judge"],
                                    cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Judge exited {result.returncode}")
        judgments = rows(EXP / "judgments.jsonl")
        summary = {}
        for arm in ("baseline", "candidate"):
            group = [row for row in judgments if row["arm"] == arm]
            summary[arm] = {"questions": len(group),
                            "score": sum(row["judge_score"] for row in group) / len(group),
                            "failures": sum(bool(row["error"]) or not row["predicted_answer"] for row in group),
                            "by_video": {video: sum(row["judge_score"] for row in group if row["video_id"] == video)
                                         / sum(row["video_id"] == video for row in group) for video in VIDEOS}}
        save(EXP / "summary.json", summary)
        status.update(status="completed", current=None)
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(EXP / "status.json", status)


if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        prepare()
    elif sys.argv[1:] == ["run"]:
        run()
    elif len(sys.argv) == 3 and sys.argv[1] == "worker" and sys.argv[2] in VIDEOS:
        sys.exit(worker(sys.argv[2]))
    elif sys.argv[1:] == ["judge"]:
        judge()
    else:
        raise ValueError("Use prepare, run, worker VIDEO, or judge")
