#!/usr/bin/env python3
"""Compare two bounded QA policies on frozen memories, then expand the safer policy."""

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(os.environ.get("M3_EVAL_ROOT", Path(__file__).resolve().parent.parent))
NAME = "m3_qa_efficiency_20261109"
RUN = ROOT / "runs" / NAME
EXP = ROOT / "experiments" / NAME
BASE_RUN = ROOT / "runs/m3_compact_replay_20261109"
BASE_EXP = ROOT / "experiments/m3_compact_replay_20261109"
SOURCE_RUN = ROOT / "runs/m3_dev_1031_ten_retry_20261108"
PYTHON = ROOT / "oGMemory/.venv/bin/python"
ACCOUNT = "m3-dev-1031-ten-retry-20261108"
PILOT = ["bedroom_04", "kitchen_01"]
EXPANSION = ["bedroom_05", "gym_03", "study_23"]
POLICIES = {
    "moderate": {"incremental_tokens": 1600, "max_input_tokens": 10000,
                 "max_search_calls": 4, "max_iterations": 5},
    "tight": {"incremental_tokens": 1000, "max_input_tokens": 8000,
              "max_search_calls": 3, "max_iterations": 4},
}


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest(path):
    return {(row["video_id"], row["question_id"]): row for row in rows(path)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    if RUN.exists() or EXP.exists():
        raise FileExistsError("Experiment already exists")
    RUN.mkdir(parents=True)
    (EXP / "private").mkdir(parents=True, mode=0o700)
    shutil.copytree(BASE_RUN / "_source", RUN / "_source",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("control_adapter.py", "compact_person.py"):
        shutil.copyfile(ROOT / "oGMemory/m3" / name, RUN / "_source/m3" / name)
    shutil.copyfile(Path(__file__), RUN / "driver.py")
    for policy, settings in POLICIES.items():
        for video in PILOT + EXPANSION:
            config = yaml.safe_load((BASE_EXP / "private" / f"{video}.yaml").read_text())
            context = config["m3"]["qa_context"]
            for key in ("incremental_tokens", "max_input_tokens", "max_search_calls"):
                context[key] = settings[key]
            context["max_top_k"] = 6
            config_path = EXP / "private" / f"{policy}-{video}.yaml"
            config_path.write_text(yaml.safe_dump(config, allow_unicode=True))
            config_path.chmod(0o600)
    files = [RUN / "_source/m3" / name for name in
             ("control_adapter.py", "compact_person.py", "context_budget.py",
              "incremental_context.py", "index_episodic.py")]
    save(EXP / "manifest.json", {
        "base_experiment": str(BASE_EXP.relative_to(ROOT)),
        "pilot": PILOT, "expansion": EXPANSION, "policies": POLICIES,
        "selection": "Among pilot policies within 0.05 mean score of baseline, choose lowest total API tokens; otherwise choose best mean score.",
        "code_sha256": {str(path.relative_to(ROOT)): digest(path) for path in files},
        "baseline_sha256": {video: digest(BASE_RUN / video / "m3_qa_results.jsonl")
                            for video in PILOT + EXPANSION},
        "design": "Frozen compact vector index, graph, memories, and fact indexes; no memory regeneration.",
    })
    save(EXP / "status.json", {"status": "prepared", "completed": [], "stages": {}})


def qa(policy, video, status):
    stage = f"qa/{policy}/{video}"
    if stage in status["completed"]:
        return
    config = EXP / "private" / f"{policy}-{video}.yaml"
    folder = RUN / policy / video
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "m3_qa_results.jsonl"
    env = dict(os.environ, OGMEM_M3_COMPACT_PERSON="1", OGMEM_M3_PERSON_ON_DEMAND="1",
               OGMEM_M3_EARLY_ANSWER="1", PYTHONUNBUFFERED="1", PYTHONHASHSEED="0",
               OGMEM_M3_MAX_WORKERS="1", OGMEM_EMBED_MAX_CONCURRENCY="1")
    env["PYTHONPATH"] = f"{SOURCE_RUN / '_instrumentation'}:{RUN / '_source'}"
    env["OGMEM_GRAPH_CONNECTION_STRING"] = yaml.safe_load(config.read_text())["storage"]["connection_string"]
    env["M3_USAGE_LOG"] = str(EXP / f"usage-{policy}-{video}.jsonl")
    env["M3_AUDIT_VIDEO"] = video
    command = [str(PYTHON), "-c",
               "import sys; from m3.qa import main_react_qa; sys.exit(main_react_qa(sys.argv[1:]))",
               "--config", str(config), "--annotations", "/data/m3_data/annotations/robot.json",
               "--video", video, "--graph-store", "sql", "--account-id", ACCOUNT,
               "--output-jsonl", str(output), "--temperature", "0",
               "--max-iterations", str(POLICIES[policy]["max_iterations"]),
               "--no-gain-limit", "2", "--timeout-seconds", "180", "--top-k", "6",
               "--candidate-k", "20", "--transcript-top-k", "20", "--transcript-fusion", "rrf",
               "--context-k", "6", "--include-neighbors", "1", "--temporal-policy", "current_clip",
               "--person-retrieval", "--memory-coverage-json",
               str(SOURCE_RUN / video / "memory_coverage.json")]
    if output.exists():
        command.append("--resume")
    status.update(status="running", current=stage)
    save(EXP / "status.json", status)
    started = time.monotonic()
    with (EXP / "private" / f"{policy}-{video}.log").open("a") as log:
        result = subprocess.run(command, cwd=RUN / "_source", env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode not in (0, 2) or not output.exists():
        raise RuntimeError(f"{stage}: process exited {result.returncode}")
    actual, reference = latest(output), latest(BASE_RUN / video / "m3_qa_results.jsonl")
    if actual.keys() != reference.keys() or any(
        actual[key].get(field) != reference[key].get(field)
        for key in reference for field in ("question", "expected_answer")
    ):
        raise ValueError(f"{stage}: question coverage mismatch")
    status["stages"][stage] = {
        "exit_code": result.returncode, "seconds": time.monotonic() - started,
        "questions": len(actual), "attempt_rows": len(rows(output)),
        "failures": sum(bool(row.get("error")) or not row.get("predicted_answer") for row in actual.values()),
    }
    status["completed"].append(stage)
    save(EXP / "status.json", status)


def judge(videos, arms, status, label):
    stage = f"judge/{label}"
    if stage in status["completed"]:
        return
    sys.path.insert(0, str(RUN / "_source"))
    from core.operation_deadline import operation_deadline
    from m3.eval import build_judge_llm, judge_answer

    tasks = []
    for video in videos:
        for arm in arms:
            path = (BASE_RUN / video / "m3_qa_results.jsonl" if arm == "baseline"
                    else RUN / arm / video / "m3_qa_results.jsonl")
            for item in latest(path).values():
                tasks.append({key: item.get(key) for key in
                              ("video_id", "question_id", "question", "expected_answer",
                               "predicted_answer", "error")} | {"arm": arm})
    random.Random(20261109).shuffle(tasks)
    frozen = EXP / f"judge-input-{label}.json"
    if frozen.exists() and json.loads(frozen.read_text()) != tasks:
        raise ValueError(f"Changed judge inputs: {label}")
    if not frozen.exists():
        save(frozen, tasks)
    output = EXP / f"judgments-{label}.jsonl"
    done = {(r["arm"], r["video_id"], r["question_id"]) for r in rows(output)} if output.exists() else set()
    status.update(status="running", current=stage)
    save(EXP / "status.json", status)
    os.environ["M3_USAGE_LOG"] = str(EXP / f"usage-judge-{label}.jsonl")
    judge_llm = build_judge_llm(config_path=str(EXP / "private" / f"moderate-{videos[0]}.yaml"))
    with output.open("a") as stream:
        for task in tasks:
            key = (task["arm"], task["video_id"], task["question_id"])
            if key in done:
                continue
            score, reasoning, error = 0.0, "Invalid or missing answer", False
            if not task["error"] and task["predicted_answer"]:
                for attempt in range(3):
                    with operation_deadline(120):
                        score, reasoning, error = judge_answer(
                            judge_llm=judge_llm, question=task["question"],
                            expected=task["expected_answer"], predicted=task["predicted_answer"])
                    if not error:
                        break
                    time.sleep(30 * (attempt + 1))
                if error:
                    raise RuntimeError(f"Judge failed: {key}")
            stream.write(json.dumps({**task, "judge_score": score, "judge_reasoning": reasoning},
                                    ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            done.add(key)
            status["stages"][stage] = {"completed": len(done), "total": len(tasks)}
            save(EXP / "status.json", status)
    status["completed"].append(stage)
    save(EXP / "status.json", status)


def choose():
    judgments = rows(EXP / "judgments-pilot.jsonl")
    means = {arm: sum(r["judge_score"] for r in judgments if r["arm"] == arm)
             / sum(r["arm"] == arm for r in judgments)
             for arm in ("baseline", *POLICIES)}
    usage = {}
    for arm in POLICIES:
        calls = [r for video in PILOT for r in rows(EXP / f"usage-{arm}-{video}.jsonl")]
        if any(r.get("status") != "returned" or not r.get("usage") for r in calls):
            raise ValueError(f"Incomplete usage for {arm}")
        usage[arm] = sum(r["usage"].get("total_tokens", 0) for r in calls)
    eligible = [arm for arm in POLICIES if means[arm] >= means["baseline"] - 0.05]
    selected = min(eligible, key=lambda arm: usage[arm]) if eligible else max(POLICIES, key=lambda arm: means[arm])
    save(EXP / "pilot-selection.json", {"means": means, "total_tokens": usage,
                                       "eligible": eligible, "selected": selected})
    return selected


def main():
    if sys.argv[1:] == ["prepare"]:
        prepare()
        return
    if sys.argv[1:] != ["run"]:
        raise ValueError("Use prepare or run")
    status = json.loads((EXP / "status.json").read_text())
    manifest = json.loads((EXP / "manifest.json").read_text())
    for video, expected in manifest["baseline_sha256"].items():
        if digest(BASE_RUN / video / "m3_qa_results.jsonl") != expected:
            raise ValueError(f"Baseline changed: {video}")
    try:
        for policy in POLICIES:
            for video in PILOT:
                qa(policy, video, status)
        judge(PILOT, ["baseline", *POLICIES], status, "pilot")
        selected = choose()
        for video in EXPANSION:
            qa(selected, video, status)
        judge(EXPANSION, ["baseline", selected], status, "expansion")
        status.update(status="completed", current=None, selected=selected)
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(EXP / "status.json", status)


if __name__ == "__main__":
    main()
