#!/usr/bin/env python3
"""Accuracy-first, resumable single-variable QA comparisons on frozen memory."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parent.parent
NAME = "m3_accuracy_first_20261109"
RUN = ROOT / "runs" / NAME
EXP = ROOT / "experiments" / NAME
BASE_RUN = ROOT / "runs/m3_compact_replay_20261109"
BASE_EXP = ROOT / "experiments/m3_compact_replay_20261109"
SOURCE_RUN = ROOT / "runs/m3_dev_1031_ten_retry_20261108"
VIDEOS = ["bedroom_04", "kitchen_01", "bedroom_05", "gym_03", "study_23"]
M3_RUN = ROOT / "experiments/m3_dev_1031_ten_retry_20261108/frozen_baseline/m3_qa"
POLICIES = {
    "person_only": {"person_on_demand": True},
    "incremental_2500": {"incremental_tokens": 2500},
    "search_6": {"max_search_calls": 6},
}
DEADLINE = time.mktime(time.strptime("2026-11-09 23:00:00", "%Y-%m-%d %H:%M:%S"))

spec = importlib.util.spec_from_file_location(
    "efficiency_driver", ROOT / "scripts/run_m3_qa_efficiency_20261109.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
driver.RUN, driver.EXP = RUN, EXP


def save_status(status):
    driver.save(EXP / "status.json", status)


def prepare():
    if EXP.exists() or RUN.exists():
        raise FileExistsError("Experiment already exists")
    RUN.mkdir(parents=True)
    (EXP / "private").mkdir(parents=True, mode=0o700)
    shutil.copytree(BASE_RUN / "_source", RUN / "_source",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("control_adapter.py", "compact_person.py"):
        shutil.copyfile(ROOT / "oGMemory/m3" / name, RUN / "_source/m3" / name)
    shutil.copyfile(Path(__file__), RUN / "driver.py")
    copy_m3_answers()
    for policy, settings in POLICIES.items():
        for video in VIDEOS:
            config = yaml.safe_load((BASE_EXP / "private" / f"{video}.yaml").read_text())
            for key, value in settings.items():
                if key not in ("person_on_demand", "early_answer", "ambiguity_guard",
                               "later_incremental_tokens"):
                    config["m3"]["qa_context"][key] = value
            path = EXP / "private" / f"{policy}-{video}.yaml"
            path.write_text(yaml.safe_dump(config, allow_unicode=True))
            path.chmod(0o600)
    first_policy = next(iter(POLICIES))
    shutil.copyfile(EXP / "private" / f"{first_policy}-{VIDEOS[0]}.yaml",
                    EXP / "private" / f"moderate-{VIDEOS[0]}.yaml")
    (EXP / "private" / f"moderate-{VIDEOS[0]}.yaml").chmod(0o600)
    driver.save(EXP / "manifest.json", {
        "videos": VIDEOS, "policies": POLICIES, "deadline_local": "2026-11-09 23:00:00",
        "baseline": str(BASE_RUN.relative_to(ROOT)),
        "baseline_sha256": {video: driver.digest(BASE_RUN / video / "m3_qa_results.jsonl")
                            for video in VIDEOS},
        "m3_agent_sha256": {video: driver.digest(M3_RUN / video / "m3_qa_results.jsonl")
                            for video in VIDEOS},
        "code_sha256": {name: driver.digest(RUN / "_source/m3" / name) for name in
                        ("control_adapter.py", "compact_person.py", "context_budget.py",
                         "incremental_context.py", "index_episodic.py")},
        "selection": "No lower overall blind score than the original compact policy, "
                     "higher score than M3 Agent, no video decline over 0.1, "
                     "no additional failures, and at least 10% fewer QA tokens.",
        "caveat": "Small sample, baseline bedroom_04 includes a retry; results are not guarantees.",
    })
    save_status({"status": "prepared", "completed": [], "stages": {}})


def copy_m3_answers():
    for video in VIDEOS:
        source = M3_RUN / video / "m3_qa_results.jsonl"
        actual, reference = driver.latest(source), driver.latest(BASE_RUN / video / "m3_qa_results.jsonl")
        if actual.keys() != reference.keys() or any(
            actual[key].get(field) != reference[key].get(field)
            for key in reference for field in ("question", "expected_answer")
        ):
            raise ValueError(f"M3 Agent question coverage mismatch: {video}")
        folder = RUN / "m3_agent" / video
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, folder / "m3_qa_results.jsonl")


def qa(policy, video, status):
    stage = f"qa/{policy}/{video}"
    if stage in status["completed"]:
        return
    config = EXP / "private" / f"{policy}-{video}.yaml"
    folder = RUN / policy / video
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "m3_qa_results.jsonl"
    env = dict(os.environ, OGMEM_M3_COMPACT_PERSON="1", PYTHONUNBUFFERED="1",
               PYTHONHASHSEED="0", OGMEM_M3_MAX_WORKERS="1", OGMEM_EMBED_MAX_CONCURRENCY="1")
    env.pop("OGMEM_M3_EARLY_ANSWER", None)
    env.pop("OGMEM_M3_PERSON_ON_DEMAND", None)
    env.pop("OGMEM_M3_AMBIGUITY_GUARD", None)
    env.pop("OGMEM_M3_LATER_INCREMENTAL_TOKENS", None)
    if POLICIES[policy].get("person_on_demand"):
        env["OGMEM_M3_PERSON_ON_DEMAND"] = "1"
    if POLICIES[policy].get("early_answer"):
        env["OGMEM_M3_EARLY_ANSWER"] = "1"
    if POLICIES[policy].get("ambiguity_guard"):
        env["OGMEM_M3_AMBIGUITY_GUARD"] = "1"
    if POLICIES[policy].get("later_incremental_tokens"):
        env["OGMEM_M3_LATER_INCREMENTAL_TOKENS"] = str(
            POLICIES[policy]["later_incremental_tokens"])
    env["PYTHONPATH"] = f"{SOURCE_RUN / '_instrumentation'}:{RUN / '_source'}"
    env["OGMEM_GRAPH_CONNECTION_STRING"] = yaml.safe_load(config.read_text())["storage"]["connection_string"]
    env["M3_USAGE_LOG"] = str(EXP / f"usage-{policy}-{video}.jsonl")
    env["M3_AUDIT_VIDEO"] = video
    command = [str(driver.PYTHON), "-c",
               "import sys; from m3.qa import main_react_qa; sys.exit(main_react_qa(sys.argv[1:]))",
               "--config", str(config), "--annotations", "/data/m3_data/annotations/robot.json",
               "--video", video, "--graph-store", "sql", "--account-id", driver.ACCOUNT,
               "--output-jsonl", str(output), "--temperature", "0",
               "--max-iterations", "6", "--timeout-seconds", "180", "--top-k", "8",
               "--candidate-k", "20", "--transcript-top-k", "20", "--transcript-fusion", "rrf",
               "--context-k", "8", "--include-neighbors", "1", "--temporal-policy", "current_clip",
               "--person-retrieval", "--memory-coverage-json",
               str(SOURCE_RUN / video / "memory_coverage.json")]
    status.update(status="running", current=stage)
    save_status(status)
    started = time.monotonic()
    retries = 0
    while True:
        remaining = DEADLINE - time.time()
        if remaining <= 0:
            raise TimeoutError(f"{stage}: deadline reached before successful completion")
        attempt_command = [*command, "--resume"] if output.exists() else command
        with (EXP / "private" / f"{policy}-{video}.log").open("a") as log:
            result = subprocess.run(attempt_command, cwd=RUN / "_source", env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=remaining)
        if result.returncode not in (0, 2) or not output.exists():
            raise RuntimeError(f"{stage}: process exited {result.returncode}")
        actual = driver.latest(output)
        reference = driver.latest(BASE_RUN / video / "m3_qa_results.jsonl")
        if actual.keys() != reference.keys() or any(
            actual[key].get(field) != reference[key].get(field)
            for key in reference for field in ("question", "expected_answer")
        ):
            raise ValueError(f"{stage}: question coverage mismatch")
        failures = sum(bool(row.get("error")) or not row.get("predicted_answer")
                       for row in actual.values())
        if not failures:
            break
        retries += 1
        delay = min(300, 30 * (2 ** min(retries - 1, 4)))
        status.update(status="waiting_retry", current=stage)
        status["stages"][stage] = {"failures": failures, "retries": retries,
                                    "retry_delay_seconds": delay}
        save_status(status)
        remaining = DEADLINE - time.time()
        if remaining <= delay:
            raise TimeoutError(f"{stage}: deadline reached while waiting to retry")
        time.sleep(delay)
        status.update(status="running", current=stage)
        save_status(status)
    status["stages"][stage] = {
        "exit_code": result.returncode, "seconds": time.monotonic() - started,
        "questions": len(actual), "attempt_rows": len(driver.rows(output)),
        "failures": failures, "retries": retries,
    }
    status["completed"].append(stage)
    save_status(status)


def summarize(policy, status):
    judgments = driver.rows(EXP / f"judgments-{policy}.jsonl")
    scores = {}
    for arm in ("baseline", "m3_agent", policy):
        scores[arm] = {}
        for video in VIDEOS:
            values = [row["judge_score"] for row in judgments
                      if row["arm"] == arm and row["video_id"] == video]
            scores[arm][video] = sum(values) / len(values)
    totals = {}
    for arm in ("baseline", "m3_agent", policy):
        values = [row["judge_score"] for row in judgments if row["arm"] == arm]
        usage = []
        for video in VIDEOS:
            path = (BASE_EXP / f"usage-qa-{video}.jsonl" if arm == "baseline"
                    else ROOT / "experiments/m3_dev_1031_ten_retry_20261108/frozen_baseline"
                    / f"usage-m3_qa-{video}.jsonl" if arm == "m3_agent"
                    else EXP / f"usage-{policy}-{video}.jsonl")
            usage.extend(driver.rows(path))
        if any(row.get("status") == "returned" and not row.get("usage") for row in usage):
            raise ValueError(f"Returned call without token usage for {arm}")
        unknown_usage_calls = sum(row.get("status") != "returned" for row in usage)
        failures = sum(bool(row.get("error")) or not row.get("predicted_answer")
                       for row in judgments if row["arm"] == arm)
        totals[arm] = {"score": sum(values) / len(values), "questions": len(values),
                       "tokens": sum((row.get("usage") or {}).get("total_tokens", 0)
                                     for row in usage),
                       "failures": failures, "unknown_usage_calls": unknown_usage_calls}
    baseline, candidate = totals["baseline"], totals[policy]
    eligible = (candidate["score"] >= baseline["score"]
                and candidate["score"] > totals["m3_agent"]["score"]
                and candidate["tokens"] <= baseline["tokens"] * 0.9
                and candidate["failures"] <= baseline["failures"]
                and candidate["unknown_usage_calls"] == 0
                and all(scores[policy][video] >= scores["baseline"][video] - 0.1
                        for video in VIDEOS))
    driver.save(EXP / f"summary-{policy}.json",
                {"totals": totals, "scores_by_video": scores, "eligible": eligible})
    status["stages"][f"summary/{policy}"] = {"eligible": eligible}
    save_status(status)


def main():
    if sys.argv[1:] == ["prepare"]:
        prepare()
        return
    if sys.argv[1:] != ["run"]:
        raise ValueError("Use prepare or run")
    status = json.loads((EXP / "status.json").read_text())
    status.pop("error", None)
    copy_m3_answers()
    manifest = json.loads((EXP / "manifest.json").read_text())
    m3_hashes = {video: driver.digest(M3_RUN / video / "m3_qa_results.jsonl")
                 for video in VIDEOS}
    if "m3_agent_sha256" in manifest and manifest["m3_agent_sha256"] != m3_hashes:
        raise ValueError("M3 Agent baseline changed")
    manifest["m3_agent_sha256"] = m3_hashes
    driver.save(EXP / "manifest.json", manifest)
    for video, expected in manifest["baseline_sha256"].items():
        if driver.digest(BASE_RUN / video / "m3_qa_results.jsonl") != expected:
            raise ValueError(f"Baseline changed: {video}")
    try:
        for policy in POLICIES:
            if time.time() >= DEADLINE:
                status.update(status="deadline_reached", current=None)
                return
            for video in VIDEOS:
                if time.time() >= DEADLINE:
                    status.update(status="deadline_reached", current=None)
                    return
                qa(policy, video, status)
            if time.time() >= DEADLINE:
                status.update(status="deadline_reached", current=None)
                return
            driver.judge(VIDEOS, ["baseline", "m3_agent", policy], status, policy)
            summarize(policy, status)
        status.update(status="completed", current=None)
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_status(status)


if __name__ == "__main__":
    main()
