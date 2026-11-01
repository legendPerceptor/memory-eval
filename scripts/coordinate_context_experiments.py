#!/usr/bin/env python3
"""Quota-aware, serial scheduling within the user's November 1 deadline.

Read-only quota checks do not change subscriptions or payment settings.
Model/QA sources stay frozen in each prepared run; no success retry selection.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import requests

ROOT = Path(os.environ.get("M3_EVAL_ROOT", Path(__file__).resolve().parent.parent))
EXP = ROOT / "experiments/m3_context_study_20261101"
DEADLINE = datetime.datetime.fromisoformat("2026-11-01T23:30:00+08:00").timestamp()
JOBS = [
    ("m3_context_budget_recovered_20261101", "2026-11-01T10:00:00+08:00"),
    ("m3_context_finalization_recovered_20261101", "2026-11-01T10:00:00+08:00"),
    ("m3_context_compact12k_guarded_20261101", "2026-11-01T15:05:00+08:00"),
    ("m3_context_grouped12k_20261101", "2026-11-01T20:05:00+08:00"),
]


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def status(**fields):
    save(EXP / "coordinator_status.json", {
        "observed_at": datetime.datetime.now().astimezone().isoformat(), **fields})


def quota():
    observed = time.time()
    try:
        response = requests.get(
            "https://www.minimax.cn/v1/token_plan/remains",
            headers={"Authorization": "Bearer " + os.environ["MINIMAX_API_KEY"]},
            timeout=30, allow_redirects=False)
        body = response.json()
        raw = {"observed_at_unix": observed, "http_status": response.status_code, "body": body}
        file = EXP / "private" / f"quota-{int(observed)}.json"
        with file.open("x") as stream:
            json.dump(raw, stream, ensure_ascii=False, indent=2)
        file.chmod(0o600)
        if response.status_code != 200 or body.get("base_resp", {}).get("status_code") != 0:
            return None
        general = next((r for r in body.get("model_remains", []) if r["model_name"] == "general"), None)
        if general is None:
            return None
        end = general["end_time"]
        end = end / 1000 if end > 1e11 else end
        safe = {
            "observed_at_unix": observed,
            "interval_remaining_percent": general["current_interval_remaining_percent"],
            "weekly_remaining_percent": general["current_weekly_remaining_percent"],
            "window_end_unix": end,
        }
        with (EXP / "quota_observations.jsonl").open("a") as stream:
            stream.write(json.dumps(safe) + "\n")
        return safe
    except Exception as error:
        with (EXP / "quota_observations.jsonl").open("a") as stream:
            stream.write(json.dumps({"observed_at_unix": observed,
                                     "query_error_type": type(error).__name__}) + "\n")
        return None


def wait_until(timestamp, label):
    while time.time() < timestamp:
        if time.time() >= DEADLINE - 900:
            status(status="deadline_stopped", pending=label)
            return False
        status(status="scheduled_wait", pending=label,
               next_start=datetime.datetime.fromtimestamp(timestamp).astimezone().isoformat())
        time.sleep(min(60, max(1, timestamp - time.time())))
    return True


def run_command(command, log, env, label):
    with log.open("x") as stream:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                              stderr=subprocess.STDOUT, start_new_session=True) as process:
            status(status="running", job=label, active_process_group=process.pid)
            try:
                return process.wait(timeout=max(1, DEADLINE - time.time() - 180))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                return 74


def main():
    if not os.environ.get("MINIMAX_API_KEY"):
        raise RuntimeError("Source ~/.bashrc first")
    python = ROOT / "oGMemory/.venv/bin/python"
    env = {**os.environ, "M3_EVAL_ROOT": str(ROOT)}
    for key in ("M3_AUDIT_VIDEO", "M3_AUDIT_QUESTION_INDEX", "M3_USAGE_LOG"):
        env.pop(key, None)
    save(EXP / "execution_schedule.json", {
        "jobs": [{"name": name, "not_before": not_before} for name, not_before in JOBS],
        "deadline": "2026-11-01T23:30:00+08:00",
        "minimum_reported_interval_remaining_percent": 25,
        "coordinator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Conservative window pacing, not a claim of exact token capacity; no purchase or key/model switching.",
    })
    for name, not_before in JOBS:
        if not wait_until(datetime.datetime.fromisoformat(not_before).timestamp(), name):
            return
        state_file = ROOT / "experiments" / name / "status.json"
        state = json.loads(state_file.read_text())
        if state["status"] == "finished":
            continue
        if state["status"] not in ("prepared", "held_provider_quota"):
            status(status="needs_recovery_review", job=name, worker_status=state["status"])
            return
        while True:
            if time.time() >= DEADLINE - 900:
                status(status="deadline_stopped", pending=name)
                return
            available = quota()
            if available and available["weekly_remaining_percent"] <= 0:
                status(status="provider_weekly_quota_exhausted", pending=name)
                return
            if available and available["interval_remaining_percent"] >= 25:
                break
            next_check = (available["window_end_unix"] + 60) if available else time.time() + 600
            if not wait_until(max(time.time() + 60, next_check), name):
                return
        rc = run_command(
            [str(python), str(ROOT / "runs" / name / "runner.py"), "--name", name, "--driver"],
            ROOT / "experiments" / name / "private/coordinator-driver.log", env, name)
        state = json.loads(state_file.read_text())
        fatal_stage = any(value not in (0, 2) for video in state.get("videos", {}).values()
                          for key, value in video.items() if key.endswith("_exit"))
        if rc or state["status"] != "finished" or fatal_stage:
            status(status="needs_recovery_review", job=name, process_exit=rc,
                   worker_status=state["status"])
            return
        subprocess.run([str(python), "scripts/summarize_context_experiment.py", name],
                       check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        quota()  # observe the actual change, without guessing a token-to-quota conversion
    source = ROOT / "runs/m3_context_compact12k_guarded_20261101"
    judge_env = {**env, "PYTHONPATH": f"{source}/_instrumentation:{source}/_source",
                 "M3_USAGE_LOG": str(ROOT / "experiments/m3_context_joint_judge_20261101/model-usage.jsonl")}
    # Freeze the final schedule's judge script before execution.
    judge_script = ROOT / "runs/context_assets/judge_context_comparison_recovery.py"
    if not judge_script.exists():
        judge_script.write_bytes((ROOT / "scripts/judge_context_comparison.py").read_bytes())
    rc = run_command([str(python), str(judge_script)],
                     EXP / "private/joint-judge-recovery-driver.log", judge_env, "joint_judge")
    joint_file = ROOT / "experiments/m3_context_joint_judge_20261101/status.json"
    joint = json.loads(joint_file.read_text()) if joint_file.exists() else {}
    status(status=("qa_and_judging_finished" if not rc and joint.get("status") == "finished"
                   else "joint_judge_needs_recovery"), judge_exit=rc, joint_status=joint)


if __name__ == "__main__":
    main()
