#!/usr/bin/env python3
"""Frozen-source, QA-only ablations on the fixed five-video cohort.

Run with oGMemory/.venv/bin/python after sourcing ~/.bashrc. Private configs
remain local. Never overwrite a run, regenerate memories, or select retries.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(os.environ.get("M3_EVAL_ROOT", Path(__file__).resolve().parent.parent))
COHORT = ["study_05", "gym_03", "bedroom_04", "study_18", "kitchen_20"]
BASE = "m3_flash_medium_20261031"
RECOVERY = "m3_study18_recovery_20261101"
ACCOUNT = "m3-flash-medium-20261031"
DEADLINE = datetime.datetime.fromisoformat("2026-11-01T23:30:00+08:00").timestamp()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    if not os.environ.get("MINIMAX_API_KEY"):
        raise RuntimeError("Source ~/.bashrc first; API key is not printed")
    exp = ROOT / "experiments" / args.name
    run = ROOT / "runs" / args.name
    if exp.exists() or run.exists():
        raise RuntimeError("Run exists; refuse to overwrite")
    if subprocess.check_output(["git", "-C", str(ROOT / "oGMemory"),
                                "status", "--porcelain"]).strip():
        raise RuntimeError("Commit oGMemory changes before freezing an experiment")
    run.mkdir(parents=True)
    (exp / "private").mkdir(parents=True, mode=0o700)
    source = run / "_source"
    source.mkdir()
    archive = subprocess.run(["git", "-C", str(ROOT / "oGMemory"), "archive", "HEAD"],
                             check=True, stdout=subprocess.PIPE).stdout
    subprocess.run(["tar", "-x", "-C", str(source)], input=archive, check=True)
    tokenizer = ROOT / "runs/context_assets/qwen-omni-tokenizer.json"
    if not tokenizer.exists():
        tokenizer.parent.mkdir(parents=True, exist_ok=True)
        remote = ("/root/.cache/huggingface/hub/"
                  "models--Qwen--Qwen2.5-Omni-7B/snapshots/"
                  "ae9e1690543ffd5c0221dc27f79834d0294cba00/tokenizer.json")
        with tokenizer.open("xb") as stream:
            subprocess.run(["docker", "exec", "t4-omni-omni-1", "cat", remote],
                           check=True, stdout=stream)
    expected = "8441917e39ae0244e06d704b95b3124795cec478e297f9afac39ba670d7e9d99"
    if digest(tokenizer) != expected:
        raise RuntimeError("Tokenizer does not match pinned local artifact")
    settings = dict(
        enabled=True, context_window=32768, max_input_tokens=24000,
        safety_tokens=2048, max_top_k=16, max_candidates=32,
        max_result_nodes=24, max_neighbor_clips=64, max_search_calls=8,
        max_selected_nodes=32, max_passages_per_node=32,
        tokenizer_path=str(tokenizer), tokenizer_multiplier=1.15,
    )
    if args.answer_reserve:
        settings["answer_reserve_seconds"] = args.answer_reserve
    for video in COHORT:
        origin = RECOVERY if video == "study_18" else BASE
        config = yaml.safe_load((ROOT / "experiments" / origin / "private/config.yaml").read_text())
        config["m3"]["qa_context"] = settings
        path = exp / "private" / f"{video}.yaml"
        path.write_text(yaml.safe_dump(config))
        path.chmod(0o600)
    manifest = dict(
        name=args.name, prepared_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        source_commit=subprocess.check_output(
            ["git", "-C", str(ROOT / "oGMemory"), "rev-parse", "HEAD"], text=True).strip(),
        source_archive_sha256=hashlib.sha256(archive).hexdigest(),
        cohort=COHORT, total_questions=65, account=ACCOUNT,
        memory_runs={v: RECOVERY if v == "study_18" else BASE for v in COHORT},
        qa_context={**settings, "tokenizer_path": "local pinned tokenizer (see SHA256)"},
        tokenizer_sha256=expected, model="MiniMax-M3.1-Flash", reasoning_effort="medium",
        thinking="adaptive", qa_output_tokens=8192, judge_output_tokens=2048,
        max_iterations=6, timeout_seconds=180, temporal_policy="current_clip",
        note="QA-only, same stored memories. Historical baseline not contemporaneous. No best-of retries.",
        authorization_deadline="2026-11-01T23:30:00+08:00",
    )
    save(exp / "manifest.json", manifest)
    # Freeze the runner and instrumentation too, so later edits cannot change it.
    (run / "runner.py").write_bytes(Path(__file__).read_bytes())
    instrumentation = run / "_instrumentation"
    instrumentation.mkdir()
    (instrumentation / "sitecustomize.py").write_bytes(
        (ROOT / "scripts/instrumentation/sitecustomize.py").read_bytes())
    save(exp / "status.json", {"status": "prepared"})
    return run


def worker(args):
    run = ROOT / "runs" / args.name
    exp = ROOT / "experiments" / args.name
    sys.path.insert(0, str(run / "_source"))
    from m3.config import load_m3_runtime_config
    from m3.qa import main_react_qa
    from m3.control_adapter import OGMemoryControlAdapter

    config = exp / "private" / f"{args.video}.yaml"
    os.environ["OGMEM_GRAPH_CONNECTION_STRING"] = load_m3_runtime_config(str(config)).graph_connection_string
    output = run / args.video
    output.mkdir()
    qa = output / "m3_qa_results.jsonl"
    original = OGMemoryControlAdapter.run
    index = 0

    def measured(self, ctx, task):
        nonlocal index
        index += 1
        if time.time() >= DEADLINE - 300:
            raise RuntimeError("Experiment scheduling cutoff reached; no new QA call started")
        os.environ["M3_AUDIT_VIDEO"] = args.video
        os.environ["M3_AUDIT_QUESTION_INDEX"] = str(index - 1)
        started = time.monotonic()
        try:
            result = original(self, ctx, task)
            task.search_stats["iteration_trace"] = [
                vars(it) for it in result.trace.iterations
            ]
            return result
        finally:
            task.search_stats["wall_seconds"] = time.monotonic() - started
    OGMemoryControlAdapter.run = measured
    rc = main_react_qa([
        "--config", str(config), "--annotations", "/data/m3_data/annotations/robot.json",
        "--video", args.video, "--graph-store", "sql", "--account-id", ACCOUNT,
        "--output-jsonl", str(qa), "--temperature", "0", "--max-iterations", "6",
        "--timeout-seconds", "180", "--top-k", "8", "--candidate-k", "20",
        "--transcript-top-k", "20", "--transcript-fusion", "rrf",
        "--context-k", "8", "--include-neighbors", "1", "--temporal-policy", "current_clip",
    ])
    save(output / "qa_exit.json", {"exit_code": rc})
    return rc


def driver(args):
    run = ROOT / "runs" / args.name
    exp = ROOT / "experiments" / args.name
    python = ROOT / "oGMemory/.venv/bin/python"
    env = {**os.environ, "M3_EVAL_ROOT": str(ROOT),
           "PYTHONPATH": f"{run}/_instrumentation:{run}/_source",
           "OGMEM_MINIMAX_MODEL": "MiniMax-M3.1-Flash",
           "M3_USAGE_LOG": str(exp / "model-usage.jsonl")}
    started = time.time()
    states = {}
    for video in COHORT:
        if time.time() >= DEADLINE - 900:
            save(exp / "status.json", {"status": "deadline_stopped", "videos": states})
            return
        save(exp / "status.json", {"status": "running", "video": video, "videos": states})
        with (exp / "private" / f"{video}.log").open("x") as log:
            try:
                result = subprocess.run([str(python), str(run / "runner.py"),
                                         "--name", args.name, "--worker", "--video", video],
                                        env=env, cwd=run / "_source", stdout=log, stderr=subprocess.STDOUT,
                                        timeout=max(1, DEADLINE - time.time() - 600))
            except subprocess.TimeoutExpired:
                states[video] = {"qa_exit": "deadline_terminated"}
                save(exp / "status.json", {"status": "deadline_stopped", "videos": states})
                return
        states[video] = {"qa_exit": result.returncode}
        qa = run / video / "m3_qa_results.jsonl"
        if qa.exists():
            states[video]["records"] = len(qa.read_text().splitlines())
            if time.time() >= DEADLINE - 600:
                save(exp / "status.json", {"status": "deadline_stopped", "videos": states})
                return
            with (exp / "private" / f"{video}-judge.log").open("x") as log:
                try:
                    judge = subprocess.run([
                    str(python), "-m", "m3.eval", "--config", str(exp / "private" / f"{video}.yaml"),
                    "--annotations", "/data/m3_data/annotations/robot.json",
                    "--video", video, "--input-qa-results", str(qa),
                    "--output-dir", str(run / video / "judge"),
                    ], env=env, cwd=run / "_source", stdout=log, stderr=subprocess.STDOUT,
                        timeout=max(1, DEADLINE - time.time() - 300))
                except subprocess.TimeoutExpired:
                    states[video]["judge_exit"] = "deadline_terminated"
                    save(exp / "status.json", {"status": "deadline_stopped", "videos": states})
                    return
            states[video]["judge_exit"] = judge.returncode
    save(exp / "status.json", {"status": "finished", "videos": states,
                               "wall_seconds": time.time() - started})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--answer-reserve", type=int, default=0)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--driver", action="store_true")
    p.add_argument("--video", choices=COHORT)
    args = p.parse_args()
    if not args.name.replace("_", "").isalnum():
        raise SystemExit("Unsafe experiment name")
    if args.worker:
        raise SystemExit(worker(args))
    if args.driver:
        driver(args)
    else:
        prepared = prepare(args)
        print(f"Prepared {prepared}; launch frozen runner with --driver")
