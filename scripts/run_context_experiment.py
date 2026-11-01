#!/usr/bin/env python3
"""Frozen-source, QA-only ablations on the fixed five-video cohort.

Run with oGMemory/.venv/bin/python after sourcing ~/.bashrc. Private configs
remain local. Never overwrite a run, regenerate memories, or select retries.
"""
import argparse
import datetime
import dataclasses
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
DEADLINE_ISO = os.environ.get("M3_EXPERIMENT_DEADLINE", "2026-11-01T23:30:00+08:00")
DEADLINE = datetime.datetime.fromisoformat(DEADLINE_ISO).timestamp()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quota_exhausted(error):
    """Match the observed provider quota condition, not every transient 429."""
    text = str(error)
    return ("已达到 Token Plan 用量上限" in text
            or ("Token Plan" in text and "(2056)" in text))


def seed_recovery(destination, origin, source_name):
    """Carry only validated successful answers/grades, never pick among retries."""
    receipt = {"source": source_name, "reused_answers": [], "reused_grades": []}
    for video in COHORT:
        old = origin / video / "m3_qa_results.jsonl"
        records = [json.loads(s) for s in old.read_text().splitlines()] if old.exists() else []
        assert len({r["question_id"] for r in records}) == len(records)
        valid = [r for r in records if (
            not r.get("error") and r.get("finish_reason") == "done"
            and isinstance(r.get("predicted_answer"), str) and r["predicted_answer"].strip()
            and not (r.get("memory_coverage") or {}).get("partial")
            and not (r.get("memory_coverage") or {}).get("failed_clips")
        )]
        selected = {r["question_id"]: r for r in valid}
        for row in records:
            if row["question_id"] not in selected and not quota_exhausted(row.get("error", "")):
                raise ValueError("Selective recovery is limited to quota failures; preserve other failures in a separate full trial")
        target = destination / video
        target.mkdir()
        (target / "m3_qa_results.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in valid))
        receipt["reused_answers"].extend({
            "video_id": video, "question_id": r["question_id"], "question_index": r["question_index"],
            "canonical_row_sha256": hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest(),
        } for r in valid)
        old_grade = origin / video / "judge/m3_eval_results.jsonl"
        if not old_grade.exists():
            continue
        grades = [json.loads(s) for s in old_grade.read_text().splitlines()]
        assert len({r["question_id"] for r in grades}) == len(grades)
        reused = []
        for grade in grades:
            if grade["question_id"] not in selected or grade.get("judge_error") or grade.get("error"):
                continue
            assert grade["predicted_answer"] == selected[grade["question_id"]]["predicted_answer"]
            reused.append(grade)
            receipt["reused_grades"].append({
                "video_id": video, "question_id": grade["question_id"],
                "canonical_row_sha256": hashlib.sha256(json.dumps(grade, sort_keys=True).encode()).hexdigest(),
            })
        (target / "judge").mkdir()
        (target / "judge/m3_eval_results.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reused))
    return receipt


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
    inherited = None
    source_from = getattr(args, "source_from", None)
    if source_from:
        inherited = json.loads((ROOT / "experiments" / source_from / "manifest.json").read_text())
    revision = inherited["source_commit"] if inherited else "HEAD"
    run.mkdir(parents=True)
    (exp / "private").mkdir(parents=True, mode=0o700)
    source = run / "_source"
    source.mkdir()
    archive = subprocess.run(["git", "-C", str(ROOT / "oGMemory"), "archive", revision],
                             check=True, stdout=subprocess.PIPE).stdout
    if inherited:
        assert hashlib.sha256(archive).hexdigest() == inherited["source_archive_sha256"]
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
        enabled=True, context_window=32768, max_input_tokens=args.input_budget,
        safety_tokens=2048, max_top_k=16, max_candidates=32,
        max_result_nodes=24, max_neighbor_clips=64, max_search_calls=8,
        max_selected_nodes=32, max_passages_per_node=32,
        tokenizer_path=str(tokenizer), tokenizer_multiplier=1.15,
        evidence_format=args.evidence_format,
    )
    if args.answer_reserve:
        settings["answer_reserve_seconds"] = args.answer_reserve
    for option in ("fact_memory", "identity_aware", "incremental_delivery"):
        if getattr(args, option, False):
            settings[option] = True
    if getattr(args, "fact_memory", False):
        settings["max_passages_per_node"] = 12
    if getattr(args, "fact_index", None):
        index_path = Path(args.fact_index).resolve()
        sys.path.insert(0, str(ROOT / "oGMemory"))
        from m3.fact_index import load_fact_index
        index_hash = digest(index_path)
        load_fact_index(str(index_path), index_hash)
        settings.update(fact_index_path=str(index_path), fact_index_sha256=index_hash)
    for video in COHORT:
        origin = source_from or (RECOVERY if video == "study_18" else BASE)
        config_name = f"{video}.yaml" if inherited else "config.yaml"
        config = yaml.safe_load((ROOT / "experiments" / origin / "private" / config_name).read_text())
        if inherited:
            settings = config["m3"]["qa_context"]
        else:
            config["m3"]["qa_context"] = settings
        path = exp / "private" / f"{video}.yaml"
        path.write_text(yaml.safe_dump(config))
        path.chmod(0o600)
    manifest = dict(
        name=args.name, prepared_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        source_commit=subprocess.check_output(
            ["git", "-C", str(ROOT / "oGMemory"), "rev-parse", revision], text=True).strip(),
        source_archive_sha256=hashlib.sha256(archive).hexdigest(),
        cohort=COHORT, total_questions=65, account=ACCOUNT,
        memory_runs={v: RECOVERY if v == "study_18" else BASE for v in COHORT},
        qa_context={**settings, "tokenizer_path": "local pinned tokenizer (see SHA256)"},
        tokenizer_sha256=expected, model="MiniMax-M3.1-Flash", reasoning_effort="medium",
        thinking="adaptive", qa_output_tokens=8192, judge_output_tokens=2048,
        max_iterations=6, timeout_seconds=180, temporal_policy="current_clip",
        note="QA-only, same stored memories. Historical baseline not contemporaneous. No best-of retries.",
        authorization_deadline=DEADLINE_ISO,
    )
    if inherited:
        manifest["source_from"] = source_from
        manifest["runner_changes"] = "Quota/deadline guards and explicit successful-row reuse; model source/config unchanged."
    if getattr(args, "independent_replication", False):
        manifest["independent_replication_of"] = source_from
        manifest["attempt_relationship"] = "independent_replication"
        manifest["note"] = "Independent full 65-question repeat; no answer/grade reuse or best-of selection."
    if getattr(args, "resume_successes", False):
        manifest["recovery"] = seed_recovery(run, ROOT / "runs" / source_from, source_from)
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
    manifest_path = exp / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    resume = bool(manifest.get("recovery"))
    output.mkdir(exist_ok=resume)
    qa = output / "m3_qa_results.jsonl"
    original = OGMemoryControlAdapter.run
    index = 0
    quota_stop = False
    pending_indices = None
    if resume:
        from m3.qa import load_questions
        from m3.qa_results import completed_keys
        done = completed_keys(qa)
        pending_indices = [q["question_index"]
                           for q in load_questions(Path("/data/m3_data/annotations/robot.json"))
                           if q["video_id"] == args.video
                           and (q["video_id"], q["question_id"]) not in done]

    def measured(self, ctx, task):
        nonlocal index, quota_stop
        if quota_stop:
            raise SystemExit(75)  # first error row was already persisted by QA
        index += 1
        if time.time() >= DEADLINE - 300:
            raise SystemExit(74)
        os.environ["M3_AUDIT_VIDEO"] = args.video
        os.environ["M3_AUDIT_QUESTION_INDEX"] = str(
            pending_indices[index - 1] if pending_indices is not None else index - 1)
        started = time.monotonic()
        try:
            result = original(self, ctx, task)
            task.search_stats["iteration_trace"] = [
                vars(it) for it in result.trace.iterations
            ]
            return result
        except Exception as error:
            if quota_exhausted(error):
                quota_stop = True
                task.search_stats["provider_failure"] = "quota_exhausted_2056"
                save(output / "quota_stop.json", {"provider_code": 2056,
                     "question_index": index - 1, "observed_at_unix": time.time()})
            raise
        finally:
            task.search_stats["wall_seconds"] = time.monotonic() - started
    OGMemoryControlAdapter.run = measured
    try:
        rc = main_react_qa([
        "--config", str(config), "--annotations", "/data/m3_data/annotations/robot.json",
        "--video", args.video, "--graph-store", "sql", "--account-id", ACCOUNT,
        "--output-jsonl", str(qa), "--temperature", "0", "--max-iterations", "6",
        "--timeout-seconds", "180", "--top-k", "8", "--candidate-k", "20",
        "--transcript-top-k", "20", "--transcript-fusion", "rrf",
        "--context-k", "8", "--include-neighbors", "1", "--temporal-policy", "current_clip",
        ] + (["--resume"] if resume else []))
    except SystemExit as stopped:
        if stopped.code not in (74, 75):
            raise
        rc = stopped.code
    if quota_stop:
        rc = 75
    save(output / "qa_exit.json", {"exit_code": rc})
    return rc


def judge_worker(args):
    run, exp = ROOT / "runs" / args.name, ROOT / "experiments" / args.name
    sys.path.insert(0, str(run / "_source"))
    from m3 import eval as evaluation
    original = evaluation.judge_answer
    stopped = False
    destination = run / args.video / "judge"
    manifest_path = exp / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    def guarded(*a, **kw):
        nonlocal stopped
        if stopped:
            raise SystemExit(75)
        if time.time() >= DEADLINE - 300:
            raise SystemExit(74)
        result = original(*a, **kw)
        if result[2] and quota_exhausted(result[1]):
            stopped = True
            save(destination / "quota_stop.json", {"provider_code": 2056,
                                                  "observed_at_unix": time.time()})
        return result  # let eval persist the first failed grade before stopping

    evaluation.judge_answer = guarded
    try:
        rc = evaluation.main([
            "--config", str(exp / "private" / f"{args.video}.yaml"),
            "--annotations", "/data/m3_data/annotations/robot.json",
            "--video", args.video, "--input-qa-results",
            str(run / args.video / "m3_qa_results.jsonl"),
            "--output-dir", str(destination),
        ] + (["--resume"] if manifest.get("recovery") else []))
    except SystemExit as error:
        if error.code not in (74, 75):
            raise
        rc = error.code
    if manifest.get("recovery") and rc not in (74, 75):
        # Upstream judge-only resume summarizes only the newly graded batch.
        # Preserve that batch summary and publish an explicitly rebuilt full
        # summary without changing any successful grading rows.
        summary_path = destination / "m3_eval_summary.json"
        batch = json.loads(summary_path.read_text())
        summary_path.rename(destination / "m3_eval_resume_batch_summary.json")
        records = [json.loads(s) for s in (destination / "m3_eval_results.jsonl").read_text().splitlines()]
        assert len({r["question_id"] for r in records}) == len(records)
        fields = {f.name for f in dataclasses.fields(evaluation.QuestionResult)}
        combined = evaluation.aggregate(
            [evaluation.QuestionResult(**{k: v for k, v in r.items() if k in fields}) for r in records],
            judge_provider=batch["judge_provider"], judge_model=batch["judge_model"])
        combined.update(
            status="failed" if combined["failed"] or combined["judge_error_count"] else "completed",
            resumed_existing_results=batch["skipped"], newly_graded_results=batch["total_questions"],
            input_qa_results=str(run / args.video / "m3_qa_results.jsonl"),
            results_path=str(destination / "m3_eval_results.jsonl"))
        save(summary_path, combined)
    return 75 if stopped else rc


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
        if result.returncode in (74, 75):
            save(exp / "status.json", {
                "status": "stopped_provider_quota" if result.returncode == 75 else "deadline_stopped",
                "videos": states})
            return
        if result.returncode not in (0, 2):
            save(exp / "status.json", {"status": "worker_failed", "videos": states})
            return
        qa = run / video / "m3_qa_results.jsonl"
        if qa.exists():
            states[video]["records"] = len(qa.read_text().splitlines())
            if time.time() >= DEADLINE - 600:
                save(exp / "status.json", {"status": "deadline_stopped", "videos": states})
                return
            with (exp / "private" / f"{video}-judge.log").open("x") as log:
                try:
                    judge = subprocess.run([
                    str(python), str(run / "runner.py"), "--name", args.name,
                    "--judge-worker", "--video", video,
                    ], env=env, cwd=run / "_source", stdout=log, stderr=subprocess.STDOUT,
                        timeout=max(1, DEADLINE - time.time() - 300))
                except subprocess.TimeoutExpired:
                    states[video]["judge_exit"] = "deadline_terminated"
                    save(exp / "status.json", {"status": "deadline_stopped", "videos": states})
                    return
            states[video]["judge_exit"] = judge.returncode
            if judge.returncode in (74, 75):
                save(exp / "status.json", {
                    "status": "stopped_provider_quota" if judge.returncode == 75 else "deadline_stopped",
                    "videos": states})
                return
            if judge.returncode not in (0, 2):
                save(exp / "status.json", {"status": "judge_worker_failed", "videos": states})
                return
    save(exp / "status.json", {"status": "finished", "videos": states,
                               "wall_seconds": time.time() - started})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--answer-reserve", type=int, default=0)
    p.add_argument("--input-budget", type=int, default=24000)
    p.add_argument("--evidence-format", choices=("passages", "grouped"), default="passages")
    p.add_argument("--fact-memory", action="store_true")
    p.add_argument("--identity-aware", action="store_true")
    p.add_argument("--incremental-delivery", action="store_true")
    p.add_argument("--fact-index", help="Pinned source-fact sidecar manifest")
    p.add_argument("--source-from", help="Reuse a prior arm's exact frozen source and private configuration")
    p.add_argument("--resume-successes", action="store_true",
                   help="With --source-from, preserve successful answers and successful matching grades")
    p.add_argument("--independent-replication", action="store_true",
                   help="Repeat the frozen source/config on all questions; do not count reference run as retry overhead")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--judge-worker", action="store_true")
    p.add_argument("--driver", action="store_true")
    p.add_argument("--video", choices=COHORT)
    args = p.parse_args()
    if not args.name.replace("_", "").isalnum():
        raise SystemExit("Unsafe experiment name")
    if args.input_budget <= 0:
        raise SystemExit("Input budget must be positive")
    if args.source_from and not args.source_from.replace("_", "").isalnum():
        raise SystemExit("Unsafe source experiment name")
    if args.source_from and (args.input_budget != 24000 or args.answer_reserve != 0
                             or args.evidence_format != "passages" or args.fact_memory
                             or args.identity_aware or args.incremental_delivery or args.fact_index):
        raise SystemExit("--source-from preserves its configuration; do not combine it with tuning overrides")
    if args.resume_successes and not args.source_from:
        raise SystemExit("--resume-successes requires --source-from")
    if args.independent_replication and (not args.source_from or args.resume_successes):
        raise SystemExit("Independent replication requires --source-from and forbids --resume-successes")
    if args.worker:
        raise SystemExit(worker(args))
    if args.judge_worker:
        raise SystemExit(judge_worker(args))
    if args.driver:
        driver(args)
    else:
        prepared = prepare(args)
        print(f"Prepared {prepared}; launch frozen runner with --driver")
