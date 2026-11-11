#!/usr/bin/env python3
"""Fresh, isolated three-branch M3 comparison with a bounded runtime."""

import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import tarfile
import time

import yaml

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "scripts/run_m3_ten_video.py").is_file())
NAME = "m3_three_branches_20261110"
EXP = ROOT / "experiments" / NAME
RUN = ROOT / "runs" / NAME
PYTHON = ROOT / "oGMemory/.venv/bin/python"
ANNOTATIONS = Path("/data/m3_data/annotations/robot.json")
VIDEOS = ("kitchen_01", "bedroom_05", "study_23")
BRANCHES = {
    "dev": "8c86c3ed5a180f6e3a43a501350c259ae8b6d50e",
    "m3_dev_1031": "8731930ec75153865df07689ee0c4bd98d182746",
    "simplify_person_info_1110": "42829a29",
}
DEADLINE = None


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def paths(arm):
    return RUN / arm, EXP / arm


def prepare():
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import make_dsn, parse_dsn

    if EXP.exists() or RUN.exists():
        raise FileExistsError("Refusing to overwrite an experiment")
    if not os.environ.get("MINIMAX_API_KEY"):
        raise ValueError("MINIMAX_API_KEY is required")
    revisions = {}
    for arm, expected in BRANCHES.items():
        revision = subprocess.check_output(
            ["git", "-C", str(ROOT / "oGMemory"), "rev-parse", arm], text=True
        ).strip()
        if not revision.startswith(expected):
            raise ValueError(f"Branch {arm} changed since design")
        revisions[arm] = revision
    for video in VIDEOS:
        if not (Path("/data/m3_data/videos/robot") / f"{video}.mp4").is_file():
            raise FileNotFoundError(video)
    original = yaml.safe_load((ROOT / "experiments/m3_paper8_fresh_20261110/private/config.yaml").read_text())
    template = yaml.safe_load((ROOT / "experiments/m3_paper8_fresh_20261110/private/qa_template.yaml").read_text())["qa_context"]
    RUN.mkdir(parents=True)
    EXP.mkdir(parents=True)
    shutil.copyfile(Path(__file__), RUN / "driver.py")
    for arm, revision in revisions.items():
        run, exp = paths(arm)
        run.mkdir()
        (exp / "private").mkdir(parents=True, mode=0o700)
        source = run / "_source"
        source.mkdir()
        archive = run / "source.tar"
        with archive.open("wb") as stream:
            subprocess.run(["git", "-C", str(ROOT / "oGMemory"), "archive", revision],
                           stdout=stream, check=True)
        with tarfile.open(archive) as bundle:
            bundle.extractall(source, filter="data")
        shutil.copytree(ROOT / "scripts/instrumentation", run / "_instrumentation",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copyfile(ROOT / "scripts/run_context_experiment.py", run / "quota_api.py")
        dbname = "ogmem_cmp3_" + {"dev": "dev", "m3_dev_1031": "m3", "simplify_person_info_1110": "simple"}[arm] + "_20261110"
        config = yaml.safe_load(yaml.safe_dump(original))
        for role in ("storage", "vector_db"):
            params = parse_dsn(config[role]["connection_string"])
            connection = psycopg2.connect(**params)
            try:
                connection.autocommit = True
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1 FROM pg_database WHERE datname=%s", (dbname,))
                    exists = bool(cursor.fetchone())
                    if exists and role == "storage":
                        raise FileExistsError(f"Database already exists: {dbname}")
                    if not exists:
                        cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
            finally:
                connection.close()
            params["dbname"] = dbname
            config[role]["connection_string"] = make_dsn(**params)
        with psycopg2.connect(config["storage"]["connection_string"]) as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        account = "cmp3-" + arm.replace("_", "-")
        config["identity"]["account_id"] = account
        config["llm"]["max_tokens"] = 16384
        for role in ("episodic_llm", "qa_llm"):
            config["m3"][role]["max_tokens"] = 16384
        config["m3"].pop("qa_context", None)
        path = exp / "private/config.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True))
        path.chmod(0o600)
        context = dict(template)
        if arm == "dev":
            allowed = {"enabled", "context_window", "max_input_tokens", "max_top_k",
                       "max_candidates", "max_result_nodes", "max_neighbor_clips",
                       "max_search_calls", "max_selected_nodes", "max_passages_per_node",
                       "tokenizer_path", "tokenizer_multiplier", "safety_tokens"}
            context = {key: value for key, value in context.items() if key in allowed}
        (exp / "private/qa_template.yaml").write_text(yaml.safe_dump({"qa_context": context}))
        (exp / "private/qa_template.yaml").chmod(0o600)
        save(exp / "manifest.json", {
            "arm": arm, "revision": revision, "database": dbname, "account": account,
            "source_archive_sha256": digest(archive), "config_sha256": digest(path),
            "safety_stop_at_unix": DEADLINE, "videos": VIDEOS,
        })
    save(EXP / "manifest.json", {
        "status": "prepared", "videos": VIDEOS, "branches": revisions,
        "deadline": DEADLINE, "annotations_sha256": digest(ANNOTATIONS),
        "question_counts": {video: json.loads(
            (ROOT / "experiments/m3_ten_fullinterval_20261102/manifest.json").read_text()
        )["question_counts"][video] for video in VIDEOS},
        "driver_sha256": digest(RUN / "driver.py"),
        "design": "Three fresh independent databases; serial video and QA; same models, output limit and common retrieval settings. "
                  "dev lacks newer fact index, person grounding, incremental context and parallel memory; native implementations differ. "
                  "Simplify enables compact person and later 2000 evidence budget.",
    })


def helper(arm):
    run, exp = paths(arm)
    spec = importlib.util.spec_from_file_location("cmp_helper", ROOT / "runs/m3_parallel_ogonly_20261103/helpers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key, value in {"ROOT": ROOT, "RUN": run, "EXP": exp, "SOURCE": run / "_source",
                       "CONFIG": exp / "private/config.yaml", "ACCOUNT": json.loads((exp / "manifest.json").read_text())["account"]}.items():
        setattr(module, key, value)
    return module


def worker(arm, phase, video):
    run, exp = paths(arm)
    sys.path.insert(0, str(run / "_source"))
    h = helper(arm)
    gate = h.QuotaGate()
    if DEADLINE is None:
        gate.deadline = float("inf")
    gate.wait()
    h.instrument(gate, memory=phase == "pipeline")
    config = exp / "private/config.yaml"
    account = json.loads((exp / "manifest.json").read_text())["account"]
    if phase == "judge":
        from core.operation_deadline import operation_deadline
        from m3.eval import build_judge_llm, judge_answer
        from m3.qa import load_questions
        questions = {row["question_id"]: row for row in load_questions(ANNOTATIONS)
                     if row["video_id"] in VIDEOS}
        tasks = []
        for branch in BRANCHES:
            branch_run, _ = paths(branch)
            for current_video in VIDEOS:
                for answer in rows(branch_run / "og_qa" / current_video / "m3_qa_results.jsonl"):
                    question = questions.get(answer["question_id"])
                    if not question or question["video_id"] != current_video or (
                        answer.get("question") != question["question"]
                        or answer.get("expected_answer") != question["answer"]
                    ):
                        raise ValueError("Mismatched or unexpected question")
                    tasks.append({"arm": branch, "video_id": current_video,
                                  "question_id": answer["question_id"], "question": question["question"],
                                  "expected_answer": question["answer"],
                                  "predicted_answer": answer.get("predicted_answer", ""),
                                  "error": answer.get("error", "")})
        if len(tasks) != len(questions) * len(BRANCHES):
            raise ValueError("Incomplete question coverage")
        random.Random(20261110).shuffle(tasks)
        judge = build_judge_llm(config_path=str(config))
        output = EXP / "judgments.jsonl"
        done = {(row["arm"], row["question_id"]) for row in rows(output)}
        with output.open("a") as stream:
            for task in tasks:
                if (task["arm"], task["question_id"]) in done:
                    continue
                score, reason, error = 0.0, "Missing or failed answer", False
                if not task["error"] and task["predicted_answer"]:
                    for attempt in range(3):
                        gate.wait()
                        with operation_deadline(120):
                            score, reason, error = judge_answer(
                                judge_llm=judge, question=task["question"],
                                expected=task["expected_answer"], predicted=task["predicted_answer"])
                        if not error:
                            break
                        time.sleep(30 * (attempt + 1))
                    if error:
                        raise RuntimeError("Judge failed after retries")
                stream.write(json.dumps({**task, "judge_score": score, "judge_reasoning": reason},
                                        ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        return 0
    work = run / video
    work.mkdir(exist_ok=True)
    if phase == "pipeline":
        from m3.run_video_pipeline import main
        args = ["--video", f"/data/m3_data/videos/robot/{video}.mp4", "--video-id", video,
                "--seconds", "0", "--work-dir", str(work), "--config", str(config),
                "--pipeline", str(run / "_source/config/m3/pipelines/video_episodic_full.yaml"),
                "--graph-store", "sql", "--account-id", account, "--no-smart-resume",
                "--max-frames", "30", "--episodic-max-frames", "16", "--disable-stage", "qa",
                "--set", "voice_asr.params.device=cuda", "--set", "speaker_diarization.params.device=cuda",
                "--set", "speaker_embedding.params.device=cuda:0"]
        if arm != "dev":
            args += ["--set", "episodic_memory.params.workers=4"]
        return main(args)
    if phase == "index":
        if arm == "dev":
            return 0
        from m3.build_fact_index import main
        return main(["--config", str(config), "--account-id", account, "--video", video,
                     "--output", str(work / "fact_index")])
    if phase != "qa":
        raise ValueError(phase)
    from m3.config import load_m3_runtime_config
    from m3.control_adapter import OGMemoryControlAdapter
    from m3.qa import main_react_qa
    os.environ["OGMEM_GRAPH_CONNECTION_STRING"] = load_m3_runtime_config(str(config)).graph_connection_string
    original = OGMemoryControlAdapter.run

    def guarded(self, ctx, task):
        gate.wait()
        return original(self, ctx, task)

    OGMemoryControlAdapter.run = guarded
    settings = yaml.safe_load(config.read_text())
    context = yaml.safe_load((exp / "private/qa_template.yaml").read_text())["qa_context"]
    if arm != "dev":
        index = work / "fact_index/manifest.json"
        context.update(fact_index_path=str(index), fact_index_sha256=digest(index))
    settings["m3"]["qa_context"] = context
    qa_config = exp / "private" / f"{video}.yaml"
    if qa_config.exists() and yaml.safe_load(qa_config.read_text()) != settings:
        raise ValueError("QA config changed")
    qa_config.write_text(yaml.safe_dump(settings, allow_unicode=True))
    qa_config.chmod(0o600)
    output = run / "og_qa" / video / "m3_qa_results.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    args = ["--config", str(qa_config), "--annotations", str(ANNOTATIONS), "--video", video,
            "--graph-store", "sql", "--account-id", account, "--output-jsonl", str(output),
            "--temperature", "0", "--max-iterations", "6", "--timeout-seconds", "180",
            "--top-k", "8", "--candidate-k", "20", "--transcript-top-k", "20",
            "--transcript-fusion", "rrf", "--context-k", "8", "--include-neighbors", "1",
            "--temporal-policy", "current_clip"]
    if arm != "dev":
        args += ["--person-retrieval", "--memory-coverage-json", str(work / "memory_coverage.json")]
    if output.exists():
        args.append("--resume")
    return main_react_qa(args)


def environment(arm, phase, video):
    run, exp = paths(arm)
    env = dict(os.environ)
    for key in ("OGMEM_M3_COMPACT_PERSON", "OGMEM_M3_LATER_INCREMENTAL_TOKENS",
                "OGMEM_M3_EARLY_ANSWER", "OGMEM_M3_AMBIGUITY_GUARD", "OGMEM_M3_PERSON_ON_DEMAND"):
        env.pop(key, None)
    env.update(PYTHONPATH=f"{run / '_instrumentation'}:{run / '_source'}",
               PYTHONHASHSEED="0", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="2",
               OGMEM_MINIMAX_MODEL="MiniMax-M3.1-Flash", OGMEM_M3_MAX_WORKERS="1",
               OGMEM_M3_FACE_WORKERS="1", OGMEM_EMBED_MAX_CONCURRENCY="1",
               OGMEM_GRAPH_POOL_SIZE="16", OGMEM_VECTOR_POOL_SIZE="8",
               M3_USAGE_LOG=str(exp / f"usage-{phase}-{video}.jsonl"), M3_AUDIT_VIDEO=video)
    libs = ROOT / "oGMemory/.venv/lib/python3.12/site-packages/nvidia"
    env["LD_LIBRARY_PATH"] = f"{libs / 'cublas/lib'}:{libs / 'cudnn/lib'}:" + env.get("LD_LIBRARY_PATH", "")
    if arm == "simplify_person_info_1110":
        env.update(OGMEM_M3_COMPACT_PERSON="1", OGMEM_M3_LATER_INCREMENTAL_TOKENS="2000")
    return env


def run():
    manifest = json.loads((EXP / "manifest.json").read_text())
    if digest(RUN / "driver.py") != manifest["driver_sha256"] or digest(ANNOTATIONS) != manifest["annotations_sha256"]:
        raise ValueError("Driver or annotations changed")
    for video in VIDEOS:
        for arm in BRANCHES:
            run_dir, exp = paths(arm)
            for phase in ("pipeline", "index", "qa"):
                receipt = exp / "stages" / f"{video}-{phase}.json"
                if receipt.exists():
                    result = json.loads(receipt.read_text())
                    if result["status"] != "completed":
                        raise RuntimeError(f"Prior failed stage requires review: {arm}/{video}/{phase}")
                    continue
                remaining = None if DEADLINE is None else DEADLINE - time.time()
                if remaining is not None and remaining <= 0:
                    save(EXP / "status.json", {"status": "deadline", "arm": arm, "video": video, "phase": phase})
                    return 74
                save(EXP / "status.json", {"status": "running", "arm": arm, "video": video, "phase": phase,
                                          "pid": os.getpid()})
                started = time.time()
                with (exp / "private" / f"{video}-{phase}.log").open("a") as log:
                    child = subprocess.Popen([str(PYTHON), str(RUN / "driver.py"), "worker", arm, phase, video],
                                             cwd=run_dir / "_source", env=environment(arm, phase, video),
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = child.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGTERM)
                        child.wait()
                        code = 74
                answer_rows = rows(run_dir / "og_qa" / video / "m3_qa_results.jsonl") if phase == "qa" else []
                unique = {row.get("question_id") for row in answer_rows}
                complete_qa = len(answer_rows) == len(unique) == manifest["question_counts"][video]
                complete = ((phase != "qa" and code == 0)
                            or (phase == "qa" and code in (0, 2) and complete_qa))
                result = {"arm": arm, "video": video, "phase": phase, "exit_code": code,
                          "started_at_unix": started, "seconds": time.time() - started,
                          "status": "completed" if complete else "failed", "answer_rows": len(answer_rows)}
                save(receipt, result)
                if not complete:
                    save(EXP / "status.json", {"status": "needs_review", **result})
                    return code or 1
    remaining = None if DEADLINE is None else DEADLINE - time.time()
    if remaining is not None and remaining <= 0:
        save(EXP / "status.json", {"status": "answers_completed_needs_blind_judging"})
        return 74
    judge_arm = "simplify_person_info_1110"
    judge_run, judge_exp = paths(judge_arm)
    save(EXP / "status.json", {"status": "judging", "pid": os.getpid()})
    with (judge_exp / "private/judge.log").open("a") as log:
        child = subprocess.Popen([str(PYTHON), str(RUN / "driver.py"), "worker", judge_arm, "judge", "all"],
                                 cwd=judge_run / "_source", env=environment(judge_arm, "judge", "all"),
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait()
            code = 74
    if code:
        save(EXP / "status.json", {"status": "needs_judge_review", "exit_code": code})
        return code
    judgments = rows(EXP / "judgments.jsonl")
    result = {}
    for arm in BRANCHES:
        group = [row for row in judgments if row["arm"] == arm]
        result[arm] = {"questions": len(group),
                       "mean_score": sum(row["judge_score"] for row in group) / len(group),
                       "failures": sum(bool(row["error"]) or not row["predicted_answer"] for row in group),
                       "by_video": {video: sum(row["judge_score"] for row in group if row["video_id"] == video)
                                    / sum(row["video_id"] == video for row in group) for video in VIDEOS}}
    save(EXP / "summary.json", result)
    save(EXP / "status.json", {"status": "completed"})
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        prepare()
    elif sys.argv[1:] == ["run"]:
        raise SystemExit(run())
    elif len(sys.argv) == 5 and sys.argv[1] == "worker":
        raise SystemExit(worker(*sys.argv[2:]))
    else:
        raise SystemExit("Use prepare, run, or worker ARM PHASE VIDEO")
