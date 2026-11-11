#!/usr/bin/env python3
"""Run a single-video, two-arm person-grounding ablation in isolated stores."""

import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parent.parent
NAME = "m3_person_ablation_study23_retry_20261105"
EXP = ROOT / "experiments" / NAME
RUN = ROOT / "runs" / NAME
SOURCE = ROOT / "runs/m3_dev_1031_ten_compare_20261104/_source/oGMemory"
PYTHON = ROOT / "oGMemory/.venv/bin/python"
REFERENCE = ROOT / "experiments/m3_dev_1031_ten_compare_20261104"
VIDEO = "study_23"
ARMS = ("off", "on")


def load_base(arm):
    path = ROOT / "runs/m3_parallel_ogonly_20261103/driver.py"
    spec = importlib.util.spec_from_file_location("ablation_base", path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    values = {"ROOT": ROOT, "NAME": NAME + "_" + arm, "EXP": EXP / arm, "RUN": RUN / arm,
              "SOURCE": SOURCE, "CONFIG": EXP / arm / "private/config.yaml",
              "ACCOUNT": "m3-person-ablation-study23-retry-20261105-" + arm,
              "DB": "ogmem_person_ablation_retry_20261105_" + arm}
    for key, value in values.items():
        setattr(base, key, value)
        setattr(base.h, key, value)
    base.WORKERS = {"OGMEM_M3_MAX_WORKERS": "1", "OGMEM_M3_FACE_WORKERS": "1",
                    "OGMEM_EMBED_MAX_CONCURRENCY": "1", "OGMEM_GRAPH_POOL_SIZE": "16",
                    "OGMEM_VECTOR_POOL_SIZE": "8"}
    return base


def prepare():
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import make_dsn, parse_dsn

    if not os.environ.get("MINIMAX_API_KEY"):
        raise RuntimeError("MINIMAX_API_KEY is missing")
    if EXP.exists() or RUN.exists():
        raise FileExistsError("Experiment already exists")
    old = json.loads((REFERENCE / "manifest.json").read_text())
    if old["source"]["commit"] != "8731930ec75153865df07689ee0c4bd98d182746":
        raise ValueError("Unexpected frozen source")
    config = yaml.safe_load((REFERENCE / "private/config.yaml").read_text())
    template = REFERENCE / "private/qa_template.yaml"
    for arm in ARMS:
        base = load_base(arm)
        base.RUN.mkdir(parents=True)
        (base.EXP / "private").mkdir(parents=True, mode=0o700)
        shutil.copytree(ROOT / "scripts/instrumentation", base.RUN / "_instrumentation",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(ROOT / "scripts/run_context_experiment.py", base.RUN / "quota_api.py")
        shutil.copyfile(template, base.EXP / "private/qa_template.yaml")
        arm_config = yaml.safe_load(yaml.safe_dump(config))
        for role in ("storage", "vector_db"):
            params = parse_dsn(arm_config[role]["connection_string"])
            conn = psycopg2.connect(**params)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (base.DB,))
                if cur.fetchone():
                    raise ValueError("Experiment database already exists")
                cur.execute(sql.SQL("CREATE DATABASE {} ").format(sql.Identifier(base.DB)))
            conn.close()
            params["dbname"] = base.DB
            arm_config[role]["connection_string"] = make_dsn(**params)
            break
        for role in ("storage", "vector_db"):
            params = parse_dsn(arm_config[role]["connection_string"])
            params["dbname"] = base.DB
            arm_config[role]["connection_string"] = make_dsn(**params)
        with psycopg2.connect(arm_config["storage"]["connection_string"]) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        arm_config["identity"]["account_id"] = base.ACCOUNT
        arm_config["llm"]["max_tokens"] = 16384
        for role in ("episodic_llm", "qa_llm"):
            arm_config["m3"][role]["max_tokens"] = 16384
        base.h.private_yaml(base.CONFIG, arm_config)
        (base.EXP / "config.redacted.yaml").write_text(yaml.safe_dump(base.h.redact(arm_config)))
        base.h.save(base.EXP / "manifest.json", {"arm": arm, "safety_stop_at_unix": time.time() + 48 * 3600})
    base.h.save(EXP / "manifest.json", {"video": VIDEO, "arms": list(ARMS), "source_commit": old["source"]["commit"],
                "source_archive_sha256": old["source"]["archive_sha256"], "annotations_sha256": base.h.digest(base.h.ANNOTATIONS),
                "memory_and_qa_max_tokens": 16384, "judge_max_tokens": 2048, "memory_workers": 4,
                "qa_workers": 1, "memory_max_frames": 16, "person_crops_on_max": 8,
                "design": "Independent full pipeline per arm; on enables person grounding and retrieval; off disables both. Not crop-only.",
                "limitation": "Perception is independently recomputed, not bit-identical shared inputs; one run per arm."})
    base.h.save(EXP / "status.json", {"status": "prepared"})


def worker(arm, phase):
    base = load_base(arm)
    sys.path.insert(0, str(SOURCE))
    if phase == "pipeline":
        import m3.run_video_pipeline as pipeline
        original = pipeline.main

        def adjusted(args):
            args = list(args)
            args[args.index("episodic_memory.params.workers=8")] = "episodic_memory.params.workers=4"
            if arm == "off":
                args += ["--set", "episodic_memory.params.person_grounding=false"]
            return original(args)

        pipeline.main = adjusted
    if phase == "og_qa":
        import m3.qa as qa
        original = qa.main_react_qa

        def adjusted(args):
            args = list(args)
            index = args.index("--workers")
            del args[index:index + 2]
            if arm == "off":
                args.remove("--person-retrieval")
            return original(args)

        qa.main_react_qa = adjusted
    return base.worker(phase, VIDEO)


def judge():
    base = load_base("on")
    sys.path.insert(0, str(SOURCE))
    from m3.eval import build_judge_llm, judge_answer
    from m3.qa import load_questions

    tasks = []
    questions = [row for row in load_questions(base.h.ANNOTATIONS) if row["video_id"] == VIDEO]
    for arm in ARMS:
        path = RUN / arm / "og_qa" / VIDEO / "m3_qa_results.jsonl"
        answers = {row["question_id"]: row for row in base.h.rows(path)}
        for question in questions:
            row = answers.get(question["question_id"], {})
            tasks.append({"arm": arm, "question_id": question["question_id"], "question": question["question"],
                          "expected": question["answer"], "predicted": row.get("predicted_answer", ""),
                          "answer_error": row.get("error", "")})
    random.Random(20261105).shuffle(tasks)
    base.h.save(EXP / "judge_inputs.json", tasks)
    gate = base.h.QuotaGate()
    base.h.instrument(gate)
    llm = build_judge_llm(config_path=str(base.CONFIG))
    result = []
    for task in tasks:
        if not task["predicted"] or task["answer_error"]:
            score, reason, error = 0.0, "Invalid or missing answer", False
        else:
            gate.wait()
            score, reason, error = judge_answer(judge_llm=llm, question=task["question"],
                                                 expected=task["expected"], predicted=task["predicted"])
        result.append({**task, "score": score, "reason": reason, "judge_error": bool(error)})
        base.h.save(EXP / "judged.json", result)
    for arm in ARMS:
        rows = [row for row in result if row["arm"] == arm]
        print(arm, "questions", len(rows), "mean", sum(row["score"] for row in rows) / len(rows),
              "judge_errors", sum(row["judge_error"] for row in rows), flush=True)
    return 0 if not any(row["judge_error"] for row in result) else 2


def run():
    history = []
    for arm in ARMS:
        base = load_base(arm)
        for phase in ("pipeline", "index", "og_qa"):
            base.h.save(EXP / "status.json", {"status": "running", "arm": arm, "phase": phase, "history": history})
            started = time.time()
            with (base.EXP / "private" / (phase + ".log")).open("a") as log:
                env = base.environment(phase, VIDEO)
                process = subprocess.run([str(PYTHON), str(Path(__file__).resolve()), "worker", "--arm", arm,
                                          "--phase", phase], cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT)
            receipt = {"arm": arm, "phase": phase, "exit_code": process.returncode, "seconds": time.time() - started}
            history.append(receipt)
            base.h.save(base.EXP / (phase + ".json"), receipt)
            if process.returncode:
                base.h.save(EXP / "status.json", {"status": "needs_review", "history": history})
                return process.returncode
    base = load_base("on")
    base.h.save(EXP / "status.json", {"status": "judging", "history": history})
    with (EXP / "judge.log").open("a") as log:
        env = base.environment("judge", VIDEO)
        process = subprocess.run([str(PYTHON), str(Path(__file__).resolve()), "judge"], cwd=SOURCE,
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
    base.h.save(EXP / "status.json", {"status": "finished" if process.returncode == 0 else "judge_failed",
                                      "history": history, "judge_exit_code": process.returncode})
    return process.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "worker", "judge"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--phase", choices=("pipeline", "index", "og_qa"))
    args = parser.parse_args()
    sys.exit(prepare() if args.action == "prepare" else run() if args.action == "run" else
             judge() if args.action == "judge" else worker(args.arm, args.phase))
