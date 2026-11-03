#!/usr/bin/env python3
"""Isolated oGMemory-only parallel rerun; frozen M3 answers are never regenerated."""

import argparse
import copy
import datetime as dt
import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("M3_EVAL_ROOT", Path(__file__).resolve().parent.parent))
NAME = "m3_parallel_ogonly_20261103"
EXP = ROOT / "experiments" / NAME
RUN = ROOT / "runs" / NAME
OLD_EXP = ROOT / "experiments/m3_ten_fullinterval_20261102"
OLD_RUN = ROOT / "runs/m3_ten_fullinterval_20261102"
SOURCE = RUN / "_source/oGMemory"
PYTHON = ROOT / "oGMemory/.venv/bin/python"
CONFIG = EXP / "private/config.yaml"
ACCOUNT = "m3-parallel-ogonly-20261103"
DB = "ogmem_m3_parallel_ogonly_20261103"
HELPER = Path(__file__).with_name("helpers.py")
if not HELPER.exists():
    HELPER = ROOT / "scripts/run_m3_ten_video.py"
spec = importlib.util.spec_from_file_location("parallel_helpers", HELPER)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
for key in (
    "ROOT",
    "NAME",
    "EXP",
    "RUN",
    "SOURCE",
    "PYTHON",
    "CONFIG",
    "ACCOUNT",
    "DB",
):
    setattr(h, key, globals()[key])

WORKERS = {
    "OGMEM_M3_MAX_WORKERS": "4",
    "OGMEM_M3_FACE_WORKERS": "2",
    "OGMEM_EMBED_MAX_CONCURRENCY": "4",
    "OGMEM_GRAPH_POOL_SIZE": "16",
    "OGMEM_VECTOR_POOL_SIZE": "8",
}


def environment(phase, video):
    return {**h.environment(phase, video), **WORKERS, "PYTHONUNBUFFERED": "1"}


def prepare():
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import make_dsn, parse_dsn

    if not os.environ.get("MINIMAX_API_KEY"):
        raise ValueError("Source ~/.bashrc; API key is never printed")
    if EXP.exists() or RUN.exists():
        raise FileExistsError("Preparation never overwrites an existing experiment")
    old = json.loads((OLD_EXP / "manifest.json").read_text())
    assert h.digest(h.ANNOTATIONS) == old["annotations_sha256"]
    for video in old["execution_order"]:
        assert Path(f"/data/m3_data/videos/robot/{video}.mp4").is_file()
    RUN.mkdir(parents=True)
    (EXP / "private").mkdir(parents=True, mode=0o700)
    source = h.archive(ROOT / "oGMemory", SOURCE)
    shutil.copyfile(Path(__file__), RUN / "driver.py")
    shutil.copyfile(HELPER, RUN / "helpers.py")
    shutil.copyfile(ROOT / "scripts/run_context_experiment.py", RUN / "quota_api.py")
    shutil.copytree(
        ROOT / "scripts/instrumentation",
        RUN / "_instrumentation",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    reference = EXP / "frozen_baseline"
    reference.mkdir()
    for name in (
        "manifest.json",
        "status.json",
        "joint_questions.jsonl",
        "comparison.json",
    ):
        shutil.copyfile(OLD_EXP / name, reference / name)
    shutil.copytree(OLD_EXP / "stages", reference / "stages")
    for path in OLD_EXP.glob("usage-*.jsonl"):
        shutil.copyfile(path, reference / path.name)
    prediction_hashes = {}
    for path in OLD_RUN.glob("m3_qa/*/m3_qa_results.jsonl"):
        target = reference / "m3_qa" / path.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        prediction_hashes[str(target.relative_to(EXP))] = h.digest(target)
    config = yaml.safe_load((OLD_EXP / "private/config.yaml").read_text())
    for role in ("storage", "vector_db"):
        params = parse_dsn(config[role]["connection_string"])
        conn = psycopg2.connect(**params)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (DB,))
            if not cur.fetchone():
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB)))
        conn.close()
        params["dbname"] = DB
        config[role]["connection_string"] = make_dsn(**params)
    with psycopg2.connect(config["storage"]["connection_string"]) as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("SELECT to_regclass('graph_nodes')")
        assert cur.fetchone()[0] is None, "Refusing populated experiment database"
    config["identity"]["account_id"] = ACCOUNT
    config["m3"].pop("qa_context", None)
    h.private_yaml(CONFIG, config)
    shutil.copyfile(
        OLD_EXP / "private/qa_template.yaml", EXP / "private/qa_template.yaml"
    )
    os.chmod(EXP / "private/qa_template.yaml", 0o600)
    (EXP / "config.redacted.yaml").write_text(
        yaml.safe_dump(h.redact(config), allow_unicode=True)
    )
    manifest = copy.deepcopy(old)
    manifest.update(
        {
            "experiment": NAME,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "account": ACCOUNT,
            "database": DB,
            "source": source,
            "baseline_source": old["source"],
            "config_sha256": h.digest(CONFIG),
            "qa_template_sha256": h.digest(EXP / "private/qa_template.yaml"),
            "driver_sha256": h.digest(RUN / "driver.py"),
            "helper_sha256": h.digest(RUN / "helpers.py"),
            "quota_api_sha256": h.digest(RUN / "quota_api.py"),
            "source_files": {
                str(p.relative_to(RUN)): h.digest(p)
                for p in (RUN / "_instrumentation").glob("*.py")
            },
            "safety_stop_at_unix": time.time() + 48 * 3600,
            "quota_policy": "12% interval gate; 5% weekly floor; new 48h safety window",
            "workers": {**WORKERS, "memory": 8, "qa": 4, "video": 1, "local_audio": 1},
            "historical_predictions_reused": True,
            "m3_inference_rerun": False,
            "m3_grades_reused": True,
            "reference_prediction_hashes": prediction_hashes,
            "reference_files": {
                str(p.relative_to(EXP)): h.digest(p)
                for p in reference.rglob("*")
                if p.is_file()
            },
            "perception": "fresh oGMemory-only; no new M3 perception/memory/QA",
            "person_grounding": True,
            "person_retrieval": True,
            "comparison_caveat": "Old source lacks new person grounding; not a pure parallelism ablation.",
            "complete_stage_subset": old["execution_order"][:8],
            "complete_stage_subset_questions": 97,
        }
    )
    h.save(EXP / "manifest.json", manifest)
    h.save(EXP / "status.json", {"status": "prepared"})
    print(
        json.dumps(
            {
                "experiment": NAME,
                "source": source,
                "videos": manifest["execution_order"],
            }
        )
    )


def preflight():
    """Real local embedding request, no paid generation; fail closed on bad services."""
    sys.path.insert(0, str(SOURCE))
    import requests
    from m3.index_episodic import build_embedder

    for url in ("http://127.0.0.1:8001/health", "http://127.0.0.1:18081/health/gpu"):
        response = requests.get(url, timeout=20)
        response.raise_for_status()
    embedder = build_embedder(None, config_path=str(CONFIG))
    result = embedder.embed_texts(
        ["M3 parallel experiment embedding service preflight"]
    )
    assert len(result) == 1 and len(result[0]) > 0
    h.save(
        EXP / "preflight.json",
        {
            "ok": True,
            "embedding_dimension": len(result[0]),
            "at": dt.datetime.now().astimezone().isoformat(),
        },
    )


def worker(phase, video):
    sys.path.insert(0, str(SOURCE))
    gate = h.QuotaGate()
    gate.wait()
    h.instrument(gate, memory=phase == "pipeline")
    work = RUN / video
    work.mkdir(exist_ok=True)
    if phase == "pipeline":
        from m3.run_video_pipeline import main

        rc = main(
            [
                "--video",
                f"/data/m3_data/videos/robot/{video}.mp4",
                "--video-id",
                video,
                "--seconds",
                "0",
                "--work-dir",
                str(work),
                "--config",
                str(CONFIG),
                "--pipeline",
                str(SOURCE / "config/m3/pipelines/video_episodic_full.yaml"),
                "--graph-store",
                "sql",
                "--account-id",
                ACCOUNT,
                "--no-smart-resume",
                "--max-frames",
                "30",
                "--episodic-max-frames",
                "16",
                "--disable-stage",
                "qa",
                "--set",
                "episodic_memory.params.workers=8",
                "--set",
                "voice_asr.params.device=cuda",
                "--set",
                "speaker_diarization.params.device=cuda",
                "--set",
                "speaker_embedding.params.device=cuda:0",
            ]
        )
        if rc:
            return rc
        (work / "perception_exit_code.txt").write_text("0\n")
        return h.audit_frames(video)
    if phase == "index":
        from m3.build_fact_index import main

        return main(
            [
                "--config",
                str(CONFIG),
                "--account-id",
                ACCOUNT,
                "--video",
                video,
                "--output",
                str(work / "fact_index"),
            ]
        )
    if phase != "og_qa":
        raise ValueError("Only oGMemory phases are permitted")
    from m3.config import load_m3_runtime_config
    from m3.control_adapter import OGMemoryControlAdapter
    from m3.qa import main_react_qa

    os.environ["OGMEM_GRAPH_CONNECTION_STRING"] = load_m3_runtime_config(
        str(CONFIG)
    ).graph_connection_string
    original = OGMemoryControlAdapter.run

    def guarded(self, ctx, task):
        # Gate before the adapter's QA timer, retain ContextVar question IDs.
        gate.wait()
        return original(self, ctx, task)

    OGMemoryControlAdapter.run = guarded
    output = RUN / "og_qa" / video / "m3_qa_results.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "--config",
        str(h.qa_config(video)),
        "--annotations",
        str(h.ANNOTATIONS),
        "--video",
        video,
        "--graph-store",
        "sql",
        "--account-id",
        ACCOUNT,
        "--output-jsonl",
        str(output),
        "--temperature",
        "0",
        "--max-iterations",
        "6",
        "--timeout-seconds",
        "180",
        "--top-k",
        "8",
        "--candidate-k",
        "20",
        "--transcript-top-k",
        "20",
        "--transcript-fusion",
        "rrf",
        "--context-k",
        "8",
        "--include-neighbors",
        "1",
        "--temporal-policy",
        "current_clip",
        "--workers",
        "4",
        "--person-retrieval",
        "--memory-coverage-json",
        str(work / "memory_coverage.json"),
    ]
    if output.exists():
        args.append("--resume")  # Retain first completed outcomes, including failures.
    return main_react_qa(args)


def judge():
    sys.path.insert(0, str(SOURCE))
    from core.operation_deadline import operation_deadline
    from m3._parallel import call_scope
    from m3.eval import build_judge_llm, judge_answer

    gate = h.QuotaGate()
    h.instrument(gate)
    manifest = json.loads((EXP / "manifest.json").read_text())
    tasks, hashes = h.fixed_tasks(manifest)
    tasks = [t for t in tasks if t["method"] == "ogmemory_strict_incremental"]
    target = EXP / "joint_questions.jsonl"
    # M3 inference and its grades are frozen, including failures, not retried.
    if not target.exists():
        frozen = [
            r
            for r in h.rows(EXP / "frozen_baseline/joint_questions.jsonl")
            if r["method"] == "m3_reference_text"
        ]
        assert len(frozen) == 128
        target.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in frozen)
        )
    existing = h.rows(target)
    done = {(r["method"], r["video_id"], r["question_id"]) for r in existing}
    assert len(done) == len(existing)
    h.save(EXP / "judging_inputs.json", {"prediction_hashes": hashes, "tasks": tasks})
    llm = build_judge_llm(config_path=str(CONFIG))
    with target.open("a") as out:
        for task in tasks:
            key = (task["method"], task["video_id"], task["question_id"])
            if key in done:
                continue
            if task["error"] or not task["predicted_answer"]:
                score, reason, error = 0.0, "Invalid QA answer; no judge call.", False
            else:
                gate.wait()
                with (
                    call_scope(
                        qa_video_id=task["video_id"],
                        qa_question_index=task["question_index"],
                    ),
                    operation_deadline(120),
                ):
                    score, reason, error = judge_answer(
                        judge_llm=llm,
                        question=task["question"],
                        expected=task["expected_answer"],
                        predicted=task["predicted_answer"],
                    )
            out.write(
                json.dumps(
                    {
                        **task,
                        "judge_score": score,
                        "judge_reasoning": reason,
                        "judge_error": error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()
    for path, digest in hashes.items():
        assert h.digest(RUN / path) == digest
    return 0


def resources(folder, videos, phases):
    usage = [
        r
        for phase in phases
        for video in videos
        for r in h.rows(folder / f"usage-{phase}-{video}.jsonl")
        if r.get("requested_model") == "MiniMax-M3.1-Flash"
    ]
    known = [r for r in usage if r.get("usage")]
    stages = [
        json.loads(p.read_text())
        for phase in phases
        for video in videos
        if (p := folder / "stages" / f"{video}-{phase}.json").exists()
    ]
    return {
        "calls": len(usage),
        "unknown_usage_calls": len(usage) - len(known),
        "prompt_tokens": sum(r["usage"].get("prompt_tokens", 0) for r in known),
        "completion_tokens": sum(r["usage"].get("completion_tokens", 0) for r in known),
        "wall_seconds_including_quota_wait": sum(r.get("seconds", 0) for r in stages),
        "failed_stages": sum(bool(r.get("exit_code")) for r in stages),
    }


def summarize():
    result = h.summarize()
    baseline = EXP / "frozen_baseline"
    old = json.loads((baseline / "comparison.json").read_text())
    # Never represent unexecuted reference inference as zero-token efficiency.
    result["methods"]["m3_reference_text"]["qa_usage"] = old["methods"][
        "m3_reference_text"
    ]["qa_usage"]
    manifest = result["cohort"]
    comparison = {}
    for group, videos in (
        ("all_128", manifest["execution_order"]),
        ("matched_first_eight_97", manifest["complete_stage_subset"]),
    ):
        comparison[group] = {}
        for phase_set, phases in (
            ("memory_and_index", ["pipeline", "index"]),
            ("qa", ["og_qa"]),
        ):
            comparison[group][phase_set] = {
                "old_ogmemory": resources(baseline, videos, phases),
                "new_ogmemory": resources(EXP, videos, phases),
            }
        comparison[group]["by_video"] = {
            v: {
                "old": resources(baseline, [v], ["pipeline", "index", "og_qa"]),
                "new": resources(EXP, [v], ["pipeline", "index", "og_qa"]),
            }
            for v in videos
        }
    result["resource_comparison"] = comparison
    h.save(EXP / "comparison.json", result)
    lines = [
        "# oGMemory 并发十视频复测",
        "",
        "仅重跑oGMemory；M3推理及评分沿用冻结结果，不是新的同步对照实验。",
        "MiniMax-M3.1-Flash / adaptive / medium；统一0/0.5/1评分口径。",
        "失败计零，评委错误另列；均分不是人工准确率。",
        "",
        "| 视频 | 题数 | 旧oGMemory | 新oGMemory | 冻结M3 |",
        "|---|---:|---:|---:|---:|",
    ]

    def score(value):
        return "评委错误" if value is None else f"{value:.4f}"

    new_og = result["methods"]["ogmemory_strict_incremental"]
    old_og = old["methods"]["ogmemory_strict_incremental"]
    reference = result["methods"]["m3_reference_text"]
    for video in manifest["execution_order"]:
        values = [
            score(m["by_video"][video]["mean_score"])
            for m in (old_og, new_og, reference)
        ]
        lines.append(
            f"| {video} | {manifest['question_counts'][video]} | "
            + " | ".join(values)
            + " |"
        )
    lines += [
        "",
        "## 资源比较",
        "",
        "| 题集/阶段 | 旧/新 | prompt tokens | completion tokens | 未知usage请求 | 墙钟秒 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group, phases in comparison.items():
        for phase in ("memory_and_index", "qa"):
            for method, metrics in phases[phase].items():
                lines.append(
                    f"| {group}/{phase} | {method} | {metrics['prompt_tokens']} | "
                    f"{metrics['completion_tokens']} | {metrics['unknown_usage_calls']} | "
                    f"{metrics['wall_seconds_including_quota_wait']:.1f} |"
                )
    lines += [
        "",
        "## 限制",
        "",
        "旧M3 study_18存在连接失败，kitchen_20缺失；旧oGMemory kitchen_20失败。",
        "全128题与前八视频97题的资源指标分开记录。不能把失败/缺失预测的少量用量解释为效率优势。",
        "总时间包含配额等待；不能将不同运行窗口的墙钟差全部归因于并发。",
        "新版本包含人物关系链，不是纯并发消融；失败和未知usage不当作零成本。",
        "并发是否减少token由实测决定，不预设结论。",
    ]
    (EXP / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def run():
    manifest = json.loads((EXP / "manifest.json").read_text())
    for path, field in (
        (CONFIG, "config_sha256"),
        (RUN / "driver.py", "driver_sha256"),
        (RUN / "helpers.py", "helper_sha256"),
        (h.ANNOTATIONS, "annotations_sha256"),
    ):
        assert h.digest(path) == manifest[field]
    for path, digest in manifest["reference_files"].items():
        assert h.digest(EXP / path) == digest
    lock = (RUN / "driver.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    history = []
    for video in manifest["execution_order"]:
        for phase in ("pipeline", "index", "og_qa"):
            receipt = EXP / "stages" / f"{video}-{phase}.json"
            if receipt.exists():
                result = json.loads(receipt.read_text())
            else:
                if time.time() >= manifest["safety_stop_at_unix"]:
                    h.save(
                        EXP / "status.json",
                        {"status": "safety_stopped", "history": history},
                    )
                    return 74
                h.save(
                    EXP / "status.json",
                    {
                        "status": "running",
                        "pid": os.getpid(),
                        "video": video,
                        "phase": phase,
                        "history": history,
                    },
                )
                started = time.time()
                with (EXP / "private" / f"{video}-{phase}.log").open("a") as log:
                    child = subprocess.run(
                        [
                            str(PYTHON),
                            str(RUN / "driver.py"),
                            "worker",
                            "--phase",
                            phase,
                            "--video",
                            video,
                        ],
                        check=False,
                        cwd=SOURCE,
                        env=environment(phase, video),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                complete_qa = (
                    phase == "og_qa"
                    and len(h.rows(RUN / "og_qa" / video / "m3_qa_results.jsonl"))
                    == manifest["question_counts"][video]
                )
                result = {
                    "video": video,
                    "phase": phase,
                    "exit_code": child.returncode,
                    "started_at_unix": started,
                    "seconds": time.time() - started,
                    "status": "completed"
                    if child.returncode == 0
                    else "completed_with_qa_failures"
                    if child.returncode == 2 and complete_qa
                    else "failed",
                }
                h.save(receipt, result)
            history.append(result)
            if result["status"] == "failed":
                h.save(
                    EXP / "status.json",
                    {
                        "status": "needs_review",
                        "history": history,
                        "interrupted": result,
                    },
                )
                return result["exit_code"] or 1
    h.save(EXP / "status.json", {"status": "judging", "history": history})
    with (EXP / "private/judge.log").open("a") as log:
        child = subprocess.run(
            [str(PYTHON), str(RUN / "driver.py"), "judge"],
            check=False,
            cwd=SOURCE,
            env=environment("judge", "all"),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if child.returncode:
        h.save(
            EXP / "status.json", {"status": "needs_judge_review", "history": history}
        )
        return child.returncode
    result = summarize()
    failures = any(r["exit_code"] for r in history) or any(
        m["judge_errors"] for m in result["methods"].values()
    )
    h.save(
        EXP / "status.json",
        {
            "status": "finished_with_failures" if failures else "finished",
            "history": history,
        },
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=["prepare", "preflight", "run", "worker", "judge", "summarize"],
    )
    parser.add_argument("--phase", choices=["pipeline", "index", "og_qa"])
    parser.add_argument("--video")
    args = parser.parse_args()
    if args.action == "worker":
        return worker(args.phase, args.video)
    result = globals()[args.action]()
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
