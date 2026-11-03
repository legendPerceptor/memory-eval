#!/usr/bin/env python3
"""Audited recovery of the parallel oGMemory-only run; frozen source is unchanged."""

import argparse
import datetime as dt
import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(os.environ.get("M3_EVAL_ROOT", Path(__file__).resolve().parent.parent))
RUN = ROOT / "runs/m3_parallel_ogonly_20261103"
EXP = ROOT / "experiments/m3_parallel_ogonly_20261103"
RECOVERY = EXP / "recovery_01"
SCRIPT = RUN / "recovery_01.py"
VIDEO = "study_09"
MEMORY_WORKERS = 2
RETRY = {"max_retries": 4, "retry_base_seconds": 30, "retry_max_seconds": 300}


def base():
    os.environ.setdefault("M3_EVAL_ROOT", str(ROOT))
    spec = importlib.util.spec_from_file_location("frozen_ogonly", RUN / "driver.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def lock():
    with (RUN / "driver.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def canonical_rows(rows):
    result = {}
    for row in rows:
        key = str(row["video_id"]), int(row["clip_id"])
        if key in result:
            raise ValueError("Duplicate memory record")
        result[key] = {
            k: v
            for k, v in row.items()
            if k not in ("resumed", "generation_retry_counts")
        }
    return result


def verify_preserved(previous, current):
    old, new = canonical_rows(previous), canonical_rows(current)
    if not all(new.get(key) == value for key, value in old.items()):
        raise ValueError("Previously successful memory changed or disappeared")
    return len(old)


def check_memory_scope(scope, failed):
    if scope.get("input_video_id") != VIDEO or str(scope.get("input_clip_id")) not in {
        str(i) for i in failed
    }:
        raise RuntimeError(
            "Recovery refuses a model request outside registered failed clips"
        )


def pipeline_args(original, video, plan):
    args = list(original)
    old = "episodic_memory.params.workers=8"
    if args.count(old) != 1:
        raise ValueError("Unexpected frozen worker arguments")
    args[args.index(old)] = f"episodic_memory.params.workers={MEMORY_WORKERS}"
    for key, value in RETRY.items():
        args += ["--set", f"episodic_memory.params.{key}={value}"]
    if video == VIDEO:
        for stage in plan["reuse_stages"]:
            args += ["--disable-stage", stage]
        args += [
            "--set",
            "episodic_memory.params.retry_failed=true",
            "--set",
            "episodic_memory.params.failure_policy=stop",
        ]
    return args


def combined_receipt(previous, attempt):
    attempts = list(previous.get("attempts", [previous])) if previous else []
    attempts.append(attempt)
    return {
        **attempt,
        "seconds": sum(a["seconds"] for a in attempts),
        "started_at_unix": attempts[0]["started_at_unix"],
        "attempts": attempts,
        "failed_attempts": sum(bool(a["exit_code"]) for a in attempts),
    }


def validate_original(b):
    manifest = json.loads((EXP / "manifest.json").read_text())
    for path, field in (
        (b.CONFIG, "config_sha256"),
        (RUN / "driver.py", "driver_sha256"),
        (RUN / "helpers.py", "helper_sha256"),
        (RUN / "quota_api.py", "quota_api_sha256"),
        (b.h.ANNOTATIONS, "annotations_sha256"),
    ):
        if b.h.digest(path) != manifest[field]:
            raise ValueError(f"Frozen input changed: {path.name}")
    for path, digest in manifest["source_files"].items():
        if b.h.digest(RUN / path) != digest:
            raise ValueError("Frozen instrumentation changed")
    for path, digest in manifest["reference_files"].items():
        if b.h.digest(EXP / path) != digest:
            raise ValueError("Frozen M3 reference changed")
    return manifest


def prepare():
    b = base()
    with lock():
        if RECOVERY.exists() or SCRIPT.exists():
            raise FileExistsError("Recovery plan already exists; never overwrite")
        manifest = validate_original(b)
        work = RUN / VIDEO
        summary = json.loads((work / "pipeline_summary.json").read_text())
        failed = b.h.rows(work / "episodic_memory.jsonl")
        failures = json.loads((work / "episodic_memory_failures.json").read_text())
        target_ids = sorted(int(r["clip_id"]) for r in failures)
        if (
            len(failed) != 71
            or len(failures) != 8
            or any(r["video_id"] != VIDEO or "529" not in r["error"] for r in failures)
            or set(target_ids) & {key[1] for key in canonical_rows(failed)}
        ):
            raise ValueError(
                "Recovery inputs differ from reviewed 71 successful / 8 overloaded clips"
            )
        if len(set(target_ids)) != 8 or (
            set(target_ids) | {key[1] for key in canonical_rows(failed)}
        ) != set(range(79)):
            raise ValueError("Incomplete or duplicate clip partition")
        stages = list(summary["stages"])
        reuse = stages[: stages.index("episodic_memory")]
        if not all(summary["stages"][s]["status"] == "success" for s in reuse):
            raise ValueError("Cannot reuse unsuccessful perception stage")
        if list((RUN / "og_qa").glob("*/m3_qa_results.jsonl")):
            raise ValueError("This recovery is registered before any new QA")
        state = json.loads((EXP / "status.json").read_text())
        if state["status"] != "needs_review":
            raise ValueError("Unexpected coordinator state")
        RECOVERY.mkdir()
        (RECOVERY / "before").mkdir()
        names = [
            "episodic_memory.jsonl",
            "episodic_memory.summary.json",
            "episodic_memory_failures.json",
            "pipeline_summary.json",
        ]
        for name in names:
            shutil.copyfile(work / name, RECOVERY / "before" / name)
        for name, path in {
            "stage.json": EXP / "stages" / f"{VIDEO}-pipeline.json",
            "status.json": EXP / "status.json",
            "preflight.json": EXP / "preflight.json",
        }.items():
            shutil.copyfile(path, RECOVERY / "before" / name)
        shutil.copyfile(Path(__file__), SCRIPT)
        inputs = [
            "data.jsonl",
            "frames.jsonl",
            "audio.jsonl",
            "faces.jsonl",
            "face_clusters.jsonl",
            "person_identities.json",
            "voice_asr.jsonl",
            "speaker_turns.jsonl",
            "speaker_embeddings.jsonl",
            "input_provenance.json",
        ]
        plan = {
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "experiment": manifest["experiment"],
            "source_commit": manifest["source"]["commit"],
            "original_manifest_sha256": b.h.digest(EXP / "manifest.json"),
            "recovery_driver_sha256": b.h.digest(SCRIPT),
            "failed_video": VIDEO,
            "failed_clip_ids": target_ids,
            "preserve_successes": 71,
            "reuse_stages": reuse,
            "memory_workers": MEMORY_WORKERS,
            "qa_workers": 4,
            "retry": RETRY,
            "m3_inference_rerun": False,
            "model_and_prompt_unchanged": True,
            "safety_stop_at_unix": manifest["safety_stop_at_unix"],
            "perception_hashes": {name: b.h.digest(work / name) for name in inputs},
            "snapshot_hashes": {
                p.name: b.h.digest(p) for p in (RECOVERY / "before").iterdir()
            },
            "source_hashes": {
                str(p.relative_to(b.SOURCE)): b.h.digest(p)
                for p in b.SOURCE.rglob("*")
                if p.is_file() and p.suffix in (".py", ".yaml", ".toml")
            },
            "cost_policy": "Append all usage; cumulative stage seconds include original failure and recovery.",
        }
        b.h.save(RECOVERY / "plan.json", plan)
        print(
            json.dumps(
                {
                    "recovery": str(RECOVERY),
                    "failed_clips": target_ids,
                    "memory_workers": 2,
                }
            )
        )


def validate_plan(b):
    validate_original(b)
    plan = json.loads((RECOVERY / "plan.json").read_text())
    if b.h.digest(EXP / "manifest.json") != plan["original_manifest_sha256"]:
        raise ValueError("Original manifest changed")
    if b.h.digest(SCRIPT) != plan["recovery_driver_sha256"]:
        raise ValueError("Recovery driver changed")
    for name, digest in plan["snapshot_hashes"].items():
        if b.h.digest(RECOVERY / "before" / name) != digest:
            raise ValueError("Recovery snapshot changed")
    for name, digest in plan["source_hashes"].items():
        if b.h.digest(b.SOURCE / name) != digest:
            raise ValueError("Frozen core source changed")
    return plan


def verify_perception(b, plan):
    for name, digest in plan["perception_hashes"].items():
        if b.h.digest(RUN / VIDEO / name) != digest:
            raise ValueError(f"Perception input changed: {name}")


def worker(phase, video):
    b = base()
    plan = validate_plan(b)
    if phase not in ("pipeline", "index", "og_qa"):
        raise ValueError("No M3 reference inference is allowed")
    if phase == "pipeline":
        import sys

        sys.path.insert(0, str(b.SOURCE))
        from m3 import run_video_pipeline

        original = run_video_pipeline.main
        if video == VIDEO:
            verify_perception(b, plan)
            from m3 import episodic_memory
            from m3._parallel import current_call_scope
            from providers.llm.openai_llm import OpenAILLM

            original_memory = episodic_memory.run

            def protected_memory(*args, **kwargs):
                # Installed after the frozen worker's quota/usage wrappers.
                completion = OpenAILLM._create_completion

                def guarded(client, **request):
                    check_memory_scope(current_call_scope(), plan["failed_clip_ids"])
                    return completion(client, **request)

                OpenAILLM._create_completion = guarded
                try:
                    rc = original_memory(*args, **kwargs)
                finally:
                    OpenAILLM._create_completion = completion
                old = b.h.rows(RECOVERY / "before/episodic_memory.jsonl")
                current = b.h.rows(RUN / VIDEO / "episodic_memory.jsonl")
                preserved = verify_preserved(old, current)
                verify_perception(b, plan)
                b.h.save(
                    RECOVERY / "memory_validation.json",
                    {
                        "preserved_successes": preserved,
                        "records_after": len(current),
                        "return_code": rc,
                        "perception_hashes_unchanged": True,
                    },
                )
                return rc

            episodic_memory.run = protected_memory
        run_video_pipeline.main = lambda args: original(
            pipeline_args(args, video, plan)
        )
    return b.worker(phase, video)


def run():
    b = base()
    plan = validate_plan(b)
    manifest = json.loads((EXP / "manifest.json").read_text())
    with lock():
        verify_perception(b, plan)
        history = []
        for video in manifest["execution_order"]:
            for phase in ("pipeline", "index", "og_qa"):
                path = EXP / "stages" / f"{video}-{phase}.json"
                previous = json.loads(path.read_text()) if path.exists() else None
                if previous and previous["status"] in (
                    "completed",
                    "completed_with_qa_failures",
                ):
                    history.append(previous)
                    continue
                if previous and not (
                    video == VIDEO
                    and phase == "pipeline"
                    and previous
                    == json.loads((RECOVERY / "before/stage.json").read_text())
                ):
                    raise RuntimeError(
                        "Unreviewed failed attempt; register a new recovery instead of looping"
                    )
                if time.time() >= plan["safety_stop_at_unix"]:
                    b.h.save(
                        EXP / "status.json",
                        {"status": "safety_stopped", "history": history},
                    )
                    return 74
                b.h.save(
                    EXP / "status.json",
                    {
                        "status": "running",
                        "pid": os.getpid(),
                        "video": video,
                        "phase": phase,
                        "recovery": "recovery_01",
                        "memory_workers": 2,
                        "qa_workers": 4,
                        "history": history,
                    },
                )
                started = time.time()
                with (EXP / "private" / f"{video}-{phase}-recovery_01.log").open(
                    "a"
                ) as log:
                    child = subprocess.run(
                        [
                            str(b.PYTHON),
                            str(SCRIPT),
                            "worker",
                            "--phase",
                            phase,
                            "--video",
                            video,
                        ],
                        cwd=b.SOURCE,
                        env=b.environment(phase, video),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                qa = (
                    b.h.rows(RUN / "og_qa" / video / "m3_qa_results.jsonl")
                    if phase == "og_qa"
                    else []
                )
                complete_qa = (
                    phase == "og_qa"
                    and len(qa) == manifest["question_counts"][video]
                    and len({(r["video_id"], r["question_id"]) for r in qa}) == len(qa)
                )
                attempt = {
                    "video": video,
                    "phase": phase,
                    "exit_code": child.returncode,
                    "seconds": time.time() - started,
                    "started_at_unix": started,
                    "recovery": "recovery_01",
                    "status": "completed"
                    if child.returncode == 0
                    else "completed_with_qa_failures"
                    if child.returncode == 2 and complete_qa
                    else "failed",
                }
                result = combined_receipt(previous, attempt)
                b.h.save(RECOVERY / f"{video}-{phase}.json", attempt)
                b.h.save(path, result)
                history.append(result)
                if result["status"] == "failed":
                    b.h.save(
                        EXP / "status.json",
                        {
                            "status": "needs_review",
                            "history": history,
                            "recovery": "recovery_01",
                            "interrupted": result,
                        },
                    )
                    return child.returncode or 1
        b.h.save(
            EXP / "status.json",
            {"status": "judging", "history": history, "recovery": "recovery_01"},
        )
        with (EXP / "private/judge-recovery_01.log").open("a") as log:
            child = subprocess.run(
                [str(b.PYTHON), str(SCRIPT), "judge"],
                cwd=b.SOURCE,
                env=b.environment("judge", "all"),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if child.returncode:
            b.h.save(
                EXP / "status.json",
                {"status": "needs_judge_review", "history": history},
            )
            return child.returncode
        result = b.summarize()
        result["execution_amendment"] = plan
        b.h.save(EXP / "comparison.json", result)
        with (EXP / "REPORT.md").open("a") as out:
            out.write(
                "\n## 服务过载恢复\n\n首次study_09使用8个记忆worker；"
                "8片段529失败后降至2，只补失败片段，其余视频也采用2。QA仍4。"
                "记忆退避30秒起、上限300秒、最多4次额外瞬时错误重试。"
                "阶段时间和usage包括首次失败及恢复，不能当作纯并发提速实验。\n"
            )
        failures = any(r["exit_code"] for r in history) or any(
            m["judge_errors"] for m in result["methods"].values()
        )
        b.h.save(
            EXP / "status.json",
            {
                "status": "finished_with_failures" if failures else "finished",
                "history": history,
                "recovery": "recovery_01",
            },
        )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["prepare", "preflight", "run", "worker", "judge"]
    )
    parser.add_argument("--phase", choices=["pipeline", "index", "og_qa"])
    parser.add_argument("--video")
    args = parser.parse_args()
    if args.action == "worker":
        return worker(args.phase, args.video)
    if args.action in ("preflight", "judge"):
        b = base()
        validate_plan(b)
        return getattr(b, args.action)()
    return globals()[args.action]()


if __name__ == "__main__":
    raise SystemExit(main())
