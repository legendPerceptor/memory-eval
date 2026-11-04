#!/usr/bin/env python3
"""Ten-video m3_dev_1031 run with isolated data and a blinded parallel-run control."""

import collections
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(os.environ.get("M3_EVAL_ROOT", Path.cwd())).resolve()
os.environ["M3_EVAL_ROOT"] = str(ROOT)
NAME = "m3_dev_1031_ten_compare_20261104"
COMMIT = "8731930ec75153865df07689ee0c4bd98d182746"
CONTROL = ROOT / "experiments/m3_parallel_8_restart_20261103"
BASE = ROOT / "runs/m3_parallel_ogonly_20261103/driver.py"
spec = importlib.util.spec_from_file_location("parallel_base", BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.NAME = NAME
base.RUN = ROOT / "runs" / NAME
base.EXP = ROOT / "experiments" / NAME
base.SOURCE = base.RUN / "_source/oGMemory"
base.CONFIG = base.EXP / "private/config.yaml"
base.ACCOUNT = "m3-dev-1031-ten-compare-20261104"
base.DB = "ogmem_m3_dev_1031_ten_compare_20261104"
base.__file__ = __file__
base.WORKERS = {"OGMEM_M3_MAX_WORKERS": "1", "OGMEM_M3_FACE_WORKERS": "1",
                "OGMEM_EMBED_MAX_CONCURRENCY": "1", "OGMEM_GRAPH_POOL_SIZE": "16",
                "OGMEM_VECTOR_POOL_SIZE": "8"}
for key in ("NAME", "RUN", "EXP", "SOURCE", "CONFIG", "ACCOUNT", "DB"):
    setattr(base.h, key, getattr(base, key))


def archive(repo, target):
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "m3_dev_1031"], text=True).strip()
    if revision != COMMIT:
        raise ValueError(f"Branch moved: expected {COMMIT}, got {revision}")
    target.mkdir(parents=True)
    archive_path = target.parent / (target.name + ".tar")
    with archive_path.open("wb") as output:
        subprocess.run(["git", "-C", str(repo), "archive", COMMIT], stdout=output, check=True)
    with tarfile.open(archive_path) as bundle:
        bundle.extractall(target, filter="data")
    return {"commit": COMMIT, "archive_sha256": base.h.digest(archive_path)}


base.h.archive = archive
original_prepare = base.prepare
original_worker = base.worker


def prepare():
    control_status = json.loads((CONTROL / "status.json").read_text())
    if control_status.get("status") not in ("finished", "finished_with_failures"):
        raise ValueError("Parallel comparison run is not finished")
    original_prepare()
    target = base.EXP / "parallel_control"
    target.mkdir()
    paths = [CONTROL / name for name in ("manifest.json", "status.json", "joint_questions.jsonl", "comparison.json")]
    paths += list(CONTROL.glob("usage-*.jsonl"))
    for path in paths:
        shutil.copyfile(path, target / path.name)
    shutil.copytree(CONTROL / "stages", target / "stages")
    manifest_path = base.EXP / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({"workers": {**base.WORKERS, "memory": 4, "qa": 1, "judge": 1, "video": 1, "local_audio": 1},
                     "comparison_control": "m3_parallel_8_restart_20261103",
                     "parallel_control_files": {str(path.relative_to(base.EXP)): base.h.digest(path)
                                                for path in target.rglob("*") if path.is_file()},
                     "comparison_caveat": "Branch comparison also changes worker counts and execution time; not a pure commit ablation.",
                     "person_grounding": True, "person_retrieval": True,
                     "judge_protocol": "Both arms, shuffled independently, blinded common judge; failures counted zero, judge errors separate."})
    base.h.save(manifest_path, manifest)


def worker(phase, video):
    sys.path.insert(0, str(base.SOURCE))
    if phase == "pipeline":
        import m3.run_video_pipeline as pipeline
        original = pipeline.main

        def adjusted(args):
            args = list(args)
            previous = "episodic_memory.params.workers=8"
            if args.count(previous) != 1:
                raise ValueError("Unexpected pipeline worker settings")
            args[args.index(previous)] = "episodic_memory.params.workers=4"
            return original(args)

        pipeline.main = adjusted
    elif phase == "og_qa":
        import m3.qa as qa
        from m3.control_adapter import OGMemoryControlAdapter
        original = qa.main_react_qa
        original_run = OGMemoryControlAdapter.run
        questions = {q["question"]: q for q in qa.load_questions(base.h.ANNOTATIONS) if q["video_id"] == video}
        if len(questions) != sum(q["video_id"] == video for q in qa.load_questions(base.h.ANNOTATIONS)):
            raise ValueError("Cannot attribute usage for duplicate questions")

        def attributed(self, ctx, task):
            question = questions.get(task.question)
            if question is None:
                raise ValueError("Cannot attribute QA usage")
            os.environ["M3_AUDIT_VIDEO"] = video
            os.environ["M3_AUDIT_QUESTION_INDEX"] = str(question["question_index"])
            return original_run(self, ctx, task)

        def adjusted(args):
            args = list(args)
            if args.count("--workers") != 1:
                raise ValueError("Unexpected QA worker settings")
            index = args.index("--workers")
            if args[index + 1] != "4":
                raise ValueError("Unexpected QA worker count")
            del args[index:index + 2]
            return original(args)

        OGMemoryControlAdapter.run = attributed
        qa.main_react_qa = adjusted
    return original_worker(phase, video)


def judge():
    sys.path.insert(0, str(base.SOURCE))
    from core.operation_deadline import operation_deadline
    from m3.eval import build_judge_llm, judge_answer
    from m3.qa import load_questions

    manifest = json.loads((base.EXP / "manifest.json").read_text())
    for path, expected in manifest["parallel_control_files"].items():
        if base.h.digest(base.EXP / path) != expected:
            raise ValueError(f"Control changed: {path}")
    previous = {(row["video_id"], row["question_id"]): row
                for row in base.h.rows(base.EXP / "parallel_control/joint_questions.jsonl")
                if row["method"] == "ogmemory_strict_incremental"}
    questions = [row for row in load_questions(base.h.ANNOTATIONS) if row["video_id"] in manifest["selected_video_ids"]]
    if len(questions) != 128 or len(previous) != 128:
        raise ValueError("Unexpected comparison cohort")
    tasks = []
    prediction_hashes = {}
    for video in manifest["execution_order"]:
        path = base.RUN / "og_qa" / video / "m3_qa_results.jsonl"
        prediction_hashes[str(path)] = base.h.digest(path)
        current = {(row["video_id"], row["question_id"]): row for row in base.h.rows(path)}
        for question in (q for q in questions if q["video_id"] == video):
            key = (video, question["question_id"])
            if key not in current or key not in previous:
                raise ValueError(f"Missing answer row: {key}")
            for arm, row in (("branch", current[key]), ("parallel", previous[key])):
                expected = question["answer"]
                if row.get("question") != question["question"] or row.get("expected_answer") != expected:
                    raise ValueError(f"Question or reference mismatch: {key} {arm}")
                tasks.append({"arm": arm, "video_id": video, "question_id": key[1],
                              "question_index": question["question_index"], "question": question["question"],
                              "expected_answer": expected, "predicted_answer": row.get("predicted_answer", ""),
                              "error": row.get("error", "")})
    random.Random(20261104).shuffle(tasks)
    input_path = base.EXP / "judging_inputs.json"
    frozen = {"prediction_hashes": prediction_hashes, "tasks": tasks}
    if input_path.exists():
        if json.loads(input_path.read_text()) != frozen:
            raise ValueError("Judge inputs changed during resume")
    else:
        base.h.save(input_path, frozen)
    gate = base.h.QuotaGate()
    base.h.instrument(gate)
    llm = build_judge_llm(config_path=str(base.CONFIG))
    path = base.EXP / "joint_questions.jsonl"
    existing = base.h.rows(path) if path.exists() else []
    done = {(r["arm"], r["video_id"], r["question_id"]) for r in existing}
    if len(done) != len(existing):
        raise ValueError("Duplicate judge result")
    with path.open("a") as stream:
        for task in tasks:
            key = (task["arm"], task["video_id"], task["question_id"])
            if key in done:
                continue
            started = time.monotonic()
            if task["error"] or not task["predicted_answer"]:
                score, reasoning, error = 0.0, "Invalid or missing answer; no judge call.", False
            else:
                gate.wait()
                os.environ["M3_AUDIT_VIDEO"] = task["video_id"]
                os.environ["M3_AUDIT_QUESTION_INDEX"] = str(task["question_index"])
                with operation_deadline(120):
                    score, reasoning, error = judge_answer(judge_llm=llm, question=task["question"],
                                                           expected=task["expected_answer"],
                                                           predicted=task["predicted_answer"])
            stream.write(json.dumps({**task, "judge_score": score, "judge_reasoning": reasoning,
                                     "judge_error": bool(error), "judge_seconds": time.monotonic() - started},
                                    ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            done.add(key)
            base.h.save(base.EXP / "judge_status.json", {"completed": len(done), "total": len(tasks)})
    if any(base.h.digest(Path(path)) != expected for path, expected in prediction_hashes.items()):
        raise ValueError("Answers changed during judging")
    return 0


def summarize():
    rows = base.h.rows(base.EXP / "joint_questions.jsonl")
    if len(rows) != 256:
        raise ValueError("Incomplete common judging")
    methods = {}
    for arm in ("branch", "parallel"):
        group = [row for row in rows if row["arm"] == arm]
        errors = sum(row["judge_error"] for row in group)
        videos = {}
        for video in sorted({row["video_id"] for row in group}):
            subset = [row for row in group if row["video_id"] == video]
            videos[video] = {"questions": len(subset), "valid_answers": sum(bool(r["predicted_answer"]) and not r["error"] for r in subset),
                             "judge_errors": sum(r["judge_error"] for r in subset),
                             "mean_score": None if any(r["judge_error"] for r in subset) else sum(r["judge_score"] for r in subset) / len(subset)}
        methods[arm] = {"questions": len(group), "valid_answers": sum(bool(r["predicted_answer"]) and not r["error"] for r in group),
                        "judge_errors": errors, "mean_score": None if errors else sum(r["judge_score"] for r in group) / len(group),
                        "by_video": videos}
    by_key = collections.defaultdict(dict)
    for row in rows:
        by_key[(row["video_id"], row["question_id"])][row["arm"]] = row
    paired = [arms for arms in by_key.values() if set(arms) == {"branch", "parallel"}
              and all(not r["error"] and r["predicted_answer"] and not r["judge_error"] for r in arms.values())]
    resources = {}
    for label, folder in (("branch", base.EXP), ("parallel", base.EXP / "parallel_control")):
        resources[label] = {}
        for phase in ("pipeline", "og_qa", "judge"):
            calls = [row for path in folder.glob(f"usage-{phase}-*.jsonl") for row in base.h.rows(path)]
            known = [row["usage"] for row in calls if row.get("usage")]
            resources[label][phase] = {"calls": len(calls), "unknown_usage_calls": len(calls) - len(known),
                                       "prompt_tokens": sum(item.get("prompt_tokens", 0) or 0 for item in known),
                                       "completion_tokens": sum(item.get("completion_tokens", 0) or 0 for item in known),
                                       "cached_tokens": sum((item.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0 for item in known),
                                       "reasoning_tokens": sum((item.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0 for item in known)}
        stages = [json.loads(path.read_text()) for path in (folder / "stages").glob("*.json")]
        resources[label]["stage_seconds"] = {phase: sum(row.get("seconds", 0) for row in stages if row["phase"] == phase)
                                              for phase in ("pipeline", "index", "og_qa")}
    paired_scores = {arm: sum(item[arm]["judge_score"] for item in paired) / len(paired) if paired else None
                     for arm in ("branch", "parallel")}
    result = {"methods": methods, "paired_valid_questions": len(paired), "paired_mean_scores": paired_scores,
              "resources": resources, "limitations": ["Automatic 0/0.5/1 scoring, not human accuracy.",
              "Models and prompts use matching configs, but code, concurrency and execution time differ.",
              "Unknown usage is not zero cost; embedding calls are not LLM tokens."]}
    base.h.save(base.EXP / "comparison.json", result)
    lines = ["# m3_dev_1031 十视频对照", "", "同批随机盲评；失败答案计零，评委异常单列。", "",
             "| 视频 | 题数 | 分支得分 | 并行版得分 |", "|---|---:|---:|---:|"]
    for video in sorted(methods["branch"]["by_video"]):
        left, right = methods["branch"]["by_video"][video], methods["parallel"]["by_video"][video]
        fmt = lambda value: "评委错误" if value is None else f"{value:.4f}"
        lines.append(f"| {video} | {left['questions']} | {fmt(left['mean_score'])} | {fmt(right['mean_score'])} |")
    lines += ["", f"两边均有效的共同题数：{len(paired)}；得分及分阶段 token、耗时详见 `comparison.json`。",
              "", "模型、源码和并发不同，不能将差异单独归因于某个并行提交；输入缓存及未知 usage 单独记录。"]
    (base.EXP / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


base.prepare = prepare
base.worker = worker
base.judge = judge
base.summarize = summarize

if __name__ == "__main__":
    raise SystemExit(base.main())
