#!/usr/bin/env python3
"""Fresh eight-video oGMemory run with frozen paper controls and blind judging."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import yaml

ROOT = Path("/home/yuanjian/Development/memory-eval")
NAME = "m3_paper8_fresh_20261110"
VIDEOS = ["study_09", "kitchen_01", "bedroom_05", "study_23",
          "kitchen_23", "study_05", "gym_03", "bedroom_04"]
PAPER_RUN = ROOT / "runs/m3_ten_fullinterval_20261102"
PAPER_EXP = ROOT / "experiments/m3_ten_fullinterval_20261102"
SOURCE = ROOT / "runs/m3_adaptive_incremental_20261109/_source"
spec = importlib.util.spec_from_file_location(
    "comparison", ROOT / "scripts/run_m3_dev_1031_ten_compare.py"
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
base = comparison.base
values = {
    "NAME": NAME,
    "RUN": ROOT / "runs" / NAME,
    "EXP": ROOT / "experiments" / NAME,
    "ACCOUNT": "m3-paper8-fresh-20261110",
    "DB": "ogmem_m3_paper8_fresh_20261110",
}
values["SOURCE"] = values["RUN"] / "_source/oGMemory"
values["CONFIG"] = values["EXP"] / "private/config.yaml"
for key, value in values.items():
    setattr(base, key, value)
    setattr(base.h, key, value)
base.__file__ = str(Path(__file__).resolve())
base.WORKERS = {"OGMEM_M3_MAX_WORKERS": "1", "OGMEM_M3_FACE_WORKERS": "1",
                "OGMEM_EMBED_MAX_CONCURRENCY": "1", "OGMEM_GRAPH_POOL_SIZE": "16",
                "OGMEM_VECTOR_POOL_SIZE": "8"}


def archive(repo, target):
    shutil.copytree(SOURCE, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return {"commit": "8731930ec75153865df07689ee0c4bd98d182746",
            "source_snapshot": str(SOURCE.relative_to(ROOT)),
            "note": "Frozen adaptive experiment snapshot including uncommitted feature files"}


base.h.archive = archive
original_environment = base.environment


def environment(phase, video):
    return {**original_environment(phase, video), "OGMEM_M3_COMPACT_PERSON": "1",
            "OGMEM_M3_LATER_INCREMENTAL_TOKENS": "2000"}


base.environment = environment
base.worker = comparison.worker


def prepare():
    comparison.original_prepare()
    config = yaml.safe_load(base.CONFIG.read_text())
    config["llm"]["max_tokens"] = 16384
    for role in ("episodic_llm", "qa_llm"):
        config["m3"][role]["max_tokens"] = 16384
    base.CONFIG.write_text(yaml.safe_dump(config, allow_unicode=True))
    os.chmod(base.CONFIG, 0o600)
    (base.EXP / "config.redacted.yaml").write_text(
        yaml.safe_dump(base.h.redact(config), allow_unicode=True))
    context = yaml.safe_load(
        (ROOT / "experiments/m3_compact_replay_20261109/private/bedroom_04.yaml").read_text()
    )["m3"]["qa_context"]
    for key in ("fact_index_path", "fact_index_sha256"):
        context.pop(key, None)
    template = base.EXP / "private/qa_template.yaml"
    template.write_text(yaml.safe_dump({"qa_context": context}, allow_unicode=True))
    os.chmod(template, 0o600)
    controls = {}
    for arm, folder in (("paper_og", "og_qa"), ("paper_m3", "m3_qa")):
        for video in VIDEOS:
            original = PAPER_RUN / folder / video / "m3_qa_results.jsonl"
            target = base.EXP / "paper_controls" / arm / video / original.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            controls[str(target.relative_to(base.EXP))] = base.h.digest(target)
    manifest_path = base.EXP / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_hashes = {str(path.relative_to(base.SOURCE)): base.h.digest(path)
                     for path in sorted(base.SOURCE.rglob("*")) if path.is_file()}
    manifest.update({
        "execution_order": VIDEOS, "selected_video_ids": VIDEOS,
        "complete_stage_subset": VIDEOS, "total_questions": 97,
        "complete_stage_subset_questions": 97,
        "config_sha256": base.h.digest(base.CONFIG),
        "qa_template_sha256": base.h.digest(template),
        "driver_sha256": base.h.digest(base.RUN / "driver.py"),
        "source_snapshot_sha256": source_hashes,
        "paper_controls_sha256": controls,
        "workers": {**base.WORKERS, "memory": 4, "qa": 1, "video": 1},
        "person_compaction": True, "later_incremental_tokens": 2000,
        "memory_and_qa_max_tokens": 16384,
        "safety_stop_at_unix": time.time() + 48 * 3600,
        "fresh_perception_memory_index_and_qa": True,
        "paper_control": "frozen paper 8d61b772 answers, rejudged jointly",
        "caveat": "Different code, max output and service window from paper; not a causal ablation.",
    })
    base.h.save(manifest_path, manifest)
    (base.EXP / "PLAN.md").write_text(
        "# Fresh paper eight-video experiment\n\n"
        "Fresh perception, memory, vector and fact indexes, and QA; no old memories reused.\n"
        "Paper controls are frozen answers, not freshly regenerated M3 inference.\n"
        "Compact-person indexing and adaptive 3500/2000 evidence increments are enabled.\n"
        "Memory concurrency 4; QA/video concurrency 1; max output 16384.\n"
        "Report missing usage, failures, retries, stage and actual elapsed times separately.\n"
        "Blind rejudge all three answer sets together; failed answers count zero.\n"
        "The paper used different code and 8192 max output, so time differences are not causal.\n"
    )


def judge():
    sys.path.insert(0, str(base.SOURCE))
    from core.operation_deadline import operation_deadline
    from m3.eval import build_judge_llm, judge_answer
    from m3.qa import load_questions

    manifest = json.loads((base.EXP / "manifest.json").read_text())
    for path, expected in manifest["paper_controls_sha256"].items():
        if base.h.digest(base.EXP / path) != expected:
            raise ValueError(f"Frozen paper control changed: {path}")
    questions = [q for q in load_questions(base.h.ANNOTATIONS) if q["video_id"] in VIDEOS]
    if len(questions) != 97:
        raise ValueError("Paper question cohort changed")
    tasks = []
    hashes = {}
    for video in VIDEOS:
        for arm in ("paper_og", "paper_m3", "fresh"):
            path = (base.RUN / "og_qa" / video / "m3_qa_results.jsonl" if arm == "fresh"
                    else base.EXP / "paper_controls" / arm / video / "m3_qa_results.jsonl")
            hashes[str(path)] = base.h.digest(path)
            actual = {row["question_id"]: row for row in base.h.rows(path)}
            for question in (item for item in questions if item["video_id"] == video):
                row = actual.get(question["question_id"])
                if row is None or row.get("question") != question["question"] or (
                        row.get("expected_answer") != question["answer"]):
                    raise ValueError(f"Missing or mismatched question {arm}/{question['question_id']}")
                tasks.append({"arm": arm, "video_id": video,
                              "question_id": question["question_id"],
                              "question": question["question"],
                              "expected_answer": question["answer"],
                              "predicted_answer": row.get("predicted_answer", ""),
                              "error": row.get("error", "")})
    random.Random(20261110).shuffle(tasks)
    frozen = {"prediction_hashes": hashes, "tasks": tasks}
    frozen_path = base.EXP / "judging_inputs.json"
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen:
        raise ValueError("Judge inputs changed")
    base.h.save(frozen_path, frozen)
    gate = base.h.QuotaGate()
    base.h.instrument(gate)
    llm = build_judge_llm(config_path=str(base.CONFIG))
    output = base.EXP / "joint_questions.jsonl"
    existing = base.h.rows(output)
    done = {(row["arm"], row["video_id"], row["question_id"]) for row in existing}
    if len(done) != len(existing):
        raise ValueError("Duplicate judgments")
    with output.open("a") as stream:
        for task in tasks:
            key = (task["arm"], task["video_id"], task["question_id"])
            if key in done:
                continue
            score, reason, error = 0.0, "Missing or failed answer", False
            if not task["error"] and task["predicted_answer"]:
                for attempt in range(3):
                    gate.wait()
                    with operation_deadline(120):
                        score, reason, error = judge_answer(
                            judge_llm=llm, question=task["question"],
                            expected=task["expected_answer"], predicted=task["predicted_answer"])
                    if not error:
                        break
                    time.sleep(30 * (attempt + 1))
                if error:
                    raise RuntimeError(f"Judge failed: {key}")
            stream.write(json.dumps({**task, "judge_score": score,
                                     "judge_reasoning": reason}, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            done.add(key)
            base.h.save(base.EXP / "judge_status.json", {"completed": len(done), "total": len(tasks)})
    for path, expected in hashes.items():
        if base.h.digest(Path(path)) != expected:
            raise ValueError(f"Judged answers changed: {path}")
    return 0


def summarize():
    rows = base.h.rows(base.EXP / "joint_questions.jsonl")
    methods = {}
    for arm in ("paper_og", "paper_m3", "fresh"):
        group = [row for row in rows if row["arm"] == arm]
        if len(group) != 97:
            raise ValueError(f"Incomplete judgments: {arm}")
        methods[arm] = {
            "questions": len(group),
            "mean_score": sum(row["judge_score"] for row in group) / len(group),
            "failures": sum(bool(row["error"]) or not row["predicted_answer"] for row in group),
            "judge_errors": 0,
            "by_video": {video: {
                "questions": len(subset),
                "mean_score": sum(row["judge_score"] for row in subset) / len(subset),
            } for video in VIDEOS
              for subset in [[row for row in group if row["video_id"] == video]]},
        }
    result = {"methods": methods,
              "fresh_resources": {phase: base.resources(base.EXP, VIDEOS, [phase])
                                  for phase in ("pipeline", "index", "og_qa")},
              "paper_resources": {phase: base.resources(PAPER_EXP, VIDEOS, [phase])
                                  for phase in ("pipeline", "index", "og_qa")},
              "caveat": "Historical paper controls; different code/config/window, not causal."}
    base.h.save(base.EXP / "comparison.json", result)
    return result


base.prepare = prepare
base.judge = judge
base.summarize = summarize

if __name__ == "__main__":
    raise SystemExit(base.main())
