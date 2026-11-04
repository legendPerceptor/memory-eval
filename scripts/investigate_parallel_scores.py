#!/usr/bin/env python3
"""Audit matched video answers and run a separately recorded blinded regrade."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs/m3_parallel_8_restart_20261103"
SOURCE = RUN / "_source/oGMemory"
EXP = ROOT / "experiments/m3_parallel_score_investigation_20261104"
BASELINE = ROOT / "experiments/m3_parallel_8_restart_20261103/frozen_baseline/joint_questions.jsonl"
CURRENT = ROOT / "experiments/m3_parallel_8_restart_20261103/joint_questions.jsonl"
CONFIG = ROOT / "experiments/m3_parallel_8_restart_20261103/private/config.yaml"
METHOD = "ogmemory_strict_incremental"


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def keyed(path):
    result = {}
    for row in read_rows(path):
        if row["method"] != METHOD:
            continue
        key = (row["video_id"], row["question_id"])
        if key in result:
            raise ValueError(f"Duplicate question {key}")
        result[key] = row
    return result


def valid(row):
    return bool(row.get("predicted_answer")) and not row.get("error") and not row.get("judge_error")


def prepare():
    if EXP.exists():
        raise FileExistsError(EXP)
    before, after = keyed(BASELINE), keyed(CURRENT)
    if set(before) != set(after):
        raise ValueError("Question sets differ")
    pairs = []
    for key in sorted(before):
        old, new = before[key], after[key]
        if old["question"] != new["question"] or old["expected_answer"] != new["expected_answer"]:
            raise ValueError(f"Question mismatch: {key}")
        if valid(old) and valid(new):
            pairs.append({"video_id": key[0], "question_id": key[1], "question": old["question"],
                          "expected_answer": old["expected_answer"],
                          "old_answer": old["predicted_answer"], "new_answer": new["predicted_answer"],
                          "old_score": old["judge_score"], "new_score": new["judge_score"],
                          "old_reason": old.get("judge_reasoning"), "new_reason": new.get("judge_reasoning")})
    if len(pairs) != 104:
        raise ValueError(f"Unexpected matched cohort: {len(pairs)}")
    tasks = []
    for pair in pairs:
        for arm in ("old", "new"):
            tasks.append({"task_id": f"{pair['question_id']}:{arm}", "video_id": pair["video_id"],
                          "question_id": pair["question_id"], "arm": arm,
                          "question": pair["question"], "expected_answer": pair["expected_answer"],
                          "answer": pair[f"{arm}_answer"]})
    random.Random(20261104).shuffle(tasks)
    EXP.mkdir(parents=True)
    save(EXP / "pairs.json", pairs)
    save(EXP / "tasks.json", tasks)
    save(EXP / "manifest.json", {
        "input_hashes": {str(path): digest(path) for path in (BASELINE, CURRENT, CONFIG)},
        "pairs": len(pairs), "tasks": len(tasks), "seed": 20261104,
        "scope": "Only questions where both original answers and judgments were valid; excludes kitchen_20.",
        "protocol": "Independently score each answer using a single judge, with randomized order and no arm label in prompts.",
        "safety_stop_at_unix": time.time() + 48 * 3600,
        "retry_policy": "No automatic regrading of completed calls, including judge errors.",
    })
    save(EXP / "status.json", {"status": "prepared", "completed": 0, "total": len(tasks)})
    summarize_original(pairs)


def summarize_original(pairs):
    diffs = [pair["new_score"] - pair["old_score"] for pair in pairs]
    summary = {"matched_pairs": len(pairs), "old_mean": sum(p["old_score"] for p in pairs) / len(pairs),
               "new_mean": sum(p["new_score"] for p in pairs) / len(pairs),
               "improved": sum(delta > 0 for delta in diffs), "worsened": sum(delta < 0 for delta in diffs),
               "unchanged": sum(delta == 0 for delta in diffs),
               "transitions": dict(Counter(f"{p['old_score']}->{p['new_score']}" for p in pairs))}
    save(EXP / "original_summary.json", summary)


def evidence_audit():
    pairs = json.loads((EXP / "pairs.json").read_text())
    regressions = [p for p in pairs if p["old_score"] == 1 and p["new_score"] == 0]
    if len(regressions) != 9:
        raise ValueError("Unexpected full-credit regression count")
    outputs = []
    for pair in regressions:
        case = {"question_id": pair["question_id"], "video_id": pair["video_id"],
                "question": pair["question"], "expected_answer": pair["expected_answer"]}
        for arm, folder in (("old", "m3_ten_fullinterval_20261102"),
                            ("new", "m3_parallel_8_restart_20261103")):
            qa_path = ROOT / "runs" / folder / "og_qa" / pair["video_id"] / "m3_qa_results.jsonl"
            rows = [row for row in read_rows(qa_path) if row["question_id"] == pair["question_id"]]
            if len(rows) != 1 or rows[0].get("predicted_answer") != pair[f"{arm}_answer"]:
                raise ValueError(f"QA provenance mismatch: {pair['question_id']} {arm}")
            row = rows[0]
            sources = [source for call in row.get("search_stats", {}).get("context_calls", [])
                       for source in call.get("selected_sources", []) if source.get("clip_id") is not None]
            case[arm] = {"qa_file": str(qa_path.relative_to(ROOT)), "qa_file_sha256": digest(qa_path),
                         "answer": row["predicted_answer"], "search_rounds": row.get("search_rounds"),
                         "queries": [entry.get("query") for entry in row.get("search_history", [])],
                         "selected_clip_ids": sorted({source["clip_id"] for source in sources}),
                         "selected_source_records": sources,
                         "memory_file": f"runs/{folder}/{pair['video_id']}/episodic_memory.jsonl",
                         "memory_file_sha256": digest(ROOT / "runs" / folder / pair["video_id"] / "episodic_memory.jsonl")}
        old_clips, new_clips = set(case["old"]["selected_clip_ids"]), set(case["new"]["selected_clip_ids"])
        case["clip_comparison"] = {"both": sorted(old_clips & new_clips), "old_only": sorted(old_clips - new_clips),
                                   "new_only": sorted(new_clips - old_clips)}
        outputs.append(case)
    save(EXP / "nine_regressions_evidence.json", outputs)


def judge():
    manifest = json.loads((EXP / "manifest.json").read_text())
    if any(digest(Path(path)) != value for path, value in manifest["input_hashes"].items()):
        raise ValueError("Original inputs changed after task freezing")
    os.environ["M3_EVAL_ROOT"] = str(ROOT)
    spec = importlib.util.spec_from_file_location("investigation_helper", RUN / "helpers.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.EXP, helper.RUN = EXP, RUN
    gate = helper.QuotaGate()
    sys.path.insert(0, str(SOURCE))
    from core.operation_deadline import operation_deadline
    from m3.eval import build_judge_llm, judge_answer
    llm = build_judge_llm(config_path=str(CONFIG))
    tasks = json.loads((EXP / "tasks.json").read_text())
    grades = EXP / "grades.jsonl"
    existing = read_rows(grades) if grades.exists() else []
    done = {row["task_id"] for row in existing}
    if len(done) != len(existing) or not done.issubset({task["task_id"] for task in tasks}):
        raise ValueError("Invalid existing grade records")
    with grades.open("a") as stream:
        for task in tasks:
            if task["task_id"] in done:
                continue
            gate.wait()
            started = time.monotonic()
            with operation_deadline(120):
                score, reasoning, error = judge_answer(
                    judge_llm=llm, question=task["question"],
                    expected=task["expected_answer"], predicted=task["answer"])
            row = {"task_id": task["task_id"], "video_id": task["video_id"],
                   "question_id": task["question_id"], "arm": task["arm"],
                   "judge_score": score, "judge_reasoning": reasoning, "judge_error": bool(error),
                   "seconds": time.monotonic() - started}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            done.add(task["task_id"])
            save(EXP / "status.json", {"status": "judging", "completed": len(done), "total": len(tasks)})
    results = read_rows(grades)
    errors = sum(row["judge_error"] for row in results)
    if errors:
        save(EXP / "status.json", {"status": "judge_errors", "completed": len(done), "total": len(tasks), "errors": errors})
        return
    grouped = {}
    for row in results:
        grouped.setdefault(row["question_id"], {})[row["arm"]] = row["judge_score"]
    if len(grouped) != len(tasks) // 2 or any(set(value) != {"old", "new"} for value in grouped.values()):
        raise ValueError("Incomplete paired grade set")
    changes = [row["new"] - row["old"] for row in grouped.values()]
    save(EXP / "blind_summary.json", {"pairs": len(grouped),
         "old_mean": sum(row["old"] for row in grouped.values()) / len(grouped),
         "new_mean": sum(row["new"] for row in grouped.values()) / len(grouped),
         "improved": sum(value > 0 for value in changes), "worsened": sum(value < 0 for value in changes),
         "unchanged": sum(value == 0 for value in changes)})
    save(EXP / "status.json", {"status": "finished", "completed": len(done), "total": len(tasks), "errors": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "evidence_audit", "judge"))
    args = parser.parse_args()
    globals()[args.action]()
