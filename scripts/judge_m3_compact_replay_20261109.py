#!/usr/bin/env python3
"""Wait for compact QA, then blind-score both fixed answer sets."""

import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(os.environ["M3_EVAL_ROOT"])
RUN = ROOT / "runs/m3_compact_replay_20261109"
EXP = ROOT / "experiments/m3_compact_replay_20261109"
BASE = ROOT / "runs/m3_dev_1031_ten_retry_20261108"
sys.path.insert(0, str(RUN / "_source"))
from m3.eval import build_judge_llm, judge_answer
from core.operation_deadline import operation_deadline


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    while True:
        status = json.loads((EXP / "status.json").read_text())
        if status["status"] == "completed":
            break
        if status["status"] == "failed":
            raise RuntimeError(f"Replay failed: {status.get('current')}")
        time.sleep(60)
    videos = json.loads((EXP / "manifest.json").read_text())["videos"]
    tasks = []
    hashes = {}
    for video in videos:
        old_path = BASE / "og_qa" / video / "m3_qa_results.jsonl"
        new_path = RUN / video / "m3_qa_results.jsonl"
        hashes[str(old_path)] = digest(old_path)
        hashes[str(new_path)] = digest(new_path)
        old = {item["question_id"]: item for item in rows(old_path)}
        new = {item["question_id"]: item for item in rows(new_path)}
        if old.keys() != new.keys():
            raise ValueError(f"Question coverage differs for {video}")
        for question_id in sorted(old):
            if any(old[question_id].get(key) != new[question_id].get(key)
                   for key in ("question", "expected_answer")):
                raise ValueError(f"Question mismatch: {question_id}")
            for arm, item in (("original", old[question_id]), ("compact", new[question_id])):
                tasks.append({key: item.get(key) for key in
                              ("video_id", "question_id", "question_index", "question",
                               "expected_answer", "predicted_answer", "error")} | {"arm": arm})
    random.Random(20261109).shuffle(tasks)
    frozen = {"prediction_sha256": hashes, "tasks": tasks}
    input_path = EXP / "judging_inputs.json"
    if input_path.exists() and json.loads(input_path.read_text()) != frozen:
        raise ValueError("Judge inputs changed")
    if not input_path.exists():
        save(input_path, frozen)
    output = EXP / "joint_questions.jsonl"
    existing = rows(output) if output.exists() else []
    done = {(item["arm"], item["video_id"], item["question_id"]) for item in existing}
    if len(done) != len(existing):
        raise ValueError("Duplicate judgments")
    judge = build_judge_llm(config_path=str(EXP / "private" / f"{videos[0]}.yaml"))
    with output.open("a") as stream:
        for task in tasks:
            key = (task["arm"], task["video_id"], task["question_id"])
            if key in done:
                continue
            started = time.monotonic()
            if task["error"] or not task["predicted_answer"]:
                score, reasoning, error = 0.0, "Invalid or missing answer; no judge call.", False
            else:
                for attempt in range(3):
                    with operation_deadline(120):
                        score, reasoning, error = judge_answer(
                            judge_llm=judge, question=task["question"],
                            expected=task["expected_answer"], predicted=task["predicted_answer"])
                    if not error:
                        break
                    time.sleep(30 * (attempt + 1))
            stream.write(json.dumps({**task, "judge_score": score, "judge_reasoning": reasoning,
                                     "judge_error": bool(error), "judge_seconds": time.monotonic() - started},
                                    ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            done.add(key)
            save(EXP / "judge_status.json", {"completed": len(done), "total": len(tasks)})
    if any(digest(Path(path)) != value for path, value in hashes.items()):
        raise ValueError("Answers changed during judging")
    return 0


if __name__ == "__main__":
    sys.exit(main())
