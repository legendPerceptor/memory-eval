#!/usr/bin/env python3
"""Resume only failed/missing joint judgments; never regrade successful rows."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(os.environ.get("M3_EVAL_ROOT", Path(__file__).resolve().parent.parent))
NAME = "m3_context_joint_judge_20261101"
DEADLINE = datetime.datetime.fromisoformat("2026-11-01T23:30:00+08:00").timestamp()


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def key(row):
    return row["method"], row["question_id"]


def audit_existing(tasks, existing):
    task_map = {key(t): t for t in tasks}
    known = {key(r): r for r in existing}
    assert len(task_map) == len(tasks) and len(known) == len(existing)
    assert known.keys() <= task_map.keys()
    for k, row in known.items():
        assert all(row.get(field) == value for field, value in task_map[k].items()), k
        assert row["judge_score"] in (0., .5, 1.)
    return known


def recover_records(tasks, existing, judge, checkpoint, should_stop=lambda: False):
    known = audit_existing(tasks, existing)
    successful = {k: dict(r) for k, r in known.items() if not r["judge_error"]}
    attempts = []
    stopped = False

    def ordered():
        return [known[key(t)] for t in tasks if key(t) in known]

    checkpoint(ordered(), attempts)
    for task in tasks:
        k = key(task)
        if k in successful:
            continue
        if should_stop():
            stopped = True
            break
        started = time.monotonic()
        called = not task.get("error") and bool(task.get("predicted_answer"))
        if not called:
            score, reason, error = 0., "QA did not produce a valid answer; no judge call.", False
        else:
            score, reason, error = judge(task)
        assert score in (0., .5, 1.)
        new = {**task, "judge_score": score, "judge_reasoning": reason,
               "judge_error": bool(error), "judge_seconds": time.monotonic() - started}
        known[k] = new
        attempts.append({"method": task["method"], "question_id": task["question_id"],
                         "model_judge_call": called,
                         "judge_error": bool(error), "judge_score": score,
                         "judge_reasoning": reason})
        checkpoint(ordered(), attempts)
        if error and ("已达到 Token Plan 用量上限" in reason
                      or ("Token Plan" in reason and "(2056)" in reason)):
            stopped = True
            break
    assert all(known[k] == old for k, old in successful.items())
    result = ordered()
    return result, {
        "preserved_successful_grades": len(successful),
        "successful_grades_unchanged": True, "predictions_and_task_fields_unchanged": True,
        "attempts": attempts, "remaining_missing": len(tasks) - len(result),
        "model_judge_calls": sum(bool(a["model_judge_call"]) for a in attempts),
        "remaining_judge_errors": sum(bool(r["judge_error"]) for r in result),
        "stopped": stopped,
    }


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    if time.time() >= DEADLINE - 180:
        raise RuntimeError("Authorization deadline is too close or has passed; no judge requests started")
    run, exp = ROOT / "runs" / NAME, ROOT / "experiments" / NAME
    with (run / ".recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / "grades_recovered.jsonl").exists():
            raise RuntimeError("Completed recovery already exists; do not regrade successful results")
        tasks = rows(run / "inputs.jsonl")
        manifest = json.loads((exp / "manifest.json").read_text())
        for paths in manifest["source_prediction_hashes"].values():
            for relative, expected in paths.items():
                assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
        assert len(tasks) == manifest["tasks"]
        state_file = exp / "recovery_state.json"
        previous = json.loads(state_file.read_text()) if state_file.exists() else {}
        source = run / previous["latest_grades"] if previous else run / "grades.jsonl"
        existing = rows(source)
        audit_existing(tasks, existing)
        attempt_id = datetime.datetime.now().strftime("recovery_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        directory = run / attempt_id
        directory.mkdir()
        destination = directory / "grades.jsonl"
        relative = str(destination.relative_to(run))

        def checkpoint(records, attempts):
            temporary = destination.with_suffix(".tmp")
            temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
            temporary.replace(destination)
            atomic_json(directory / "attempts.json", attempts)
            atomic_json(state_file, {"status": "running", "latest_grades": relative,
                                    "source_grades": str(source.relative_to(run)),
                                    "known_records": len(records), "pid": os.getpid()})

        source_tree = ROOT / "runs/m3_context_compact12k_guarded_20261101/_source"
        sys.path.insert(0, str(source_tree))
        from core.operation_deadline import operation_deadline
        from m3.eval import build_judge_llm, judge_answer
        llm = build_judge_llm(config_path=str(
            ROOT / "experiments/m3_context_compact12k_guarded_20261101/private/study_05.yaml"))

        def judge(task):
            with operation_deadline(min(120., max(.1, DEADLINE - time.time() - 180))):
                return judge_answer(judge_llm=llm, question=task["question"],
                                    expected=task["expected_answer"], predicted=task["predicted_answer"])

        records, audit = recover_records(
            tasks, existing, judge, checkpoint,
            should_stop=lambda: time.time() >= DEADLINE - 180)
        complete = not audit["remaining_missing"] and not audit["remaining_judge_errors"]
        atomic_json(directory / "receipt.json", audit)
        if complete:
            # The attempt file is immutable after publication. Keep originals.
            os.link(destination, run / "grades_recovered.jsonl")
        atomic_json(state_file, {
            "status": "finished" if complete else "incomplete",
            "latest_grades": relative, "source_grades": str(source.relative_to(run)),
            "known_records": len(records), **{k: v for k, v in audit.items() if k != "attempts"}})
        print(json.dumps({k: v for k, v in audit.items() if k != "attempts"}, ensure_ascii=False, indent=2))
        return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
