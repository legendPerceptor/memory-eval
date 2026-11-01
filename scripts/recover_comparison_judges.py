#!/usr/bin/env python3
"""Regrade only API-failed judge rows; predictions and successful grades stay fixed."""
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs/m3_agent_matched_20261101"
EXP = ROOT / "experiments/m3_agent_matched_20261101"
sys.path.insert(0, str(RUN / "_source/oGMemory"))
from m3.eval import build_judge_llm, judge_answer, aggregate, QuestionResult


def main():
    source = RUN / "_common_judge_ogmemory"
    destination = RUN / "_common_judge_ogmemory_recovered"
    destination.mkdir(exist_ok=False)
    original = [json.loads(s) for s in (source / "m3_eval_results.jsonl").read_text().splitlines()]
    rows = [dict(r) for r in original]
    llm = build_judge_llm(config_path=str(EXP / "private/config.yaml"))
    audit = []
    for row in rows:
        if not row.get("judge_error"):
            continue
        before = dict(row)
        attempts = []
        start = time.monotonic()
        for attempt in range(3):
            score, reason, error = judge_answer(
                judge_llm=llm, question=row["question"], expected=row["expected_answer"],
                predicted=row["predicted_answer"])
            attempts.append({"judge_error": error, "reason": reason})
            if not error:
                row.update(judge_score=score, judge_reasoning=reason, judge_error=False,
                           latency_seconds=time.monotonic() - start)
                break
            if attempt < 2:
                time.sleep(20)
        audit.append({"question_id": row["question_id"], "old_error": before["judge_reasoning"],
                      "attempts": attempts, "recovered": not row["judge_error"],
                      "score": row["judge_score"]})
    for old, new in zip(original, rows):
        assert old["predicted_answer"] == new["predicted_answer"]
        if not old.get("judge_error"):
            assert old == new
    (destination / "m3_eval_results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    previous = json.loads((source / "m3_eval_summary.json").read_text())
    fields = {f.name for f in dataclasses.fields(QuestionResult)}
    summary = aggregate([QuestionResult(**{k: v for k, v in r.items() if k in fields}) for r in rows],
                        judge_provider=previous["judge_provider"], judge_model=previous["judge_model"])
    summary.update(status="failed" if summary["failed"] or summary["judge_error_count"] else "completed",
                   prediction_coverage=1.0, missing_from_input=0, grading_recovery_count=len(audit))
    (destination / "m3_eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (EXP / "judge_recovery.json").write_text(json.dumps({
        "only_judge_errors_retried": True, "predictions_unchanged": True,
        "successful_grades_unchanged": True, "records": audit,
        "remaining_judge_errors": summary["judge_error_count"],
        "corrected_score": summary["overall_mean_score"],
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    if summary["judge_error_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
