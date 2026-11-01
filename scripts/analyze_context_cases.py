#!/usr/bin/env python3
"""Paired case inventory; automatic scores are diagnostics, not causal labels."""
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments/m3_context_study_20261101"
VIDEOS = ["study_05", "gym_03", "bedroom_04", "study_18", "kitchen_20"]
ARMS = {
    "budget_only": "m3_context_budget_recovered_20261101",
    "finalization": "m3_context_finalization_recovered_20261101",
    "compact12k": "m3_context_compact12k_guarded_20261101",
    "grouped12k": "m3_context_grouped12k_20261101",
}


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main():
    original = {}
    for video in VIDEOS:
        run = "m3_study18_recovery_20261101" if video == "study_18" else "m3_flash_medium_20261031"
        for r in rows(ROOT / "runs" / run / video / "m3_qa_results.jsonl"):
            original[r["question_id"]] = r
    baseline_grades = {r["question_id"]: r for r in rows(
        ROOT / "runs/m3_agent_matched_20261101/_common_judge_ogmemory_recovered/m3_eval_results.jsonl")}
    joint_root = ROOT / "runs/m3_context_joint_judge_20261101"
    joint_file = joint_root / "grades_recovered.jsonl"
    if not joint_file.exists():
        joint_file = joint_root / "grades.jsonl"
    joint = rows(joint_file) if joint_file.exists() else []
    joint_complete = len(joint) == 390 and not any(r["judge_error"] for r in joint)
    if joint_complete:
        baseline_grades = {r["question_id"]: r for r in joint if r["method"] == "historical_ogmemory"}
    methods = {}
    for method, name in ARMS.items():
        metrics_path = ROOT / "experiments" / name / "metrics.json"
        if not metrics_path.exists() or not json.loads(metrics_path.read_text())["complete"]:
            continue
        grade_rows = ([r for r in joint if r["method"] == method] if joint_complete else
                      [r for v in VIDEOS for r in rows(ROOT / "runs" / name / v / "judge/m3_eval_results.jsonl")])
        grades = {r["question_id"]: r for r in grade_rows}
        answers = [r for v in VIDEOS for r in rows(ROOT / "runs" / name / v / "m3_qa_results.jsonl")]
        cases = []
        for answer in answers:
            qid = answer["question_id"]
            old, before, after = original[qid], baseline_grades[qid], grades[qid]
            assert before["predicted_answer"] == old["predicted_answer"]
            assert after["predicted_answer"] == answer["predicted_answer"]
            difference = after["judge_score"] - before["judge_score"]
            contexts = answer["search_stats"].get("context_calls", [])
            cases.append({
                "video_id": answer["video_id"], "question_id": qid,
                "question": answer["question"], "expected_answer": answer["expected_answer"],
                "original_answer": old["predicted_answer"], "new_answer": answer["predicted_answer"],
                "original_score": before["judge_score"], "new_score": after["judge_score"],
                "score_delta": difference,
                "score_change": "higher" if difference > 0 else "lower" if difference < 0 else "equal",
                "original_finish_reason": old["finish_reason"], "new_finish_reason": answer["finish_reason"],
                "timeout_recovered": old["finish_reason"] == "timeout" and not answer["error"]
                                     and bool(answer["predicted_answer"]),
                "selected_sources_on_final_call": contexts[-1]["selected_sources"] if contexts else [],
                "source_retrieval_queries": [s["query"] for s in answer["search_history"]],
            })
        methods[method] = {
            "score_changes": dict(Counter(r["score_change"] for r in cases)),
            "timeout_recoveries": [
                {k: r[k] for k in ("question_id", "new_score")} for r in cases if r["timeout_recovered"]],
            "full_score_count": sum(r["new_score"] == 1 for r in cases),
            "partial_score_count": sum(r["new_score"] == .5 for r in cases),
            "zero_score_count": sum(r["new_score"] == 0 for r in cases),
            "cases": cases,
        }
    result = {
        "grading": "joint interleaved pass" if joint_complete else "independent/previous judge passes (diagnostic)",
        "interpretation": "A score change is not automatically a retrieval/packing error or proof of improvement. Review source excerpts and grading variation.",
        "methods": methods,
    }
    (OUT / "case_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({m: {k: v for k, v in s.items() if k != "cases"} for m, s in methods.items()},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
