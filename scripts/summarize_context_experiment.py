#!/usr/bin/env python3
"""Summarize all recorded attempts; never modify QA or judge outputs."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent
COHORT = ["study_05", "gym_03", "bedroom_04", "study_18", "kitchen_20"]


def rows(path):
    if not path.exists():
        return []
    # Active writer may have an incomplete last line. Do not hide malformed
    # complete records; this is only a live-progress convenience.
    text = path.read_text()
    lines = text.splitlines()
    if text and not text.endswith("\n"):
        lines = lines[:-1]
    return [json.loads(s) for s in lines if s.strip()]


def distribution(values):
    if not values:
        return {"count": 0}
    values = sorted(values)
    return {"count": len(values), "min": values[0], "max": values[-1],
            "mean": statistics.mean(values), "median": statistics.median(values),
            "p95_nearest_rank": values[max(0, __import__("math").ceil(.95 * len(values)) - 1)]}


def summarize(name):
    exp, run = ROOT / "experiments" / name, ROOT / "runs" / name
    manifest = json.loads((exp / "manifest.json").read_text())
    qa = [r for v in COHORT for r in rows(run / v / "m3_qa_results.jsonl")]
    assert len({(r["video_id"], r["question_id"]) for r in qa}) == len(qa)
    historical = {}
    for video in COHORT:
        source = "m3_study18_recovery_20261101" if video == "study_18" else "m3_flash_medium_20261031"
        for r in rows(ROOT / "runs" / source / video / "m3_qa_results.jsonl"):
            historical[(video, r["question_id"])] = r
    scope_mismatches = []
    if len(historical) == 65:
        for r in qa:
            previous = historical[(r["video_id"], r["question_id"])]
            assert r["question"] == previous["question"]
            if r["temporal_scope"] != previous["temporal_scope"]:
                scope_mismatches.append(r["question_id"])
    judge = [r for v in COHORT for r in rows(run / v / "judge/m3_eval_results.jsonl")]
    grade_map = {(r["video_id"], r["question_id"]): r for r in judge}
    assert len(grade_map) == len(judge)
    usage = rows(exp / "model-usage.jsonl")
    qa_calls = [r for r in usage if r["process"] == "runner.py" and r.get("qa_video_id")]
    known = [r for r in qa_calls if r.get("usage")]
    counter_exceedances, input_exceedances, pairing_errors = [], [], []
    records = []
    for r in qa:
        calls = sorted([c for c in qa_calls if c["qa_video_id"] == r["video_id"]
                        and int(c["qa_question_index"]) == r["question_index"]],
                       key=lambda c: c["started_at_unix"])
        key = (r["video_id"], r["question_id"])
        g = grade_map.get(key)
        if g:
            assert g["predicted_answer"] == r["predicted_answer"], key
        packed = r["search_stats"].get("context_calls", [])
        if len(packed) != len(calls):
            pairing_errors.append({"question_id": r["question_id"],
                                   "packed": len(packed), "calls": len(calls)})
        else:
            for index, (p, c) in enumerate(zip(packed, calls)):
                if not c.get("usage"):
                    continue
                actual = c["usage"]["prompt_tokens"]
                if actual > p["estimated_input_tokens"]:
                    counter_exceedances.append([r["question_id"], index, actual, p["estimated_input_tokens"]])
                if actual > p["input_limit"]:
                    input_exceedances.append([r["question_id"], index, actual, p["input_limit"]])
        records.append({
            "video_id": r["video_id"], "question_id": r["question_id"],
            "question": r["question"], "expected_answer": r["expected_answer"],
            "predicted_answer": r["predicted_answer"], "error": r["error"],
            "finish_reason": r["finish_reason"],
            "valid_answer": bool(r["predicted_answer"]) and not r["error"],
            "judge_score": g["judge_score"] if g else None,
            "judge_error": g["judge_error"] if g else None,
            "judge_reasoning": g["judge_reasoning"] if g else None,
            "model_calls": len(calls),
            "prompt_tokens": sum(c.get("usage", {}).get("prompt_tokens", 0) for c in calls if c.get("usage")),
            "peak_prompt_tokens": max((c.get("usage", {}).get("prompt_tokens", 0)
                                      for c in calls if c.get("usage")), default=0),
            "model_seconds": sum(c["seconds"] for c in calls),
            "packing_seconds": sum(p["packing_seconds"] for p in packed),
            "retrieval_seconds": sum(s["duration_ms"] for s in r["search_history"]) / 1000,
            "wall_seconds": r["search_stats"].get("wall_seconds"),
            "finalization_reason": r["search_stats"].get("finalization_reason"),
            "stop_reason": r["search_stats"].get("stop_reason"),
            "searches": r["search_stats"].get("search_requests"),
            "temporal_scope": r["temporal_scope"],
        })
    all_packed = [p for r in qa for p in r["search_stats"].get("context_calls", [])]
    score_sum = sum(g["judge_score"] for g in judge if not g["judge_error"])
    metrics = {
        "name": name, "source_commit": manifest["source_commit"],
        "expected_questions": 65, "qa_records": len(qa), "judge_records": len(judge),
        "valid_answers": sum(r["valid_answer"] for r in records),
        "judge_errors": sum(bool(g["judge_error"]) for g in judge),
        "complete": len(qa) == len(judge) == 65 and not any(g["judge_error"] for g in judge),
        "score_sum": score_sum,
        "mean_score_all_65": score_sum / 65,
        "mean_score_observed_judged": score_sum / len(judge) if judge else None,
        "finish_reasons": dict(Counter(r["finish_reason"] for r in qa)),
        "finalization_reasons": dict(Counter(r["finalization_reason"] or "not_triggered" for r in records)),
        "qa_usage": {
            "logical_calls": len(qa_calls),
            "unknown_usage_calls": len(qa_calls) - len(known),
            "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in known),
            "completion_tokens": sum(r["usage"]["completion_tokens"] for r in known),
            "reasoning_tokens": sum((r["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
                                    for r in known),
            "peak_prompt_tokens": max((r["usage"]["prompt_tokens"] for r in known), default=0),
            "cached_prompt_tokens": sum((r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
                                        for r in known),
            "model_seconds_sum": sum(r["seconds"] for r in qa_calls),
        },
        "packing_seconds": distribution([r["packing_seconds"] for r in all_packed]),
        "question_wall_seconds": distribution([r["wall_seconds"] for r in records if r["wall_seconds"] is not None]),
        "counter_exceedances": counter_exceedances,
        "input_limit_exceedances": input_exceedances,
        "usage_pairing_errors": pairing_errors,
        "historical_scope_audit_available": len(historical) == 65,
        "temporal_scope_mismatches": scope_mismatches,
        "model_setting_deviations": [
            {"video": c["qa_video_id"], "question_index": c["qa_question_index"],
             "started_at_unix": c["started_at_unix"]}
            for c in qa_calls if (
                c.get("requested_model") != "MiniMax-M3.1-Flash"
                or c.get("max_tokens") != 8192
                or c.get("reasoning_effort") != "medium"
                or c.get("thinking") != {"type": "adaptive"}
            )
        ],
        "by_video": {},
        "qa_files_sha256": {
            v: hashlib.sha256((run / v / "m3_qa_results.jsonl").read_bytes()).hexdigest()
            for v in COHORT if (run / v / "m3_qa_results.jsonl").exists()
        },
        "interpretation": "Fixed-memories QA development ablation, not independent human accuracy or an end-to-end memory regeneration.",
    }
    for video in COHORT:
        selected = [r for r in records if r["video_id"] == video]
        graded = [r for r in selected if r["judge_score"] is not None]
        metrics["by_video"][video] = {
            "records": len(selected), "valid_answers": sum(r["valid_answer"] for r in selected),
            "judged": len(graded),
            "mean_score": statistics.mean(r["judge_score"] for r in graded) if graded else None,
            "prompt_tokens": sum(r["prompt_tokens"] for r in selected),
            "model_calls": sum(r["model_calls"] for r in selected),
            "judge_errors": sum(bool(r["judge_error"]) for r in graded),
        }
    (exp / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    (exp / "questions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    if records:
        with (exp / "questions.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        raise SystemExit("Unsafe experiment name")
    result = summarize(args.name)
    print(json.dumps({k: result[k] for k in (
        "name", "qa_records", "judge_records", "valid_answers", "mean_score_all_65",
        "complete", "qa_usage", "input_limit_exceedances", "usage_pairing_errors")}, indent=2))
