#!/usr/bin/env python3
"""Audit old QA costs by sequential call order, without changing predictions."""
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments/m3_context_study_20261101"
VIDEOS = ["study_05", "gym_03", "bedroom_04", "study_18", "kitchen_20"]
BASE = "m3_flash_medium_20261031"
RECOVERY = "m3_study18_recovery_20261101"


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main():
    joined = []
    for source, videos in [(BASE, [v for v in VIDEOS if v != "study_18"]),
                           (RECOVERY, ["study_18"])]:
        calls = sorted([r for r in rows(ROOT / "experiments" / source / "model-usage.jsonl")
                        if r["process"] == "run_video_pipeline.py" and not r.get("images")],
                       key=lambda r: r["started_at_unix"])
        offset = 0
        for video in videos:
            answers = rows(ROOT / "runs" / source / video / "m3_qa_results.jsonl")
            log = (ROOT / "runs" / source / video / "pipeline.log").read_text(errors="replace")
            timeout_spans = iter(float(s) for s in re.findall(r"ReAct timeout \(([\d.]+)s >", log))
            for answer in answers:
                count = answer["search_stats"]["model_rounds"]
                selected = calls[offset:offset + count]
                offset += count
                assert len(selected) == count and all(r.get("usage") for r in selected)
                occurrences = sum(len(r["node_ids"]) for r in answer["search_history"])
                unique = len({n for r in answer["search_history"] for n in r["node_ids"]})
                joined.append({
                    "video_id": video, "question_id": answer["question_id"],
                    "question": answer["question"], "finish_reason": answer["finish_reason"],
                    "model_rounds": count,
                    "prompt_tokens": [r["usage"]["prompt_tokens"] for r in selected],
                    "model_seconds": [r["seconds"] for r in selected],
                    "model_time_sum": sum(r["seconds"] for r in selected),
                    "retrieval_seconds": sum(r["duration_ms"] for r in answer["search_history"]) / 1000,
                    "searches": len(answer["search_history"]),
                    "node_occurrences": occurrences, "unique_node_ids": unique,
                    "overlap_fraction": 1 - unique / occurrences if occurrences else 0,
                    "observed_loop_timeout_seconds": (
                        next(timeout_spans) if answer["finish_reason"] == "timeout" else None),
                })
        assert offset == len(calls), (source, offset, len(calls))
    timeouts = [q for q in joined if q["finish_reason"] == "timeout"]
    total = sum(sum(q["prompt_tokens"]) for q in joined)
    assert len(joined) == 65 and len(timeouts) == 6 and total == 19905749
    result = {
        "attribution": "Inferred from sequential QA call order, checked against all recorded model-round counts; historical logs lack question IDs.",
        "questions": joined, "timeouts": timeouts,
        "prompt_tokens": total,
        "peak_prompt_tokens": max(max(q["prompt_tokens"]) for q in joined),
        "model_calls": sum(q["model_rounds"] for q in joined),
        "median_node_overlap_fraction": statistics.median(q["overlap_fraction"] for q in joined),
        "note": "Retrieved node overlap is not exactly wire-payload byte duplication. Timings sum logical calls including transport retries.",
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "baseline_diagnosis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for q in timeouts:
        print(q["question_id"], "input", q["prompt_tokens"],
              "model seconds", round(q["model_time_sum"], 2),
              "retrieval seconds", round(q["retrieval_seconds"], 2),
              "loop", q["observed_loop_timeout_seconds"],
              "overlap", round(q["overlap_fraction"], 3))


if __name__ == "__main__":
    main()
