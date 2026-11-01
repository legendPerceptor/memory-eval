#!/usr/bin/env python3
"""Actual recorded experiment-call ledger, separate from selected QA trajectories."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = [
    "m3_context_budget_only_20261101",
    "m3_context_budget_recovered_20261101",
    "m3_context_finalization_20261101",
    "m3_context_finalization_recovered_20261101",
    "m3_context_compact12k_guarded_20261101",
    "m3_context_grouped12k_v2_20261101",
    "m3_context_finalization_repeat_20261101",
    "m3_context_joint_judge_20261101",
]


def summarize(calls):
    known = [c for c in calls if c.get("usage")]
    prompt = sum(c["usage"]["prompt_tokens"] for c in known)
    completion = sum(c["usage"]["completion_tokens"] for c in known)
    return {
        "recorded_logical_calls": len(calls),
        "statuses": dict(Counter(c["status"] for c in calls)),
        "unknown_usage_calls": len(calls) - len(known),
        "received_prompt_tokens": prompt,
        "received_completion_tokens": completion,
        "received_total_tokens": prompt + completion,
        "recorded_api_seconds_sum": sum(c["seconds"] for c in calls),
    }


def main():
    calls, source_hashes, by_run = [], {}, {}
    for name in RUNS:
        path = ROOT / "experiments" / name / "model-usage.jsonl"
        data = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        calls.extend(data)
        source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        by_run[name] = summarize(data)
    roles = {
        "qa": summarize([c for c in calls if c.get("qa_video_id")]),
        "judge": summarize([c for c in calls if not c.get("qa_video_id")]),
    }
    probe = json.loads((ROOT / "experiments/m3_context_study_20261101/quota_recovery_probe.json").read_text())
    result = {
        "scope": "Context-study calls only; excludes prior memory generation and earlier original/M3 benchmark runs.",
        "by_run": by_run, "by_role": roles, "total_recorded": summarize(calls),
        "separate_availability_probe": {
            "calls": 1, "received_usage": probe["usage"], "benchmark_answer": False},
        "source_log_sha256": source_hashes,
        "cautions": [
            "Each actual log is counted once. Reused successful trajectories are not double-counted.",
            "Includes interrupted original A/B and their recorded failed requests, not just the final chosen trajectories.",
            "Missing usage is unknown, not zero billing. A forcibly interrupted in-flight call may lack a completion log.",
            "Logical call duration may include SDK retries; sums are not overall wall time or a bill.",
            "Read-only quota queries, local embedding service and configuration/tests are not chat-generation token calls.",
        ],
    }
    path = ROOT / "experiments/m3_context_study_20261101/recorded_call_ledger.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"by_role": roles, "total": result["total_recorded"],
                      "probe": result["separate_availability_probe"]}, indent=2))


if __name__ == "__main__":
    main()
