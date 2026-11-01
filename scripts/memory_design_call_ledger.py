#!/usr/bin/env python3
"""Count all recorded new calls, including optional/failed trials and probes."""
import hashlib
import json
from pathlib import Path

from summarize_context_costs import summarize

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments/m3_memory_design_20261102"


def main():
    calls, sources, by_run = [], {}, {}
    for path in sorted((ROOT / "experiments").glob("m3_*20261102/model-usage.jsonl")):
        records = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        calls.extend(records)
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        by_run[path.parent.name] = summarize(records)
    probe = json.loads((OUT / "answer_choice_probe.json").read_text())
    result = {
        "scope": "All newly recorded November 2 chat calls, not only completed method trajectories.",
        "total_recorded": summarize(calls), "by_run": by_run,
        "by_role": {
            "qa": summarize([r for r in calls if r.get("qa_video_id")]),
            "judge": summarize([r for r in calls if not r.get("qa_video_id")]),
        },
        "source_usage_log_sha256": sources,
        "separate_nonbenchmark_tool_choice_probe": {
            "calls": len(probe["results"]),
            "received_prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in probe["results"]),
            "received_completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in probe["results"]),
            "unknown_usage_calls": sum(not r.get("usage") for r in probe["results"]),
        },
        "cautions": [
            "Counts each actual log once; repeated trajectories are independent calls, not reused answers.",
            "A missing usage field is unknown billing, not a free failed request.",
            "An interrupted in-flight call may have no completed log.",
            "API seconds can include SDK retries and are not experiment wall time or money.",
            "Local GPU embedding, read-only quota queries and prior audiovisual memory generation are excluded.",
            "Protocol probes do not establish reasoning quality and are not benchmark scores.",
        ],
    }
    (OUT / "recorded_call_ledger.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["total_recorded"], indent=2))


if __name__ == "__main__":
    main()
