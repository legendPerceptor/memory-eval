#!/usr/bin/env python3
"""Separate total, cached and uncached input; no price or billing assumptions."""
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments/m3_context_study_20261101"
ARMS = {
    "budget_only": "m3_context_budget_recovered_20261101",
    "finalization": "m3_context_finalization_recovered_20261101",
    "compact12k": "m3_context_compact12k_guarded_20261101",
    "grouped12k": "m3_context_grouped12k_v2_20261101",
    "finalization_repeat": "m3_context_finalization_repeat_20261101",
}


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def observed_usage(paths, process):
    calls = [r for p in paths for r in rows(p)
             if r["process"] == process and not r.get("images")]
    known = [r for r in calls if r.get("usage")]
    unknown_cache = sum((r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") is None
                        for r in known)
    if unknown_cache:
        raise ValueError("Cannot claim exact uncached input without cache usage fields")
    if any(not 0 <= r["usage"]["prompt_tokens_details"]["cached_tokens"] <= r["usage"]["prompt_tokens"]
           for r in known):
        raise ValueError("Invalid provider cached-token count")
    total = sum(r["usage"]["prompt_tokens"] for r in known)
    cache = sum(r["usage"]["prompt_tokens_details"]["cached_tokens"] for r in known)
    return {"logical_calls": len(calls), "known_usage_calls": len(known),
            "prompt_tokens": total, "cached_prompt_tokens": cache,
            "uncached_prompt_tokens": total - cache,
            "cache_fraction": cache / total if total else 0,
            "unknown_usage_calls": len(calls) - len(known)}


def main():
    baseline = observed_usage([
        ROOT / "experiments/m3_flash_medium_20261031/model-usage.jsonl",
        ROOT / "experiments/m3_study18_recovery_20261101/model-usage.jsonl",
    ], "run_video_pipeline.py")
    methods = {"historical_ogmemory": baseline}
    methods["m3_reference_text"] = observed_usage([
        ROOT / "experiments/m3_agent_reference_text_20261101/model-usage.jsonl",
    ], "reference_text.py")
    for method, name in ARMS.items():
        path = ROOT / "experiments" / name / "metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text())
        if not metrics["complete"]:
            continue
        usage = metrics["qa_usage"]
        if usage.get("unknown_cache_usage_calls", 0):
            raise ValueError(f"{name} has unknown cache usage")
        total, cached = usage["prompt_tokens"], usage["cached_prompt_tokens"]
        methods[method] = {
            "logical_calls": usage["logical_calls"], "prompt_tokens": total,
            "cached_prompt_tokens": cached, "uncached_prompt_tokens": total - cached,
            "cache_fraction": cached / total if total else 0,
            "unknown_usage_calls": usage["unknown_usage_calls"],
            "total_input_reduction_vs_original": 1 - total / baseline["prompt_tokens"],
            "uncached_input_reduction_vs_original": 1 - (total - cached) / baseline["uncached_prompt_tokens"],
            "prior_attempts": metrics.get("prior_attempts", {}),
        }
    report = {
        "observed_at": datetime.datetime.now().astimezone().isoformat(),
        "methods": methods,
        "interpretation": [
            "All completed oGMemory trajectories cover the same 65 questions and stored memories.",
            "Historical loop failures still contributed received usage; provider failures with unknown usage remain separately recorded.",
            "Per-round evidence repacking modifies earlier tool payloads. Reduced prefix stability is a plausible explanation for the lower observed cache fraction, not a controlled proof of provider internals.",
            "The reference M3 controller appends new clip knowledge rather than rewriting earlier tool payloads.",
            "Cached and uncached tokens may have different costs; these are observed usage counts, not a bill or a token-plan conversion formula.",
        ],
    }
    (OUT / "cache_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(methods, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
