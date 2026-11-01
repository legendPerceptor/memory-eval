#!/usr/bin/env python3
"""Exercise real scoped fact hydration without asking/scoring benchmark QA."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "oGMemory"))
from core.models import RequestContext
from m3.config import load_m3_runtime_config
from m3.context_budget import ContextBudget
from m3.control_adapter import Search, SearchConfig
from m3.index_episodic import build_embedder
from m3.load_clips import build_graph_service
from m3.vector_index import build_m3_vector_index


def main():
    index = ROOT / "runs/m3_source_facts_20261102/manifest.json"
    budget = ContextBudget(
        fact_memory=True, identity_aware=True, incremental_delivery=True,
        fact_index_path=str(index), fact_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest())
    output = {}
    for video in ("study_05", "gym_03", "bedroom_04", "study_18", "kitchen_20"):
        name = "m3_study18_recovery_20261101" if video == "study_18" else "m3_flash_medium_20261031"
        config = ROOT / "experiments" / name / "private/config.yaml"
        runtime = load_m3_runtime_config(str(config))
        os.environ["OGMEM_GRAPH_CONNECTION_STRING"] = runtime.graph_connection_string
        vector = build_m3_vector_index(runtime)
        graph, _ = build_graph_service("sql", runtime.graph_connection_string, vector_index=vector)
        ctx = RequestContext(account_id="m3-flash-medium-20261031", user_id="smoke",
                             agent_id="smoke", session_id="fact-scope-smoke", trace_id=video)
        result = Search("What did the person say?", top_k=3, ctx=ctx, graph=graph,
                        embedder=build_embedder(None, config_path=str(config)),
                        search_config=SearchConfig(
                            video_id=video, before_clip=2, before_sec=90.167,
                            candidate_k=8, context_budget=budget))
        branches = [r for r in result["retrieval_branches"] if r["kind"] == "source_fact_vector"]
        assert branches and branches[0]["verified_nodes"] > 0
        assert branches[0]["stale_nodes"] == 0
        for hit in result["hits"]:
            assert hit["video_id"] == video and hit["clip_id"] <= 2 and hit["source_end_sec"] <= 90.167
        output[video] = {"fact_branch": branches[0],
                         "selected_clips": [h["clip_id"] for h in result["hits"]],
                         "within_scope": True, "question_or_expected_answer_read": False}
    path = ROOT / "experiments/m3_memory_design_20261102/fact_index_smoke.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({v: r["fact_branch"]["verified_nodes"] for v, r in output.items()}))


if __name__ == "__main__":
    main()
