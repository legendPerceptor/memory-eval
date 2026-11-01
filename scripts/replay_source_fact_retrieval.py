#!/usr/bin/env python3
"""Replay recorded queries with/without fact recall; no QA/judge or gold access."""
import argparse
from dataclasses import replace
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments/m3_memory_design_20261102"
NAME = "m3_source_fact_index_v2_20261102"
VIDEOS = ["study_05", "gym_03", "bedroom_04", "study_18", "kitchen_20"]


class CachedEmbedding:
    def __init__(self, inner):
        self.inner = inner
        self.cache = {}
        self.new_queries = 0

    def embed_texts(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            vectors = self.inner.embed_texts(missing)
            if len(vectors) != len(missing):
                raise ValueError("Embedding response count mismatch")
            self.cache.update(zip(missing, vectors))
            self.new_queries += len(missing)
        return [self.cache[t] for t in texts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-searches", type=int, default=20,
                        help="First N executed searches, or zero for every executed query")
    parser.add_argument("--output", default="source_retrieval_replay.json")
    args = parser.parse_args()
    if args.max_searches < 0 or Path(args.output).name != args.output:
        raise ValueError("Invalid replay arguments")
    exp, run = ROOT / "experiments" / NAME, ROOT / "runs" / NAME
    if json.loads((exp / "status.json").read_text())["status"] != "finished":
        raise ValueError("Wait for immutable complete source predictions")
    sys.path.insert(0, str(run / "_source"))
    from core.models import RequestContext
    from m3.config import load_m3_runtime_config
    from m3.context_budget import load_context_budget
    from m3.control_adapter import Search, SearchConfig
    from m3.index_episodic import build_embedder
    from m3.load_clips import build_graph_service
    from m3.vector_index import build_m3_vector_index
    import m3.fact_index as facts
    if Path(sys.modules["m3.control_adapter"].__file__).resolve() != (
        run / "_source/m3/control_adapter.py").resolve():
        raise ValueError("Replay did not load the frozen controller")

    started = time.monotonic()
    records, hashes = [], {}
    embedding_queries = 0
    for video in VIDEOS:
        path = run / video / "m3_qa_results.jsonl"
        hashes[video] = hashlib.sha256(path.read_bytes()).hexdigest()
        config = exp / "private" / f"{video}.yaml"
        runtime = load_m3_runtime_config(str(config))
        os.environ["OGMEM_GRAPH_CONNECTION_STRING"] = runtime.graph_connection_string
        graph, _ = build_graph_service(
            "sql", runtime.graph_connection_string, vector_index=build_m3_vector_index(runtime))
        embedder = CachedEmbedding(build_embedder(None, config_path=str(config)))
        budget = load_context_budget(str(config))
        ctx = RequestContext(account_id="m3-flash-medium-20261031", user_id="replay",
                             agent_id="replay", session_id="source-fact-replay", trace_id=video)
        rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        for row in rows:
            for search in row["search_history"]:
                if not search.get("executed") or search.get("status") != "success":
                    continue
                if args.max_searches and len(records) >= args.max_searches:
                    break
                scope = row["temporal_scope"]
                settings = SearchConfig(
                    video_id=video, before_clip=scope.get("resolved_before_clip"),
                    before_sec=scope.get("effective_before_sec"), default_top_k=8,
                    candidate_k=20, transcript_top_k=20, transcript_fusion="rrf",
                    include_neighbors=1, context_k=8, context_budget=budget)
                captured = {}
                original = facts.fuse_fact_candidates

                def capture(base, branch):
                    captured["base"] = {hit.node.node_id for hit in base}
                    captured["facts"] = {hit.node.node_id for hit in branch}
                    return original(base, branch)

                with patch.object(facts, "fuse_fact_candidates", capture):
                    enriched = Search(search["query"], top_k=search["top_k"],
                                      ctx=ctx, graph=graph, embedder=embedder, search_config=settings)
                plain = Search(search["query"], top_k=search["top_k"],
                               ctx=ctx, graph=graph, embedder=embedder, search_config=replace(
                                   settings, context_budget=replace(
                                       budget, fact_index_path=None, fact_index_sha256=None)))
                with_index = {hit["node_id"] for hit in enriched["hits"]}
                without_index = {hit["node_id"] for hit in plain["hits"]}
                previous = set(search["node_ids"]) - set(search["context_node_ids"]) - set(search["boundary_node_ids"])
                records.append({
                    "video_id": video, "question_id": row["question_id"], "search_round": search["round"],
                    "query": search["query"], "top_k": search["top_k"],
                    "scope": scope, "replay_matches_recorded_primary_set": with_index == previous,
                    "with_index_primary": sorted(with_index), "without_index_primary": sorted(without_index),
                    "changed_primary_nodes": sorted(with_index - without_index),
                    "base_candidate_count": len(captured["base"]),
                    "fact_candidates_outside_base_pool": sorted(captured["facts"] - captured["base"]),
                    "selected_primary_outside_base_pool": sorted(with_index - captured["base"]),
                })
        embedding_queries += embedder.new_queries
        if args.max_searches and len(records) >= args.max_searches:
            break
    result = {
        "observed_at": datetime.datetime.now().astimezone().isoformat(), "source_run": NAME,
        "source_commit": json.loads((exp / "manifest.json").read_text())["source_commit"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prediction_hashes": hashes, "max_searches": args.max_searches, "executed_queries": len(records),
        "primary_set_replay_matches": sum(r["replay_matches_recorded_primary_set"] for r in records),
        "queries_with_changed_primary_selection": sum(bool(r["changed_primary_nodes"]) for r in records),
        "queries_selecting_nodes_outside_original_candidate_pool": sum(
            bool(r["selected_primary_outside_base_pool"]) for r in records),
        "new_local_embedding_queries": embedding_queries,
        "chat_model_calls": 0, "expected_answers_used": False,
        "wall_seconds": time.monotonic() - started,
        "caution": "Retrieval-path audit, not relevance labels or proof that changed evidence improves answers.",
        "records": records,
    }
    (OUT / args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
