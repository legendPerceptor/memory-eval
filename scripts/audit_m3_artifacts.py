#!/usr/bin/env python3
"""Read-only database and artifact audit. Run with the oGMemory Python environment."""
import argparse
import datetime
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import psycopg2
import yaml
from psycopg2 import sql


def jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--output", type=Path, help="Optional snapshot directory; defaults to --experiment")
    a = p.parse_args()
    destination = a.output or a.experiment
    destination.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((a.experiment / "private/config.yaml").read_text())
    manifest = json.loads((a.run / "sample_manifest.json").read_text())
    artifacts = {}
    for video in manifest["selected_video_ids"]:
        source = a.run / video
        target = destination / "video_summaries" / video
        target.mkdir(parents=True, exist_ok=True)
        entries = {}
        for path in sorted(source.glob("*.json")):
            if "summary" in path.name or "failures" in path.name or path.name == "input_provenance.json":
                # These are structured stage diagnostics, never YAML credentials.
                shutil.copy2(path, target / path.name)
        for name in ("data.jsonl", "frames.jsonl", "audio.jsonl", "voice_asr.jsonl",
                     "episodic_memory.jsonl", "episodic_index.jsonl", "m3_qa_results.jsonl",
                     "semantic_memories.jsonl", "speaker_turns.jsonl", "speaker_embeddings.jsonl"):
            path = source / name
            if path.exists():
                entries[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "bytes": path.stat().st_size, "records": len(jsonl(path))}
        starts, final = [], None
        log = source / "pipeline.log"
        if log.exists():
            for line in log.read_text(errors="replace").splitlines():
                match = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}).*Running stage (\w+)", line)
                if match:
                    starts.append((match[2], datetime.datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S,%f")))
                timestamp = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})", line)
                if timestamp:
                    final = datetime.datetime.strptime(timestamp[1], "%Y-%m-%d %H:%M:%S,%f")
        # Adjacent stage-start deltas include stage IO overhead. Last stage
        # only has an observed-log span; do not claim it is a precise timer.
        times = {name: (starts[i + 1][1] - time).total_seconds()
                 for i, (name, time) in enumerate(starts[:-1])}
        if starts and final:
            entries["observed_log_span_seconds"] = (final - starts[0][1]).total_seconds()
            entries["last_stage_observed_span_seconds"] = (final - starts[-1][1]).total_seconds()
            entries["last_stage"] = starts[-1][0]
        entries["stage_start_deltas_seconds"] = times
        artifacts[video] = entries
    (destination / "artifact_audit.json").write_text(json.dumps(artifacts, indent=2) + "\n")

    db_audit = {"snapshot_time": datetime.datetime.now().astimezone().isoformat()}
    with psycopg2.connect(config["storage"]["connection_string"]) as db:
        db.set_session(readonly=True, isolation_level="REPEATABLE READ")
        with db.cursor() as cur:
            def fetch(name, query, params=None):
                cur.execute(query, params)
                columns = [d[0] for d in cur.description]
                db_audit[name] = [dict(zip(columns, row)) for row in cur.fetchall()]
            fetch("accounts", "SELECT account_id,count(*) AS nodes FROM graph_nodes GROUP BY account_id")
            fetch("node_counts", """SELECT node_type,raw_type,extractor_type,extraction_status,
                count(*) AS count FROM graph_nodes GROUP BY 1,2,3,4 ORDER BY 1,2,3,4""")
            fetch("edge_counts", "SELECT edge_type,count(*) AS count FROM graph_edges GROUP BY 1 ORDER BY 1")
            fetch("invalid_edges", """SELECT count(*) AS count FROM graph_edges e
                LEFT JOIN graph_nodes s ON e.source_id=s.node_id
                LEFT JOIN graph_nodes t ON e.target_id=t.node_id
                WHERE s.node_id IS NULL OR t.node_id IS NULL
                   OR s.account_id<>e.account_id OR t.account_id<>e.account_id""")
            fetch("episodic", """SELECT
                coalesce(extracted_content->>'source_video_id',metadata->>'video_id') AS video,
                count(*) AS nodes,
                count(*) FILTER (WHERE extraction_status='completed') AS completed,
                count(*) FILTER (WHERE jsonb_typeof(metadata->'embedding')='array'
                    AND jsonb_array_length(metadata->'embedding')=1024) AS embeddings_1024
                FROM graph_nodes WHERE extractor_type='episodic_memory_generator' GROUP BY 1""")
    with psycopg2.connect(config["vector_db"]["connection_string"]) as db:
        db.set_session(readonly=True)
        with db.cursor() as cur:
            cur.execute(sql.SQL("""SELECT account_id,video_id,count(*) AS count,
                min(vector_dims(embedding)) AS min_dim,max(vector_dims(embedding)) AS max_dim
                FROM {} GROUP BY 1,2 ORDER BY 1,2""").format(
                    sql.Identifier(config["m3"]["vector_index"]["table_name"])))
            columns = [d[0] for d in cur.description]
            db_audit["vectors"] = [dict(zip(columns, row)) for row in cur.fetchall()]
    (destination / "database_audit.json").write_text(json.dumps(db_audit, indent=2, default=str) + "\n")

    usage = jsonl(a.experiment / "model-usage.jsonl")
    groups = defaultdict(list)
    for row in usage:
        role = "memory" if row["images"] else "qa" if row["tools"] else "judge_or_other_text"
        groups[role].append(row)
    result = {}
    for role, rows in groups.items():
        def total(key):
            return sum((r.get("usage") or {}).get(key) or 0 for r in rows)
        result[role] = {
            "logical_calls": len(rows), "statuses": dict(Counter(r["status"] for r in rows)),
            "models": dict(Counter(r.get("reported_model") for r in rows)),
            "prompt_tokens": total("prompt_tokens"), "completion_tokens": total("completion_tokens"),
            "reasoning_tokens": sum(((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") or 0 for r in rows),
            "cached_prompt_tokens": sum(((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0 for r in rows),
            "peak_prompt_tokens": max(((r.get("usage") or {}).get("prompt_tokens") or 0 for r in rows), default=0),
            "finish_reasons": dict(Counter(f for r in rows for f in r.get("finish_reasons", []))),
            "sum_logical_call_seconds": sum(r["seconds"] for r in rows),
            "missing_usage_responses": sum(r["status"] == "returned" and not r.get("usage") for r in rows),
        }
        events = [(r["started_at_unix"], 1) for r in rows]
        events += [(r["started_at_unix"] + r["seconds"], -1) for r in rows]
        active = peak = 0
        for _, change in sorted(events):
            active += change
            peak = max(peak, active)
        result[role]["observed_peak_inflight_logical_calls"] = peak
        result[role]["timing_note"] = "HTTP logical-call time includes network, queueing, inference and SDK retries; summed concurrent time is not wall time."
    (destination / "usage_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Saved artifact, read-only database, and logical model-call audits.")


if __name__ == "__main__":
    main()
