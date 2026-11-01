#!/usr/bin/env python3
"""Reconstruct actually delivered source excerpts; never inspect expected answers."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

import psycopg2
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "oGMemory"))
from m3.context_budget import dumps
from m3.fact_index import content_hash, load_fact_index
from replay_grouped_evidence import restore_document

OUT = ROOT / "experiments/m3_memory_design_20261102"
VIDEOS = ["study_05", "gym_03", "bedroom_04", "study_18", "kitchen_20"]


def stores():
    result = {}
    for name in ("m3_flash_medium_20261031", "m3_study18_recovery_20261101"):
        config = yaml.safe_load((ROOT / "experiments" / name / "private/config.yaml").read_text())
        with psycopg2.connect(config["storage"]["connection_string"]) as db:
            db.set_session(readonly=True, isolation_level="REPEATABLE READ")
            with db.cursor() as cur:
                cur.execute("""SELECT node_id,extracted_content FROM graph_nodes
                    WHERE account_id=%s AND extractor_type='episodic_memory_generator'""",
                            ("m3-flash-medium-20261031",))
                for node, body in cur.fetchall():
                    video = body.get("source_video_id")
                    if video in VIDEOS and (video == "study_18") == (name == "m3_study18_recovery_20261101"):
                        result[(video, node)] = body
    return result


def audit_index(data):
    path = ROOT / "runs/m3_source_facts_20261102/manifest.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = json.loads((OUT / "fact_index_receipt.json").read_text())
    assert digest == receipt["manifest_sha256"]
    index = load_fact_index(str(path), digest)
    count, sources = 0, set()
    for record in index.records:
        key = (record["source_video_id"], record["node_id"])
        body = data[key]
        assert record["account_id"] == "m3-flash-medium-20261031"
        assert record["content_sha256"] == content_hash(body)
        for field in ("source_clip_id", "source_start_sec", "source_duration_sec"):
            assert record[field] == body[field]
        values = []
        for selector in record["selectors"]:
            field = selector["field"]
            if field == "description":
                start, end = selector["start_char"], selector["end_char"]
                assert 0 <= start < end <= len(body[field])
                values.append(body[field][start:end])
            else:
                index_ = selector["index"]
                assert 0 <= index_ < len(body[field])
                values.append(dumps(body[field][index_]))
        assert record["text"] == " ".join(values)
        count += 1
        sources.add(key)
    return {"fact_windows_verified": count, "source_nodes_verified": len(sources),
            "source_nodes_by_video": {v: sum(k[0] == v for k in sources) for v in VIDEOS},
            "all_original_text_and_scope_metadata_exact": True}


def audit_run(name, data, require_complete=False):
    result = {"records": 0, "context_calls": 0, "source_occurrences": 0, "ranges": 0}
    hashes = {}
    for video in VIDEOS:
        path = ROOT / "runs" / name / video / "m3_qa_results.jsonl"
        if not path.exists():
            continue
        raw = path.read_text()
        lines = raw.splitlines()
        if raw and not raw.endswith("\n"):
            lines = lines[:-1]
        if require_complete:
            hashes[video] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in lines:
            row = json.loads(line)
            result["records"] += 1
            scope = row["temporal_scope"]
            cutoff = scope.get("effective_before_sec")
            clip_bound = scope.get("resolved_before_clip")
            for call in row["search_stats"].get("context_calls", []):
                result["context_calls"] += 1
                visible = set()
                for source in call["selected_sources"]:
                    assert source["video_id"] == video
                    if clip_bound is not None:
                        assert source["clip_id"] <= clip_bound
                    body = data[(video, source["node_id"])]
                    doc, _ = restore_document(source, body, cutoff)
                    assert doc is not None, (name, row["question_id"], source["node_id"])
                    for selector in source["ranges"]:
                        key = (source["content_sha256"], dumps(selector))
                        assert key not in visible, "Duplicate visible source selector"
                        visible.add(key)
                    result["source_occurrences"] += 1
                    result["ranges"] += len(source["ranges"])
                assert len(visible) == call["selected_passages"]
    if require_complete:
        assert result["records"] == 65
        result["prediction_hashes"] = hashes
    result["all_delivered_ranges_exact_and_in_scope"] = True
    result["no_duplicate_visible_source_positions"] = True
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    if Path(args.output).name != args.output:
        raise ValueError("Expected local audit filename")
    data = stores()
    result = {"observed_at": datetime.datetime.now().astimezone().isoformat(),
              "index": audit_index(data),
              "runs": {name: audit_run(name, data, args.complete) for name in args.names}}
    (OUT / args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
