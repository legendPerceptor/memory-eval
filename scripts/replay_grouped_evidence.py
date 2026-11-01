#!/usr/bin/env python3
"""Offline, read-only replay of exactly admitted evidence from arm A.

No model or embedding calls. Reconstruct sources by immutable content hash,
including boundary transcript projection, then compare presentation overhead.
This does not estimate answer quality or claim exact full-request API tokens.
"""
import hashlib
import datetime
import json
from pathlib import Path
import statistics
import sys
import subprocess

import psycopg2
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "oGMemory"))
from core.temporal import boundary_transcript, temporal_metadata
from m3.context_budget import _local_tokenizer, dumps, model_document

OUT = ROOT / "experiments/m3_context_study_20261101"
RUN = ROOT / "runs/m3_context_budget_only_20261101"
FIELDS = ("description", "speech_transcript", "events", "characters", "entities")


def body_for(content, source, boundary=False):
    body = {
        "node_id": source["node_id"], "video_id": source["video_id"],
        "clip_id": source["clip_id"], **temporal_metadata(content),
        **{k: content.get(k, "" if k == "description" else []) for k in FIELDS},
    }
    if boundary:
        body.update(source_start_sec=content["source_start_sec"], source_end_sec=content["source_end_sec"])
    return body


def restore_document(source, content, cutoff):
    body = body_for(content, source)
    boundary = False
    if hashlib.sha256(dumps(body).encode()).hexdigest() != source["content_sha256"]:
        projected = boundary_transcript(content, cutoff) if cutoff is not None else {}
        if not projected:
            return None, False
        body = body_for(projected, source, boundary=True)
        boundary = True
        if hashlib.sha256(dumps(body).encode()).hexdigest() != source["content_sha256"]:
            return None, False
    if cutoff is not None:
        assert body["source_end_sec"] <= cutoff + 1e-6
    passages = []
    for selector in source["ranges"]:
        field = selector["field"]
        if field == "description":
            start, end = selector["start_char"], selector["end_char"]
            assert 0 <= start <= end <= len(body[field])
            passages.append({**selector, "text": body[field][start:end]})
        else:
            assert 0 <= selector["index"] < len(body[field])
            passages.append({**selector, "value": body[field][selector["index"]]})
    return {**{k: v for k, v in body.items() if k not in FIELDS},
            "content_sha256": source["content_sha256"], "passages": passages}, boundary


def verify_grouped(plain, grouped):
    assert {k: v for k, v in plain.items() if k not in ("passages", "content_sha256")} == {
        k: v for k, v in grouped.items() if k not in FIELDS}
    reconstructed = {}
    for passage in plain["passages"]:
        field = passage["field"]
        value = ([passage["start_char"], passage["end_char"], passage["text"]]
                 if field == "description" else [passage["index"], passage["value"]])
        reconstructed.setdefault(field, []).append(value)
    assert reconstructed == {k: v for k, v in grouped.items() if k in FIELDS}


def main():
    stores = {}
    for run in ("m3_flash_medium_20261031", "m3_study18_recovery_20261101"):
        config = yaml.safe_load((ROOT / "experiments" / run / "private/config.yaml").read_text())
        with psycopg2.connect(config["storage"]["connection_string"]) as db:
            db.set_session(readonly=True, isolation_level="REPEATABLE READ")
            with db.cursor() as cur:
                cur.execute("""SELECT node_id,extracted_content FROM graph_nodes
                    WHERE account_id=%s AND extractor_type='episodic_memory_generator'""",
                            ("m3-flash-medium-20261031",))
                stores[run] = {str(r[0]): r[1] for r in cur.fetchall()}
    tokenizer, digest = _local_tokenizer(str(ROOT / "runs/context_assets/qwen-omni-tokenizer.json"))
    records, missing = [], []
    restored, boundary_count, passages = 0, 0, 0
    versions = set()
    for path in sorted(RUN.glob("*/m3_qa_results.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            store = stores["m3_study18_recovery_20261101" if row["video_id"] == "study_18"
                           else "m3_flash_medium_20261031"]
            cutoff = row["temporal_scope"].get("effective_before_sec")
            for i, stats in enumerate(row["search_stats"].get("context_calls", [])):
                plain, grouped = {}, {}
                for source in stats["selected_sources"]:
                    content = store.get(source["node_id"])
                    document, projected = restore_document(source, content, cutoff) if content else (None, False)
                    if document is None:
                        missing.append({"question_id": row["question_id"], "call": i,
                                        "node_id": source["node_id"], "content_sha256": source["content_sha256"]})
                        continue
                    compact = model_document(document, "grouped")
                    verify_grouped(document, compact)
                    restored += 1
                    versions.add(source["content_sha256"])
                    boundary_count += projected
                    passages += len(document["passages"])
                    plain[source["id"]], grouped[source["id"]] = document, compact
                if not plain:
                    continue
                p, g = dumps({"evidence": plain}), dumps({"evidence": grouped})
                records.append({
                    "video_id": row["video_id"], "question_id": row["question_id"], "call": i,
                    "documents": len(plain),
                    "plain_utf8_bytes": len(p.encode()), "grouped_utf8_bytes": len(g.encode()),
                    "plain_proxy_tokens": len(tokenizer.encode(p, add_special_tokens=False).ids),
                    "grouped_proxy_tokens": len(tokenizer.encode(g, add_special_tokens=False).ids),
                })
    ptotal = sum(r["plain_proxy_tokens"] for r in records)
    gtotal = sum(r["grouped_proxy_tokens"] for r in records)
    result = {
        "kind": "offline same-admission evidence-only format comparison",
        "observed_at": datetime.datetime.now().astimezone().isoformat(),
        "source_git_commit": subprocess.check_output(
            ["git", "-C", str(ROOT / "oGMemory"), "rev-parse", "HEAD"], text=True).strip(),
        "source_files_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), ROOT / "oGMemory/m3/context_budget.py",
                         ROOT / "oGMemory/core/temporal.py")
        },
        "live_model_calls": 0, "live_embedding_calls": 0,
        "database_access": "read-only repeatable-read",
        "tokenizer_sha256": digest, "calls_with_evidence": len(records),
        "source_occurrences_verified": restored, "boundary_projections_verified": boundary_count,
        "unique_source_versions_verified": len(versions),
        "admitted_passages_verified": passages,
        "unverified_sources": missing, "all_sources_verified": not missing,
        "plain_proxy_tokens": ptotal, "grouped_proxy_tokens": gtotal,
        "evidence_proxy_token_reduction": 1 - gtotal / ptotal if ptotal else None,
        "median_per_call_reduction": statistics.median(
            1 - r["grouped_proxy_tokens"] / r["plain_proxy_tokens"] for r in records),
        "content_values_and_source_selectors_unchanged": True,
        "caution": "Same selected excerpts only; excludes system prompts, tool schema, other history, refs and format instructions. Not measured API input savings or a QA quality result.",
        "records": records,
    }
    (OUT / "grouped_evidence_replay.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
