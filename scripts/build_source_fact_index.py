#!/usr/bin/env python3
"""Build immutable, source-grounded BGE-M3 sidecar without benchmark questions."""
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import psycopg2
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "oGMemory"))
from m3.fact_index import source_facts
from m3.index_episodic import build_embedder

OUT = ROOT / "runs/m3_source_facts_20261102"
EXP = ROOT / "experiments/m3_memory_design_20261102"
ACCOUNT = "m3-flash-medium-20261031"


def main():
    OUT.mkdir()  # Never silently overwrite an index.
    started = time.monotonic()
    records, source_counts = [], {}
    for name in ("m3_flash_medium_20261031", "m3_study18_recovery_20261101"):
        path = ROOT / "experiments" / name / "private/config.yaml"
        config = yaml.safe_load(path.read_text())
        with psycopg2.connect(config["storage"]["connection_string"]) as db:
            db.set_session(readonly=True, isolation_level="REPEATABLE READ")
            with db.cursor() as cur:
                cur.execute("""SELECT node_id,extracted_content FROM graph_nodes
                    WHERE account_id=%s AND extractor_type='episodic_memory_generator'
                    ORDER BY node_id""", (ACCOUNT,))
                for node, body in cur.fetchall():
                    video = body.get("source_video_id")
                    if video not in ("study_05", "gym_03", "bedroom_04", "study_18", "kitchen_20"):
                        continue
                    if (video == "study_18") != (name == "m3_study18_recovery_20261101"):
                        continue
                    source_counts[video] = source_counts.get(video, 0) + 1
                    records.extend(source_facts(node, ACCOUNT, body))
    embedder = build_embedder(None, config_path=str(
        ROOT / "experiments/m3_flash_medium_20261031/private/config.yaml"))
    vectors = []
    for offset in range(0, len(records), 32):
        batch = records[offset:offset + 32]
        vectors.extend(embedder.embed_texts([r["text"] for r in batch]))
        (OUT / "progress.json").write_text(json.dumps(
            {"embedded": len(vectors), "total": len(records), "seconds": time.monotonic() - started}))
    matrix = np.asarray(vectors, dtype=np.float32)
    assert matrix.shape == (len(records), 1024)
    assert np.isfinite(matrix).all() and np.all(np.linalg.norm(matrix, axis=1) > 0)
    np.save(OUT / "vectors.npy", matrix, allow_pickle=False)
    vector_hash = hashlib.sha256((OUT / "vectors.npy").read_bytes()).hexdigest()
    manifest = {"format": "ogmemory-source-facts-v1", "embedding_model": "BAAI/bge-m3",
                "dimension": 1024, "vectors_file": "vectors.npy",
                "vectors_sha256": vector_hash, "records": records}
    path = OUT / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n")
    receipt = {"created_at": datetime.datetime.now().astimezone().isoformat(),
               "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "vectors_sha256": vector_hash, "fact_windows": len(records),
               "source_clips": source_counts, "dimension": 1024,
               "index_bytes": path.stat().st_size + (OUT / "vectors.npy").stat().st_size,
               "build_wall_seconds": time.monotonic() - started,
               "questions_or_reference_answers_read": False,
               "external_chat_model_calls": 0,
               "note": "Local existing BGE-M3 service; read-only SQL inputs. No perception regeneration."}
    (EXP / "fact_index_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
