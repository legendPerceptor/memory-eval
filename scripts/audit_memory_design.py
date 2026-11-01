#!/usr/bin/env python3
"""Read-only input fingerprint and source-fact inventory; no gold-answer access."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

import psycopg2
from psycopg2 import sql
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "oGMemory"))
from m3.fact_memory import fact_units

OUT = ROOT / "experiments/m3_memory_design_20261102"
VIDEOS = ["study_05", "gym_03", "bedroom_04", "study_18", "kitchen_20"]


def collect():
    result = {"observed_at": datetime.datetime.now().astimezone().isoformat(),
              "databases": {}, "videos": {}}
    for name in ("m3_flash_medium_20261031", "m3_study18_recovery_20261101"):
        config = yaml.safe_load((ROOT / "experiments" / name / "private/config.yaml").read_text())
        tables = {}
        with psycopg2.connect(config["storage"]["connection_string"]) as db:
            db.set_session(readonly=True, isolation_level="REPEATABLE READ")
            with db.cursor() as cur:
                for table in ("graph_nodes", "graph_edges", "m3_episodic_vectors"):
                    cur.execute(sql.SQL("SELECT md5(row_to_json(t)::text) FROM {} t ORDER BY 1")
                                .format(sql.Identifier(table)))
                    hashes = [r[0] for r in cur.fetchall()]
                    tables[table] = {"rows": len(hashes), "sorted_row_md5_sha256": hashlib.sha256(
                        "\n".join(hashes).encode()).hexdigest()}
                cur.execute("""SELECT node_id,extracted_content FROM graph_nodes
                    WHERE account_id=%s AND extractor_type='episodic_memory_generator'
                    ORDER BY node_id""", ("m3-flash-medium-20261031",))
                data = cur.fetchall()
        result["databases"][name] = tables
        for video in VIDEOS:
            if (video == "study_18") != (name == "m3_study18_recovery_20261101"):
                continue
            bodies = [body for _, body in data if body.get("source_video_id") == video]
            units = [bundle for body in bodies for _, bundle in fact_units(body, 12)]
            selectors = set()
            for index, body in enumerate(bodies):
                for _, bundle in fact_units(body, 12):
                    for p in bundle:
                        selectors.add((index, json.dumps(
                            {k: v for k, v in p.items() if k not in ("text", "value")}, sort_keys=True)))
                        if p["field"] == "description":
                            assert body["description"][p["start_char"]:p["end_char"]] == p["text"]
                        else:
                            assert body[p["field"]][p["index"]] == p["value"]
            result["videos"][video] = {"clips": len(bodies), "candidate_fact_windows": len(units),
                                       "unique_source_selectors": len(selectors),
                                       "all_selectors_reconstruct_exact_original": True}
    return result


if __name__ == "__main__":
    label = sys.argv[1]
    if not label.replace("_", "").isalnum():
        raise ValueError("Unsafe label")
    result = collect()
    with (OUT / f"input_audit_{label}.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result["videos"], indent=2))
