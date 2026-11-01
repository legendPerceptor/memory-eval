#!/usr/bin/env python3
"""Read-only row-content fingerprints; publish no DSNs or row contents."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import psycopg2
from psycopg2 import sql
import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments/m3_context_study_20261101"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    args = p.parse_args()
    if not args.label.replace("_", "").isalnum():
        raise SystemExit("Unsafe snapshot label")
    result = {"observed_at": datetime.datetime.now().astimezone().isoformat(), "databases": {}}
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
                    tables[table] = {"rows": len(hashes),
                                     "sorted_row_md5_sha256": hashlib.sha256(
                                         "\n".join(hashes).encode()).hexdigest()}
        result["databases"][name] = tables
    path = OUT / f"database_{args.label}.json"
    with path.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(path)


if __name__ == "__main__":
    main()
