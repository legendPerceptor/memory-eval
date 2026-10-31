#!/usr/bin/env python3
"""Create an isolated run config/database; no API credentials are printed."""
import hashlib
import argparse
import json
import os
import subprocess
from pathlib import Path

import psycopg2
import yaml
from psycopg2 import sql
from psycopg2.extensions import make_dsn, parse_dsn

ROOT = Path(__file__).resolve().parent.parent
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="m3_minimax_flash_20261031")
    parser.add_argument("--database", default="ogmem_m3_flash_20261031")
    parser.add_argument("--account", default="m3-flash-20261031")
    parser.add_argument("--reasoning-effort", choices=["medium", "high", "max"], default="max")
    args = parser.parse_args()
    if not all(c.isalnum() or c in "_-" for c in args.experiment_id):
        raise SystemExit("Experiment ID must be a single safe path component")
    EXP = ROOT / "experiments" / args.experiment_id
    DB, ACCOUNT = args.database, args.account
    if not os.environ.get("MINIMAX_API_KEY"):
        raise SystemExit("Source ~/.bashrc and export MINIMAX_API_KEY first")
    path = EXP / "private/config.yaml"
    if path.exists():
        raise SystemExit("Config exists; refusing to replace pinned experiment settings")
    config = yaml.safe_load((ROOT / "oGMemory/data/ogmem_test_lyj.resolved.yaml").read_text())
    for role in ("storage", "vector_db"):
        params = parse_dsn(config[role]["connection_string"])
        db = psycopg2.connect(**params)
        db.autocommit = True
        with db.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname=%s", (DB,))
            if not cursor.fetchone():
                cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB)))
        db.close()
        params["dbname"] = DB
        config[role]["connection_string"] = make_dsn(**params)
    with psycopg2.connect(config["storage"]["connection_string"]) as db:
        with db.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute("SELECT to_regclass('graph_nodes')")
            if cursor.fetchone()[0] is not None:
                cursor.execute("SELECT count(*) FROM graph_nodes")
                if cursor.fetchone()[0]:
                    raise SystemExit("Target database is not empty; choose a new experiment")
    model = {
        "provider": "minimax", "api_key": "${MINIMAX_API_KEY}",
        "base_url": "https://api.minimax.cn/v1",
        "model": "MiniMax-M3.1-Flash", "max_tokens": 8192,
        "temperature": 0, "request_timeout_seconds": 300,
        "is_reasoning_model": True, "reasoning_effort": args.reasoning_effort,
        "extra_body": {"thinking": {"type": "adaptive"}},
    }
    config["llm"] = dict(model)
    for role in ("episodic_llm", "qa_llm", "judge_llm"):
        config["m3"][role] = {**model, "max_tokens": 2048 if role == "judge_llm" else 8192}
    config["identity"]["account_id"] = ACCOUNT
    config["m3"]["vector_index"]["table_name"] = "m3_episodic_vectors"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(yaml.safe_dump(config, allow_unicode=True))
    path.chmod(0o600)

    def redact(value):
        if isinstance(value, dict):
            return {k: "<REDACTED>" if k in ("api_key", "connection_string", "password", "hf_token")
                    else redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [redact(v) for v in value]
        return value

    (EXP / "config.redacted.yaml").write_text(yaml.safe_dump(redact(config), allow_unicode=True))
    metadata = {
        "status": "prepared", "date": "2026-10-31",
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT / "oGMemory"), "rev-parse", "HEAD"], text=True).strip(),
        "source_branch": "dev", "source_dirty": True,
        "source_change": "Optional reasoning settings and bounded independent-clip workers; defaults unchanged.",
        "database": DB, "account": ACCOUNT,
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model": model["model"], "endpoint": model["base_url"],
        "seed": 42, "sample_size": 5,
        "reasoning": {"thinking": "adaptive", "reasoning_effort": args.reasoning_effort,
                      "temperature_sent": False, "memory_max_completion_tokens": 8192,
                      "qa_max_completion_tokens": 8192, "judge_max_completion_tokens": 2048},
        "previous_run": "m3_baseline_20261031 interrupted at user request; no comparative score",
    }
    (EXP / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Prepared isolated Flash experiment; keys remain in environment/private configuration.")


if __name__ == "__main__":
    main()
