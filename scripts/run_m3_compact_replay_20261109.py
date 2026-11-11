#!/usr/bin/env python3
"""Reindex frozen memories and rerun QA using an isolated compact vector table."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(os.environ.get("M3_EVAL_ROOT", Path(__file__).resolve().parent.parent))
NAME = "m3_compact_replay_20261109"
SOURCE_RUN = ROOT / "runs/m3_dev_1031_ten_retry_20261108"
SOURCE_EXP = ROOT / "experiments/m3_dev_1031_ten_retry_20261108"
RUN = ROOT / "runs" / NAME
EXP = ROOT / "experiments" / NAME
SNAPSHOT = RUN / "_source"
PYTHON = ROOT / "oGMemory/.venv/bin/python"
ACCOUNT = "m3-dev-1031-ten-retry-20261108"
TABLE = "m3_compact_replay_20261109"
VIDEOS = ["bedroom_04", "bedroom_05", "gym_03", "kitchen_01", "kitchen_20",
          "kitchen_23", "study_05", "study_09", "study_18", "study_23"]


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    import psycopg2

    if RUN.exists() or EXP.exists():
        raise FileExistsError("Experiment already exists")
    config = yaml.safe_load((SOURCE_EXP / "private/config.yaml").read_text())
    with psycopg2.connect(config["vector_db"]["connection_string"]) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regclass(%s)", (TABLE,))
            if cursor.fetchone()[0] is not None:
                raise FileExistsError("Experiment vector table already exists")
    RUN.mkdir(parents=True)
    (EXP / "private").mkdir(parents=True, mode=0o700)
    shutil.copytree(
        SOURCE_RUN / "_source/oGMemory", SNAPSHOT,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ("compact_person.py", "control_adapter.py", "index_episodic.py"):
        shutil.copyfile(ROOT / "oGMemory/m3" / name, SNAPSHOT / "m3" / name)
    shutil.copyfile(Path(__file__), RUN / "driver.py")
    source_hashes = {}
    for video in VIDEOS:
        source = SOURCE_RUN / video / "episodic_memory.jsonl"
        source_hashes[video] = digest(source)
        config = yaml.safe_load((SOURCE_EXP / "private" / f"{video}.yaml").read_text())
        config["m3"]["vector_index"]["table_name"] = TABLE
        target = EXP / "private" / f"{video}.yaml"
        target.write_text(yaml.safe_dump(config, allow_unicode=True))
        target.chmod(0o600)
    save(EXP / "manifest.json", {
        "source_run": str(SOURCE_RUN.relative_to(ROOT)), "videos": VIDEOS,
        "memory_sha256": source_hashes, "vector_table": TABLE,
        "graph_account": ACCOUNT, "person_compaction": True,
        "code_sha256": {name: digest(SNAPSHOT / "m3" / name)
                        for name in ("compact_person.py", "control_adapter.py", "index_episodic.py")},
        "design": "Reuse source graph and fact indexes; isolate compact vector table and QA outputs.",
    })
    save(EXP / "status.json", {"status": "prepared", "completed": []})


def main():
    if sys.argv[1:] == ["prepare"]:
        prepare()
        return 0
    if sys.argv[1:] != ["run"]:
        raise ValueError("Use prepare or run")
    manifest = json.loads((EXP / "manifest.json").read_text())
    status = json.loads((EXP / "status.json").read_text())
    env = dict(os.environ, OGMEM_M3_COMPACT_PERSON="1", PYTHONUNBUFFERED="1",
               PYTHONHASHSEED="0", OGMEM_M3_MAX_WORKERS="1",
               OGMEM_EMBED_MAX_CONCURRENCY="1")
    env["PYTHONPATH"] = f"{SOURCE_RUN / '_instrumentation'}:{SNAPSHOT}"
    for phase in ("index", "qa"):
        for video in VIDEOS:
            key = f"{phase}/{video}"
            if key in status["completed"]:
                continue
            source = SOURCE_RUN / video / "episodic_memory.jsonl"
            if digest(source) != manifest["memory_sha256"][video]:
                raise ValueError(f"Source memory changed: {video}")
            folder = RUN / video
            folder.mkdir(exist_ok=True)
            config = EXP / "private" / f"{video}.yaml"
            env["OGMEM_GRAPH_CONNECTION_STRING"] = yaml.safe_load(
                config.read_text()
            )["storage"]["connection_string"]
            env["M3_USAGE_LOG"] = str(EXP / f"usage-{phase}-{video}.jsonl")
            env["M3_AUDIT_VIDEO"] = video
            if phase == "index":
                command = [str(PYTHON), "-m", "m3.index_episodic", "--config", str(config),
                           "--episodic-jsonl", str(source), "--output-index",
                           str(folder / "episodic_index.jsonl"), "--account-id", ACCOUNT]
            else:
                original_status = json.loads((SOURCE_EXP / "status.json").read_text())
                if original_status.get("status") == "judging":
                    status.update(status="waiting_for_original_judge", current=key)
                    save(EXP / "status.json", status)
                    while json.loads((SOURCE_EXP / "status.json").read_text()).get("status") == "judging":
                        time.sleep(60)
                command = [str(PYTHON), "-c",
                           "import sys; from m3.qa import main_react_qa; sys.exit(main_react_qa(sys.argv[1:]))",
                           "--config", str(config), "--annotations", "/data/m3_data/annotations/robot.json",
                           "--video", video, "--graph-store", "sql", "--account-id", ACCOUNT,
                           "--output-jsonl", str(folder / "m3_qa_results.jsonl"), "--temperature", "0",
                           "--max-iterations", "6", "--timeout-seconds", "180", "--top-k", "8",
                           "--candidate-k", "20", "--transcript-top-k", "20", "--transcript-fusion", "rrf",
                           "--context-k", "8", "--include-neighbors", "1", "--temporal-policy", "current_clip",
                           "--person-retrieval", "--memory-coverage-json",
                           str(SOURCE_RUN / video / "memory_coverage.json")]
                if (folder / "m3_qa_results.jsonl").exists():
                    command.append("--resume")
            status.update(status="running", current=key)
            save(EXP / "status.json", status)
            started = time.time()
            with (EXP / "private" / f"{video}-{phase}.log").open("a") as log:
                result = subprocess.run(command, cwd=SNAPSHOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            status.setdefault("stages", {})[key] = {"exit_code": result.returncode, "seconds": time.time() - started}
            complete_qa = False
            if phase == "qa" and result.returncode in (0, 2):
                output = folder / "m3_qa_results.jsonl"
                reference = SOURCE_RUN / "og_qa" / video / "m3_qa_results.jsonl"
                if output.exists():
                    actual = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
                    expected = [json.loads(line) for line in reference.read_text().splitlines() if line.strip()]
                    actual_latest = {(row["video_id"], row["question_id"]): row for row in actual}
                    expected_latest = {(row["video_id"], row["question_id"]): row for row in expected}
                    complete_qa = actual_latest.keys() == expected_latest.keys() and all(
                        actual_latest[question_id].get(field) == expected_latest[question_id].get(field)
                        for question_id in expected_latest for field in ("question", "expected_answer")
                    )
                    status["stages"][key]["attempt_rows"] = len(actual)
                    status["stages"][key]["retry_rows"] = len(actual) - len(actual_latest)
                    status["stages"][key]["question_failures"] = sum(
                        bool(row.get("error")) or not row.get("predicted_answer") for row in actual_latest.values()
                    )
            if result.returncode and not complete_qa:
                status.update(status="failed", current=key)
                save(EXP / "status.json", status)
                return result.returncode
            if phase == "qa" and not complete_qa:
                status.update(status="failed", current=key)
                save(EXP / "status.json", status)
                return 1
            status["completed"].append(key)
            save(EXP / "status.json", status)
    status.update(status="completed", current=None)
    save(EXP / "status.json", status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
