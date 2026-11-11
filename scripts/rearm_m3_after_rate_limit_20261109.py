#!/usr/bin/env python3
"""Preserve incomplete judgments and requeue failed QA stages."""

import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parent.parent
for name in ("early_answer", "meaning_guard"):
    exp = ROOT / "experiments" / f"m3_{name}_20261109"
    run = ROOT / "runs" / f"m3_{name}_20261109" / name
    path = exp / "status.json"
    status = json.loads(path.read_text())
    archive = exp / "interrupted_judgments" / time.strftime("%Y%m%d-%H%M%S")
    for pattern in (f"judge-input-{name}.json", f"judgments-{name}.jsonl",
                    f"usage-judge-{name}.jsonl", f"summary-{name}.json"):
        source = exp / pattern
        if source.exists():
            archive.mkdir(parents=True, exist_ok=True)
            shutil.move(source, archive / source.name)
    requeued = []
    for answer_file in sorted(run.glob("*/m3_qa_results.jsonl")):
        latest = {}
        for line in answer_file.read_text().splitlines():
            row = json.loads(line)
            latest[row["question_id"]] = row
        if any(row.get("error") or not row.get("predicted_answer") for row in latest.values()):
            stage = f"qa/{name}/{answer_file.parent.name}"
            requeued.append(stage)
            status["completed"] = [item for item in status["completed"] if item != stage]
    status["completed"] = [item for item in status["completed"]
                           if item not in (f"judge/{name}", f"summary/{name}")]
    status["recovery"] = {"reason": "API 429 rate and plan usage limits",
                          "requeued": requeued,
                          "archived_judgments": str(archive.relative_to(ROOT))}
    status.update(status="waiting_for_api_quota", current=None)
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(name, requeued)
