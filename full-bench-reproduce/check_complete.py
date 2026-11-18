#!/usr/bin/env python3
"""Termination gate for full_run_loop.sh (runbook §5.1 analogue).

Exit 0 only when EVERY video in the annotations has a complete, error-free
``m3_qa_results.jsonl`` under ``<work-root>/<video_id>/``. Mirrors the
driver's own `already_done` semantics (run_m3_bench_subset.py) so the loop
stops exactly when the driver would also consider every video done.

Usage:
    python3 check_complete.py <annotations.json> <work_dir_root> [-v]

Exit codes:
    0  every expected (video, question) answered, no error rows
    1  still incomplete (prints a per-video gap summary)
    2  bad arguments / unreadable inputs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load_annotations(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"annotations is not a dict: {path}")
    return data


def completed_row(row: dict) -> bool:
    ans = row.get("predicted_answer")
    return (
        not row.get("error")
        and isinstance(ans, str)
        and bool(ans.strip())
    )


def video_status(work_root: Path, video_id: str, expected_qids: set[str]) -> tuple[bool, dict]:
    jsonl = work_root / video_id / "m3_qa_results.jsonl"
    if not jsonl.exists():
        return False, {"missing_file": True, "answered": 0, "expected": len(expected_qids)}
    latest: dict[str, dict] = {}
    bad = 0
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if not isinstance(rec, dict):
            bad += 1
            continue
        if str(rec.get("video_id")) != video_id:
            continue
        qid = rec.get("question_id")
        if qid is None:
            continue
        latest[str(qid)] = rec
    answered = sum(1 for q in expected_qids if q in latest and completed_row(latest[q]))
    info = {
        "missing_file": False,
        "answered": answered,
        "expected": len(expected_qids),
        "bad_lines": bad,
        "error_rows": sum(1 for q in expected_qids if q in latest and latest[q].get("error")),
        "missing_qids": sorted(expected_qids - {q for q, r in latest.items() if completed_row(r)}),
    }
    return (answered == len(expected_qids) and bad == 0), info


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: check_complete.py <annotations.json> <work_dir_root> [-v]", file=sys.stderr)
        return 2
    ann_path, work_root = Path(argv[1]), Path(argv[2])
    verbose = "-v" in argv[3:]
    if not ann_path.exists():
        print(f"annotations not found: {ann_path}", file=sys.stderr)
        return 2
    ann = load_annotations(ann_path)
    if not work_root.exists():
        print(f"work root not found: {work_root}", file=sys.stderr)
        return 1

    total_v = len(ann)
    done_v = 0
    total_q = 0
    done_q = 0
    incomplete: list[tuple[str, dict]] = []
    for vid, entry in ann.items():
        qids = {str(q["question_id"]) for q in (entry.get("qa_list") or []) if q.get("question_id") is not None}
        expected_n = len(qids)
        total_q += expected_n
        ok, info = video_status(work_root, vid, qids)
        done_q += info["answered"]
        if ok:
            done_v += 1
        else:
            incomplete.append((vid, info))

    pct = (100.0 * done_q / total_q) if total_q else 0.0
    print(f"videos complete: {done_v}/{total_v}   questions answered: {done_q}/{total_q} ({pct:.1f}%)")
    if incomplete:
        incomplete.sort(key=lambda kv: (kv[1]["answered"] / max(1, kv[1]["expected"])))
        for vid, info in incomplete[: (len(incomplete) if verbose else 25)]:
            extra = " [MISSING FILE]" if info["missing_file"] else (
                f" [bad_lines={info['bad_lines']} error_rows={info['error_rows']}]" if (info["bad_lines"] or info["error_rows"]) else ""
            )
            print(f"  - {vid}: {info['answered']}/{info['expected']}{extra}")
        if not verbose and len(incomplete) > 25:
            print(f"  ... and {len(incomplete) - 25} more (use -v)")
        return 1
    print("ALL COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
