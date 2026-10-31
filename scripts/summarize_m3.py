#!/usr/bin/env python3
"""Audit a fixed M3 cohort; never silently drop missing questions or judge errors.

Standard library only. Source JSONL remains untouched. Duplicate/out-of-cohort
records are rejected instead of selecting a best/latest answer implicitly.
"""
import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def keyed(rows):
    result = {}
    for row in rows:
        key = str(row["video_id"]), str(row["question_id"])
        if key in result:
            raise ValueError(f"duplicate question {key}: select an explicit attempt first")
        result[key] = row
    return result


def bucket(rows):
    n = len(rows)
    known = [r["score"] for r in rows if r["score"] is not None]
    return {
        "expected": n,
        "valid_answer_rate": sum(r["valid_answer"] for r in rows) / n if n else None,
        "qa_records": sum(r["qa_present"] for r in rows),
        "valid_answers": sum(r["valid_answer"] for r in rows),
        "judge_records": sum(r["judge_present"] for r in rows),
        "judge_errors": sum(r["judge_error"] for r in rows),
        "unscored_valid_answers": sum(r["valid_answer"] and r["score"] is None for r in rows),
        "missing_qa": sum(not r["qa_present"] for r in rows),
        "qa_failures": sum(r["qa_present"] and not r["valid_answer"] for r in rows),
        "score_counts": dict(Counter(str(x) for x in known)),
        # None means valid answers remain ungraded / judge service failed.
        "mean_score": sum(known) / n if n and len(known) == n else None,
        "full_credit_rate": sum(s == 1.0 for s in known) / n if n and len(known) == n else None,
        "operational_score_missing_judges_zero": sum(known) / n if n else None,
        "mean_score_valid_answers_only": (
            statistics.fmean(r["score"] for r in rows if r["valid_answer"] and r["score"] is not None)
            if any(r["valid_answer"] and r["score"] is not None for r in rows) else None
        ),
        "finish_reasons": dict(Counter(r["finish_reason"] for r in rows)),
    }


def summarize(run, annotations, output):
    manifest = read_json(run / "sample_manifest.json")
    if manifest is None:
        raise ValueError("sample_manifest.json required")
    raw = annotations.read_bytes()
    ann = json.loads(raw)
    # The upstream manifest hashes normalized JSON, not the raw file bytes.
    from_hash = hashlib.sha256(
        json.dumps(ann, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    if from_hash != manifest["annotations_sha256"]:
        raise ValueError("annotations differ from pinned manifest")
    expected = {}
    for video in manifest["selected_video_ids"]:
        for i, q in enumerate(ann[video].get("qa_list") or ann[video].get("questions") or []):
            if not (q.get("question") or q.get("query")):
                continue
            qid = str(q.get("question_id") or q.get("id") or f"{video}_Q{i:04d}")
            if (video, qid) in expected:
                raise ValueError("duplicate annotation question")
            expected[video, qid] = q
    if len(expected) != manifest["total_questions"]:
        raise ValueError("annotation count differs from pinned manifest")
    # Inspect per-video QA files, not an aggregate that might be stale.
    qa = keyed([r for v in manifest["selected_video_ids"]
                for r in read_jsonl(run / v / "m3_qa_results.jsonl")])
    judged = keyed(read_jsonl(run / "_aggregate/m3_eval_results.jsonl"))
    if (set(qa) | set(judged)) - set(expected):
        raise ValueError("results contain questions outside the pinned cohort")
    rows, videos, types, labels = [], defaultdict(list), defaultdict(list), defaultdict(list)
    for (video, qid), q in expected.items():
        a, j = qa.get((video, qid)), judged.get((video, qid))
        valid = bool(a and str(a.get("predicted_answer") or "").strip() and not a.get("error"))
        if j is not None and a is not None:
            if str(j.get("predicted_answer") or "").strip() != str(a.get("predicted_answer") or "").strip():
                raise ValueError(f"judge prediction differs from QA: {video}/{qid}")
        judge_error = bool(j and j.get("judge_error"))
        score = None
        if not valid:
            score = 0.0
        elif j and not judge_error:
            score = float(j["judge_score"])
            if score not in (0.0, 0.5, 1.0):
                raise ValueError(f"invalid judge score {score}")
        type_value = q.get("type") or q.get("category") or []
        type_label = ",".join(type_value) if isinstance(type_value, list) else str(type_value)
        row = {
            "video_id": video, "question_id": qid, "type": type_label,
            "question": q.get("question") or q.get("query"),
            "expected_answer": q.get("answer") or q.get("gt_answer") or q.get("ground_truth") or "",
            "predicted_answer": (a or {}).get("predicted_answer", ""),
            "qa_present": a is not None, "valid_answer": valid,
            "judge_present": j is not None, "judge_error": judge_error,
            "score": score, "judge_reasoning": (j or {}).get("judge_reasoning", ""),
            "error": (a or {}).get("error", "") if a else "missing_qa",
            "finish_reason": (a or {}).get("finish_reason", "missing_qa"),
            "iterations": (a or {}).get("iterations", 0),
            "search_rounds": (a or {}).get("search_rounds", 0),
            "search_stats": (a or {}).get("search_stats", {}),
            "temporal_scope": (a or {}).get("temporal_scope", {}),
        }
        rows.append(row)
        videos[video].append(row)
        types[type_label].append(row)
        for label in (type_value if isinstance(type_value, list) else [type_label]):
            labels[str(label)].append(row)
    by_video = {}
    for video, rs in videos.items():
        p = run / video
        summary = read_json(p / "pipeline_summary.json", {})
        stages = summary.get("stages", {})
        by_video[video] = {
            **bucket(rs),
            "clips": sum(len(r["raw_nodes"]) if "raw_nodes" in r else 1
                         for r in read_jsonl(p / "data.jsonl")),
            "frames": sum(r.get("child_count", 0) for r in read_jsonl(p / "frames.jsonl")),
            "episodic_records": len(read_jsonl(p / "episodic_memory.jsonl")),
            "index_records": len(read_jsonl(p / "episodic_index.jsonl")),
            "episodic_summary": read_json(p / "episodic_memory.summary.json", {}),
            "stage_statuses": {k: v.get("status") for k, v in stages.items()},
            "failed_stages": [k for k, v in stages.items() if not v.get("ok")],
            "pipeline_summary_present": bool(summary),
        }
    means = [v["mean_score"] for v in by_video.values()]
    driver_report = read_json(run / "_aggregate/run_report.json", {})
    metrics = {
        "status": "finished" if driver_report.get("status") in ("completed", "failed") else "snapshot",
        "driver_status": driver_report.get("status", "not_finished"),
        "run_root": str(run.resolve()),
        "annotation_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "annotation_canonical_sha256": from_hash,
        "manifest": manifest, "overall": bucket(rows), "by_video": by_video,
        "by_joint_type": {t: bucket(rs) for t, rs in sorted(types.items())},
        "by_label_overlapping": {t: bucket(rs) for t, rs in sorted(labels.items())},
        "label_note": "by_label_overlapping is multi-label: one question may appear in multiple buckets; never sum these counts as the overall denominator.",
        "video_macro_mean": statistics.fmean(means) if all(v is not None for v in means) else None,
        "interpretation": "Same-model diagnostic {0,0.5,1} score, not independently verified accuracy. Missing QA scores zero; judge failures are unknown.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    (output / "questions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    fields = ["video_id", "question_id", "type", "qa_present", "valid_answer",
              "judge_present", "judge_error", "score", "finish_reason", "iterations",
              "question", "expected_answer", "predicted_answer", "judge_reasoning", "error"]
    with (output / "questions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    lines = ["| 视频 | 题数 | QA 记录 | 有效答案 | 记忆/片段 | 自动均分 |",
             "|---|---:|---:|---:|---:|---:|"]
    for video, b in by_video.items():
        score = f'{b["mean_score"]:.4f}' if b["mean_score"] is not None else "未完成评分"
        lines.append(f'| {video} | {b["expected"]} | {b["qa_records"]} | {b["valid_answers"]} | {b["episodic_records"]}/{b["clips"]} | {score} |')
    (output / "table.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(metrics["overall"], ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=Path("/data/m3_data/annotations/robot.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run, args.annotations, args.output)
