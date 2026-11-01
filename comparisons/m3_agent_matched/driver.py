"""Frozen-cohort end-to-end driver and common-judge comparison."""
import datetime
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ["M3_EVAL_ROOT"])
RUN = ROOT / "runs/m3_agent_matched_20261101"
EXP = ROOT / "experiments/m3_agent_matched_20261101"
SOURCE = RUN / "_source/oGMemory"
CONFIG = EXP / "private/config.yaml"
PY = ROOT / "oGMemory/.venv/bin/python"


def load_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def status(phase, **kwargs):
    write(EXP / "status.json", {"phase": phase, "pid": os.getpid(),
                               "observed_at": datetime.datetime.now().astimezone().isoformat(), **kwargs})


def judge(input_path, output):
    output.mkdir(parents=True, exist_ok=True)
    command = [str(PY), "-m", "m3.eval", "--config", str(CONFIG),
               "--annotations", "/data/m3_data/annotations/robot.json",
               "--video", ",".join(VIDEOS), "--input-qa-results", str(input_path),
               "--output-dir", str(output)]
    with (output / "judge.log").open("w") as log:
        result = subprocess.run(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT)
    return result.returncode


manifest = json.loads((RUN / "sample_manifest.json").read_text())
VIDEOS = manifest["selected_video_ids"]


def main():
    if (EXP / "exit_code.txt").exists():
        raise RuntimeError("This experiment already ended; do not overwrite its results")
    report = []
    for video in VIDEOS:
        work = RUN / video
        status("waiting_for_fresh_perception", video=video, per_video=report)
        deadline = time.monotonic() + 7200
        while not (work / "perception_exit_code.txt").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Fresh perception receipt missing for {video}")
            time.sleep(15)
        if (work / "perception_exit_code.txt").read_text().strip() != "0":
            report.append({"video": video, "status": "failed", "reason": "perception_failed"})
            continue
        status("memory_and_qa", video=video, per_video=report)
        started = time.monotonic()
        with (work / "agent.log").open("w") as log:
            result = subprocess.run([str(PY), str(RUN / "_adapter/runner.py"), "--video", video],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        answers = load_rows(work / "m3_qa_results.jsonl")
        coverage = json.loads((work / "memory_summary.json").read_text()) if (work / "memory_summary.json").exists() else {}
        valid = sum(bool(q.get("predicted_answer")) and not q.get("error") for q in answers)
        ok = result.returncode == 0 and valid == manifest["question_counts"][video] and not coverage.get("failed_clips")
        report.append({"video": video, "status": "completed" if ok else "degraded_or_failed",
                       "agent_exit_code": result.returncode, "agent_seconds": time.monotonic() - started,
                       "qa_records": len(answers), "valid_answers": valid, "memory": coverage})
        write(EXP / "execution.json", report)
    aggregate = RUN / "_aggregate"
    aggregate.mkdir(exist_ok=True)
    predictions = [q for v in VIDEOS for q in load_rows(RUN / v / "m3_qa_results.jsonl")]
    merged = aggregate / "all_m3_qa_results.jsonl"
    merged.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in predictions))
    status("common_judge_m3_agent", per_video=report)
    rc = judge(merged, aggregate)
    # Use exactly the fixed oGMemory predictions: four original videos plus the
    # one targeted recovery. Do not rerun, choose best answers, or drop timeouts.
    og_predictions = []
    for video in VIDEOS:
        root = ROOT / ("runs/m3_study18_recovery_20261101" if video == "study_18"
                       else "runs/m3_flash_medium_20261031")
        qs = load_rows(root / video / "m3_qa_results.jsonl")
        if len(qs) != manifest["question_counts"][video]:
            raise RuntimeError(f"Reference oGMemory predictions incomplete: {video}")
        og_predictions.extend(qs)
    og_path = aggregate / "fixed_ogmemory_predictions.jsonl"
    og_path.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in og_predictions))
    status("common_judge_fixed_ogmemory", per_video=report)
    og_dir = RUN / "_common_judge_ogmemory"
    og_rc = judge(og_path, og_dir)
    results = {}
    for method, directory in [("m3_agent_matched", aggregate), ("ogmemory_recovered", og_dir)]:
        summary = json.loads((directory / "m3_eval_summary.json").read_text())
        graded = load_rows(directory / "m3_eval_results.jsonl")
        if len(graded) != manifest["total_questions"] or len({(q["video_id"], q["question_id"]) for q in graded}) != manifest["total_questions"]:
            raise RuntimeError("Common judge did not cover the fixed cohort exactly once")
        results[method] = {"summary": summary, "predictions_sha256": hashlib.sha256(
            (merged if method == "m3_agent_matched" else og_path).read_bytes()).hexdigest()}
        compact = [{k: q.get(k) for k in ["video_id", "question_id", "question", "expected_answer",
                    "predicted_answer", "judge_score", "judge_error", "judge_reasoning", "error"]}
                   for q in graded]
        (EXP / f"{method}_questions.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in compact))
    write(EXP / "comparison.json", {"protocol": "Matched MiniMax, common 0/0.5/1 judge; method names absent from judge prompts.",
                                   "reference_is_adapted_not_official_weights": True,
                                   "perception": "fresh shared GPU pipeline; M3 adds tagged face crops",
                                   "cohort": manifest, "methods": results, "execution": report})
    lines = ["# M3-Agent 适配版与 oGMemory：固定五视频对照", "",
             "两组使用固定预测、同一 MiniMax 配置和同一评分提示重新评分，不择优选取历史评分。"
             "M3-Agent 为共享感知和模型接口的适配版，不是官方权重复现。", "",
             "| 视频 | oGMemory（恢复后固定答案） | M3-Agent 适配版 |", "|---|---:|---:|"]
    for video in VIDEOS:
        scores = [results[m]["summary"]["by_video"][video]["mean_score"] for m in ["ogmemory_recovered", "m3_agent_matched"]]
        lines.append(f"| {video} | {scores[0]:.4f} | {scores[1]:.4f} |")
    for method, record in results.items():
        s = record["summary"]
        lines += ["", f"- {method}: 65题均分 {s['overall_mean_score']:.4f}；"
                  f"有效答案 {s['answered']}/65；评委错误 {s['judge_error_count']}。"]
    lines += ["", "有评委错误时质量均分不可直接解释；请查看逐题文件。缺失答案和超时计零。"
              "人物裁剪、实体身份算法及检索粒度仍是方法差异；不能称完全相同输入的单变量实验。"
              "本次运行包含感知/生成的重叠执行及开发预检，不能用总墙钟直接宣称严格加速比。",
              "", "详细适配清单见 PLAN.md，源码和定义指纹分别见 adapter_source_hashes.json、reference_definitions.json。"]
    (EXP / "REPORT.md").write_text("\n".join(lines) + "\n")
    failed = any(r["status"] != "completed" for r in report) or rc != 0 or og_rc != 0
    (EXP / "exit_code.txt").write_text("2\n" if failed else "0\n")
    status("finished_with_report", per_video=report, comparison="comparison.json",
           has_failures=failed)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("needs_attention", error_type=type(error).__name__, error=str(error))
        (EXP / "exit_code.txt").write_text("1\n")
        raise
