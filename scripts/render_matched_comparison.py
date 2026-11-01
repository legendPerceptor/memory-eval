#!/usr/bin/env python3
"""Consolidate immutable predictions and fully valid common-judge grades."""
import collections
import csv
import datetime
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments/m3_end_to_end_comparison_20261101"


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def qa_usage(paths, process):
    calls = [r for p in paths for r in rows(p) if r.get("process") == process and not r.get("images")]
    known = [r for r in calls if r.get("usage")]
    return {
        "logical_calls": len(calls),
        "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in known),
        "completion_tokens": sum(r["usage"]["completion_tokens"] for r in known),
        "peak_prompt_tokens": max((r["usage"]["prompt_tokens"] for r in known), default=0),
        "unknown_usage_calls": len(calls) - len(known),
    }


def main():
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / "experiments/m3_flash_medium_20261031/sample_manifest.json").read_text())
    paths = {
        "ogmemory_recovered": ROOT / "runs/m3_agent_matched_20261101/_common_judge_ogmemory_recovered",
        "m3_agent_reference_text": ROOT / "runs/m3_agent_reference_text_20261101/_aggregate",
        "m3_agent_native_tools": ROOT / "runs/m3_agent_matched_20261101/_aggregate",
    }
    data = {}
    for method, directory in paths.items():
        records = rows(directory / "m3_eval_results.jsonl")
        assert len(records) == 65 and not any(r["judge_error"] for r in records)
        keyed = {(r["video_id"], r["question_id"]): r for r in records}
        assert len(keyed) == 65
        data[method] = keyed
    expected = set(data["ogmemory_recovered"])
    assert all(set(r) == expected for r in data.values())
    raw_sources = {
        "ogmemory_recovered": [
            ROOT / ("runs/m3_study18_recovery_20261101" if v == "study_18" else "runs/m3_flash_medium_20261031")
            / v / "m3_qa_results.jsonl" for v in manifest["selected_video_ids"]],
        "m3_agent_native_tools": [
            ROOT / "runs/m3_agent_matched_20261101" / v / "m3_qa_results.jsonl"
            for v in manifest["selected_video_ids"]],
        "m3_agent_reference_text": [
            ROOT / "runs/m3_agent_reference_text_20261101" / v / "m3_qa_results.jsonl"
            for v in manifest["selected_video_ids"]],
    }
    scopes = {method: {(r["video_id"], r["question_id"]): r["temporal_scope"]
                      for path in files for r in rows(path)} for method, files in raw_sources.items()}
    for key in expected:
        baseline = scopes["ogmemory_recovered"][key]
        for method in ("m3_agent_native_tools", "m3_agent_reference_text"):
            scope = scopes[method][key]
            assert scope["policy"] == baseline["policy"], (method, key)
            assert scope["resolved_before_clip"] == baseline["resolved_before_clip"], (method, key)
            a, b = scope["effective_before_sec"], baseline["effective_before_sec"]
            assert (a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-6), (method, key, a, b)
    combined = []
    for key in sorted(expected):
        original = data["ogmemory_recovered"][key]
        row = {k: original[k] for k in ("video_id", "question_id", "question", "expected_answer")}
        row["type"] = original.get("type", "")
        for method, mapping in data.items():
            r = mapping[key]
            assert r["question"] == original["question"] and r["expected_answer"] == original["expected_answer"]
            row[method] = {k: r[k] for k in ("predicted_answer", "judge_score", "judge_reasoning",
                                            "judge_error", "error", "finish_reason")}
        combined.append(row)
    metrics = {}
    for method, mapping in data.items():
        records = list(mapping.values())
        by_video = {}
        for video in manifest["selected_video_ids"]:
            group = [r for r in records if r["video_id"] == video]
            by_video[video] = {
                "questions": len(group), "valid_answers": sum(bool(r["predicted_answer"]) and not r["error"] for r in group),
                "mean_score": statistics.fmean(r["judge_score"] for r in group),
                "finish_reasons": dict(collections.Counter(r["finish_reason"] for r in group)),
            }
        metrics[method] = {
            "questions": 65,
            "valid_answers": sum(bool(r["predicted_answer"]) and not r["error"] for r in records),
            "mean_score": statistics.fmean(r["judge_score"] for r in records),
            "video_macro_mean": statistics.fmean(r["mean_score"] for r in by_video.values()),
            "judge_errors": 0, "by_video": by_video,
            "score_counts": dict(collections.Counter(str(r["judge_score"]) for r in records)),
            "source_results_sha256": hashlib.sha256((paths[method] / "m3_eval_results.jsonl").read_bytes()).hexdigest(),
        }
        labels = collections.defaultdict(list)
        for r in records:
            types = r.get("type") or []
            for label in (types.split(",") if isinstance(types, str) else types):
                labels[label.strip()].append(r["judge_score"])
        metrics[method]["by_label_overlapping"] = {
            label: {"count": len(values), "mean_score": statistics.fmean(values)}
            for label, values in sorted(labels.items())}
    metrics["ogmemory_recovered"]["qa_usage"] = qa_usage([
        ROOT / "experiments/m3_flash_medium_20261031/model-usage.jsonl",
        ROOT / "experiments/m3_study18_recovery_20261101/model-usage.jsonl"], "run_video_pipeline.py")
    metrics["m3_agent_reference_text"]["qa_usage"] = qa_usage([
        ROOT / "experiments/m3_agent_reference_text_20261101/model-usage.jsonl"], "reference_text.py")
    metrics["m3_agent_native_tools"]["qa_usage"] = qa_usage([
        ROOT / "experiments/m3_agent_matched_20261101/model-usage.jsonl"], "runner.py")
    # Verify that the new common judging did not silently regenerate oG answers.
    old = rows(ROOT / "experiments/m3_study18_recovery_20261101/combined-fixed-grades.jsonl")
    drift = []
    for r in old:
        now = data["ogmemory_recovered"][r["video_id"], r["question_id"]]
        assert r["predicted_answer"] == now["predicted_answer"]
        if r["judge_score"] != now["judge_score"]:
            drift.append({"question_id": r["question_id"], "previous_score": r["judge_score"],
                          "common_judge_score": now["judge_score"]})
    (OUT / "metrics.json").write_text(json.dumps({
        "observed_at": datetime.datetime.now().astimezone().isoformat(),
        "cohort": manifest, "methods": metrics, "ogmemory_predictions_unchanged": True,
        "all_65_temporal_cutoffs_match": True,
        "ogmemory_grading_drift": drift,
        "interpretation": "Five-video development comparison of adapted implementations, not official checkpoint reproduction or an architecture-only causal effect.",
    }, ensure_ascii=False, indent=2) + "\n")
    (OUT / "questions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in combined))
    flat = []
    for r in combined:
        item = {k: r[k] for k in ("video_id", "question_id", "type", "question", "expected_answer")}
        for method in paths:
            for key in ("predicted_answer", "judge_score", "error", "finish_reason"):
                item[f"{method}_{key}"] = r[method][key]
        flat.append(item)
    with (OUT / "questions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    labels = {"ogmemory_recovered": "oGMemory（恢复后固定答案）",
              "m3_agent_reference_text": "M3-Agent 适配：参考 Action 文本协议",
              "m3_agent_native_tools": "M3-Agent 适配：原生工具移植诊断"}
    lines = ["# 同一 MiniMax 的五视频端到端对照", "",
             "生成/回答/评分均为 MiniMax-M3.1-Flash，adaptive/medium。"
             "oGMemory 使用四个首轮视频与 study_18 恢复后的固定预测；M3 从原视频重建感知和记忆。"
             "同一份新 M3 记忆搭配两种控制器，分别运行全部 65 题，没有择优拼接。",
             "", "## 统一评分（评委错误已清零）", "",
             "| 方法 | 有效答案 | 65 题均分 | 视频宏平均 |", "|---|---:|---:|---:|"]
    for method, m in metrics.items():
        lines.append(f"| {labels[method]} | {m['valid_answers']}/65 | **{m['mean_score']:.4f}** | {m['video_macro_mean']:.4f} |")
    lines += ["", "| 视频 | oGMemory | M3 参考文本协议 | M3 原生工具诊断 |",
              "|---|---:|---:|---:|"]
    for video in manifest["selected_video_ids"]:
        values = [metrics[k]["by_video"][video]["mean_score"] for k in paths]
        lines.append(f"| {video} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |")
    delta = metrics["ogmemory_recovered"]["mean_score"] - metrics["m3_agent_reference_text"]["mean_score"]
    lines += ["", "## 实际 QA 请求量（不含记忆、评委及 SDK 内部重试逐次计数）", "",
              "| 方法 | 逻辑调用 | 累计输入 tokens | 单次输入峰值 |",
              "|---|---:|---:|---:|"]
    for method, m in metrics.items():
        u = m["qa_usage"]
        lines.append(f"| {labels[method]} | {u['logical_calls']} | {u['prompt_tokens']} | {u['peak_prompt_tokens']} |")
    lines += ["", "累计输入包含模型报告的缓存命中，不等于计费金额；这些是不同执行时段的观测，"
              "不能据此宣称严格墙钟加速。M3 两种控制器共用一轮记忆生成，不应重复计算实际记忆成本。",
              "", "## 可以支持的结论", "",
              f"- 在这组开发样本和当前适配设置下，oGMemory 比 M3 参考文本协议版的软均分高 {delta:.4f}；"
              "这不是官方 M3-Agent 权重结论，也不是完整 M3-Bench 排名。",
              "- M3 两种控制器使用同一份记忆，但完成率和分数明显不同，说明控制协议不能被忽略。"
              "不能把原生工具迁移版的低分简单归因于记忆图。",
              "- 参考文本协议保留末轮允许合理猜测的指令。65/65 非空答案不代表 65 题正确，"
              "也不能据此宣称检索或事实依据全部充分。",
              "- oGMemory 仍有 6 题原始超时，没有为对照挑选重试后的最佳答案。",
              "", "## 评分与输入审计", "",
              "- oGMemory 共同评分的两条 429 故障仅针对失败评委项补评；所有预测及其他成功分值不变。"
              "原始故障文件保留，详见 `../m3_agent_matched_20261101/judge_recovery.json`。",
              "- 先前恢复组合分为 0.5538，本次共同评分为 0.5462。差异是评委重评，不是改答案；"
              f"检测到 {len(drift)} 题分值变化，明细在 metrics.json。",
              "- 五视频的采样帧和 WAV 逐文件字节一致；四视频 ASR 文本/时段一致。"
              "study_18 的 clip 61 有 4 条 ASR 转写/时间差异，原文差异已保存。说话人分段选定字段一致。",
              "- M3 有额外身份裁剪输入；使用共享感知替换了原版 Gemini/ERes2NetV2 与 5-fps 质量过滤，"
              "检索阈值未针对 BGE 调优。预算和可见时间相同不代表所有输入与策略完全相同。",
              "- 本轮不是严格墙钟性能试验：包含开发预检、不同时间的服务负载、共享资源和一次限流。",
              "- 参考源码 AST 原函数及源文件哈希已记录；不执行其全局安装或官方权重。",
              "- 后续发现并修复了适配器身份更新日志中原始内容列表的别名问题，未改本轮冻结输入或预测。"
              "对全部实际问题时间窗口审计后，QA 使用的文本、嵌入、边及身份映射投影一致；"
              "原始图像/声纹内容列表不是此检索路径的输入，详见 prefix_metadata_audit.json。",
              "", "## 文件", "",
              "- `questions.csv` / `questions.jsonl`：65 题三列预测、评分、原因和失败状态。",
              "- `metrics.json`：固定分母统计、来源哈希及评分波动。",
              "- 详细方法偏离：`../m3_agent_matched_20261101/PLAN.md`。",
              "- 参考文本控制协议：`../m3_agent_reference_text_20261101/PLAN.md`。",
              "- 原首轮成绩、定向恢复与本次共同评分均独立保留。", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    latex = [
        r"\section{同模型 M3-Agent 适配对照（2026 年 11 月 1 日）}",
        "本节保留首次同模型共同评分批次；后续上下文消融的交错重评另列。"
        "不同批次不能择高拼接，重评分数变化不等于原答案发生变化。",
        r"\label{sec:matched-comparison}",
        "本组从原视频重新计算 M3 感知和记忆，不复用 oGMemory 的情景/语义记忆或数据集预生成图。"
        "同一份新 M3 记忆搭配原生工具移植和参考 Action 文本协议两种控制器，"
        "分别运行全部 65 题，不择优拼接。两组官方检查点没有部署，因此这是适配实验。",
        r"\begin{table}[htbp]\centering\small\begin{tabular}{lrr}\toprule",
        r"方法 & 有效答案 & 65 题参考均分\\\midrule",
    ]
    for method, m in metrics.items():
        latex.append(f"{labels[method]} & {m['valid_answers']}/65 & {m['mean_score']:.4f} \\\\")
    latex += [r"\bottomrule\end{tabular}\end{table}",
              r"\begin{table}[htbp]\centering\small\begin{tabular}{lrrr}\toprule",
              r"视频 & oGMemory & M3 参考文本 & M3 原生工具\\\midrule"]
    for video in manifest["selected_video_ids"]:
        v = [metrics[k]["by_video"][video]["mean_score"] for k in paths]
        latex.append(f"\\code{{{video}}} & {v[0]:.4f} & {v[1]:.4f} & {v[2]:.4f} \\\\")
    latex += [
        r"\bottomrule\end{tabular}\end{table}",
        f"本组 oGMemory 的均分比 M3 参考文本协议版高 {delta:.4f}。"
        "这仅适用于当前五视频和适配设置，不能声称官方 M3-Agent 被全面超越。"
        "原生工具诊断版和参考文本版的记忆完全相同，但控制器完成率差别明显；"
        "文本版保留末轮允许合理猜测的指令，有答案不等于有充分事实依据。",
        "统一评委的两条限流失败仅补评失败项，最终三组评委错误均为零。"
        "oGMemory 预测没有修改；共同评分 0.5462 与此前组合评分 0.5538 的差异来自重评波动。"
        "所有原始故障和先前评分均保留。",
        "65 题的实际时间截止点已逐项核对一致。五视频的采样帧和 WAV 逐文件一致；"
        "四视频转写一致，study\\_18 的 clip 61 有四条"
        "转写/时间差异。M3 额外使用身份裁剪，感知后端、控制提示和原版策略存在已记录适配，"
        "故不是严格单变量架构实验。服务负载、开发停顿和限流也使总墙钟不可直接作严格加速比。",
        "详细三列逐题结果、原函数指纹、输入核对与方法偏离见 "
        r"\code{experiments/m3_end_to_end_comparison_20261101/}。"
    ]
    (ROOT / "latex/matched_comparison_results.tex").write_text("\n".join(latex) + "\n")
    print(json.dumps({k: {"mean": v["mean_score"], "valid": v["valid_answers"]} for k, v in metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
