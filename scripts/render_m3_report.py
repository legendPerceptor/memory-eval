#!/usr/bin/env python3
"""Render numerical report sections from audited fixed-cohort metrics."""
import argparse
import json
from pathlib import Path


def number(value, digits=4):
    return "未完成有效评分" if value is None else f"{value:.{digits}f}"


def tex(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(c, c) for c in str(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--latex", type=Path, default=Path("latex/results.tex"))
    a = parser.parse_args()
    m = json.loads((a.experiment / "metrics.json").read_text())
    if m["status"] != "finished":
        raise SystemExit("Refusing to render an unfinished snapshot as final results")
    overall = m["overall"]
    videos = m["by_video"]
    usage = json.loads((a.experiment / "usage_summary.json").read_text())
    database = json.loads((a.experiment / "database_audit.json").read_text())
    artifacts = json.loads((a.experiment / "artifact_audit.json").read_text())
    lines = [
        "# MiniMax-M3.1-Flash 五视频技术评估",
        "",
        "正式配置：adaptive/medium；记忆 4 workers；记忆/QA 输出预算 8192，"
        "评委 2048；完整视频，全部原始音频，原 wrapper 检索与时间设置。",
        "",
        "## 核心结果",
        "",
        f"- 管线状态：`{m['driver_status']}`；不能把进程状态等同于质量验收。",
        f"- 固定题数：{overall['expected']}；QA 记录：{overall['qa_records']}；"
        f"非空且无运行错误的答案：{overall['valid_answers']}。",
        f"- 题目加权自动均分：**{number(overall['mean_score'])}**；"
        f"视频宏平均：**{number(m['video_macro_mean'])}**。",
        f"- 满分题比例：{number(overall['full_credit_rate'])}；"
        f"评分分布：`{overall['score_counts']}`。",
        f"- 评委错误：{overall['judge_errors']}；"
        f"有效答案中未完成评分：{overall['unscored_valid_answers']}。",
        "",
        "均分为 `{0,0.5,1}` 同模型自动评分，**不是独立人工准确率**。"
        "缺失/执行失败按固定全集计零，评委故障为未知，不能伪装成模型错误。",
        "",
        (a.experiment / "table.md").read_text(),
        "## 题型（多标签，题数不可相加）",
        "",
        "| 标签 | 题数 | 有效答案 | 自动均分 |",
        "|---|---:|---:|---:|",
    ]
    latex = [
        f"管线结束状态为 \\code{{{m['driver_status']}}}。"
        f"固定 {overall['expected']} 题中保存 {overall['qa_records']} 条 QA 记录，"
        f"{overall['valid_answers']} 条答案非空且没有运行错误。"
        f"题目加权自动均分为 \\textbf{{{number(overall['mean_score'])}}}，"
        f"视频宏平均为 \\textbf{{{number(m['video_macro_mean'])}}}。"
        f"评委错误 {overall['judge_errors']} 个。",
        r"\begin{table}[htbp]\centering\small",
        r"\begin{tabular}{lrrrrr}\toprule",
        r"视频 & 题数 & 有效答案 & 记忆/片段 & 索引 & 自动均分\\\midrule",
    ]
    for video, b in videos.items():
        latex.append(
            f"\\code{{{video}}} & {b['expected']} & {b['valid_answers']} & "
            f"{b['episodic_records']}/{b['clips']} & {b['index_records']} & "
            f"{number(b['mean_score'])} \\\\")
    latex += [r"\bottomrule\end{tabular}\end{table}",
              r"\begin{table}[htbp]\centering\small",
              r"\begin{tabular}{p{0.51\linewidth}rrr}\toprule",
              r"标签 & 题数 & 有效答案 & 自动均分\\\midrule"]
    for label, b in m["by_label_overlapping"].items():
        lines.append(f"| {label} | {b['expected']} | {b['valid_answers']} | {number(b['mean_score'])} |")
        latex.append(f"{tex(label)} & {b['expected']} & {b['valid_answers']} & {number(b['mean_score'])} \\\\")
    latex += [r"\bottomrule\end{tabular}\end{table}",
              "分类表按原始标签分别计分，多标签题重复进入多个分组；"
              "不能将分类题数相加作为总分母。总体均分仍严格按固定全集计算。"]
    lines += ["", "## 记忆、图与执行状态", ""]
    for video, b in videos.items():
        lines.append(f"- `{video}`：{b['clips']} 片段、{b['frames']} 帧、"
                     f"{b['episodic_records']} 条情景记忆、{b['index_records']} 条索引；"
                     f"失败阶段：{b['failed_stages']}。")
    invalid_edges = database["invalid_edges"][0]["count"]
    lines += [f"- SQL 账号分布：`{database['accounts']}`。",
              f"- 悬空或跨账号关系：{invalid_edges}。",
              "- 详细阶段 success/degraded/skipped/failed 及超短声纹片段等警告见 "
              "`video_summaries/`；不能仅凭总退出码概括全部执行细节。",
              "", "## 模型用量与耗时", "",
              "| 环节 | 逻辑调用 | 输入 tokens | 输出 tokens | 推理 tokens | 输入峰值 |",
              "|---|---:|---:|---:|---:|---:|"]
    latex += [
        f"SQL 审计发现悬空或跨账号关系 {invalid_edges} 条。"
        "各视频的阶段状态、失败清单和向量覆盖见机器审计文件。",
        r"\begin{table}[htbp]\centering\small",
        r"\begin{tabular}{lrrrr}\toprule",
        r"环节 & 逻辑调用 & 输入 tokens & 输出 tokens & 推理 tokens\\\midrule",
    ]
    role_names = {"memory": "记忆", "qa": "问答", "judge_or_other_text": "评委/其他文本"}
    for role, b in usage.items():
        name = role_names.get(role, role)
        lines.append(f"| {name} | {b['logical_calls']} | {b['prompt_tokens']} | "
                     f"{b['completion_tokens']} | {b['reasoning_tokens']} | {b['peak_prompt_tokens']} |")
        latex.append(f"{tex(name)} & {b['logical_calls']} & {b['prompt_tokens']} & "
                     f"{b['completion_tokens']} & {b['reasoning_tokens']} \\\\")
    latex += [r"\bottomrule\end{tabular}\end{table}",
              "用量来自实际收到的 API usage，不等于账单金额。"
              "逻辑调用耗时包含网络、排队、推理与 SDK 重试，"
              "并发请求耗时和不等于整轮墙钟时间。预检与前序中断尝试另行保存，不混入本表。",
              r"\IfFileExists{findings.tex}{\input{findings.tex}}{}"]
    lines += ["", "用量只统计本次正式运行的逻辑请求回执。预检、max 故障诊断和 Omni "
              "中断试运行分别保存，不混入本表。SDK 内部重试没有逐个 HTTP 记账；"
              "这些 token 计数不等于账单金额。",
              "", "## 观察到的阶段耗时（秒）", "",
              "| 视频 | 前处理至记忆开始 | 记忆阶段 | QA 日志跨度 |",
              "|---|---:|---:|---:|"]
    for video, entry in artifacts.items():
        times = entry.get("stage_start_deltas_seconds", {})
        pre = 0
        for stage, seconds in times.items():
            if stage == "episodic_memory":
                break
            pre += seconds
        qa_span = entry.get("last_stage_observed_span_seconds") if entry.get("last_stage") == "qa" else None
        lines.append(f"| {video} | {pre:.1f} | {number(times.get('episodic_memory'), 1)} | {number(qa_span, 1)} |")
    lines += ["", "阶段耗时由相邻阶段开始日志估算，含 IO 和调度开销。"
              "QA 最后阶段只报告最后一条日志的观察跨度，不伪装成精确逐题计时；"
              "judge-only 输出中的 latency 是评分调用耗时，不是原始 QA 用时。",
              "", "## 方法与限制", "",
              "- 多模态证据 → 情景 ExtractedNode → 片段/视频 MemoryNode；"
              "BGE-M3 向量、转写词法召回、RRF、邻近片段、ReAct QA。",
              "- 默认 QA 主检索片段情景记忆，不能将身份图/语义汇总的生成"
              "等同于已经验证它们有质量增益。",
              "- 每 30 秒片段的画面主要覆盖前约 0–15 秒；完整音频不弥补所有视觉遗漏。",
              "- current_clip 允许看到问题所在片段末尾，非严格因果评测。",
              "- 只测五个 robot 视频，study_05 为已用开发样本，不代表整个 M3-Bench。",
              "- 正式模型使用 medium 推理强度；前序 max 和 Omni 没有完整质量成绩，"
              "不能计算新模型相对它们的提升百分比。",
              "- 暂未启用有界 QA 上下文；后续可冻结记忆做独立消融。",
              "", "## 验证与产物", "",
              "源码测试：`source-tests.log`；汇总测试：`summary-tests.log`。"
              "逐题明细见 `questions.csv` / `questions.jsonl`；"
              "代码协议见 `PLAN.md`、`architecture.md`、`source.patch`；"
              "LaTeX 主文档见 `../../latex/report.tex`。",
              ""]
    if (a.experiment / "findings.md").exists():
        lines += ["", (a.experiment / "findings.md").read_text()]
    (a.experiment / "REPORT.md").write_text("\n".join(lines))
    a.latex.parent.mkdir(parents=True, exist_ok=True)
    a.latex.write_text("\n".join(latex) + "\n")
    print("Rendered Markdown report and LaTeX numerical section.")


if __name__ == "__main__":
    main()
