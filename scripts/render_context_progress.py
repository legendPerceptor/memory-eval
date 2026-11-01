#!/usr/bin/env python3
"""Publish a clearly provisional section; never replace a finalized section."""
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARMS = {
    "A：预算（恢复）": "m3_context_budget_recovered_20261101",
    "B：预算+收尾": "m3_context_finalization_recovered_20261101",
    "C：12k预算": "m3_context_compact12k_guarded_20261101",
    "D：12k分组": "m3_context_grouped12k_v2_20261101",
    "B独立重复": "m3_context_finalization_repeat_20261101",
}


def main():
    target = ROOT / "latex/context_optimization_results.tex"
    if target.exists() and "CONTEXT_REPORT_STATUS: final" in target.read_text():
        raise RuntimeError("Refuse to replace final context results with a progress report")
    now = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %z")
    replay_path = ROOT / "experiments/m3_context_study_20261101/grouped_evidence_replay_full65.json"
    replay = json.loads(replay_path.read_text())
    lines = [
        "% CONTEXT_REPORT_STATUS: provisional",
        r"\section{上下文预算与 QA 收尾消融（中期）}",
        r"\label{sec:context-optimization}",
        r"\textbf{本节为中期记录，共同交错重评尚未完成，不是最终排名。}",
        f"更新于 {now}。用户指定本轮自主实验截止时间为2026年11月1日23:30（北京时间）。",
        "固定原五视频和65题，读取同一份已生成记忆，仅改变QA上下文与控制方式。"
        "模型为MiniMax-M3.1-Flash，adaptive/medium，QA输出8192，6轮/180秒，"
        "时间范围仍为current\\_clip。没有重新采帧、转写或生成记忆。",
        r"\begin{center}\small\begin{tabular}{lrrrrr}\toprule",
        r"条件 & 有效回答 & 独立/恢复评分 & 输入/M & 峰值/k & QA调用\\\midrule",
        r"原固定答案 & 59/65 & 0.5462 & 19.906 & 156.69 & 319\\",
        r"M3参考协议适配 & 65/65 & 0.4615 & 0.570 & 4.74 & 310\\",
    ]
    completed = []
    for label, name in ARMS.items():
        path = ROOT / "experiments" / name / "metrics.json"
        metrics = json.loads(path.read_text()) if path.exists() else {}
        if metrics.get("complete"):
            u = metrics["qa_usage"]
            lines.append(
                f"{label} & {metrics['valid_answers']}/65 & {metrics['mean_score_all_65']:.4f} & "
                f"{u['prompt_tokens']/1e6:.3f} & {u['peak_prompt_tokens']/1e3:.2f} & {u['logical_calls']}" + r"\\")
            completed.append(name)
        else:
            lines.append(label + r" & \multicolumn{5}{c}{未完成，暂不报告总体分数}\\")
    lines += [
        r"\bottomrule\end{tabular}\end{center}",
        "原固定答案与M3行沿用前次共同评委成绩；A/B等行是各组独立评分或明确记录的恢复评分。"
        "后续455个（条件，问题）对将以固定顺序种子交错重评，评委不接收条件名称；"
        "所有旧评分保留，不择高、不把不同运行的最好答案拼成一组。",
        r"\subsection{已定位的问题与实现}",
        "原6道超时题模型调用累计183--230秒，而检索累计仅1.1--1.7秒；"
        "首轮约1250输入tokens，随后完整片段对象跨查询重复累积至9万--12万。"
        "现有预算模块在原正式基线中未开启，本轮A显式启用内容版本去重、连续原文选择"
        "和完整请求计数；有效输入预算22528，预留8192输出及2048安全空间。",
        "B新增可选45秒回答预留，并在最后一轮实际关闭Search；压缩时保留无新增证据提示。"
        "本批B的54题执行了最后一轮收尾，时间预留没有触发，不能将其分数变化归功于45秒机制。"
        "C降至12000输入估计预算；D在同样预算下按字段分组已选原文，不改写事实。"
        "另做B完整独立重复，检验分数稳定性，不选取较高的一轮。",
        r"\subsection{额度事件与已验证边界}",
        "MiniMax曾返回429/2056额度耗尽。立即停止后续请求后，只读接口显示新窗口，"
        "209-token小请求确认服务恢复；没有购买、升级或更换模型。"
        "A保留55条成功答案和48条成功评分，仅补额度失败部分；"
        "B原尝试没有有效答案，另立目录完整运行。先前失败请求的未知usage不视作零费用。",
        f"分组格式在A完整65题轨迹上离线回放，核验了{replay['unique_source_versions_verified']}个源内容版本、"
        f"{replay['source_occurrences_verified']}次源节点出现和{replay['admitted_passages_verified']}条原文；"
        f"在相同入选内容上，证据正文代理tokens减少{100*replay['evidence_proxy_token_reduction']:.2f}\\%。"
        "这不是完整API请求的实测降幅，也不是D的质量成绩。",
        "当前提交的M3/provider/core单元检查1041项通过。"
        "此前扩大检查中的8项失败在修改前版本同样复现，另有1项既有音频问题。"
        "当前版本排除全部9项已知问题后为1388通过、108跳过、9排除，"
        "不能称整个仓库全部通过。",
        r"\subsection{缓存与质量不能忽略}",
        "A固定答案轨迹的总输入下降76.3\\%，但未缓存输入只下降42.2\\%；"
        "缓存比例由原65.2\\%变为15.3\\%。重打包修改前部工具消息可能削弱稳定前缀，"
        "这一解释不是对供应商内部实现的控制实验，更不能据此直接换算账单。",
        r"\begin{figure}[htbp]\centering",
        r"\includegraphics[width=\linewidth]{figures/context_cache.pdf}",
        r"\caption{已完成条件的实际输入用量；深色为未缓存输入，浅色为缓存输入。}\end{figure}",
        "A恢复了原6题的回答覆盖，但剪刀计数仍错误；人物角色、弹力带与护具区分、"
        "容器/层数等仍有失误。相近答案也可能跨评分批次得到不同半分。"
        "M3-Agent为同模型适配而非官方权重复现，额外人脸裁剪和单片段ASR差异仍需保留。",
        r"完整协议与状态见\path{experiments/m3_context_study_20261101/}；"
        r"设计与案例见\path{docs/m3_context_optimization.md}及"
        r"\path{docs/m3_context_case_review.md}。",
    ]
    target.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "provisional", "completed_arms": completed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
