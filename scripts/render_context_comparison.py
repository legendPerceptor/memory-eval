#!/usr/bin/env python3
"""Render completed context ablations, preserving first and joint judge passes."""
from collections import Counter
import itertools
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ["study_05", "gym_03", "bedroom_04", "study_18", "kitchen_20"]
ARMS = {
    "budget_only": "m3_context_budget_recovered_20261101",
    "finalization": "m3_context_finalization_recovered_20261101",
    "compact12k": "m3_context_compact12k_guarded_20261101",
    "grouped12k": "m3_context_grouped12k_v2_20261101",
    "finalization_repeat": "m3_context_finalization_repeat_20261101",
}
LABELS = {
    "historical_ogmemory": "oGMemory 原固定答案",
    "m3_reference_text": "M3-Agent 参考文本协议适配",
    "budget_only": "A：仅上下文预算（额度恢复）",
    "finalization": "B：预算 + 回答收尾",
    "compact12k": "C：12k预算 + 回答收尾",
    "grouped12k": "D：12k分组原文 + 回答收尾",
    "finalization_repeat": "B独立重复：相同配置",
}


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] * (high - index) + values[high] * (index - low) if high != low else values[low]


def clustered_interval(deltas, counts):
    """Enumerate the 5^5 bootstrap samples of five whole video clusters."""
    if len(deltas) != len(counts) or not counts or any(n <= 0 for n in counts):
        raise ValueError("Expected matching positive video-cluster sizes")
    n = len(counts)
    values = [
        sum(deltas[i] for i in sample) / sum(counts[i] for i in sample)
        for sample in itertools.product(range(n), repeat=n)
    ]
    return {"lower": percentile(values, .025), "upper": percentile(values, .975),
            "samples": len(values), "clusters": n,
            "caution": "Only five development videos; illustrative cluster bootstrap, not strong generalization evidence."}


def replication_summary(grades):
    first = {r["question_id"]: r for r in grades if r["method"] == "finalization"}
    second = {r["question_id"]: r for r in grades if r["method"] == "finalization_repeat"}
    assert first and first.keys() == second.keys()
    scores = [statistics.mean(r["judge_score"] for r in group.values()) for group in (first, second)]
    identical = [qid for qid in first if first[qid]["predicted_answer"] == second[qid]["predicted_answer"]]
    return {
        "questions_per_run": len(first), "scores": scores,
        "mean_of_two_run_scores": statistics.mean(scores),
        "absolute_score_difference": abs(scores[1] - scores[0]),
        "question_score_disagreements": sum(first[q]["judge_score"] != second[q]["judge_score"] for q in first),
        "identical_predictions": len(identical),
        "identical_predictions_with_different_judge_scores": [
            q for q in identical if first[q]["judge_score"] != second[q]["judge_score"]],
        "best_of_selection": False,
        "note": "Same 65 questions twice, not 130 new questions. The two-run mean is descriptive, not a selected single-run score.",
    }


def main():
    output = ROOT / "experiments/m3_context_study_20261101"
    common = ROOT / "runs/m3_context_joint_judge_20261101"
    grade_path = common / "grades.jsonl"
    if (common / "grades_recovered.jsonl").exists():
        grade_path = common / "grades_recovered.jsonl"
    grades = rows(grade_path)
    assert len(grades) == 65 * len(LABELS) and not any(r["judge_error"] for r in grades), "Joint judging incomplete"
    from verify_context_artifacts import verify_joint
    verify_joint(ROOT, grades)
    methods = {}
    for method in LABELS:
        selected = [r for r in grades if r["method"] == method]
        assert len(selected) == 65 and len({r["question_id"] for r in selected}) == 65
        methods[method] = {
            "mean_score": statistics.mean(r["judge_score"] for r in selected),
            "valid_answers": sum(not r["error"] and bool(r["predicted_answer"]) for r in selected),
            "judge_errors": 0,
            "by_video": {v: statistics.mean(r["judge_score"] for r in selected if r["video_id"] == v)
                         for v in VIDEOS},
            "score_counts": dict(Counter(str(r["judge_score"]) for r in selected)),
        }
    previous = json.loads((ROOT / "experiments/m3_end_to_end_comparison_20261101/metrics.json").read_text())
    methods["historical_ogmemory"]["first_pass_score"] = previous["methods"]["ogmemory_recovered"]["mean_score"]
    methods["historical_ogmemory"]["qa_usage"] = previous["methods"]["ogmemory_recovered"]["qa_usage"]
    methods["m3_reference_text"]["first_pass_score"] = previous["methods"]["m3_agent_reference_text"]["mean_score"]
    methods["m3_reference_text"]["qa_usage"] = previous["methods"]["m3_agent_reference_text"]["qa_usage"]
    for method, name in ARMS.items():
        stats = json.loads((ROOT / "experiments" / name / "metrics.json").read_text())
        assert stats["complete"] and stats["historical_scope_audit_available"]
        assert not stats["usage_pairing_errors"]
        assert not stats["input_limit_exceedances"]
        assert not stats["temporal_scope_mismatches"]
        assert not stats["model_setting_deviations"]
        assert not stats["tokenizer_hash_mismatches"]
        methods[method].update(
            first_pass_score=stats["mean_score_all_65"],
            qa_usage=stats["qa_usage"], finish_reasons=stats["finish_reasons"],
            finalization_reasons=stats["finalization_reasons"],
            packing_seconds=stats["packing_seconds"],
            wall_seconds=stats["question_wall_seconds"],
            source_commit=stats["source_commit"],
            counter_exceedances=stats["counter_exceedances"],
        )
    keyed = {(r["method"], r["question_id"]): r for r in grades}
    baseline = [r for r in grades if r["method"] == "historical_ogmemory"]
    for method in ARMS:
        delta_sums = [
            sum(keyed[(method, r["question_id"])]["judge_score"] - r["judge_score"]
                for r in baseline if r["video_id"] == v)
            for v in VIDEOS
        ]
        counts = [sum(r["video_id"] == v for r in baseline) for v in VIDEOS]
        methods[method]["paired_delta_vs_original"] = sum(delta_sums) / 65
        methods[method]["video_cluster_interval"] = clustered_interval(delta_sums, counts)
        original_usage = methods["historical_ogmemory"]["qa_usage"]
        methods[method]["input_token_reduction"] = (
            1 - methods[method]["qa_usage"]["prompt_tokens"] / original_usage["prompt_tokens"])
        methods[method]["peak_input_reduction"] = (
            1 - methods[method]["qa_usage"]["peak_prompt_tokens"] / original_usage["peak_prompt_tokens"])
    first_manifest = json.loads((ROOT / "experiments" / ARMS["finalization"] / "manifest.json").read_text())
    repeat_manifest = json.loads((ROOT / "experiments" / ARMS["finalization_repeat"] / "manifest.json").read_text())
    assert repeat_manifest["independent_replication_of"] == ARMS["finalization"]
    assert not repeat_manifest.get("recovery")
    for field in ("source_archive_sha256", "qa_context", "cohort", "memory_runs",
                  "model", "reasoning_effort", "thinking", "qa_output_tokens",
                  "max_iterations", "timeout_seconds", "temporal_policy"):
        assert first_manifest[field] == repeat_manifest[field], field
    repeat = replication_summary(grades)
    result = {"methods": methods, "replication": repeat,
              "joint_grade_source": str(grade_path.relative_to(ROOT)),
              "interpretation": "Same stored memories for oGMemory arms; fixed prior M3 predictions. All 65 questions retained. Development/automatic-judge comparison."}
    (output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (output / "joint_questions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in grades))
    md = [
        "# 上下文与 QA 收尾消融：完成结果",
        "",
        "固定5视频65题。oGMemory各组使用同一份记忆，只重新执行QA；M3-Agent为既有参考协议适配答案。",
        "共同评委将455个(方案,问题)对以seed=42打乱交错评分，方案名称不交付给评委。",
        "全部初评分保留；下面共同评分不是按题挑选最高分。所有共同评分judge错误为0。",
        "",
        "| 方案 | 有效回答 | 独立/恢复评分 | 共同评分 | QA输入tokens | 峰值输入 | QA调用 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    tex = [
        "% CONTEXT_REPORT_STATUS: final",
        r"\section{上下文预算与 QA 收尾消融}",
        r"\label{sec:context-optimization}",
        "本节固定已生成的多模态记忆，仅重新运行 QA。"
        "旧 oGMemory 和 M3-Agent 为固定的既有答案；A/B/C/D及B的完整重复均保留，未择高。"
        "将七组共455条答案交错打乱后用相同 MiniMax 评委重新评分，评委不接收方案名称。"
        "初次评分仍保留；共同重评批次与前节不同，旧答案本身没有修改。",
        r"\begin{center}\small",
        r"\begin{tabular}{lrrrrrr}\toprule",
        r"方案 & 有效回答 & 独立评分 & 共同评分 & 输入/M & 峰值/k & 调用\\\midrule",
    ]
    short = {"historical_ogmemory": "原固定答案", "m3_reference_text": "M3参考协议",
             "budget_only": "A：仅预算", "finalization": "B：预算+收尾",
             "compact12k": "C：12k+收尾", "grouped12k": "D：12k分组+收尾",
             "finalization_repeat": "B独立重复"}
    for method, stats in methods.items():
        u = stats["qa_usage"]
        md.append(f"| {LABELS[method]} | {stats['valid_answers']}/65 | {stats['first_pass_score']:.4f} | "
                  f"{stats['mean_score']:.4f} | {u['prompt_tokens']:,} | {u['peak_prompt_tokens']:,} | {u['logical_calls']} |")
        tex.append(f"{short[method]} & {stats['valid_answers']}/65 & {stats['first_pass_score']:.4f} & "
                   f"{stats['mean_score']:.4f} & {u['prompt_tokens']/1e6:.3f} & "
                   f"{u['peak_prompt_tokens']/1e3:.2f} & {u['logical_calls']}" + r"\\")
    tex += [r"\bottomrule\end{tabular}\end{center}",
            "A曾遇到套餐额度中断：恢复时保留55条成功答案和48条成功评分，"
            "只补做额度导致的失败与未评分部分。B的前次尝试没有有效答案，另立目录完整运行。"
            "原失败记录和未知usage的请求开销均保留，未按题挑选高分。",
            "评分为0、0.5、1自动评分的全65题均值，不是人工准确率。输入包含缓存tokens，"
            "不等于费用；跨时段服务负载不同，不能据此声称严格墙钟加速比。",
            r"\subsection{逐视频共同评分}",
            r"\begin{center}\small\begin{tabular}{lrrrrrrr}\toprule",
            r"视频 & 原答案 & M3参考 & A & B & C & D & B重复\\\midrule"]
    md += ["", "## 逐视频共同评分", "",
           "| 视频 | 原答案 | M3参考 | A | B | C | D | B重复 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for video in VIDEOS:
        values = [f"{methods[m]['by_video'][video]:.4f}" for m in LABELS]
        md.append("| " + " | ".join([video, *values]) + " |")
        tex.append(video.replace("_", r"\_") + " & " + " & ".join(values) + r"\\")
    tex += [r"\bottomrule\end{tabular}\end{center}",
            r"\subsection{诊断、机制与限制}",
            "原6道超时题的模型调用累计183--230秒，检索仅1.1--1.7秒。"
            "首轮约1250输入tokens，随后重复的完整片段对象使输入累积至9万--12万。"
            "A复用原仓已有但正式基线未开启的上下文预算模块：内容版本去重、"
            "连续原文选择、有效引用和完整请求计数。22528输入预算预留8192输出及2048安全空间。"
            "B增加45秒回答预留，并在最后一轮关闭Search；C将输入估计上限降为12000。"
            "D在相同12000预算下将已选原文按字段分组，去除重复包装和模型无需读取的内容哈希；"
            "原始值、来源范围、人物与时间元数据不改写，完整哈希保留在审计中。"
            "计数器使用本地Qwen tokenizer代理，不等同于MiniMax的精确模板。",
            ""]
    md += ["", "## 相对原固定答案的变化", ""]
    for method in ARMS:
        s = methods[method]
        ci = s["video_cluster_interval"]
        md.append(f"- {LABELS[method]}：共同评分差 {s['paired_delta_vs_original']:+.4f}；"
                  f"输入减少 {s['input_token_reduction']:.1%}，峰值减少 {s['peak_input_reduction']:.1%}；"
                  f"5视频整簇bootstrap区间 [{ci['lower']:+.4f}, {ci['upper']:+.4f}]。")
        tex.append(f"{short[method]}相对原答案评分差为{s['paired_delta_vs_original']:+.4f}，"
                   f"累计输入下降{s['input_token_reduction']*100:.1f}\\%，"
                   f"峰值下降{s['peak_input_reduction']*100:.1f}\\%。"
                   f"按5个完整视频簇枚举重采样的95\\%区间为"
                   f"[{ci['lower']:+.4f}, {ci['upper']:+.4f}]。" + "\n")
    md += [
        "",
        "只有5个开发视频，整簇bootstrap仅帮助展示不确定性，不是泛化能力的强证据。",
        "回答覆盖不等于正确率；证据不足也可能形成有效回答而得到0分。",
        "收尾是软预算，不是对正在执行的HTTP请求作硬取消。",
        "M3-Agent仍是同模型适配，不是官方checkpoint复现，存在额外人脸裁剪及一个ASR片段差异。",
        "完整方案和代码位置见 docs/m3_context_optimization.md；失败和验证边界见 VALIDATION.md。",
        "",
        "## 收尾触发情况", "",
    ]
    for method in ARMS:
        md.append(f"- {LABELS[method]}：`{json.dumps(methods[method]['finalization_reasons'], ensure_ascii=False)}`")
    md += ["", "## B的完整独立重复", "",
           f"两次共同评分分别为{repeat['scores'][0]:.4f}和{repeat['scores'][1]:.4f}，"
           f"差值绝对值{repeat['absolute_score_difference']:.4f}；"
           f"{repeat['question_score_disagreements']}题的自动评分不同。",
           f"两轮均分的描述性平均为{repeat['mean_of_two_run_scores']:.4f}，"
           "不选取较高一轮、不拼接更好答案，也不把同一65题重复称为130个新问题。",
           f"有{repeat['identical_predictions']}题输出文本完全相同，其中"
           f"{len(repeat['identical_predictions_with_different_judge_scores'])}题仍得到不同评分。",
           "只有两次运行，不能据此精确估计随机波动或证明泛化。"]
    tex += [
        r"\subsection{重复运行与评分波动}",
        f"B两次共同评分为{repeat['scores'][0]:.4f}与{repeat['scores'][1]:.4f}，"
        f"绝对差{repeat['absolute_score_difference']:.4f}；"
        f"{repeat['question_score_disagreements']}题的自动评分不同。"
        f"两轮均分的描述性平均为{repeat['mean_of_two_run_scores']:.4f}，"
        "不择高、不拼接答案。同一65题的重复不等于130个独立新问题，"
        "仅两轮不能精确估计随机波动或支持泛化结论。",
        "上述区间仅用于呈现小样本不确定性：五个视频均为开发样本，不是独立留出集。"
        "未触发时间预留的运行不能证明45秒机制带来收益；"
        "提前收尾也不能保证单个极慢HTTP不会越过软截止。"
        "M3-Agent仍为同模型适配而非官方权重复现。"
        "原文被预算省略不等于事实不存在，计数题尤其不能把入选片段数当作真实次数。",
        "源帧定性检查还发现：study\\_18第14片段的输入止于435秒远景，"
        "445--446秒出现的货架近景与取纸动作没有进入16帧前缀采样。"
        "这提示采帧/记忆细节的上游限制，但这些尾段帧在434秒问题时间之后，"
        "只在本次current\\_clip宽限内，不能用于严格因果评测。"
        "辅助检查没有改变任何本轮模型输入或答案。",
        r"详细结果、逐题变化和验证记录位于\path{experiments/m3_context_study_20261101/}；"
        r"设计说明见\path{docs/m3_context_optimization.md}。",
    ]
    cache_path = output / "cache_analysis.json"
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())["methods"]
        md += ["", "## 缓存与未缓存输入", "",
               "| 方案 | 总输入 | 缓存输入 | 未缓存输入 | 缓存比例 |",
               "|---|---:|---:|---:|---:|"]
        for method in LABELS:
            c = cache[method]
            md.append(f"| {LABELS[method]} | {c['prompt_tokens']:,} | {c['cached_prompt_tokens']:,} | "
                      f"{c['uncached_prompt_tokens']:,} | {c['cache_fraction']:.1%} |")
        md += [
            "",
            "重打包会改变较早的工具消息，稳定前缀变短可能解释缓存比例下降；"
            "这与实际usage一致，但不是对供应商内部缓存实现的控制实验。",
            "因此总输入下降比例不等于费用下降比例，也不直接等于Token Plan额度消耗下降比例。",
        ]
        tex += [
            r"\subsection{缓存与未缓存输入}",
            r"\begin{figure}[htbp]\centering",
            r"\includegraphics[width=\linewidth]{figures/context_cache.pdf}",
            r"\caption{固定答案轨迹的实际输入用量。深色为未缓存输入，浅色为缓存输入；"
            r"未知usage的额度失败请求另行保留，不计作零费用。}\end{figure}",
            "重打包会改变较早的工具消息，可能削弱稳定前缀。"
            "实际usage显示缓存比例改变，因此总输入降幅不能直接当成费用或套餐额度降幅。"
            "缓存内部实现与跨时段服务负载没有被本实验严格控制。",
        ]
    md += [
        "", "## 代码与使用边界", "",
        "oGMemory当前配置工具可用 `python -m m3.configure_qa_context` 生成0600新副本，"
        "不更换模型或连接，不覆盖原配置。它应用于ReAct QA/pipeline，不是旧的"
        "`python -m m3.qa`单次入口；跨策略必须使用新的QA输出目录。",
        "当前发布代码的M3/provider/core单元检查1041通过；扩大检查排除9项已复现的"
        "既有问题后1388通过、108跳过、9排除，不是整个仓库全部通过。",
        "每个实验仍使用自身manifest记录的冻结版本，不用当前HEAD改写历史结果。",
    ]
    tex += [
        r"\subsection{验证与配置使用}",
        "当前发布代码的M3/provider/core单元检查1041项通过；"
        "扩大检查排除9项已复现的既有问题后为1388通过、108跳过、9排除。"
        "不能将排除等同于修复，也不能称整个仓库全部通过。"
        "各实验使用各自冻结源码，源码文件与Git归档逐项核对；"
        "共同评分输入和原预测文件在发布前重新校验，而不只信任状态标记。",
        r"可用\code{python -m m3.configure_qa_context}生成新的0600配置副本，"
        "保留模型与连接设置。它针对ReAct QA/pipeline，不是旧单次QA入口；"
        "切换策略必须使用新的输出目录，不应跨配置resume旧答案。",
    ]
    (output / "REPORT.md").write_text("\n".join(md) + "\n")
    (ROOT / "latex/context_optimization_results.tex").write_text("\n".join(tex) + "\n")
    print(json.dumps({m: s["mean_score"] for m, s in methods.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
