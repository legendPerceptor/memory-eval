#!/usr/bin/env python3
"""Publish complete, common-judge comparisons and QA cost/behavior diagnostics."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics

from judge_memory_design import load_inputs, VIDEOS, REFERENCES, NAME
from render_context_comparison import clustered_interval
from summarize_context_experiment import summarize

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments/m3_memory_design_20261102"
LABELS = {
    "balanced_prior": "B：前次 balanced",
    "balanced_prior_repeat": "B：前次独立重复",
    "m3_reference": "M3-Agent 同模型参考适配",
    "facts": "F：原文事实窗口",
    "facts_identity": "FI：事实 + 显式人物条件",
    "facts_identity_incremental": "FIP：事实 + 人物 + 增量",
    "source_index": "S：来源事实索引 + 增量修正",
    "source_index_repeat": "S：独立重复",
    "balanced_contemporary": "B：本轮同期独立重复",
    "source_index_scene": "H：S + 场景锚点 + 原生收尾",
    "source_index_scene_repeat": "H：独立重复",
    "incremental_only": "I：balanced 表示 + 增量交付",
    "m3_reference_repeat": "M3-Agent：本轮同记忆重复",
    "strict_incremental": "E：严格增量3500 + 原生收尾",
    "strict_incremental_repeat": "E：独立重复",
}


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def verify(tasks, grades, hashes, manifest, root=ROOT):
    if hashes != manifest["source_prediction_hashes"]:
        raise ValueError("Prediction files changed since judge registration")
    fields = ("method", "video_id", "question_id", "question", "expected_answer",
              "predicted_answer", "error", "finish_reason")
    keyed = {(r["method"], r["video_id"], r["question_id"]): r for r in tasks}
    scored = {(r["method"], r["video_id"], r["question_id"]): r for r in grades}
    if len(scored) != len(grades) or set(scored) != set(keyed):
        raise ValueError("Missing or duplicate judge rows")
    for key, grade in scored.items():
        if any(grade.get(field) != keyed[key].get(field) for field in fields):
            raise ValueError("Judge task fields changed")
        if grade["judge_error"] or grade["judge_score"] not in (0, .5, 1):
            raise ValueError("Unresolved judge error or invalid score")
    return scored


def behavior(name):
    qa = [r for v in VIDEOS for r in rows(ROOT / "runs" / name / v / "m3_qa_results.jsonl")]
    contexts = [c for r in qa for c in r["search_stats"].get("context_calls", [])]
    branches = [b for r in qa for s in r["search_history"] for b in s.get("retrieval_branches", [])]
    identity = [b for b in branches if b["kind"] == "explicit_character_soft_rank"]
    facts = [b for b in branches if b["kind"] == "source_fact_vector"]
    comparisons, preserved = 0, 0
    for r in qa:
        calls = r["search_stats"].get("context_calls", [])
        for previous, current in zip(calls, calls[1:]):
            p, c = previous.get("model_message_sha256"), current.get("model_message_sha256")
            if p and c:
                comparisons += 1
                preserved += int(c[:len(p)] == p)
    return {
        "epoch_rebuild_calls": sum(bool(c.get("epoch_rebuilt")) for c in contexts),
        "calls_reusing_tool_messages": sum(c.get("reused_tool_messages", 0) > 0 for c in contexts),
        "visible_fact_no_gain_finalizations": sum(
            r["search_stats"].get("finalization_reason") == "consecutive_no_new_visible_facts" for r in qa),
        "required_answer_calls": sum(c.get("tool_choice") == "required" for c in contexts),
        "identity_rank_searches": len(identity),
        "identity_rank_searches_with_explicit_matches": sum(bool(b.get("matches")) for b in identity),
        "fact_index_searches": len(facts),
        "fact_index_proposed_nodes": sum(b["proposed_nodes"] for b in facts),
        "fact_index_verified_nodes": sum(b["verified_nodes"] for b in facts),
        "fact_index_stale_nodes": sum(b["stale_nodes"] for b in facts),
        "consecutive_message_prefix_comparisons": comparisons,
        "consecutive_message_prefix_preserved": preserved,
        "note": "Message-prefix hashes exclude tool schema; cached-token usage is measured separately.",
    }


def paired_contrast(grades, left, right):
    """Average repeated trajectories per question; never select the best answer."""
    methods = set(left + right)
    data = {m: {} for m in methods}
    for row in grades:
        if row["method"] in data:
            key = row["video_id"], row["question_id"]
            if key in data[row["method"]] or row.get("judge_error"):
                raise ValueError("Duplicate/error grade in contrast")
            data[row["method"]][key] = row["judge_score"]
    keys = set(data[left[0]])
    if not keys or any(set(rows) != keys for rows in data.values()):
        raise ValueError("Contrast requires identical complete question sets")
    deltas = {
        key: statistics.mean(data[m][key] for m in left) - statistics.mean(data[m][key] for m in right)
        for key in keys}
    videos = sorted({key[0] for key in keys})
    sums = [sum(delta for key, delta in deltas.items() if key[0] == video) for video in videos]
    counts = [sum(key[0] == video for key in keys) for video in videos]
    return {
        "left_methods": left, "right_methods": right, "questions": len(keys),
        "mean_paired_delta": sum(sums) / sum(counts),
        "video_cluster_interval": clustered_interval(sums, counts),
        "interpretation": "Exploratory paired video-cluster resampling; repeats averaged, not independent new questions.",
    }


def main():
    methods = json.loads((OUT / "methods.json").read_text())
    tasks, hashes = load_inputs()
    judge = ROOT / "experiments" / NAME
    manifest = json.loads((judge / "manifest.json").read_text())
    grades = rows(ROOT / "runs" / NAME / "grades.jsonl")
    keyed = verify(tasks, grades, hashes, manifest)
    previous = json.loads((ROOT / "experiments/m3_context_study_20261101/comparison.json").read_text())
    result = {}
    for method, name in methods.items():
        scores = [r for r in grades if r["method"] == method]
        if method == "m3_reference":
            usage = previous["methods"]["m3_reference_text"]["qa_usage"]
            metrics = {}
        elif method == "m3_reference_repeat":
            metrics = json.loads((ROOT / "experiments" / name / "metrics.json").read_text())
            usage = metrics["qa_usage"]
            if metrics["model_setting_deviations"] or not metrics["frozen_inputs_unchanged"]:
                raise ValueError("Reference repetition validation failed")
        else:
            if method not in REFERENCES:
                summarize(name)
            metrics = json.loads((ROOT / "experiments" / name / "metrics.json").read_text())
            usage = metrics["qa_usage"]
            for key in ("temporal_scope_mismatches", "usage_pairing_errors",
                        "input_limit_exceedances", "tokenizer_hash_mismatches", "model_setting_deviations",
                        "increment_budget_exceedances"):
                if metrics.get(key):
                    raise ValueError(f"{name}: audit failure {key}")
        prompt, cached = usage["prompt_tokens"], usage.get("cached_prompt_tokens")
        # Reference trajectory cache totals are recorded in the preceding audit.
        if method == "m3_reference" and cached is None:
            cache_audit = json.loads((ROOT / "experiments/m3_context_study_20261101/cache_analysis.json").read_text())
            reference_cache = cache_audit["methods"]["m3_reference_text"]
            if reference_cache["prompt_tokens"] != prompt:
                raise ValueError("Reference cache audit and trajectory usage disagree")
            cached = reference_cache["cached_prompt_tokens"]
        result[method] = {
            "run": name, "label": LABELS[method], "questions": len(scores),
            "valid_answers": sum(bool(r["predicted_answer"]) and not r["error"] for r in scores),
            "mean_score": sum(r["judge_score"] for r in scores) / 65,
            "score_counts": dict(Counter(str(r["judge_score"]) for r in scores)),
            "by_video": {v: statistics.mean(r["judge_score"] for r in scores if r["video_id"] == v)
                         for v in VIDEOS},
            "qa_usage": {**usage, "cached_prompt_tokens": cached,
                         "uncached_prompt_tokens": prompt - cached if cached is not None else None,
                         "cache_fraction": cached / prompt if prompt and cached is not None else None},
            "initial_grade": (metrics.get("mean_score_all_65")
                              if metrics.get("judge_records") == 65 else None),
            "finish_reasons": metrics.get("finish_reasons"),
            "wall_seconds": metrics.get("question_wall_seconds"),
            "behavior": behavior(name) if method not in REFERENCES and method != "m3_reference_repeat" else None,
        }
    for method in methods:
        deltas, counts = [], []
        for video in VIDEOS:
            selected = [r for r in grades if r["method"] == method and r["video_id"] == video]
            deltas.append(sum(r["judge_score"] - keyed[(
                "balanced_contemporary", video, r["question_id"])]["judge_score"] for r in selected))
            counts.append(len(selected))
        result[method]["paired_delta_vs_contemporary_balanced"] = sum(deltas) / 65
        result[method]["video_cluster_interval_vs_contemporary_balanced"] = clustered_interval(deltas, counts)
    replication = {
        "source_index_scores": [result[m]["mean_score"] for m in ("source_index", "source_index_repeat")],
        "source_index_mean": statistics.mean(result[m]["mean_score"] for m in ("source_index", "source_index_repeat")),
        "source_index_mean_prompt_tokens": statistics.mean(
            result[m]["qa_usage"]["prompt_tokens"] for m in ("source_index", "source_index_repeat")),
        "prior_balanced_mean": statistics.mean(
            result[m]["mean_score"] for m in ("balanced_prior", "balanced_prior_repeat")),
        "note": "Descriptive repeated trajectories on the same 65 questions; not independent dataset samples.",
    }
    if "source_index_scene" in result:
        optional = [m for m in ("source_index_scene", "source_index_scene_repeat") if m in result]
        replication["scene_scores"] = [result[m]["mean_score"] for m in optional]
        replication["scene_mean"] = statistics.mean(replication["scene_scores"])
        replication["scene_runs"] = len(optional)
    if "m3_reference_repeat" in result:
        replication["m3_reference_scores"] = [
            result[m]["mean_score"] for m in ("m3_reference", "m3_reference_repeat")]
        replication["m3_reference_mean"] = statistics.mean(replication["m3_reference_scores"])
    if "strict_incremental" in result:
        names = [m for m in ("strict_incremental", "strict_incremental_repeat") if m in result]
        replication["strict_incremental_scores"] = [result[m]["mean_score"] for m in names]
        replication["strict_incremental_mean"] = statistics.mean(replication["strict_incremental_scores"])
        replication["strict_incremental_runs"] = len(names)
    contrasts = {
        "FI_minus_F": paired_contrast(grades, ["facts_identity"], ["facts"]),
        "FIP_minus_FI": paired_contrast(grades, ["facts_identity_incremental"], ["facts_identity"]),
        "S_mean_minus_FIP": paired_contrast(
            grades, ["source_index", "source_index_repeat"], ["facts_identity_incremental"]),
        "S_mean_minus_B_now": paired_contrast(
            grades, ["source_index", "source_index_repeat"], ["balanced_contemporary"]),
        "S_mean_minus_M3_mean": paired_contrast(
            grades, ["source_index", "source_index_repeat"],
            [m for m in ("m3_reference", "m3_reference_repeat") if m in result]),
        "B_now_minus_B_prior_mean": paired_contrast(
            grades, ["balanced_contemporary"], ["balanced_prior", "balanced_prior_repeat"]),
    }
    if "incremental_only" in result:
        contrasts["I_minus_B_now"] = paired_contrast(
            grades, ["incremental_only"], ["balanced_contemporary"])
    if "source_index_scene" in result:
        contrasts["H_mean_minus_S_mean"] = paired_contrast(
            grades, [m for m in ("source_index_scene", "source_index_scene_repeat") if m in result],
            ["source_index", "source_index_repeat"])
    if "m3_reference_repeat" in result:
        contrasts["M3_repeat_minus_prior"] = paired_contrast(
            grades, ["m3_reference_repeat"], ["m3_reference"])
    if "strict_incremental" in result:
        names = [m for m in ("strict_incremental", "strict_incremental_repeat") if m in result]
        contrasts["E_mean_minus_S_mean"] = paired_contrast(
            grades, names, ["source_index", "source_index_repeat"])
        contrasts["E_mean_minus_B_now"] = paired_contrast(grades, names, ["balanced_contemporary"])
    output = {"methods": result, "replication": replication, "joint_judge_tasks": len(grades),
              "exploratory_paired_contrasts": contrasts,
              "judge_seed": manifest["seed"], "prediction_hashes": hashes,
              "limitations": [
                  "Fixed-memory QA development study, not a new audiovisual end-to-end run.",
                  "Automatic scores 0/0.5/1, not human accuracy.",
                  "Same five development videos; no held-out generalization claim.",
                  "M3 reference is a same-model adaptation, not official checkpoint reproduction.",
                  "Token counts are QA only; input ratios are not monetary or latency ratios.",
                  "S also changes visible-fact no-gain control; not a single-factor index ablation.",
              ]}
    (OUT / "comparison.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    with (OUT / "joint_questions.jsonl").open("w") as stream:
        stream.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in grades)
    with (OUT / "joint_questions.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(grades[0]))
        writer.writeheader()
        writer.writerows(grades)
    report = [
        "# 来源事实、人物条件与增量交付：最终共同评分",
        "", "固定5视频65题；下列为同批交错自动0/0.5/1评分，非人工准确率。",
        "已执行的初评分保留在各组原目录，不择高拼接；仅共同评分的组不把缺失初评记为0。", "",
        "| 方案 | 有效答案 | 共同评分 | QA输入token | 未缓存输入 | 输入峰值 | 调用 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for value in result.values():
        u = value["qa_usage"]
        report.append(f'| {value["label"]} | {value["valid_answers"]}/65 | {value["mean_score"]:.4f} '
                      f'| {u["prompt_tokens"]:,} | {u["uncached_prompt_tokens"]:,} '
                      f'| {u["peak_prompt_tokens"]:,} | {u["logical_calls"]} |')
    report += ["", "## 分视频", "", "| 方案 | " + " | ".join(VIDEOS) + " |",
               "|---|" + "---:|" * len(VIDEOS)]
    for value in result.values():
        report.append("| " + value["label"] + " | " + " | ".join(
            f'{value["by_video"][v]:.4f}' for v in VIDEOS) + " |")
    report += ["", "## 重复与解释", "",
               f'S两次得分{replication["source_index_scores"]}，描述性均值'
               f'{replication["source_index_mean"]:.4f}。同期B为'
               f'{result["balanced_contemporary"]["mean_score"]:.4f}；前次B两次答案在本批评委下'
               f'均值{replication["prior_balanced_mean"]:.4f}。',
               "详细行为、缓存、逐题结果、成对差异和五视频cluster bootstrap区间见comparison.json。",
               "逐方法区间对照同期B；组合对照在下表明确标出双方。重复先按题平均，不择高。",
               "样本小且有多重开发比较，区间不作为确认性显著性结论。",
               "", "| 探索性对照 | 平均分差 | 视频聚类区间 |", "|---|---:|---:|"]
    for name, comparison in contrasts.items():
        interval = comparison["video_cluster_interval"]
        report.append(f'| {name} | {comparison["mean_paired_delta"]:+.4f} '
                      f'| [{interval["lower"]:+.4f}, {interval["upper"]:+.4f}] |')
    report += [
               "", "## 限制", *["- " + s for s in output["limitations"]]]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n")
    latex = [
        "% MEMORY_DESIGN_REPORT_STATUS: final",
        r"\section{来源事实、人物条件与增量交付（11月2日）}",
        r"\label{sec:memory-design}",
        "本轮固定原有340个片段记忆与65题，不重新生成感知或记忆。新增可追溯原文事实窗口、",
        "显式人物条件软排序和前缀稳定的增量交付。S额外使用GPU BGE-M3构建的7965个",
        "事实窗口旁路索引，按账号/视频/时间筛选并回源核对内容哈希；同时改为按实际",
        "新增可见事实判断停搜。人物条件不等价于已核实的人脸/声音身份解析。",
        r"\begin{center}\small\begin{tabular}{lrrrr}\toprule",
        r"方案 & 有效回答 & 共同评分 & QA输入token & 调用\\\midrule",
    ]
    for value in result.values():
        u = value["qa_usage"]
        latex.append(f'{value["label"]} & {value["valid_answers"]}/65 & {value["mean_score"]:.4f} '
                     f'& {u["prompt_tokens"]:,} & {u["logical_calls"]}' + r"\\")
    latex += [r"\bottomrule\end{tabular}\end{center}",
              f'同批评委共{len(grades)}条固定答案，seed={manifest["seed"]}随机交错，不向评委发送方案名。',
              f'S独立运行两次的描述性均值为{replication["source_index_mean"]:.4f}，'
              f'同期B为{result["balanced_contemporary"]["mean_score"]:.4f}。',
              "这些是小规模开发集自动评分，不是人工准确率或官方M3-Agent复现。",
              r"全部消融、失败、逐题评分与用量见\code{experiments/m3_memory_design_20261102/REPORT.md}。",
              "索引构建约63秒、37MB，QA用量不包含历史记忆生成。输入含缓存，",
              "不能将输入token比例当成费用或端到端加速比。",
              r"\IfFileExists{figures/memory_design_tradeoff.pdf}{",
              r"\begin{figure}[htbp]\centering",
              r"\includegraphics[width=\linewidth]{figures/memory_design_tradeoff.pdf}",
              r"\caption{同批自动评分与QA输入。每点为完整65题轨迹，重复运行分别显示；不是费用或端到端开销。}",
              r"\end{figure}}{}",
              r"实现及安全边界见\code{docs/m3_memory_design.md}。"]
    (ROOT / "latex/memory_design_results.tex").write_text("\n".join(latex) + "\n")
    print(json.dumps({m: r["mean_score"] for m, r in result.items()}, indent=2))


if __name__ == "__main__":
    main()
