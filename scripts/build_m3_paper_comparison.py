#!/usr/bin/env python3
"""Read frozen results and build the paper's two-method tables/figures. No inference."""

import hashlib
import json
from pathlib import Path

from compare_m3_eight import usage_summary
from plot_m3_stage_times import CATEGORIES, collect

ROOT = Path(__file__).resolve().parent.parent
NAME = "m3_ten_fullinterval_20261102"
OUTPUT = ROOT / "experiments/m3_paper_paired8_20261103"
METHODS = {
    "ogmemory": ("ogmemory_strict_incremental", "og_qa", "pipeline"),
    "m3_agent": ("m3_reference_text", "m3_qa", "m3_memory"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def metrics(records):
    if not records or any(r["judge_error"] for r in records):
        raise ValueError("Empty or incomplete judging")
    if any(r["judge_score"] not in (0, 0.5, 1) for r in records):
        raise ValueError("Unexpected scoring scale")
    if any(
        (r.get("error") or not r["predicted_answer"]) and r["judge_score"] != 0
        for r in records
    ):
        raise ValueError("Invalid answers must retain zero score")
    if len({(r["video_id"], r["question_id"]) for r in records}) != len(records):
        raise ValueError("Duplicate questions")
    return {
        "questions": len(records),
        "valid_answers": sum(
            not r.get("error") and bool(r["predicted_answer"]) for r in records
        ),
        "mean_score": sum(r["judge_score"] for r in records) / len(records),
        "full_credit_fraction": sum(r["judge_score"] == 1 for r in records)
        / len(records),
    }


def build_data():
    exp = ROOT / "experiments" / NAME
    run = ROOT / "runs" / NAME
    manifest = json.loads((exp / "manifest.json").read_text())
    cohort = manifest["execution_order"][:8]
    assert (
        len(cohort) == 8 and sum(manifest["question_counts"][v] for v in cohort) == 97
    )
    grades_path = exp / "joint_questions.jsonl"
    grades = rows(grades_path)
    assert len(grades) == 256
    timing = collect(ROOT, NAME)
    timed = {v["video"]: v for v in timing["videos"] if v["video"] in cohort}
    assert set(timed) == set(cohort)
    hashes = {
        str(grades_path.relative_to(ROOT)): digest(grades_path),
        str((exp / "manifest.json").relative_to(ROOT)): digest(exp / "manifest.json"),
    }
    hashes.update(timing["source_sha256"])
    result = {
        "source_experiment": NAME,
        "source_commit": manifest["source"]["commit"],
        "cohort": cohort,
        "questions_per_method": 97,
        "score_source": "Final ten-video common judging, not the earlier eight-video judge.",
        "model": manifest["model"],
        "reasoning_effort": manifest["reasoning_effort"],
        "selection": "Previously reported first eight videos in the registered execution order; retain all 97 questions.",
        "methods": {},
        "all_ten": {},
        "source_sha256": hashes,
        "excluded_from_main": {
            "study_18": "M3 has 11 connection failures; only 3/14 valid answers.",
            "kitchen_20": "Embedding/pipeline failure; neither method completed predictions.",
        },
        "inference_calls": 0,
    }
    for method, (grade_method, qa_phase, memory_phase) in METHODS.items():
        all_rows = [r for r in grades if r["method"] == grade_method]
        selected = [r for r in all_rows if r["video_id"] in cohort]
        assert len(selected) == 97 and len(all_rows) == 128
        for video in cohort:
            path = run / qa_phase / video / "m3_qa_results.jsonl"
            predictions = {(r["video_id"], r["question_id"]): r for r in rows(path)}
            assert len(predictions) == manifest["question_counts"][video]
            for grade in [r for r in selected if r["video_id"] == video]:
                prediction = predictions[(video, grade["question_id"])]
                for key in ("question", "expected_answer", "predicted_answer", "error"):
                    assert prediction.get(key, "") == grade.get(key, "")
            hashes[str(path.relative_to(ROOT))] = digest(path)
        usage, combined = {}, []
        for label, phase in (("qa", qa_phase), ("memory", memory_phase)):
            calls = []
            for video in cohort:
                path = exp / f"usage-{phase}-{video}.jsonl"
                part = rows(path)
                assert all(r["requested_model"] == "MiniMax-M3.1-Flash" for r in part)
                calls.extend(part)
                hashes[str(path.relative_to(ROOT))] = digest(path)
            usage[label] = usage_summary(calls)
            combined.extend(calls)
        usage["memory_plus_qa"] = usage_summary(combined)
        assert usage["qa"]["unknown_usage_calls"] == 0
        times = {
            key: sum(timed[v]["methods"][method][key] for v in cohort)
            for key in (*CATEGORIES, "allocated_total")
        }
        result["methods"][method] = {
            **metrics(selected),
            "by_video": {
                v: metrics([r for r in selected if r["video_id"] == v]) for v in cohort
            },
            "usage": usage,
            "time_seconds": times,
            "stage_share": {k: times[k] / times["allocated_total"] for k in CATEGORIES},
        }
        result["all_ten"][method] = metrics(all_rows)
    result["ogmemory_face_clustering_seconds"] = sum(
        timed[v]["stage_seconds"]["face_clustering"] for v in cohort
    )
    a, b = result["methods"]["ogmemory"], result["methods"]["m3_agent"]
    result["differences"] = {
        "score_percentage_points": 100 * (a["mean_score"] - b["mean_score"]),
        "qa_prompt_ratio": a["usage"]["qa"]["prompt_tokens"]
        / b["usage"]["qa"]["prompt_tokens"],
        "time_ratio": a["time_seconds"]["allocated_total"]
        / b["time_seconds"]["allocated_total"],
    }
    result["limitations"] = [
        "Auto scores, not human accuracy; one run per method, small development/extension cohort.",
        "Two original oGMemory timeouts retained as zero; no replacement from recovery experiments.",
        "Recorded memory tokens omit one M3 request with unknown usage; not a complete bill.",
        "Shared perception executed once, allocated equally to each standalone-method time.",
        "Timings include initialization, retries and within-stage waits; exclude judge and between-phase gaps.",
        "Source predates new person grounding and PR300 parallel integration; do not attribute those benefits.",
    ]
    return result


def figures(data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42, "ps.fonttype": 42})
    folder = ROOT / "latex/figures"
    folder.mkdir(exist_ok=True)
    a, b = [data["methods"][m] for m in METHODS]
    outputs = []

    def save(fig, name):
        for ext in ("pdf", "png"):
            path = folder / f"{name}.{ext}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            outputs.append(path)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 3.5), layout="constrained")
    videos = data["cohort"]
    x = np.arange(len(videos) + 1)
    for offset, stats, label, color in (
        (-0.19, a, "oGMemory", "#2878b5"),
        (0.19, b, "M3-Agent (matched model)", "#ed8b23"),
    ):
        values = [stats["by_video"][v]["mean_score"] for v in videos] + [
            stats["mean_score"]
        ]
        ax.bar(x + offset, values, 0.36, label=label, color=color)
    ax.axvline(7.5, color="#999999", linestyle=":", linewidth=1)
    ax.set_xticks(x, [v.replace("_", "\n") for v in videos] + ["Weighted\nmean"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean automatic score (0 / 0.5 / 1)")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.legend(ncols=2, loc="upper left", frameon=False, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    save(fig, "paired8_final_scores")

    fig = plt.figure(figsize=(10.5, 4.1), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.65])
    token_ax = fig.add_subplot(grid[:, 0])
    time_ax = fig.add_subplot(grid[0, 1])
    share_ax = fig.add_subplot(grid[1, 1])
    colors = {
        "shared_perception": "#aab4be",
        "memory_and_index": "#2878b5",
        "qa": "#ed8b23",
    }
    x = np.arange(2)
    memory = [m["usage"]["memory"]["total_tokens"] / 1e6 for m in (a, b)]
    qa = [m["usage"]["qa"]["total_tokens"] / 1e6 for m in (a, b)]
    token_ax.bar(x, memory, 0.53, color=colors["memory_and_index"])
    token_ax.bar(x, qa, 0.53, bottom=memory, color=colors["qa"])
    for i, (mem, q) in enumerate(zip(memory, qa)):
        token_ax.text(
            i,
            mem / 2,
            f"{mem:.3f}",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
        )
        token_ax.text(i, mem + q / 2, f"{q:.3f}", ha="center", va="center", fontsize=11)
        token_ax.text(
            i,
            mem + q + 0.17,
            f"{mem + q:.3f}" + ("*" if i else ""),
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    token_ax.set_xticks(x, ["oGMemory", "M3-Agent*"], fontsize=11)
    token_ax.set_ylim(0, 8.5)
    token_ax.set_ylabel("Recorded input + output tokens (M)")
    token_ax.set_title("(a) Memory + QA tokens", fontsize=12, loc="left")
    for ax, fraction in ((time_ax, False), (share_ax, True)):
        for row, method in enumerate((a, b)):
            left = 0
            times = method["time_seconds"]
            for key in CATEGORIES:
                value = (
                    100 * times[key] / times["allocated_total"]
                    if fraction
                    else times[key] / 60
                )
                ax.barh(
                    row,
                    value,
                    left=left,
                    height=0.48,
                    color=colors[key],
                    edgecolor="white",
                )
                if fraction:
                    ax.text(
                        left + value / 2,
                        row,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        color="white" if key == "memory_and_index" else "#222222",
                        fontsize=10,
                    )
                left += value
            if not fraction:
                ax.text(left + 8, row, f"{left:.1f}", va="center", fontsize=11)
        ax.set_yticks([0, 1], ["oGMemory", "M3-Agent"], fontsize=11)
        ax.invert_yaxis()
        ax.set_xlim(0, 100 if fraction else 800)
        ax.set_xlabel(
            "Share of per-method time (%)" if fraction else "Per-method time (minutes)",
            fontsize=11,
        )
        ax.set_title(
            "(c) Stage shares" if fraction else "(b) Stage wall time",
            loc="left",
            fontsize=12,
        )
    for ax in (token_ax, time_ax, share_ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y" if ax is token_ax else "x", alpha=0.15)
    fig.legend(
        handles=[
            Patch(
                color=colors["shared_perception"], label="Shared perception (time only)"
            ),
            Patch(color=colors["memory_and_index"], label="Memory / graph / index"),
            Patch(color=colors["qa"], label="QA"),
        ],
        loc="outside lower center",
        ncols=3,
        frameon=False,
        fontsize=11,
    )
    save(fig, "paired8_final_resources")
    return outputs


def latex(data):
    a, b = [data["methods"][m] for m in METHODS]
    lines = [
        "% Generated by scripts/build_m3_paper_comparison.py; final common judge only.",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{逐视频自动评分，以百分数表示；分差按未舍入值计算。保留全部97题，失败计零。}",
        r"\label{tab:videos}\begin{tabular}{lrrrr}\toprule",
        r"视频 & 题数 & oGMemory & M3-Agent & 分差（百分点）\\\midrule",
    ]
    for video in data["cohort"]:
        av, bv = a["by_video"][video], b["by_video"][video]
        lines.append(
            r"\code{"
            + video
            + "} & "
            + str(av["questions"])
            + f" & {100 * av['mean_score']:.1f}\\% & {100 * bv['mean_score']:.1f}\\%"
            + f" & {100 * (av['mean_score'] - bv['mean_score']):+.1f}"
            + r"\\"
        )
    lines += [
        r"\midrule",
        f"题目加权合计 & 97 & {100 * a['mean_score']:.1f}\\% & {100 * b['mean_score']:.1f}\\%"
        + f" & {100 * (a['mean_score'] - b['mean_score']):+.1f}"
        + r"\\",
        r"\bottomrule\end{tabular}\end{table}",
        r"\begin{figure}[htbp]\centering",
        r"\includegraphics[width=\linewidth]{figures/paired8_final_scores.pdf}",
        (
            r"\caption{两方法逐视频表现及题目加权均值。纵轴为自动评分而非人工准确率；"
            r"不拼接早期临时评分或重跑答案。}\label{fig:scores}\end{figure}"
        ),
    ]
    lines += [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{同一八视频子集的质量与资源开销。M为百万token，时间单位为分钟。}",
        r"\label{tab:resources}\begin{tabular}{lrr}\toprule",
        r"指标 & oGMemory & M3-Agent\\\midrule",
    ]
    pairs = [
        ("有效回答", f"{a['valid_answers']}/97", f"{b['valid_answers']}/97"),
        (
            r"全分题比例（\%）",
            f"{100 * a['full_credit_fraction']:.2f}",
            f"{100 * b['full_credit_fraction']:.2f}",
        ),
    ]
    for label, key in [
        ("QA输入（M）", "prompt_tokens"),
        ("其中未缓存输入（M）", "uncached_prompt_tokens"),
        ("QA输出，含推理（M）", "completion_tokens"),
    ]:
        pairs.append(
            (
                label,
                f"{a['usage']['qa'][key] / 1e6:.3f}",
                f"{b['usage']['qa'][key] / 1e6:.3f}",
            )
        )
    pairs.append(
        (
            "已记录建忆+QA总token（M）",
            f"{a['usage']['memory_plus_qa']['total_tokens'] / 1e6:.3f}",
            f"{b['usage']['memory_plus_qa']['total_tokens'] / 1e6:.3f}" + r"$^{*}$",
        )
    )
    for label, key in [
        ("共享感知与启动", "shared_perception"),
        ("记忆、图谱及索引", "memory_and_index"),
        ("QA耗时", "qa"),
        ("各自计入感知后的总时间", "allocated_total"),
    ]:
        pairs.append(
            (
                label,
                f"{a['time_seconds'][key] / 60:.2f}",
                f"{b['time_seconds'][key] / 60:.2f}",
            )
        )
    lines += [" & ".join(p) + r"\\" for p in pairs]
    lines += [
        r"\bottomrule\end{tabular}",
        (
            r"\par\smallskip\footnotesize $^{*}$M3建忆有1次失败请求未返回usage，"
            r"该总量仅表示已记录用量，不把未知值当作零；两方法QA用量完整。"
        ),
        r"\end{table}",
        r"\begin{figure}[htbp]\centering",
        r"\includegraphics[width=\linewidth]{figures/paired8_final_resources.pdf}",
        (
            r"\caption{资源开销及阶段分布。左图为建忆与QA已记录的输入、输出总token，"
            r"不含embedding和评委；星号表示M3有未知建忆用量。右图将同一次共享感知"
            r"分别计入两条方法路径；占比由累计耗时计算，不能将两方法总时间相加"
            r"当作实验墙钟时间。}\label{fig:resources}\end{figure}"
        ),
    ]
    (ROOT / "latex/paired8_results.tex").write_text("\n".join(lines) + "\n")


def main():
    data = build_data()
    outputs = figures(data)
    latex(data)
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "comparison.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    )
    receipt = {
        "script_sha256": digest(Path(__file__)),
        "inference_calls": 0,
        "artifacts": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                *outputs,
                ROOT / "latex/paired8_results.tex",
                OUTPUT / "comparison.json",
            ]
        },
    }
    (OUTPUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                m: {
                    "score": d["mean_score"],
                    "valid": d["valid_answers"],
                    "total_minutes": d["time_seconds"]["allocated_total"] / 60,
                }
                for m, d in data["methods"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
