#!/usr/bin/env python3
"""Build paper figures for the m3_parallel4_paper8_20261113 experiment.

- paired8_final_resources : refresh oGMemory resources (new run); M3-Agent held.
- qa_budget_pareto       : overall QA-token-budget Pareto front (97 questions).
- qa_budget_pareto_per_video : per-video QA-token-budget curves (8 videos, one figure).

Reads only frozen artifacts; never launches inference or judging.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path("/home/yuanjian/Development/memory-eval")
NAME = "m3_parallel4_paper8_20261113"
EXP = ROOT / "experiments" / NAME
FIGS = ROOT / "latex" / "figures"
OLD = json.loads((ROOT / "experiments/m3_paper_paired8_20261103/comparison.json").read_text())

plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.unicode_minus": False})


def save(fig, name):
    FIGS.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def usage_sum(path):
    pin = pout = 0
    for ln in Path(path).read_text().splitlines():
        if not ln.strip():
            continue
        u = json.loads(ln).get("usage") or {}
        pin += u.get("prompt_tokens", 0) or 0
        pout += u.get("completion_tokens", 0) or 0
    return pin, pout


def og_new():
    split = json.loads((EXP / "og_stage_split.json").read_text())
    t = split["totals"]
    vids = json.loads((EXP / "qa_budget_per_video.json").read_text())["videos"]
    qa_in = qa_out = 0
    for v in vids:
        i, o = usage_sum(EXP / f"usage-qa-{v}.jsonl")
        qa_in += i; qa_out += o
    return {
        "memory_tokens": 2510031 + 785051,
        "qa_tokens": qa_in + qa_out,
        "shared": t["shared"], "mem": t["mem"], "qa_seconds": t["qa"],
        "total": t["shared"] + t["mem"] + t["qa"],
        "qa_prompt": qa_in, "qa_completion": qa_out,
    }


def plot_resources():
    og = og_new()
    m3 = OLD["methods"]["m3_agent"]
    mem = [og["memory_tokens"] / 1e6, m3["usage"]["memory"]["total_tokens"] / 1e6]
    qa = [og["qa_tokens"] / 1e6, m3["usage"]["qa"]["total_tokens"] / 1e6]
    times = [
        {"shared_perception": og["shared"], "memory_and_index": og["mem"],
         "qa": og["qa_seconds"], "allocated_total": og["total"]},
        m3["time_seconds"],
    ]
    CATS = ("shared_perception", "memory_and_index", "qa")
    colors = {"shared_perception": "#aab4be", "memory_and_index": "#2878b5", "qa": "#ed8b23"}

    fig = plt.figure(figsize=(10.5, 4.1), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.65])
    token_ax = fig.add_subplot(grid[:, 0])
    time_ax = fig.add_subplot(grid[0, 1])
    share_ax = fig.add_subplot(grid[1, 1])
    x = np.arange(2)
    token_ax.bar(x, mem, 0.53, color=colors["memory_and_index"])
    token_ax.bar(x, qa, 0.53, bottom=mem, color=colors["qa"])
    for i, (m, q) in enumerate(zip(mem, qa)):
        token_ax.text(i, m / 2, f"{m:.3f}", ha="center", va="center", color="white", fontsize=11)
        token_ax.text(i, m + q / 2, f"{q:.3f}", ha="center", va="center", fontsize=11)
        token_ax.text(i, m + q + 0.17, f"{m + q:.3f}" + ("*" if i else ""),
                      ha="center", fontsize=12, fontweight="bold")
    token_ax.set_xticks(x, ["oGMemory", "M3-Agent*"], fontsize=11)
    token_ax.set_ylim(0, 8.5)
    token_ax.set_ylabel("Recorded input + output tokens (M)")
    token_ax.set_title("(a) Memory + QA tokens", fontsize=12, loc="left")

    for ax, fraction in ((time_ax, False), (share_ax, True)):
        for row, method in enumerate(times):
            left = 0
            for key in CATS:
                value = 100 * method[key] / method["allocated_total"] if fraction else method[key] / 60
                ax.barh(row, value, left=left, height=0.48, color=colors[key], edgecolor="white")
                if fraction:
                    ax.text(left + value / 2, row, f"{value:.1f}", ha="center", va="center",
                            color="white" if key == "memory_and_index" else "#222222", fontsize=10)
                left += value
            if not fraction:
                ax.text(left + 8, row, f"{left:.1f}", va="center", fontsize=11)
        ax.set_yticks([0, 1], ["oGMemory", "M3-Agent"], fontsize=11)
        ax.invert_yaxis()
        ax.set_xlim(0, 100 if fraction else 800)
        ax.set_xlabel("Share of per-method time (%)" if fraction else "Per-method time (minutes)", fontsize=11)
        ax.set_title("(c) Stage shares" if fraction else "(b) Stage wall time", loc="left", fontsize=12)
    for ax in (token_ax, time_ax, share_ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y" if ax is token_ax else "x", alpha=0.15)
    fig.legend(handles=[
        Patch(color=colors["shared_perception"], label="Shared perception (time only)"),
        Patch(color=colors["memory_and_index"], label="Memory / graph / index"),
        Patch(color=colors["qa"], label="QA")],
        loc="outside lower center", ncols=3, frameon=False, fontsize=11)
    save(fig, "paired8_final_resources")
    return og



def og_by_video():
    import collections
    vids = json.loads((EXP / "qa_budget_per_video.json").read_text())["videos"]
    by = collections.defaultdict(lambda: [0, 0.0])
    for ln in (EXP / "judgments.jsonl").read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        by[r["video_id"]][0] += 1
        by[r["video_id"]][1] += r.get("score", 0)
    return vids, {v: by[v][1] / by[v][0] for v in vids}, sum(r["score"] for r in map(json.loads, filter(str.strip, (EXP/"judgments.jsonl").read_text().splitlines())))/97


def plot_scores():
    vids, ogv, ogmean = og_by_video()
    m3 = OLD["methods"]["m3_agent"]
    x = np.arange(len(vids) + 1)
    fig, ax = plt.subplots(figsize=(10.5, 3.5), layout="constrained")
    og_vals = [ogv[v] for v in vids] + [ogmean]
    m3_vals = [m3["by_video"][v]["mean_score"] for v in vids] + [m3["mean_score"]]
    ax.bar(x - 0.19, og_vals, 0.36, label="oGMemory (4-way parallel)", color="#2878b5")
    ax.bar(x + 0.19, m3_vals, 0.36, label="M3-Agent (matched model)", color="#ed8b23")
    ax.axvline(7.5, color="#999999", linestyle=":", linewidth=1)
    ax.set_xticks(x, [v.replace("_", "\n") for v in vids] + ["Weighted\nmean"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean automatic score (0 / 0.5 / 1)")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.legend(ncols=2, loc="upper left", frameon=False, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    save(fig, "paired8_final_scores")
    return ogv, ogmean


def plot_overall_pareto():
    res = json.loads((EXP / "qa_budget_pareto_results.json").read_text())
    rows = sorted(res["points"], key=lambda r: r["qa_input_tokens"])
    pareto = res["pareto_front"]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    xs = [r["qa_input_tokens"] / 1e6 for r in rows]
    ys = [r["accuracy"] * 100 for r in rows]
    ax.plot(xs, ys, "o-", color="#4C72B0", label="QA budget settings", zorder=2)
    if pareto:
        ax.plot([r["qa_input_tokens"] / 1e6 for r in pareto], [r["accuracy"] * 100 for r in pareto],
                "s--", color="#C44E52", ms=9, label="Pareto front", zorder=3)
    for r in rows:
        ax.annotate(f"{r['budget'] / 1000:.1f}k", (r["qa_input_tokens"] / 1e6, r["accuracy"] * 100),
                    xytext=(7, 6), textcoords="offset points", fontsize=9)
    ax.axhline(57.216, color="gray", ls=":", lw=1, label="paper baseline 57.2%")
    ax.set_xlabel("QA-stage input tokens (M, prompt)")
    ax.set_ylabel("Accuracy (%) on 97 questions")
    ax.set_title("QA token budget vs accuracy (overall Pareto front)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    save(fig, "qa_budget_pareto")
    return rows


def plot_per_video_pareto():
    """Accuracy vs QA token-budget SETTING: one line per video + a thicker average line."""
    d = json.loads((EXP / "qa_budget_per_video.json").read_text())
    budgets = [int(b) for b in d["budgets"]]
    vids = d["videos"]
    grid = {int(k): v for k, v in d["grid"].items()}
    sw = {int(p["budget"]): p for p in
          json.loads((EXP / "qa_budget_sweep_results.json").read_text())["points"]}

    pos = list(range(len(budgets)))          # evenly spaced budget settings
    xticklabels = [f"{b/1000:g}k" for b in budgets]

    fig, ax = plt.subplots(figsize=(10.2, 6.0), layout="constrained")
    cmap = plt.get_cmap("tab10")
    for i, v in enumerate(vids):
        ys = [grid[b][v] * 100 for b in budgets]
        ax.plot(pos, ys, "-o", color=cmap(i % 10), lw=1.9, ms=5.5, alpha=0.95,
                label=v, zorder=2)
    # thicker overall average line (97-question weighted accuracy)
    avg = [sw[b]["accuracy"] * 100 for b in budgets]
    ax.plot(pos, avg, "-s", color="black", lw=4.2, ms=9, label="Average (97-Q weighted)",
            zorder=5)

    ax.set_xticks(pos, xticklabels)
    ax.set_xlabel("QA token budget setting (max_input_tokens)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-video accuracy vs QA token budget (8 videos + average)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.28)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, ncols=1, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False)
    save(fig, "qa_budget_pareto_per_video")
    return vids, budgets


def main():
    og = plot_resources()
    ogv, ogmean = plot_scores()
    rows = plot_overall_pareto()
    vids, budgets = plot_per_video_pareto()
    print("resources og:", {k: round(v, 1) if isinstance(v, float) else v for k, v in og.items()})
    print("overall pareto rows:", [(r["budget"], round(r["accuracy"] * 100, 2)) for r in rows])
    print("per-video figure done for", len(vids), "videos x", len(budgets), "budgets")


if __name__ == "__main__":
    main()
