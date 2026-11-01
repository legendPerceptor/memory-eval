#!/usr/bin/env python3
"""Vector plots of common-judge scores and measured QA input, not monetary cost."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments/m3_memory_design_20261102"
LABELS = {
    "balanced_prior": ("B old 1", "#777777", "o"),
    "balanced_prior_repeat": ("B old 2", "#777777", "o"),
    "balanced_contemporary": ("B now", "#333333", "s"),
    "m3_reference": ("M3 adapted", "#1976a3", "*"),
    "facts": ("F", "#be7e18", "o"),
    "facts_identity": ("FI", "#bd4a26", "o"),
    "facts_identity_incremental": ("FIP", "#34884a", "o"),
    "source_index": ("S1", "#7a4b9b", "D"),
    "source_index_repeat": ("S2", "#7a4b9b", "D"),
    "incremental_only": ("I", "#007b76", "s"),
    "source_index_scene": ("H1", "#c52c75", "^"),
    "source_index_scene_repeat": ("H2", "#c52c75", "^"),
    "m3_reference_repeat": ("M3 now", "#1976a3", "*"),
    "strict_incremental": ("E1", "#c52c75", "^"),
    "strict_incremental_repeat": ("E2", "#c52c75", "^"),
}


def validate(data):
    methods = data["methods"]
    if data["joint_judge_tasks"] != 65 * len(methods):
        raise ValueError("Plot requires complete common judging")
    for name, row in methods.items():
        if name not in LABELS or row["questions"] != 65 or not 0 <= row["mean_score"] <= 1:
            raise ValueError("Invalid plotting method or score")
        usage = row["qa_usage"]
        if (usage["prompt_tokens"] <= 0 or usage.get("uncached_prompt_tokens") is None
                or not 0 <= usage["uncached_prompt_tokens"] <= usage["prompt_tokens"]
                or usage.get("unknown_usage_calls", 0)):
            raise ValueError("Plot requires fully observed input usage")
    return methods


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = OUT / "comparison.json"
    methods = validate(json.loads(source.read_text()))
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    offsets = [(6, 6), (6, -12), (-24, 7), (6, 8), (-16, -14), (6, 6),
               (6, -13), (6, 7), (-22, -15), (6, 7), (-22, 8), (6, -14)]
    for ax, field, title in zip(axes, ("prompt_tokens", "uncached_prompt_tokens"),
                                ("All QA input (includes cached)", "Uncached QA input")):
        for i, (method, row) in enumerate(methods.items()):
            label, color, marker = LABELS[method]
            x, y = row["qa_usage"][field] / 1e6, row["mean_score"]
            ax.scatter(x, y, marker=marker, color=color, s=60 if marker == "*" else 32, zorder=3)
            ax.annotate(label, (x, y), xytext=offsets[i % len(offsets)], textcoords="offset points",
                        fontsize=8, color=color)
        ax.set_xlabel("Input tokens (millions)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=.25, linestyle="--")
        ax.margins(x=.15, y=.23)
    axes[0].set_ylabel("Common automatic score (0 / 0.5 / 1)")
    fig.suptitle("Fixed five-video development cohort: quality vs. QA context", fontsize=11)
    fig.text(.5, .02, "Each point is one complete 65-question trajectory. QA stage only; not money or end-to-end cost.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .95))
    target = ROOT / "latex/figures/memory_design_tradeoff.pdf"
    target.parent.mkdir(exist_ok=True)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    receipt = {
        "comparison_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "plot_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "plot": str(target.relative_to(ROOT)),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "points": len(methods), "uses_common_judge_only": True,
        "interpretation": "Measured QA token usage, not price, latency or full pipeline cost.",
    }
    (OUT / "plot_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
