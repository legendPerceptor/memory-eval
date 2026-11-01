#!/usr/bin/env python3
"""Vector plots of received usage, never inferred prices or fabricated latency."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments/m3_context_study_20261101"
LABELS = {
    "historical_ogmemory": "Original\noGMemory",
    "m3_reference_text": "M3 reference\nadaptation",
    "budget_only": "A\nbudget",
    "finalization": "B\n+ finalization",
    "compact12k": "C\n12k budget",
    "grouped12k": "D\n12k grouped",
}


def main():
    source = EXP / "cache_analysis.json"
    metrics = json.loads(source.read_text())["methods"]
    names = [name for name in LABELS if name in metrics]
    uncached = [metrics[name]["uncached_prompt_tokens"] / 1e6 for name in names]
    cached = [metrics[name]["cached_prompt_tokens"] / 1e6 for name in names]
    fig, ax = plt.subplots(figsize=(7.2, 3.65))
    indices = range(len(names))
    ax.bar(indices, uncached, color="#245c86", label="Uncached input")
    ax.bar(indices, cached, bottom=uncached, color="#acd1e6", label="Cached input")
    for i, name in enumerate(names):
        total = uncached[i] + cached[i]
        ax.text(i, total + .3, f"{total:.2f}M", ha="center", fontsize=8)
    ax.set_xticks(list(indices), [LABELS[name] for name in names], fontsize=8)
    ax.set_ylabel("Received input tokens (millions)", fontsize=9)
    ax.set_ylim(0, max(a + b for a, b in zip(uncached, cached)) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.text(.5, .015,
             "Fixed 65-question trajectories. Prior quota failures with unknown usage are recorded separately.\n"
             "Usage counts are not billing amounts; M3 uses its independently generated reference memories.",
             ha="center", va="bottom", fontsize=7)
    fig.tight_layout(rect=(0, .1, 1, 1))
    target = ROOT / "latex/figures"
    target.mkdir(exist_ok=True)
    fig.savefig(target / "context_cache.pdf")
    plt.close(fig)
    receipt = {
        "matplotlib_version": matplotlib.__version__,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "methods": names, "output": "latex/figures/context_cache.pdf",
        "values": {name: metrics[name] for name in names},
    }
    (EXP / "plot_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
