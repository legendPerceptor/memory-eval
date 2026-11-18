#!/usr/bin/env python3
"""Generate latex/figures/paired8_final_bytype.{pdf,png}: 8-video oGMemory vs
M3-Agent accuracy by question type. Reads the frozen by-type JSON only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/yuanjian/Development/memory-eval")
d = json.loads((ROOT / "full-bench-reproduce/results/paper-comparison-8v/paired8_bytype.json").read_text())
rows = d["per_type"]

order = ["Person Understanding", "Multi-Detail Reasoning", "Cross-Modal Reasoning",
         "General Knowledge Extraction", "Multi-Hop Reasoning"]
short = {"Person Understanding": "Person\nUnderstanding",
         "Multi-Detail Reasoning": "Multi-Detail\nReasoning",
         "Cross-Modal Reasoning": "Cross-Modal\nReasoning",
         "General Knowledge Extraction": "General Knowledge\nExtraction",
         "Multi-Hop Reasoning": "Multi-Hop\nReasoning"}
by = {r["type"]: r for r in rows}
rows = [by[t] for t in order]

labels = [short[r["type"]] for r in rows]
og = [100 * r["og"] for r in rows]
m3 = [100 * r["m3"] for r in rows]
n = [r["n"] for r in rows]
delta = [100 * r["delta"] for r in rows]

plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, ax = plt.subplots(figsize=(10.5, 4.9))
fig.subplots_adjust(left=0.085, right=0.985, top=0.78, bottom=0.17)

x = np.arange(len(rows)); w = 0.38
b1 = ax.bar(x - w/2, og, w, label="oGMemory", color="#2878b5")
b2 = ax.bar(x + w/2, m3, w, label="M3-Agent (matched model)", color="#ed8b23")
for bars, vals in ((b1, og), (b2, m3)):
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width()/2, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
for i, dv in enumerate(delta):
    ax.text(i, max(og[i], m3[i]) + 8.5, f"+{dv:.1f} pp", ha="center", va="bottom",
            fontsize=9, color="#1a7f37", fontweight="bold")

ax.set_xticks(x, [f"{l}\n(n={nn})" for l, nn in zip(labels, n)], fontsize=9.5)
ax.set_ylim(0, 106)
ax.set_ylabel("Mean automatic score (%)")
ax.set_title("8-video M3-Bench: accuracy by question type (97 questions)", fontsize=12.5, pad=46)

# Legend ABOVE the plotting area (below the title) so it never collides with
# the rightmost (Multi-Hop) bar value / +pp annotations.
ax.legend(ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.005),
          frameon=False, fontsize=10.5, handlelength=1.6, columnspacing=2.0)

ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.2); ax.set_axisbelow(True)

fig.text(0.5, 0.015,
    "Multi-label: a question counts in each of its annotated types (166 tags / 97 questions). "
    "Multi-Hop has only 5 questions; interpret with caution. Soft score 0 / 0.5 / 1.",
    fontsize=8.4, color="#555", va="bottom", ha="center")

folder = ROOT / "latex/figures"
fig.savefig(folder / "paired8_final_bytype.pdf", bbox_inches="tight")
fig.savefig(folder / "paired8_final_bytype.png", dpi=180, bbox_inches="tight")
print("saved", folder / "paired8_final_bytype.pdf")
