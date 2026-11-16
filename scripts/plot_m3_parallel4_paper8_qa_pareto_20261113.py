#!/usr/bin/env python3
"""Plot the QA token-budget Pareto front for the 8-video 4-way parallel run."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/yuanjian/Development/memory-eval')
EXP = ROOT / 'experiments/m3_parallel4_paper8_20261113'
res = json.loads((EXP / 'qa_budget_sweep_results.json').read_text())
rows = sorted(res['points'], key=lambda r: r['qa_input_tokens'])

pareto, best = [], -1.0
for r in rows:
    if r['accuracy'] is not None and r['score'] > best:
        pareto.append(r)
        best = r['score']

out = {'points': rows, 'pareto_front': pareto}
(EXP / 'qa_budget_pareto_results.json').write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')

fig, ax = plt.subplots(figsize=(8, 5.5))
xs = [r['qa_input_tokens'] for r in rows]
ys = [r['accuracy'] * 100 for r in rows]
ax.plot(xs, ys, 'o-', color='#4C72B0', label='QA budget points', zorder=2)
if pareto:
    ax.plot([r['qa_input_tokens'] for r in pareto], [r['accuracy'] * 100 for r in pareto],
            's--', color='#C44E52', ms=9, label='Pareto front', zorder=3)
for r in rows:
    note = f"{r['budget']/1000:.1f}k"
    if r['effective_budget'] != r['budget']:
        note += f"→{r['effective_budget']/1000:.1f}k"
    ax.annotate(note, (r['qa_input_tokens'], r['accuracy'] * 100),
                xytext=(6, 6), textcoords='offset points', fontsize=8)
ax.axhline(57.216, color='gray', ls=':', lw=1, label='paper baseline 57.2%')
ax.set_xlabel('QA-stage input tokens (prompt)')
ax.set_ylabel('Accuracy (%) on 97 questions')
ax.set_title('8-video 4-way parallel: QA token budget vs accuracy (Pareto front)')
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(EXP / 'qa_budget_pareto_20261113.png', dpi=180)
print(json.dumps(out, indent=2, ensure_ascii=False))
