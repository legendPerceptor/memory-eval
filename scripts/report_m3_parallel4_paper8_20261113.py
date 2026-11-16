#!/usr/bin/env python3
"""Generate REPORT.md for the 8-video 4-way parallel experiment."""
import json
import re
from pathlib import Path

ROOT = Path('/home/yuanjian/Development/memory-eval')
NAME = 'm3_parallel4_paper8_20261113'
EXP = ROOT / 'experiments' / NAME
VIDEOS = ['study_09', 'kitchen_01', 'bedroom_05', 'study_23',
          'kitchen_23', 'study_05', 'gym_03', 'bedroom_04']
PAPER = {'acc': 57.216, 'per_video': {'study_09': 58.33, 'kitchen_01': 38.46, 'bedroom_05': 41.67,
         'study_23': 67.86, 'kitchen_23': 54.17, 'study_05': 45.83, 'gym_03': 72.73, 'bedroom_04': 81.82}}


def rows(p):
    if not Path(p).exists():
        return []
    out = []
    for ln in Path(p).read_text().splitlines():
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return out


def usage_sum(path):
    pin = pout = pcache = 0
    for r in rows(path):
        u = r.get('usage') or {}
        pin += u.get('prompt_tokens', 0) or 0
        pout += u.get('completion_tokens', 0) or 0
        det = u.get('prompt_tokens_details') or {}
        pcache += det.get('cached_tokens', 0) or 0
    return pin, pout, pcache


def parse_stages(logpath):
    """Return list of (stage, start_unix) from 'Running stage X' log lines."""
    out = []
    if not Path(logpath).exists():
        return out
    rx = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) INFO m3\.pipeline\.runner: Running stage (\S+)')
    import datetime
    for ln in Path(logpath).read_text().splitlines():
        m = rx.search(ln)
        if m:
            ts = datetime.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S,%f').timestamp()
            out.append((m.group(2), ts))
    return out


def main():
    L = []
    a = L.append
    a(f'# 8-video 4-way parallel experiment ({NAME})\n')
    a('Model: MiniMax-M3.1-Flash (thinking=adaptive, reasoning_effort=medium). '
      'Branch: rychkov/m3-parallel. Fresh pgvector DB. 97 questions, 8 videos.\n')

    # --- accuracy ---
    js = rows(EXP / 'judgments.jsonl')
    total = len(js)
    score = sum(x.get('score', 0) for x in js)
    acc = score / total * 100 if total else float('nan')
    a('## Accuracy (approx. reproduction)\n')
    a(f'- Questions judged: **{total}**  (missing/errored treated as 0)')
    a(f'- Mean score (oGMemory): **{acc:.2f}%**  vs paper baseline **{PAPER["acc"]}%** '
      f'(delta {acc - PAPER["acc"]:+.2f} pp)')
    a('')
    a('| video | questions | score | acc % | paper acc % | delta pp |')
    a('|---|---|---|---|---|---|')
    for v in VIDEOS:
        vs = [x for x in js if x['video_id'] == v]
        n = len(vs)
        s = sum(x.get('score', 0) for x in vs)
        va = s / n * 100 if n else 0
        pa = PAPER['per_video'].get(v, 0)
        a(f'| {v} | {n} | {s:.2f} | {va:.2f} | {pa} | {va - pa:+.2f} |')
    a('')

    # --- token usage by stage ---
    a('## Token usage by stage\n')
    a('| stage | videos | prompt tok | completion tok | cached tok |')
    a('|---|---|---|---|---|')
    grand = [0, 0, 0]
    for stage, key in (('pipeline', 'usage-pipeline-{v}.jsonl'), ('qa', 'usage-qa-{v}.jsonl')):
        pin = pout = pc = 0
        for v in VIDEOS:
            i, o, c = usage_sum(EXP / key.format(v=v))
            pin += i; pout += o; pc += c
        grand[0] += pin; grand[1] += pout; grand[2] += pc
        a(f'| {stage} | {len(VIDEOS)} | {pin} | {pout} | {pc} |')
    ji, jo, jc = usage_sum(EXP / 'usage-judge-all.jsonl')
    grand[0] += ji; grand[1] += jo; grand[2] += jc
    a(f'| judge | 97 | {ji} | {jo} | {jc} |')
    a(f'| **total** | | **{grand[0]}** | **{grand[1]}** | **{grand[2]}** |')
    a('')

    # --- timing ---
    a('## Wall-clock timing\n')
    a('| stage | seconds | minutes |')
    a('|---|---|---|')
    receipts = sorted((EXP / 'stages').glob('*.json')) if (EXP / 'stages').exists() else []
    tot = 0.0
    by_phase = {'pipeline': 0.0, 'qa': 0.0}
    for rp in receipts:
        r = json.loads(rp.read_text())
        sec = r.get('seconds', 0)
        tot += sec
        by_phase[r['phase']] = by_phase.get(r['phase'], 0) + sec
    a(f'| {len(VIDEOS)}x pipeline | {by_phase["pipeline"]:.1f} | {by_phase["pipeline"]/60:.1f} |')
    a(f'| {len(VIDEOS)}x qa | {by_phase["qa"]:.1f} | {by_phase["qa"]/60:.1f} |')
    a(f'| **pipeline+qa total** | **{tot:.1f}** | **{tot/60:.1f}** |')
    a('')

    # per-video timing from authoritative stage receipts (NOT raw logs,
    # which can contain earlier failed runs, e.g. kitchen_01 on 11-13).
    a('### Per-video stage timings (authoritative receipts)\n')
    a('| video | pipeline (s) | qa (s) | pipeline (min) |')
    a('|---|---|---|---|')
    for v in VIDEOS:
        psec = qsec = 0.0
        rp = EXP / 'stages' / f'{v}-pipeline.json'
        rq = EXP / 'stages' / f'{v}-qa.json'
        if rp.exists():
            psec = json.loads(rp.read_text()).get('seconds', 0) or 0
        if rq.exists():
            qsec = json.loads(rq.read_text()).get('seconds', 0) or 0
        a(f'| {v} | {psec:.1f} | {qsec:.1f} | {psec/60:.1f} |')
    a('')

    # --- pareto ---
    pr = EXP / 'qa_budget_pareto_results.json'
    if pr.exists():
        d = json.loads(pr.read_text())
        a('## QA token-budget Pareto front\n')
        a('Same frozen graph; only `m3.qa_context.max_input_tokens` varies. '
          'Effective cap = min(budget, 32768-8192-2048=22528).\n')
        a('| budget | effective | QA input tok | QA output tok | accuracy % |')
        a('|---|---|---|---|---|')
        for r in sorted(d['points'], key=lambda x: x['budget']):
            a(f"| {r['budget']} | {r['effective_budget']} | {r['qa_input_tokens']} | "
              f"{r['qa_output_tokens']} | {r['accuracy']*100:.2f} |")
        a('')
        a(f'Pareto-front points: {[p["budget"] for p in d["pareto_front"]]}')
        a(f'\n![pareto](qa_budget_pareto_20261113.png)\n')

    a('## Caveats\n')
    a('- This branch dropped `fact_index` and uses a reduced `qa_context` schema, so this is an '
      '*approximate* re-run, not byte-identical to the 57.2% paper baseline.')
    a('- QA budgets above 22528 are clamped by the runtime (context_window - qa_out - safety).')
    a('- Judge = MiniMax-M3.1-Flash, temperature 0, deterministic seed 20261113.')

    (EXP / 'REPORT.md').write_text('\n'.join(L) + '\n')
    print('wrote', EXP / 'REPORT.md')


if __name__ == '__main__':
    main()
