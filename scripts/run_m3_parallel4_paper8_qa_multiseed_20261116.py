#!/usr/bin/env python3
"""Multi-seed QA token-budget sweep on the frozen 8-video graph.

Reuses the already-built pgvector graph in ``m3_parallel4_paper8_20261113`` and
ONLY re-runs the QA stage + judge at different ``m3.qa_context.max_input_tokens``
budgets, with multiple stochastic repeats per budget. Goal: quantify QA-budget
noise so the Pareto figure can carry error bars and the low-budget (3000/4000)
regime can be tested.

Outputs (all under ``experiments/m3_parallel4_paper8_20261113/``):
  - per-run QA:      runs/.../qa_budget_{b}_r{rep}/og_qa/{video}/m3_qa_results.jsonl
  - per-run usage:   usage-qa-budget-{b}-r{rep}-{video}.jsonl
  - per-run judge:   judgments_qa_budget_{b}_r{rep}.jsonl
  - status:          qa_budget_multiseed_status.json
  - results:         qa_budget_multiseed_results.json  (per-point + aggregate)

This script never touches the existing single-seed ``qa_budget_*`` artifacts or
the frozen paper tables; it only writes the ``_r{rep}`` family + a new summary.
"""
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import yaml

ROOT = Path('/home/yuanjian/Development/memory-eval')
os.environ['M3_EVAL_ROOT'] = str(ROOT)
NAME = 'm3_parallel4_paper8_20261113'
RUN, EXP = ROOT / 'runs' / NAME, ROOT / 'experiments' / NAME
PYTHON = ROOT / 'oGMemory/.venv/bin/python'

BUDGETS = [3000, 4000, 6000, 9000, 12000, 16000, 22528]
REPS = 2
VIDEOS = ['study_09', 'kitchen_01', 'bedroom_05', 'study_23',
          'kitchen_23', 'study_05', 'gym_03', 'bedroom_04']
JUDGE_SEED_BASE = 20261113


def load_driver():
    spec = importlib.util.spec_from_file_location('base_driver', RUN / 'driver.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def write_budget_config(b):
    cfg = yaml.safe_load((EXP / 'private/config.yaml').read_text())
    cfg.setdefault('m3', {}).setdefault('qa_context', {})['max_input_tokens'] = b
    p = EXP / 'private' / f'config_qa_{b}.yaml'
    if p.exists():
        got = yaml.safe_load(p.read_text())
        if got.get('m3', {}).get('qa_context', {}).get('max_input_tokens') != b:
            got.setdefault('m3', {}).setdefault('qa_context', {})['max_input_tokens'] = b
            p.write_text(yaml.safe_dump(got, allow_unicode=True))
            p.chmod(0o600)
        return p
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    p.chmod(0o600)
    return p


def write_wrapper(b, rep):
    tag = f'qa_budget_{b}_r{rep}'
    wp = RUN / tag / '_wrapper.py'
    wp.parent.mkdir(parents=True, exist_ok=True)
    wp.write_text(
        'import importlib.util, sys\n'
        'from pathlib import Path\n'
        f'BASE=Path({str(RUN)!r}); EXP=Path({str(EXP)!r})\n'
        's=importlib.util.spec_from_file_location("d",BASE/"driver.py")\n'
        'm=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n'
        f'm.RUN=BASE/"{tag}"; m.CONFIG=EXP/"private/config_qa_{b}.yaml"\n'
        'sys.path.insert(0,str(m.SOURCE))\n'
        'raise SystemExit(m.worker("qa", sys.argv[1]) or 0)\n'
    )
    return wp


def run_qa_video(m, b, rep, video, wp):
    tag = f'qa_budget_{b}_r{rep}'
    qpath = RUN / tag / 'og_qa' / video / 'm3_qa_results.jsonl'
    if qpath.exists() and qpath.stat().st_size > 0:
        return False
    qpath.parent.mkdir(parents=True, exist_ok=True)
    env = m.environment('qa', video)
    env['M3_USAGE_LOG'] = str(EXP / f'usage-qa-budget-{b}-r{rep}-{video}.jsonl')
    env['M3_EVAL_ROOT'] = str(ROOT)
    log = (EXP / 'private' / f'qa_budget_{b}-r{rep}-{video}.log').open('ab', buffering=0)
    p = subprocess.run([str(PYTHON), str(wp), video], cwd=str(m.SOURCE), env=env,
                       stdout=log, stderr=subprocess.STDOUT)
    (EXP / 'private' / f'qa_budget_{b}-r{rep}-{video}.exit').write_text(str(p.returncode) + '\n')
    if not qpath.exists():
        raise RuntimeError(f'budget {b} r{rep} {video}: no QA output (exit {p.returncode})')
    return True


def judge_run(m, b, rep, cfgpath):
    sys.path.insert(0, str(m.SOURCE))
    from m3.qa import load_questions
    from m3.eval import build_judge_llm, judge_answer
    m._recover_key()
    llm = build_judge_llm(config_path=str(cfgpath))
    qs = [q for q in load_questions(m.h.ANNOTATIONS) if q['video_id'] in VIDEOS]
    tag = f'qa_budget_{b}_r{rep}'
    tasks = []
    for video in VIDEOS:
        answers = {r['question_id']: r for r in m.h.rows(
            RUN / tag / 'og_qa' / video / 'm3_qa_results.jsonl')}
        for q in (x for x in qs if x['video_id'] == video):
            r = answers.get(q['question_id'], {})
            tasks.append({'video_id': video, 'question_id': q['question_id'],
                          'question': q['question'], 'expected_answer': q['answer'],
                          'predicted_answer': r.get('predicted_answer', ''),
                          'error': r.get('error', '')})
    random.Random(JUDGE_SEED_BASE + 1000 * b + rep).shuffle(tasks)
    out = EXP / f'judgments_qa_budget_{b}_r{rep}.jsonl'
    done = {(x['video_id'], x['question_id']) for x in m.h.rows(out)}
    gate = m.h.QuotaGate()
    with out.open('a') as fh:
        for t in tasks:
            if (t['video_id'], t['question_id']) in done:
                continue
            if not t['predicted_answer'] or t['error']:
                score, reason = 0.0, 'Missing or failed answer'
            else:
                gate.wait()
                score, reason, err = judge_answer(judge_llm=llm, question=t['question'],
                                                  expected=t['expected_answer'],
                                                  predicted=t['predicted_answer'])
                if err:
                    raise RuntimeError(f'judge failed {b} r{rep} {t["question_id"]}')
            fh.write(json.dumps({**t, 'score': score, 'reason': reason}, ensure_ascii=False) + '\n')
            fh.flush()
            os.fsync(fh.fileno())


def summarize(m, b, rep):
    tag = f'qa_budget_{b}_r{rep}'
    js = m.h.rows(EXP / f'judgments_qa_budget_{b}_r{rep}.jsonl')
    qa_in = qa_out = n_ans = 0
    for video in VIDEOS:
        for x in m.h.rows(EXP / f'usage-qa-budget-{b}-r{rep}-{video}.jsonl'):
            u = x.get('usage') or {}
            qa_in += u.get('prompt_tokens', 0) or 0
            qa_out += u.get('completion_tokens', 0) or 0
        for r in m.h.rows(RUN / tag / 'og_qa' / video / 'm3_qa_results.jsonl'):
            n_ans += 1
    vsum = {v: [0, 0.0] for v in VIDEOS}
    for x in js:
        vsum[x['video_id']][0] += 1
        vsum[x['video_id']][1] += x.get('score', 0)
    per_video = {v: (vsum[v][1] / vsum[v][0] if vsum[v][0] else None) for v in VIDEOS}
    score = sum(x.get('score', 0) for x in js)
    return {'budget': b, 'rep': rep, 'effective_budget': min(b, 22528),
            'questions': len(js), 'answers': n_ans, 'score': score,
            'accuracy': score / len(js) if js else None,
            'qa_input_tokens': qa_in, 'qa_output_tokens': qa_out,
            'per_video': per_video}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    mu = sum(xs) / len(xs)
    return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def aggregate(points):
    agg = {}
    for b in sorted({p['budget'] for p in points}):
        rows = [p for p in points if p['budget'] == b]
        per_video_mean, per_video_std = {}, {}
        for v in VIDEOS:
            vals = [p['per_video'].get(v) for p in rows]
            per_video_mean[v] = _mean(vals)
            per_video_std[v] = _std(vals)
        agg[str(b)] = {
            'n_reps': len(rows),
            'accuracy_mean': _mean([p['accuracy'] for p in rows]),
            'accuracy_std': _std([p['accuracy'] for p in rows]),
            'qa_input_tokens_mean': _mean([p['qa_input_tokens'] for p in rows]),
            'qa_output_tokens_mean': _mean([p['qa_output_tokens'] for p in rows]),
            'qa_seconds_mean': _mean([p.get('qa_seconds') for p in rows]),
            'per_video_mean': per_video_mean,
            'per_video_std': per_video_std,
        }
    return agg


def write_results(points):
    agg = aggregate(points)
    doc = {'budgets': sorted({p['budget'] for p in points}),
           'reps': sorted({p['rep'] for p in points}),
           'videos': VIDEOS, 'points': points, 'aggregate': agg}
    (EXP / 'qa_budget_multiseed_results.json').write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + '\n')
    return doc


EFFECTIVE_CEILING = 22528


def _import_single_seed():
    """Import the existing single-seed sweep + baseline artifacts as seed points.

    baseline ``judgments.jsonl`` (max_input_tokens=24000, clamped to 22528) and the
    ``qa_budget_sweep_results.json`` points are each one independent QA sampling.
    We fold them in as rep=-1 points so the aggregate has real replicates without
    re-running the slow 22528 budget.
    """
    import collections
    seed = []
    # (1) baseline, effective 22528
    jpath = EXP / 'judgments.jsonl'
    if jpath.exists():
        js = [json.loads(x) for x in jpath.read_text().splitlines() if x.strip()]
        if len(js) == 97:
            by = collections.defaultdict(lambda: [0, 0.0])
            for r in js:
                by[r['video_id']][0] += 1
                by[r['video_id']][1] += r.get('score', 0)
            qa_in = qa_out = 0
            for v in VIDEOS:
                for x in _rows(EXP / f'usage-qa-{v}.jsonl'):
                    u = x.get('usage') or {}
                    qa_in += u.get('prompt_tokens', 0) or 0
                    qa_out += u.get('completion_tokens', 0) or 0
            seed.append({'budget': 22528, 'rep': -1, 'effective_budget': 22528,
                         'questions': len(js), 'answers': len(js),
                         'score': sum(r.get('score', 0) for r in js),
                         'accuracy': sum(r.get('score', 0) for r in js) / len(js),
                         'qa_input_tokens': qa_in, 'qa_output_tokens': qa_out,
                         'qa_seconds': None, 'source': 'baseline_judgments',
                         'per_video': {v: (by[v][1] / by[v][0] if by[v][0] else None) for v in VIDEOS}})
    # (2) single-seed sweep points
    swp = EXP / 'qa_budget_sweep_results.json'
    if swp.exists():
        for pt in json.loads(swp.read_text())['points']:
            b = int(pt['budget'])
            jb = EXP / f'judgments_qa_budget_{b}.jsonl'
            per_video = None
            if jb.exists():
                js = [json.loads(x) for x in jb.read_text().splitlines() if x.strip()]
                by = collections.defaultdict(lambda: [0, 0.0])
                for r in js:
                    by[r['video_id']][0] += 1
                    by[r['video_id']][1] += r.get('score', 0)
                per_video = {v: (by[v][1] / by[v][0] if by[v][0] else None) for v in VIDEOS}
            seed.append({'budget': b, 'rep': -2, 'effective_budget': pt.get('effective_budget', min(b, EFFECTIVE_CEILING)),
                         'questions': pt.get('questions'), 'answers': pt.get('answers'),
                         'score': pt.get('score'), 'accuracy': pt.get('accuracy'),
                         'qa_input_tokens': pt.get('qa_input_tokens'),
                         'qa_output_tokens': pt.get('qa_output_tokens'),
                         'qa_seconds': pt.get('qa_seconds'), 'source': 'single_seed_sweep',
                         'per_video': per_video})
    return seed


def _rows(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _load_prior_points():
    out = {}
    p = EXP / 'qa_budget_multiseed_results.json'
    if p.exists():
        for pt in json.loads(p.read_text()).get('points', []):
            out[(int(pt['budget']), int(pt['rep']))] = pt
    return out


def main():
    import itertools
    # MS_SPEC lets us request (budget, rep) pairs; default = BUDGETS x REPS.
    spec_env = os.environ.get('MS_SPEC', '').strip()
    if spec_env:
        pairs = []
        for tok in spec_env.split(','):
            b, r = tok.split(':')
            pairs.append((int(b), int(r)))
    else:
        budgets = [int(x) for x in (os.environ.get('MS_BUDGETS') or '').split(',') if x.strip()] or list(BUDGETS)
        reps = [int(x) for x in (os.environ.get('MS_REPS') or '').split(',') if x.strip()] or list(range(REPS))
        pairs = list(itertools.product(budgets, reps))
    include_seed = os.environ.get('MS_INCLUDE_SEED', '1') == '1'
    print(f'[multiseed] pairs={pairs} include_seed={include_seed}', flush=True)
    m = load_driver()
    status_path = EXP / 'qa_budget_multiseed_status.json'
    points = list(_load_prior_points().values())
    if include_seed:
        for sp in _import_single_seed():
            points = [p for p in points if not (int(p['budget']) == int(sp['budget']) and int(p['rep']) == -1)]
            points.append(sp)
    done = {(int(p['budget']), int(p['rep'])) for p in points}
    for b, rep in pairs:
        cfgpath = write_budget_config(b)
        wp = write_wrapper(b, rep)
        if (b, rep) in done:
            print(f'[multiseed] skip b={b} r={rep} (already have point)', flush=True)
            continue
        status_path.write_text(json.dumps(
            {'status': 'running', 'budget': b, 'rep': rep,
             'done_points': len(points), 'points': points}, indent=2, ensure_ascii=False) + '\n')
        t0 = time.time()
        for video in VIDEOS:
            run_qa_video(m, b, rep, video, wp)
        judge_run(m, b, rep, cfgpath)
        row = summarize(m, b, rep)
        row['qa_seconds'] = round(time.time() - t0, 1)
        points = [p for p in points if not (int(p['budget']) == b and int(p['rep']) == rep)]
        points.append(row)
        doc = write_results(points)
        print(f"[multiseed] done b={b} r={rep} acc={row['accuracy']:.4f} "
              f"qa_in={row['qa_input_tokens']} sec={row['qa_seconds']}", flush=True)
        status_path.write_text(json.dumps(
            {'status': 'running', 'budget': b, 'rep': rep,
             'done_points': len(points), 'aggregate': doc['aggregate'],
             'points': points}, indent=2, ensure_ascii=False) + '\n')
    status_path.write_text(json.dumps(
        {'status': 'finished', 'done_points': len(points),
         'aggregate': aggregate(points), 'points': points}, indent=2, ensure_ascii=False) + '\n')
    print('[multiseed] FINISHED', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
