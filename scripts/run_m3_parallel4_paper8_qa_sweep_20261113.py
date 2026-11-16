#!/usr/bin/env python3
"""QA-stage token-budget sweep (Pareto front) on the fixed 8-video graph.

Precondition: the main run ``m3_parallel4_paper8_20261113`` finished its
pipeline+qa, so the frozen pgvector graph already holds all 8 videos' memories.
This script REUSES that graph (same DB/account) and only re-runs the QA stage
at different ``m3.qa_context.max_input_tokens`` budgets, then re-judges all 97
questions per budget and records QA token usage + accuracy for a Pareto front.

Effective input cap is ``min(max_input_tokens, context_window - qa_out - safety)``
= min(max_input_tokens, 32768 - 8192 - 2048) = min(., 22528). Budgets > 22528
are clamped by the runtime and will be reported as such.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path('/home/yuanjian/Development/memory-eval')
os.environ['M3_EVAL_ROOT'] = str(ROOT)
NAME = 'm3_parallel4_paper8_20261113'
RUN, EXP = ROOT / 'runs' / NAME, ROOT / 'experiments' / NAME
PYTHON = ROOT / 'oGMemory/.venv/bin/python'
BUDGETS = [6000, 9000, 12000, 16000, 22528]
VIDEOS = ['study_09', 'kitchen_01', 'bedroom_05', 'study_23',
          'kitchen_23', 'study_05', 'gym_03', 'bedroom_04']


def load_driver():
    spec = importlib.util.spec_from_file_location('base_driver', RUN / 'driver.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def write_budget_config(b):
    cfg = yaml.safe_load((EXP / 'private/config.yaml').read_text())
    cfg.setdefault('m3', {}).setdefault('qa_context', {})['max_input_tokens'] = b
    p = EXP / 'private' / f'config_qa_{b}.yaml'
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    p.chmod(0o600)
    return p


def write_wrapper(b):
    wp = RUN / f'qa_budget_{b}' / '_wrapper.py'
    wp.parent.mkdir(parents=True, exist_ok=True)
    wp.write_text(
        'import importlib.util, sys\n'
        f'from pathlib import Path\n'
        f'BASE=Path({str(RUN)!r}); EXP=Path({str(EXP)!r})\n'
        's=importlib.util.spec_from_file_location("d",BASE/"driver.py")\n'
        'm=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n'
        f'm.RUN=BASE/"qa_budget_{b}"; m.CONFIG=EXP/"private/config_qa_{b}.yaml"\n'
        'sys.path.insert(0,str(m.SOURCE))\n'
        'raise SystemExit(m.worker("qa", sys.argv[1]) or 0)\n'
    )
    return wp


def run_qa_video(m, b, video, wp):
    qpath = RUN / f'qa_budget_{b}' / 'og_qa' / video / 'm3_qa_results.jsonl'
    if qpath.exists():
        return
    qpath.parent.mkdir(parents=True, exist_ok=True)
    env = m.environment('qa', video)
    env['M3_USAGE_LOG'] = str(EXP / f'usage-qa-budget-{b}-{video}.jsonl')
    env['M3_EVAL_ROOT'] = str(ROOT)
    log = (EXP / 'private' / f'qa_budget_{b}-{video}.log').open('ab', buffering=0)
    p = subprocess.run([str(PYTHON), str(wp), video], cwd=str(m.SOURCE), env=env,
                       stdout=log, stderr=subprocess.STDOUT)
    (EXP / 'private' / f'qa_budget_{b}-{video}.exit').write_text(str(p.returncode) + '\n')
    if not qpath.exists():
        raise RuntimeError(f'budget {b} {video}: no QA output (exit {p.returncode})')


def judge_budget(m, b, cfgpath):
    sys.path.insert(0, str(m.SOURCE))
    from m3.qa import load_questions
    from m3.eval import build_judge_llm, judge_answer
    import random
    m._recover_key()
    llm = build_judge_llm(config_path=str(cfgpath))
    qs = [q for q in load_questions(m.h.ANNOTATIONS) if q['video_id'] in VIDEOS]
    tasks = []
    for video in VIDEOS:
        answers = {r['question_id']: r for r in m.h.rows(
            RUN / f'qa_budget_{b}' / 'og_qa' / video / 'm3_qa_results.jsonl')}
        for q in (x for x in qs if x['video_id'] == video):
            r = answers.get(q['question_id'], {})
            tasks.append({'video_id': video, 'question_id': q['question_id'],
                          'question': q['question'], 'expected_answer': q['answer'],
                          'predicted_answer': r.get('predicted_answer', ''),
                          'error': r.get('error', '')})
    random.Random(20261113).shuffle(tasks)
    out = EXP / f'judgments_qa_budget_{b}.jsonl'
    done = {(x['video_id'], x['question_id']) for x in m.h.rows(out)}
    gate = m.h.QuotaGate()
    with out.open('a') as fh:
        for t in tasks:
            if (t['video_id'], t['question_id']) in done:
                continue
            if not t['predicted_answer'] or t['error']:
                score, reason, err = 0.0, 'Missing or failed answer', False
            else:
                gate.wait()
                score, reason, err = judge_answer(judge_llm=llm, question=t['question'],
                                                  expected=t['expected_answer'],
                                                  predicted=t['predicted_answer'])
                if err:
                    raise RuntimeError(f'judge failed {b} {t["question_id"]}')
            fh.write(json.dumps({**t, 'score': score, 'reason': reason}, ensure_ascii=False) + '\n')
            fh.flush()
            os.fsync(fh.fileno())


def summarize(m, b):
    js = m.h.rows(EXP / f'judgments_qa_budget_{b}.jsonl')
    qa_in = qa_out = 0
    n_ans = 0
    for video in VIDEOS:
        for x in m.h.rows(EXP / f'usage-qa-budget-{b}-{video}.jsonl'):
            u = x.get('usage') or {}
            qa_in += u.get('prompt_tokens', 0) or 0
            qa_out += u.get('completion_tokens', 0) or 0
        for r in m.h.rows(RUN / f'qa_budget_{b}' / 'og_qa' / video / 'm3_qa_results.jsonl'):
            n_ans += 1
    score = sum(x.get('score', 0) for x in js)
    return {'budget': b, 'effective_budget': min(b, 22528), 'questions': len(js),
            'answers': n_ans, 'score': score, 'accuracy': score / len(js) if js else None,
            'qa_input_tokens': qa_in, 'qa_output_tokens': qa_out}


def main():
    status = EXP / 'status.json'
    while True:
        if status.exists():
            s = json.loads(status.read_text()).get('status')
            if s in ('finished', 'judge_failed', 'needs_review', 'safety_stopped'):
                break
        time.sleep(30)
    final = json.loads(status.read_text()).get('status')
    if final != 'finished':
        (EXP / 'qa_budget_sweep_status.json').write_text(json.dumps(
            {'status': 'blocked_by_baseline', 'baseline': final}, indent=2) + '\n')
        return 2
    m = load_driver()
    points = []
    for b in BUDGETS:
        cfgpath = write_budget_config(b)
        wp = write_wrapper(b)
        (EXP / 'qa_budget_sweep_status.json').write_text(json.dumps(
            {'status': 'running', 'budget': b, 'points': points}, indent=2) + '\n')
        t0 = time.time()
        for video in VIDEOS:
            run_qa_video(m, b, video, wp)
        judge_budget(m, b, cfgpath)
        row = summarize(m, b)
        row['qa_seconds'] = round(time.time() - t0, 1)
        points.append(row)
        (EXP / 'qa_budget_sweep_results.json').write_text(json.dumps(
            {'points': points}, indent=2, ensure_ascii=False) + '\n')
    (EXP / 'qa_budget_sweep_status.json').write_text(json.dumps(
        {'status': 'finished', 'points': points}, indent=2, ensure_ascii=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
