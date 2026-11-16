#!/usr/bin/env python3
"""Preview judge: score the already-completed videos (same judge LLM/config).

Writes to judgments_completed5_preview.jsonl (does NOT touch the official
judgments.jsonl that the baseline's judge() step will produce). Resumable.
"""
import importlib.util, json, os, sys, time
from pathlib import Path

ROOT = Path('/home/yuanjian/Development/memory-eval')
os.environ['M3_EVAL_ROOT'] = str(ROOT)
NAME = 'm3_parallel4_paper8_20261113'
RUN, EXP = ROOT/'runs'/NAME, ROOT/'experiments'/NAME
VIDEOS = ['study_09', 'kitchen_01', 'bedroom_05', 'study_23', 'kitchen_23']

spec = importlib.util.spec_from_file_location('d', ROOT/'scripts'/'run_m3_parallel4_paper8_20261113.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sys.path.insert(0, str(m.SOURCE))
from m3.qa import load_questions
from m3.eval import build_judge_llm, judge_answer
from m3.qa_results import latest_answers

m._recover_key(); m.verify_model_settings()
llm = build_judge_llm(config_path=str(m.CONFIG))
qs = [q for q in load_questions(m.h.ANNOTATIONS) if q['video_id'] in VIDEOS]

tasks = []
for video in VIDEOS:
    latest = latest_answers(RUN/'og_qa'/video/'m3_qa_results.jsonl')
    for q in (x for x in qs if x['video_id'] == video):
        r = latest.get((video, q['question_id']), {})
        tasks.append({'video_id': video, 'question_id': q['question_id'], 'question': q['question'],
                      'expected_answer': q['answer'], 'predicted_answer': r.get('predicted_answer',''),
                      'error': r.get('error','')})

import random
random.Random(20261113).shuffle(tasks)
out = EXP/'judgments_completed5_preview.jsonl'
done = {(x['video_id'], x['question_id']) for x in m.h.rows(out)}
from m3.qa_results import completed_answer
done = {(x['video_id'], x['question_id']) for x in m.h.rows(out) if 'score' in x}
gate = m.h.QuotaGate()
with out.open('a') as fh:
    for t in tasks:
        if (t['video_id'], t['question_id']) in done: continue
        if not t['predicted_answer'] or t['error']:
            score, reason, err = 0.0, 'Missing or failed answer', False
        else:
            gate.wait()
            score, reason, err = judge_answer(judge_llm=llm, question=t['question'],
                                              expected=t['expected_answer'], predicted=t['predicted_answer'])
        fh.write(json.dumps({**t, 'score': score, 'reason': reason, 'judge_error': bool(err)}, ensure_ascii=False)+'\n')
        fh.flush(); os.fsync(fh.fileno())
        if err: raise RuntimeError('judge error preserved')

rows = m.h.rows(out)
tot = len(rows); s = sum(x.get('score',0) for x in rows)
print(f'\n=== OVERALL: {s:.4f}/{tot} = {100*s/tot:.2f}%  (paper 57.22%) ===')
per = {}
for x in rows:
    per.setdefault(x['video_id'], [0,0]); per[x['video_id']][0]+=x.get('score',0); per[x['video_id']][1]+=1
paper={'study_09':58.33,'kitchen_01':38.46,'bedroom_05':41.67,'study_23':67.86,'kitchen_23':54.17}
for v in VIDEOS:
    if v in per:
        sc,n=per[v]; print(f'  {v:12s} {sc:.2f}/{n} = {100*sc/n:5.2f}%   (paper {paper[v]:.2f}%)')
