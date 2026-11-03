#!/usr/bin/env python3
"""Frozen first-eight-video interim comparison; original answers incl. timeouts.

Separate from the ten-video final judge. No memory regeneration, QA retries, or
replacement of original predictions. Usage and timing are read from receipts.
"""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(os.environ.get('M3_EVAL_ROOT', Path(__file__).resolve().parent.parent))
BASE = 'm3_ten_fullinterval_20261102'
NAME = 'm3_eight_interim_20261103'
EXP = ROOT / 'experiments' / NAME
RUN = ROOT / 'runs' / NAME
SOURCE = ROOT / 'runs' / BASE / '_source/oGMemory'
METHODS = {'ogmemory': ('og_qa', 'pipeline'), 'm3_agent': ('m3_qa', 'm3_memory')}


def rows(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp'); t.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n'); t.replace(p)


def usage_summary(calls):
    known = [c for c in calls if c.get('usage')]
    cached = [(c['usage'].get('prompt_tokens_details') or {}).get('cached_tokens') for c in known]
    prompt = sum(c['usage']['prompt_tokens'] for c in known)
    completion = sum(c['usage']['completion_tokens'] for c in known)
    unknown_cache = sum(x is None for x in cached)
    return {'calls': len(calls), 'api_errors': sum(c.get('status') != 'returned' for c in calls),
            'unknown_usage_calls': len(calls) - len(known), 'prompt_tokens': prompt,
            'completion_tokens': completion, 'total_tokens': prompt + completion,
            'cached_prompt_tokens': sum(x for x in cached if x is not None),
            'uncached_prompt_tokens': None if unknown_cache or len(known) != len(calls) else prompt-sum(cached),
            'unknown_cache_usage_calls': unknown_cache, 'complete_usage': len(calls) == len(known),
            'scope': 'Received response usage; missing usage not treated as zero; completion includes reasoning.'}


def prepare():
    if EXP.exists() or RUN.exists():
        raise FileExistsError('Use judge/summarize on this snapshot; never replace it')
    from plot_m3_stage_times import collect
    sys.path.insert(0, str(SOURCE))
    from m3.qa import load_questions
    base_exp, base_run = ROOT/'experiments'/BASE, ROOT/'runs'/BASE
    full = json.loads((base_exp/'manifest.json').read_text())
    cohort = full['execution_order'][:8]
    timing = collect(ROOT, BASE)
    assert set(cohort) <= set(timing['selected_videos'])
    timing['videos'] = [v for v in timing['videos'] if v['video'] in cohort]
    timing['selected_videos'] = cohort
    timing['questions_per_method'] = sum(v['questions'] for v in timing['videos'])
    timing['totals_seconds'] = {m: {k: sum(v['methods'][m][k] for v in timing['videos'])
                                       for k in ('shared_perception','memory_and_index','qa','allocated_total')}
                               for m in METHODS}
    from plot_m3_stage_times import SHARED_GROUPS
    timing['shared_breakdown_seconds'] = {label: sum(v['stage_seconds'][s] for v in timing['videos'] for s in stages)
                                          for label,stages in SHARED_GROUPS.items()}
    timing['actual_experiment_phase_seconds'] = sum(v['actual_experiment_phase_seconds'] for v in timing['videos'])
    annotation = Path('/data/m3_data/annotations/robot.json')
    assert digest(annotation) == full['annotations_sha256']
    questions = [q for q in load_questions(annotation) if q['video_id'] in cohort]
    canonical = {(q['video_id'],q['question_id']):q for q in questions}
    assert len(questions) == len(canonical) == 97
    tasks, hashes, usage, raw_calls = [], {}, {}, {}
    for method, (qa_phase, memory_phase) in METHODS.items():
        found = set()
        for video in cohort:
            path = base_run/qa_phase/video/'m3_qa_results.jsonl'
            hashes[str(path)] = digest(path)
            predicted = rows(path)
            assert len(predicted) == full['question_counts'][video]
            for row in predicted:
                key = row['video_id'],row['question_id']
                assert key in canonical and key not in found
                found.add(key); gold = canonical[key]
                assert row['question'] == gold['question'] and row['expected_answer'] == gold['answer']
                tasks.append({**row, 'method': method})
        assert found == set(canonical)
        raw_calls[method] = {}
        usage[method] = {}
        for label, phase in [('qa',qa_phase),('memory',memory_phase)]:
            calls = []
            for video in cohort:
                path = base_exp/f'usage-{phase}-{video}.jsonl'
                hashes[str(path)] = digest(path); calls.extend(rows(path))
            assert all(c['requested_model']=='MiniMax-M3.1-Flash' for c in calls)
            assert all(c.get('reported_model')=='MiniMax-M3.1-Flash' for c in calls if c.get('status')=='returned')
            raw_calls[method][label] = calls
            usage[method][label] = usage_summary(calls)
        usage[method]['memory_plus_qa'] = usage_summary(raw_calls[method]['qa']+raw_calls[method]['memory'])
    by_method = {m: {(t['video_id'],t['question_id']):t for t in tasks if t['method']==m} for m in METHODS}
    for k in canonical:
        assert by_method['ogmemory'][k]['temporal_scope'] == by_method['m3_agent'][k]['temporal_scope']
    random.Random(1103).shuffle(tasks)
    EXP.mkdir(parents=True); RUN.mkdir(parents=True); (EXP/'private').mkdir(mode=0o700)
    config = EXP/'private/config.yaml'; shutil.copyfile(base_exp/'private/config.yaml', config); config.chmod(0o600)
    shutil.copyfile(Path(__file__), RUN/'driver.py')
    shutil.copyfile(ROOT/'scripts/plot_m3_stage_times.py', RUN/'plot_m3_stage_times.py')
    manifest = {'created_at':dt.datetime.now().astimezone().isoformat(), 'origin':BASE,
                'cohort':cohort, 'questions':97, 'judge_tasks':194, 'seed':1103,
                'selection':'First eight completed in preregistered execution order, including kitchen_01 two original timeouts.',
                'model':'MiniMax-M3.1-Flash', 'reasoning_effort':'medium','thinking':'adaptive','judge_output_tokens':2048,
                'qa_failures':'Original failed answers score zero; no substitution from timeout-recovery run.',
                'judging':'Separate interim common pass; method names not in prompts; final ten-video scores may differ.',
                'config_sha256':digest(config),'source_eval_sha256':digest(SOURCE/'m3/eval.py'),
                'driver_sha256':digest(RUN/'driver.py'),'source_hashes':hashes}
    save(EXP/'manifest.json',manifest);save(EXP/'inputs.json',tasks);save(EXP/'usage.json',usage);save(EXP/'timing.json',timing)
    save(EXP/'status.json',{'status':'prepared','questions':97,'judge_tasks':194})
    print('Prepared eight videos / 97 questions / 194 common grading tasks.')


def judge():
    sys.path.insert(0,str(SOURCE))
    from providers.llm.openai_llm import OpenAILLM
    from m3.eval import build_judge_llm, judge_answer
    from core.operation_deadline import operation_deadline
    assert getattr(OpenAILLM._create_completion,'_m3_usage_audit',False), 'usage hook required'
    m=json.loads((EXP/'manifest.json').read_text());tasks=json.loads((EXP/'inputs.json').read_text())
    assert digest(EXP/'private/config.yaml')==m['config_sha256']
    assert all(digest(Path(p))==h for p,h in m['source_hashes'].items())
    out=EXP/'joint_questions.jsonl'
    if out.exists():raise FileExistsError('Do not regrade successful tasks or overwrite grades')
    spec=importlib.util.spec_from_file_location('ten_driver',ROOT/'runs'/BASE/'driver.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver);gate=driver.QuotaGate()
    llm=build_judge_llm(config_path=str(EXP/'private/config.yaml'))
    errors=0
    with out.open('x') as stream:
        for i,t in enumerate(tasks):
            os.environ['M3_AUDIT_VIDEO']=t['video_id'];os.environ['M3_AUDIT_QUESTION_INDEX']=str(t['question_index'])
            started=time.monotonic()
            if t.get('error') or not t.get('predicted_answer'):
                score,reason,error=0.,'Original QA failed; no judge call.',False
            else:
                gate.wait()
                with operation_deadline(120):
                    score,reason,error=judge_answer(judge_llm=llm,question=t['question'],expected=t['expected_answer'],predicted=t['predicted_answer'])
            result={**t,'judge_score':score,'judge_reasoning':reason,'judge_error':error,'judge_seconds':time.monotonic()-started}
            stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush();errors+=bool(error)
            save(EXP/'status.json',{'status':'judging','completed':i+1,'total':len(tasks),'judge_errors':errors})
    assert all(digest(Path(p))==h for p,h in m['source_hashes'].items())
    save(EXP/'status.json',{'status':'graded' if not errors else 'graded_with_errors','completed':len(tasks),'judge_errors':errors})


def score_metrics(group):
    errors=sum(bool(r['judge_error']) for r in group)
    counts=Counter(str(r['judge_score']) for r in group)
    return {'questions':len(group),'valid_answers':sum(not r.get('error') and bool(r.get('predicted_answer')) for r in group),
            'judge_errors':errors,'mean_score':None if errors else sum(r['judge_score'] for r in group)/len(group),
            'full_credit_fraction':None if errors else sum(r['judge_score']==1 for r in group)/len(group),
            'score_counts':dict(counts)}


def summarize():
    m=json.loads((EXP/'manifest.json').read_text());g=rows(EXP/'joint_questions.jsonl')
    assert len(g)==len({(r['method'],r['video_id'],r['question_id']) for r in g})==194
    usage=json.loads((EXP/'usage.json').read_text());timing=json.loads((EXP/'timing.json').read_text())
    result={'scope':'Eight-video interim common judging; not final ten-video result.','cohort':m['cohort'],'questions_per_method':97,'methods':{}}
    for method in METHODS:
        group=[r for r in g if r['method']==method]
        result['methods'][method]={**score_metrics(group),'by_video':{v:score_metrics([r for r in group if r['video_id']==v]) for v in m['cohort']},
                                   'usage':usage[method],'time_seconds':timing['totals_seconds'][method]}
    result['ogmemory_face_clustering_seconds']=sum(v['stage_seconds']['face_clustering'] for v in timing['videos'])
    result['actual_experiment_phase_seconds']=timing['actual_experiment_phase_seconds']
    save(EXP/'comparison.json',result)
    a,b=[result['methods'][key] for key in METHODS]
    def fmt(x):return '待核验评委错误' if x is None else f'{x:.4f}'
    lines=['# 前八视频：oGMemory 与 M3-Agent 阶段性对比','',
           '固定97题、两方法共194条评分；采用本批共同评委，不拼接历史或恢复答案。',
           '模型MiniMax-M3.1-Flash / adaptive / medium；此为单独的阶段性评分，最终十视频共同评分仍将单独进行。','',
           '| 指标 | oGMemory | M3-Agent参考适配 |','|---|---:|---:|',
           f"| 自动评分均值（0/0.5/1） | {fmt(a['mean_score'])} | {fmt(b['mean_score'])} |",
           f"| 全分题比例 | {fmt(a['full_credit_fraction'])} | {fmt(b['full_credit_fraction'])} |",
           f"| 有效回答 | {a['valid_answers']}/97 | {b['valid_answers']}/97 |"]
    for label,key in [('QA输入token','prompt_tokens'),('QA未缓存输入token','uncached_prompt_tokens'),('QA输出token（含推理）','completion_tokens'),('QA输入+输出token','total_tokens'),('QA调用次数','calls')]:
        lines.append(f"| {label} | {a['usage']['qa'][key]:,} | {b['usage']['qa'][key]:,} |")
    for label,key in [('共享感知及启动（分钟）','shared_perception'),('记忆、图谱及索引（分钟）','memory_and_index'),('QA（分钟）','qa'),('各自计入共享感知后的总时间（分钟）','allocated_total')]:
        lines.append(f"| {label} | {a['time_seconds'][key]/60:.2f} | {b['time_seconds'][key]/60:.2f} |")
    lines+=['','## 分视频共同评分','','| 视频 | 题数 | oGMemory | M3-Agent |','|---|---:|---:|---:|']
    for v in m['cohort']:
        lines.append(f"| {v} | {a['by_video'][v]['questions']} | {fmt(a['by_video'][v]['mean_score'])} | {fmt(b['by_video'][v]['mean_score'])} |")
    lines+=['','## 建忆与QA合计用量（不含评委及embedding）','', '| 指标 | oGMemory | M3-Agent |','|---|---:|---:|']
    for label,key in [('记忆输入','prompt_tokens'),('记忆输出（含推理）','completion_tokens')]:
        lines.append(f"| {label} | {a['usage']['memory'][key]:,} | {b['usage']['memory'][key]:,} |")
    lines.append(f"| 已记录建忆+QA输入输出总量 | {a['usage']['memory_plus_qa']['total_tokens']:,} | {b['usage']['memory_plus_qa']['total_tokens']:,} |")
    lines+=['','## 口径与限制','',
            '- 自动评分不是人工准确率。题目加权均分包含全部97题，kitchen_01的两条首次超时计零，未排除该视频。',
            '- 两题恢复实验单独保留：均完成但诊断评分均为0，不能偷偷用恢复答案替换主结果。',
            '- QA用量完整；M3记忆阶段有1次失败请求没有usage，建忆合计只表示已记录用量，不能当成完整账单。',
            '- 所有token均为API返回值；输出已含推理token，不重复相加；token比例不等于费用比例。',
            '- 共享感知实际执行一次，比较单方法路径时分别计入同一份实测时间；两方法总时间不可直接相加。',
            f"- oGMemory的累计人脸聚类共 **{result['ogmemory_face_clustering_seconds']/60:.2f} 分钟**，各视频重复处理账号累计人脸。此项反映当前实现开销，不是各视频独立构建成本。",
            '- 记忆与索引阶段包含身份组织、图谱处理等，不是纯LLM时延；M3的细分时间无法可靠拆开。',
            '- 耗时为阶段墙钟时间，含其中的初始化、网络等待、重试；不含最终评委和阶段间驱动间隙。',
            '- 这些是同模型参考适配结果，不是M3官方权重复现；新增五视频加原三视频，不代表全M3-Bench。']
    (EXP/'REPORT.md').write_text('\n'.join(lines)+'\n')
    from plot_m3_stage_times import plot
    figures=plot(timing, ROOT/'latex/figures'/f'{NAME}_stage_times')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.8),layout='constrained')
    colors=['#3b78a8','#e69f43']
    panels=[('Common automatic score',[a['mean_score'],b['mean_score']],'Score (0 / 0.5 / 1)'),
            ('QA input tokens',[a['usage']['qa']['prompt_tokens']/1e6,b['usage']['qa']['prompt_tokens']/1e6],'Million tokens'),
            ('Processing time incl. shared input',[a['time_seconds']['allocated_total']/3600,b['time_seconds']['allocated_total']/3600],'Hours (excluding judging)')]
    if all(r['mean_score'] is not None for r in (a,b)):
        for ax,(title,values,label) in zip(axes,panels):
            ax.bar(['oGMemory','M3-Agent\nadapted'],values,color=colors,width=.58)
            for i,value in enumerate(values):ax.text(i,value,f'{value:.3f}',ha='center',va='bottom')
            ax.set_title(title,fontsize=10);ax.set_ylabel(label);ax.set_ylim(0,max(values)*1.2)
            ax.spines[['top','right']].set_visible(False)
        fig.suptitle('INTERIM: eight videos / 97 questions; original failures included',fontsize=12)
        p=ROOT/'latex/figures'/f'{NAME}_comparison'
        for ext in ('pdf','png'):fig.savefig(p.with_suffix('.'+ext),dpi=180,bbox_inches='tight');figures.append(p.with_suffix('.'+ext))
    plt.close(fig)
    save(EXP/'plot_receipt.json',{'artifacts':{str(p.relative_to(ROOT)):digest(p) for p in figures},'comparison_sha256':digest(EXP/'comparison.json')})
    calls=rows(EXP/'model-usage.jsonl')
    save(EXP/'validation.json',{'judge_records':len(g),'actual_judge_calls':len(calls),'judge_errors':sum(bool(r['judge_error']) for r in g),
         'judge_unknown_usage':sum(not r.get('usage') for r in calls),'source_predictions_unchanged':all(digest(Path(p))==h for p,h in m['source_hashes'].items()),
         'all_judge_model_settings_match':all(r.get('requested_model')=='MiniMax-M3.1-Flash' and r.get('reported_model')=='MiniMax-M3.1-Flash' and r.get('reasoning_effort')=='medium' and r.get('max_tokens')==2048 for r in calls)})
    save(EXP/'status.json',{'status':'finished' if not any(r['judge_error'] for r in g) else 'finished_with_judge_errors','questions':97,'judge_records':194})
    print('\n'.join(lines[:20]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','judge','summarize','run']);args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='judge':judge()
    elif args.action=='summarize':summarize()
    else:judge();summarize()


if __name__=='__main__':main()
