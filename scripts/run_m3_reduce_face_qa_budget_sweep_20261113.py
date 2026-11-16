#!/usr/bin/env python3
"""After the 4-worker baseline finishes, rerun QA on the frozen graph at budgets."""
import importlib.util, json, os, subprocess, sys, time
from pathlib import Path
import yaml
ROOT=Path('/home/yuanjian/Development/memory-eval')
NAME='m3_reduce_face_qa_budget_20261113'
EXP=ROOT/'experiments'/NAME; BASE=ROOT/'runs'/NAME
PYTHON=ROOT/'oGMemory/.venv/bin/python'
BUDGETS=[9000,10000,12000,14000,16000]

def load_driver():
    spec=importlib.util.spec_from_file_location('baseline_driver',BASE/'driver.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def write_qa_wrapper(m,b):
    cfg=EXP/'private'/f'config_qa_{b}.yaml'
    base=yaml.safe_load((EXP/'private/config.yaml').read_text())
    base.setdefault('m3',{})['qa_context']={'enabled':True,'max_input_tokens':b}
    cfg.write_text(yaml.safe_dump(base,allow_unicode=True)); cfg.chmod(0o600)
    qrun=BASE/f'qa_budget_{b}'
    wrapper=BASE/f'qa_budget_{b}.py'
    wrapper.write_text(f'''import importlib.util, os, sys\nfrom pathlib import Path\nROOT=Path({str(ROOT)!r}); BASE=ROOT/'runs'/'{NAME}'; EXP=ROOT/'experiments'/'{NAME}'\ns=importlib.util.spec_from_file_location('d',BASE/'driver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\nm.RUN=BASE/'qa_budget_{b}'; m.CONFIG=EXP/'private/config_qa_{b}.yaml'\nos.environ.update({{'M3_USAGE_LOG':str(EXP/'usage-qa-budget-{b}-study_09.jsonl'),'QA_EXPECTED_BUDGET':str({b}),'PYTHONUNBUFFERED':'1'}})\nsys.path.insert(0,str(m.SOURCE)); sys.path.insert(0,str(BASE/'_instrumentation'))\nsi=importlib.util.spec_from_file_location('audit',BASE/'_instrumentation/sitecustomize.py'); a=importlib.util.module_from_spec(si); si.loader.exec_module(a)\nraise SystemExit(m.worker('qa','study_09') or 0)\n''')
    return cfg,qrun,wrapper

def judge(m,b,qrun,cfg):
    sys.path.insert(0,str(m.SOURCE))
    from m3.eval import build_judge_llm, judge_answer
    from m3.qa import load_questions
    llm=build_judge_llm(config_path=str(cfg))
    qs=[q for q in load_questions(m.h.ANNOTATIONS) if q['video_id']=='study_09']
    answers={r['question_id']:r for r in m.h.rows(qrun/'og_qa/study_09/m3_qa_results.jsonl')}
    out=EXP/f'judgments_qa_budget_{b}.jsonl'
    if out.exists(): return
    with out.open('w') as fh:
        for q in qs:
            r=answers.get(q['question_id'],{})
            if not r.get('predicted_answer') or r.get('error'):
                score,reason=0.0,'Missing or failed answer'
                err=False
            else:
                score,reason,err=judge_answer(judge_llm=llm,question=q['question'],expected=q['answer'],predicted=r['predicted_answer'])
                if err: raise RuntimeError(f'judge failed for {q["question_id"]}')
            fh.write(json.dumps({'video_id':'study_09','question_id':q['question_id'],'question':q['question'],
                'expected_answer':q['answer'],'predicted_answer':r.get('predicted_answer',''),'error':r.get('error',''),
                'score':score,'reason':reason},ensure_ascii=False)+'\n'); fh.flush()

def main():
    status=EXP/'status.json'
    while True:
        if status.exists():
            s=json.loads(status.read_text()).get('status')
            if s in ('finished','judge_failed','needs_review'): break
        time.sleep(30)
    if json.loads(status.read_text()).get('status')!='finished':
        (EXP/'qa_budget_sweep_status.json').write_text(json.dumps({'status':'blocked_by_baseline','baseline':json.loads(status.read_text())},indent=2,ensure_ascii=False)+'\n')
        return 2
    m=load_driver(); rows=[]
    for b in BUDGETS:
        cfg,qrun,wrapper=write_qa_wrapper(m,b)
        result=BASE/f'qa_budget_{b}.json'
        if not (qrun/'og_qa/study_09/m3_qa_results.jsonl').exists():
            env=os.environ.copy(); env.update({'M3_EVAL_ROOT':str(ROOT),'M3_QA_DB_PASSWORD':'m3qa_20261113_isolated','M3_QA_DB_PORT':'31876','PYTHONUNBUFFERED':'1'})
            log=(EXP/'private'/f'qa_budget_{b}.log').open('ab',buffering=0)
            p=subprocess.run([str(PYTHON),str(wrapper)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            (EXP/'private'/f'qa_budget_{b}_exit.txt').write_text(str(p.returncode)+'\n')
        qpath=qrun/'og_qa/study_09/m3_qa_results.jsonl'
        qrows=m.h.rows(qpath)
        if len(qrows)!=12:
            raise RuntimeError(f'budget {b}: expected 12 QA rows, got {len(qrows)}')
        judge(m,b,qrun,cfg)
        js=m.h.rows(EXP/f'judgments_qa_budget_{b}.jsonl')
        usage=m.h.rows(EXP/f'usage-qa-budget-{b}-study_09.jsonl')
        rows.append({'budget':b,'qa_exit':int((EXP/'private'/f'qa_budget_{b}_exit.txt').read_text()) if (EXP/'private'/f'qa_budget_{b}_exit.txt').exists() else 0,
            'questions':len(qrows),'errors':sum(bool(x.get('error')) for x in qrows),'score':sum(x.get('score',0) for x in js),
            'accuracy':sum(x.get('score',0) for x in js)/12,'input_tokens':sum((x.get('usage') or {}).get('prompt_tokens',0) or 0 for x in usage),
            'output_tokens':sum((x.get('usage') or {}).get('completion_tokens',0) or 0 for x in usage)})
        (EXP/'qa_budget_sweep_results.json').write_text(json.dumps({'points':rows},indent=2,ensure_ascii=False)+'\n')
    (EXP/'qa_budget_sweep_status.json').write_text(json.dumps({'status':'finished','points':rows},indent=2,ensure_ascii=False)+'\n')

if __name__=='__main__': raise SystemExit(main())
