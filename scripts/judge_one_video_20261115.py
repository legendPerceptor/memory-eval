import importlib.util, json, os, sys
from pathlib import Path
ROOT=Path('/home/yuanjian/Development/memory-eval'); os.environ['M3_EVAL_ROOT']=str(ROOT)
NAME='m3_parallel4_paper8_20261113'; RUN,EXP=ROOT/'runs'/NAME,ROOT/'experiments'/NAME
spec=importlib.util.spec_from_file_location('d',ROOT/'scripts'/'run_m3_parallel4_paper8_20261113.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sys.path.insert(0,str(m.SOURCE))
from m3.qa import load_questions
from m3.eval import build_judge_llm, judge_answer
from m3.qa_results import latest_answers
m._recover_key(); llm=build_judge_llm(config_path=str(m.CONFIG))
video='kitchen_01'
qpath=Path(sys.argv[1]); outpath=Path(sys.argv[2])
qs=[q for q in load_questions(m.h.ANNOTATIONS) if q['video_id']==video]
latest=latest_answers(qpath)
gate=m.h.QuotaGate(); tot=0;n=0
with outpath.open('w') as fh:
    for q in qs:
        r=latest.get((video,q['question_id']),{})
        pred=r.get('predicted_answer',''); err=r.get('error','')
        if not pred or err: score,reason=0.0,'missing'
        else:
            gate.wait(); score,reason,_=judge_answer(judge_llm=llm,question=q['question'],expected=q['answer'],predicted=pred)
        tot+=score;n+=1
        fh.write(json.dumps({'qid':q['question_id'],'score':score,'pred':pred,'gold':q['answer'],'reason':reason},ensure_ascii=False)+'\n')
print(f'{video} RERUN accuracy: {tot:.2f}/{n} = {100*tot/n:.2f}%')
