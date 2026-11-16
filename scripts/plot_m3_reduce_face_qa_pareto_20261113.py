import json
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path('/home/yuanjian/Development/memory-eval'); EXP=ROOT/'experiments/m3_reduce_face_one_w8_budget_20261112'; BASE=ROOT/'runs/m3_reduce_face_one_w8_budget_20261112'
rows=[]
for b in [9000,10000,12000,14000,16000]:
 d=EXP/f'judgments_qa_budget_{b}_20261113.jsonl'
 if b==12000 and not d.exists(): d=EXP/'judgments_qa_rerun_20261113.jsonl'
 if not d.exists(): continue
 j=[json.loads(x) for x in d.read_text().splitlines() if x.strip()]; u=EXP/(f'usage-judge-budget-{b}-20261113.jsonl' if b!=12000 else 'usage-qa-rerun-study_09.jsonl')
 # QA usage is separate from judge usage for 12k; other points use matching QA audit.
 uq=EXP/(f'usage-qa-budget-{b}-study_09.jsonl' if b!=12000 else 'usage-qa-rerun-study_09.jsonl'); z=[json.loads(x) for x in uq.read_text().splitlines() if x.strip()]
 q=BASE/(f'qa_budget_{b}_20261113' if b!=12000 else 'qa_rerun_20261113')/'og_qa/study_09/m3_qa_results.jsonl'
 qa=[json.loads(x) for x in q.read_text().splitlines() if x.strip()]
 errors=[x for x in qa if x.get('error') or x.get('status') in ('failed','error')]
 valid=(len(qa)==12 and not errors)
 rows.append({'budget':b,'input_tokens':sum((x.get('usage') or {}).get('prompt_tokens',0) or 0 for x in z),'output_tokens':sum((x.get('usage') or {}).get('completion_tokens',0) or 0 for x in z),'score':sum(x.get('score',0) for x in j) if valid else None,'accuracy':(sum(x.get('score',0) for x in j)/12) if valid else None,'answered':len(qa)-len(errors),'errors':len(errors),'valid':valid})
rows.sort(key=lambda x:x['input_tokens']); pareto=[]; best=-1
for r in rows:
 if r['valid'] and r['score']>best: pareto.append(r);best=r['score']
(EXP/'qa_budget_sweep_results.json').write_text(json.dumps({'points':rows,'pareto_front':pareto},indent=2,ensure_ascii=False)+'\n')
valid_rows=[r for r in rows if r['valid']]
invalid_rows=[r for r in rows if not r['valid']]
fig,ax=plt.subplots(figsize=(7,5));ax.plot([r['input_tokens'] for r in valid_rows],[r['accuracy']*100 for r in valid_rows],'o-',label='complete budget points');ax.scatter([r['input_tokens'] for r in pareto],[r['accuracy']*100 for r in pareto],s=90,label='Pareto front',zorder=3)
if invalid_rows:
 ax.scatter([r['input_tokens'] for r in invalid_rows],[0 for r in invalid_rows],marker='x',s=80,label='incomplete / excluded',zorder=3)
for r in valid_rows: ax.annotate(f"{r['budget']/1000:.0f}k",(r['input_tokens'],r['accuracy']*100),xytext=(5,5),textcoords='offset points')
for r in invalid_rows: ax.annotate(f"{r['budget']/1000:.0f}k (invalid)",(r['input_tokens'],0),xytext=(5,5),textcoords='offset points')
ax.set_xlabel('QA input tokens');ax.set_ylabel('Score / accuracy (%)');ax.set_title('study_09 QA budget–accuracy tradeoff');ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(EXP/'qa_budget_pareto_20261113.png',dpi=180);print(json.dumps({'points':rows,'pareto_front':pareto},indent=2))
