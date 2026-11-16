#!/usr/bin/env python3
import importlib.util, os, subprocess, sys
from pathlib import Path
ROOT=Path('/home/yuanjian/Development/memory-eval'); NAME='m3_reduce_face_one_w8_budget_20261112'; BASE=ROOT/'runs'/NAME; EXP=ROOT/'experiments'/NAME; PY=ROOT/'oGMemory/.venv/bin/python'
BUDGETS=[9000,10000,12000,14000,16000]
for b in BUDGETS:
 q=BASE/(f'qa_budget_{b}_20261113' if b!=12000 else 'qa_rerun_20261113')/'og_qa/study_09/m3_qa_results.jsonl'
 if not q.exists() or sum(1 for x in q.read_text().splitlines() if x.strip())<12: print('not ready',b); continue
 src=(BASE/'qa_driver_generic.py').read_text().replace("path=EXP/'judgments.jsonl'",f"path=EXP/'judgments_qa_budget_{b}_20261113.jsonl'")
 driver=BASE/f'judge_budget_{b}_20261113.py'; driver.write_text(src)
 wrapper=BASE/f'judge_budget_wrapper_{b}_20261113.py'
 run='qa_rerun_20261113' if b==12000 else f'qa_budget_{b}_20261113'
 wrapper.write_text(f'''import importlib.util,os,sys\nfrom pathlib import Path\nROOT=Path({str(ROOT)!r}); BASE=ROOT/'runs'/'{NAME}'\ns=importlib.util.spec_from_file_location('d',BASE/'judge_budget_{b}_20261113.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\nm.RUN=BASE/'{run}';m.CONFIG=m.EXP/'private/config_qa_{"12000" if b==12000 else str(b)}_20261113.yaml'\nos.environ['M3_USAGE_LOG']=str(m.EXP/'usage-judge-budget-{b}-20261113.jsonl')\nsys.path.insert(0,str(m.SOURCE));sys.path.insert(0,str(BASE/'_instrumentation'))\nsi=importlib.util.spec_from_file_location('audit',BASE/'_instrumentation/sitecustomize.py');a=importlib.util.module_from_spec(si);si.loader.exec_module(a)\nraise SystemExit(m.judge() or 0)\n''')
 env=os.environ.copy();env['M3_EVAL_ROOT']=str(ROOT);env['PYTHONUNBUFFERED']='1'
 log=(EXP/'private'/f'judge_budget_{b}_20261113.log').open('ab',buffering=0)
 p=subprocess.run([str(PY),str(wrapper)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 print('judged',b,p.returncode)
