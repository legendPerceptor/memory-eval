import importlib.util
from pathlib import Path
ROOT=Path('/home/yuanjian/Development/memory-eval'); BASE=ROOT/'runs/m3_reduce_face_one_w8_budget_20261112'
s=importlib.util.spec_from_file_location('d',BASE/'qa_judge_20261113.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.RUN=BASE/'qa_rerun_20261113'; m.CONFIG=m.EXP/'private/config_qa_12000.yaml'
import os,sys
os.environ['M3_USAGE_LOG']=str(m.EXP/'usage-judge-qa-rerun-20261113.jsonl')
sys.path.insert(0,str(m.SOURCE));sys.path.insert(0,str(BASE/'_instrumentation'))
import importlib.util
si=importlib.util.spec_from_file_location('m3_audit_sitecustomize',BASE/'_instrumentation/sitecustomize.py');sm=importlib.util.module_from_spec(si);si.loader.exec_module(sm)
raise SystemExit(m.judge() or 0)
