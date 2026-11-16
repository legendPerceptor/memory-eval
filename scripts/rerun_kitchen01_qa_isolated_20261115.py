#!/usr/bin/env python3
"""Isolated kitchen_01 QA rerun (read-only on graph, separate output dir).

Does NOT touch the live experiment run or existing answers. Uses the same
config/graph; writes to runs/.../og_qa_rerun_kitchen01/. Greedy (temperature 0).
Purpose: test whether kitchen_01's low accuracy is stable (systematic) or variance.
"""
import importlib.util, json, os, sys
from pathlib import Path
ROOT = Path('/home/yuanjian/Development/memory-eval')
os.environ['M3_EVAL_ROOT'] = str(ROOT)
NAME='m3_parallel4_paper8_20261113'
RUN, EXP = ROOT/'runs'/NAME, ROOT/'experiments'/NAME
spec=importlib.util.spec_from_file_location('d', ROOT/'scripts'/'run_m3_parallel4_paper8_20261113.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sys.path.insert(0, str(m.SOURCE))
m._recover_key(); m.verify_model_settings()
from m3.qa import main_react_qa
from m3.config import load_m3_runtime_config
from m3.control_adapter import OGMemoryControlAdapter
os.environ['OGMEM_GRAPH_CONNECTION_STRING'] = load_m3_runtime_config(str(m.CONFIG)).graph_connection_string
gate = m.h.QuotaGate()
orig = OGMemoryControlAdapter.run
def guarded(self, ctx, task):
    gate.wait()
    return orig(self, ctx, task)
OGMemoryControlAdapter.run = guarded
out = RUN/'og_qa_rerun_kitchen01'/'m3_qa_results.jsonl'
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    print('output exists, abort to preserve'); raise SystemExit(3)
rc = main_react_qa([
    '--config', str(m.CONFIG), '--annotations', str(m.h.ANNOTATIONS), '--video','kitchen_01',
    '--graph-store','sql','--account-id',m.ACCOUNT,'--output-jsonl',str(out),
    '--temperature','0','--max-iterations','6','--timeout-seconds','180',
    '--top-k','8','--candidate-k','20','--transcript-top-k','20',
    '--transcript-fusion','rrf','--context-k','8','--include-neighbors','1',
    '--temporal-policy','current_clip'])
print('rerun rc=',rc)
