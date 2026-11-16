#!/usr/bin/env python3
"""Standalone QA rerun on completed w8 memory graph; never rebuilds memory."""
import importlib.util, os, shutil, sys
from pathlib import Path
import yaml
ROOT=Path('/home/yuanjian/Development/memory-eval')
NAME='m3_reduce_face_one_w8_budget_20261112'
EXP=ROOT/'experiments'/NAME
BASE=ROOT/'runs'/NAME
QA_RUN=BASE/'qa_rerun_20261113'
CONFIG=EXP/'private/config_qa_12000.yaml'
PYTHON=ROOT/'oGMemory/.venv/bin/python'
# Reuse frozen source, instrumentation, database/account, and guarded QA worker.
spec=importlib.util.spec_from_file_location('recovery', BASE/'qa_driver_12000.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
if QA_RUN.exists(): raise SystemExit('QA rerun directory already exists')
config=yaml.safe_load((EXP/'private/config.yaml').read_text())
config['m3']['qa_context']['max_input_tokens']=12000
CONFIG.write_text(yaml.safe_dump(config,allow_unicode=True))
QA_RUN.mkdir(parents=True)
mod.RUN=QA_RUN
mod.CONFIG=CONFIG
# worker() writes QA output under QA_RUN and usage under the original EXP.
env=mod.h.environment('qa_rerun','study_09')
env.update({'PYTHONUNBUFFERED':'1','M3_USAGE_LOG':str(EXP/'usage-qa-rerun-study_09.jsonl')})
if not os.environ.get('MINIMAX_API_KEY'):
    raise SystemExit('MINIMAX_API_KEY must be present in launcher environment')
os.environ.update(env)
sys.path.insert(0, str(QA_RUN/'_instrumentation')) if (QA_RUN/'_instrumentation').exists() else None
# Load the frozen instrumentation explicitly in this launcher process.
sys.path.insert(0, str(mod.SOURCE))
sys.path.insert(0, str(BASE/'_instrumentation'))
import sitecustomize  # noqa: F401
raise SystemExit(mod.worker('qa','study_09') or 0)
