import importlib.util, os, sys
from pathlib import Path
ROOT=Path('/home/yuanjian/Development/memory-eval'); os.environ['M3_EVAL_ROOT']=str(ROOT)
NAME='m3_parallel4_paper8_20261113'; RUN,EXP=ROOT/'runs'/NAME,ROOT/'experiments'/NAME
idx=sys.argv[1]
spec=importlib.util.spec_from_file_location('d',ROOT/'scripts'/'run_m3_parallel4_paper8_20261113.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sys.path.insert(0,str(m.SOURCE))
m._recover_key()
from m3.qa import main_react_qa
from m3.config import load_m3_runtime_config
from m3.control_adapter import OGMemoryControlAdapter
os.environ['OGMEM_GRAPH_CONNECTION_STRING']=load_m3_runtime_config(str(m.CONFIG)).graph_connection_string
gate=m.h.QuotaGate(); orig=OGMemoryControlAdapter.run
def g(self,ctx,task): gate.wait(); return orig(self,ctx,task)
OGMemoryControlAdapter.run=g
out=RUN/f'og_qa_sample{idx}_kitchen01'/'m3_qa_results.jsonl'
out.parent.mkdir(parents=True,exist_ok=True)
rc=main_react_qa(['--config',str(m.CONFIG),'--annotations',str(m.h.ANNOTATIONS),'--video','kitchen_01',
 '--graph-store','sql','--account-id',m.ACCOUNT,'--output-jsonl',str(out),'--temperature','0',
 '--max-iterations','6','--timeout-seconds','180','--top-k','8','--candidate-k','20','--transcript-top-k','20',
 '--transcript-fusion','rrf','--context-k','8','--include-neighbors','1','--temporal-policy','current_clip'])
print('sample',idx,'rc',rc)
