#!/usr/bin/env python3
"""Isolated eight-video reduce_face run, frozen dev plus working-tree patch."""
import hashlib, importlib.util, json, os, random, shutil, subprocess, sys, tarfile, time
from pathlib import Path
import yaml
ROOT = Path(os.environ.get('M3_EVAL_ROOT', Path(__file__).resolve().parent.parent))
NAME = 'm3_reduce_face_8_reasoning_v2_20261112'
RUN, EXP = ROOT/'runs'/NAME, ROOT/'experiments'/NAME
SOURCE = RUN/'_source/oGMemory'
CONFIG = EXP/'private/config.yaml'
PYTHON = ROOT/'oGMemory/.venv/bin/python'
VIDEOS = ['study_09','kitchen_01','bedroom_05','study_23','kitchen_23','study_05','gym_03','bedroom_04']
ACCOUNT = 'm3-reduce-face-8-reasoning-v2-20261112'
DB = 'ogmem_m3_reduce_face_8_reasoning_v2_20261112'
spec = importlib.util.spec_from_file_location('helper', ROOT/'scripts/run_m3_ten_video.py')
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
for key in ('ROOT','NAME','RUN','EXP','SOURCE','CONFIG','PYTHON','ACCOUNT','DB'): setattr(h,key,globals()[key])

def prepare():
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import parse_dsn, make_dsn
    if not os.environ.get('MINIMAX_API_KEY'): raise RuntimeError('Missing API key')
    if RUN.exists() or EXP.exists(): raise FileExistsError('Experiment already exists')
    rev = subprocess.check_output(['git','-C',str(ROOT/'oGMemory'),'rev-parse','HEAD'],text=True).strip()
    assert subprocess.check_output(['git','-C',str(ROOT/'oGMemory'),'branch','--show-current'],text=True).strip() == 'reduce_face'
    RUN.mkdir(parents=True); (EXP/'private').mkdir(parents=True,mode=0o700); SOURCE.mkdir(parents=True)
    with (RUN/'source.tar').open('wb') as f: subprocess.run(['git','-C',str(ROOT/'oGMemory'),'archive',rev],stdout=f,check=True)
    with tarfile.open(RUN/'source.tar') as t: t.extractall(SOURCE,filter='data')
    patch = subprocess.check_output(['git','-C',str(ROOT/'oGMemory'),'diff','--binary','HEAD'])
    (RUN/'source.patch').write_bytes(patch)
    subprocess.run(['patch','--batch','-p1','-i',str(RUN/'source.patch')],cwd=SOURCE,check=True)
    for path in ('m3/config.py','m3/llm_factory.py','providers/llm/minimax_llm.py','m3/face_clustering.py','m3/pipeline/stages.py'):
        if (SOURCE/path).read_bytes() != (ROOT/'oGMemory'/path).read_bytes():
            raise RuntimeError(f'Frozen source mismatch: {path}')
    shutil.copytree(ROOT/'scripts/instrumentation',RUN/'_instrumentation',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(ROOT/'scripts/run_context_experiment.py',RUN/'quota_api.py')
    shutil.copyfile(Path(__file__),RUN/'driver.py')
    config=yaml.safe_load((ROOT/'experiments/m3_ten_fullinterval_20261102/private/config.yaml').read_text())
    params=parse_dsn(config['storage']['connection_string'])
    conn=psycopg2.connect(**params); conn.autocommit=True
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT 1 FROM pg_database WHERE datname=%s',(DB,))
            if cur.fetchone(): raise RuntimeError('Database exists; refusing overwrite')
            cur.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(DB)))
    finally: conn.close()
    for role in ('storage','vector_db'):
        params=parse_dsn(config[role]['connection_string']); params['dbname']=DB
        config[role]['connection_string']=make_dsn(**params)
    with psycopg2.connect(config['storage']['connection_string']) as conn:
        with conn.cursor() as cur: cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
    config['identity']['account_id']=ACCOUNT
    config['m3'].pop('qa_context',None)
    h.private_yaml(CONFIG,config)
    (EXP/'config.redacted.yaml').write_text(yaml.safe_dump(h.redact(config),allow_unicode=True))
    sys.path.insert(0,str(SOURCE)); from m3.qa import load_questions
    qs=[q for q in load_questions(h.ANNOTATIONS) if q['video_id'] in VIDEOS]
    assert len(qs)==97
    h.save(EXP/'manifest.json',{'source_commit':rev,'patch_sha256':h.digest(RUN/'source.patch'),
        'driver_sha256':h.digest(RUN/'driver.py'),'config_sha256':h.digest(CONFIG),
        'annotations_sha256':h.digest(h.ANNOTATIONS),'videos':VIDEOS,'questions':len(qs),
        'account':ACCOUNT,'database':DB,'safety_stop_at_unix':time.time()+48*3600,
        'design':'fresh dev reduce_face pipeline and native QA; no fact index or person grounding; not paired dev baseline',
        'worker_settings':'video 1, memory 4, QA 1; frames 30 extraction / 16 model'})
    h.save(EXP/'status.json',{'status':'prepared'})

def verify_model_settings():
    from m3.llm_factory import build_m3_llm
    for role in ('episodic_llm', 'qa_llm', 'judge_llm'):
        llm = build_m3_llm(config_path=str(CONFIG), config_role=role)
        params = llm._completion_parameters(0.2)
        expected_max = 2048 if role == 'judge_llm' else 8192
        if (llm._model != 'MiniMax-M3.1-Flash'
            or params.get('max_completion_tokens') != expected_max
            or params.get('reasoning_effort') != 'medium'
            or params.get('extra_body', {}).get('thinking') != {'type': 'adaptive'}):
            raise RuntimeError(f'{role} model request configuration mismatch')

def worker(phase,video):
    sys.path.insert(0,str(SOURCE)); verify_model_settings(); gate=h.QuotaGate(); gate.wait(); h.instrument(gate,memory=phase=='pipeline')
    work=RUN/video; work.mkdir(exist_ok=True)
    if phase=='pipeline':
        from m3.run_video_pipeline import main
        return main(['--video',f'/data/m3_data/videos/robot/{video}.mp4','--video-id',video,'--seconds','0',
          '--work-dir',str(work),'--config',str(CONFIG),'--pipeline',str(SOURCE/'config/m3/pipelines/video_episodic_full.yaml'),
          '--graph-store','sql','--account-id',ACCOUNT,'--no-smart-resume','--max-frames','30','--episodic-max-frames','16',
          '--disable-stage','qa','--set','episodic_memory.params.workers=4','--set','voice_asr.params.device=cuda',
          '--set','speaker_diarization.params.device=cuda','--set','speaker_embedding.params.device=cuda:0'])
    if phase=='qa':
        from m3.qa import main_react_qa
        from m3.control_adapter import OGMemoryControlAdapter
        from m3.config import load_m3_runtime_config
        os.environ['OGMEM_GRAPH_CONNECTION_STRING']=load_m3_runtime_config(str(CONFIG)).graph_connection_string
        original=OGMemoryControlAdapter.run
        def guarded(self,ctx,task): gate.wait(); return original(self,ctx,task)
        OGMemoryControlAdapter.run=guarded
        output=RUN/'og_qa'/video/'m3_qa_results.jsonl'; output.parent.mkdir(parents=True,exist_ok=True)
        if output.exists(): raise FileExistsError('Preserve first answers')
        return main_react_qa(['--config',str(CONFIG),'--annotations',str(h.ANNOTATIONS),'--video',video,
          '--graph-store','sql','--account-id',ACCOUNT,'--output-jsonl',str(output),'--temperature','0',
          '--max-iterations','6','--timeout-seconds','180','--top-k','8','--candidate-k','20',
          '--transcript-top-k','20','--transcript-fusion','rrf','--context-k','8','--include-neighbors','1',
          '--temporal-policy','current_clip'])
    raise ValueError(phase)

def judge():
    sys.path.insert(0,str(SOURCE)); from m3.qa import load_questions
    from m3.eval import build_judge_llm, judge_answer
    verify_model_settings(); gate=h.QuotaGate(); h.instrument(gate); llm=build_judge_llm(config_path=str(CONFIG))
    qs=[q for q in load_questions(h.ANNOTATIONS) if q['video_id'] in VIDEOS]
    tasks=[]
    for video in VIDEOS:
        answers={r['question_id']:r for r in h.rows(RUN/'og_qa'/video/'m3_qa_results.jsonl')}
        for q in (x for x in qs if x['video_id']==video):
            r=answers.get(q['question_id'],{})
            tasks.append({'video_id':video,'question_id':q['question_id'],'question':q['question'],
              'expected_answer':q['answer'],'predicted_answer':r.get('predicted_answer',''),'error':r.get('error','')})
    random.Random(20261111).shuffle(tasks); h.save(EXP/'judging_inputs.json',tasks)
    path=EXP/'judgments.jsonl'; existing=h.rows(path); done={(x['video_id'],x['question_id']) for x in existing}
    with path.open('a') as out:
        for t in tasks:
            if (t['video_id'],t['question_id']) in done: continue
            if not t['predicted_answer'] or t['error']: score,reason,error=0.,'Missing or failed answer',False
            else:
                gate.wait(); score,reason,error=judge_answer(judge_llm=llm,question=t['question'],expected=t['expected_answer'],predicted=t['predicted_answer'])
            out.write(json.dumps({**t,'score':score,'reason':reason,'judge_error':bool(error)},ensure_ascii=False)+'\n'); out.flush(); os.fsync(out.fileno())
            if error: raise RuntimeError('Judge error, preserved for review')
    return 0

def run():
    manifest=json.loads((EXP/'manifest.json').read_text()); assert h.digest(CONFIG)==manifest['config_sha256']
    assert h.digest(h.ANNOTATIONS)==manifest['annotations_sha256']; history=[]
    for v in VIDEOS:
        for phase in ('pipeline','qa'):
            receipt=EXP/'stages'/f'{v}-{phase}.json'
            if receipt.exists():
                r=json.loads(receipt.read_text())
                if r['exit_code']: raise RuntimeError('Prior failed stage needs review')
            else:
                h.save(EXP/'status.json',{'status':'running','pid':os.getpid(),'video':v,'phase':phase,'history':history})
                start=time.time(); env=h.environment(phase,v)
                env.update({'OGMEM_M3_MAX_WORKERS':'1','OGMEM_M3_FACE_WORKERS':'1','OGMEM_EMBED_MAX_CONCURRENCY':'1','PYTHONUNBUFFERED':'1'})
                with (EXP/'private'/f'{v}-{phase}.log').open('a') as log:
                    p=subprocess.run([str(PYTHON),str(Path(__file__)), 'worker',phase,v],cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT)
                r={'video':v,'phase':phase,'exit_code':p.returncode,'seconds':time.time()-start}; h.save(receipt,r)
                if p.returncode:
                    h.save(EXP/'status.json',{'status':'needs_review','failed':r,'history':history}); return p.returncode
            history.append(r)
    h.save(EXP/'status.json',{'status':'judging','history':history})
    with (EXP/'private/judge.log').open('a') as log:
        p=subprocess.run([str(PYTHON),str(Path(__file__)),'judge'],cwd=SOURCE,env=h.environment('judge','all'),stdout=log,stderr=subprocess.STDOUT)
    h.save(EXP/'status.json',{'status':'finished' if p.returncode==0 else 'judge_failed','history':history,'judge_exit_code':p.returncode})
    return p.returncode

if __name__=='__main__':
    action=sys.argv[1]; ret=worker(sys.argv[2],sys.argv[3]) if action=='worker' else globals()[action]()
    raise SystemExit(ret or 0)
