#!/usr/bin/env python3
"""Eight-video paper cohort, 4-way parallel, fresh PG database.

Goal: on branch ``rychkov/m3-parallel`` run the eight videos from the LaTeX
paper table (97 questions) with MiniMax-M3.1-Flash, four-way in-stage
parallelism, into a brand-new pgvector database, then judge with the same
model. Reproduce (as closely as this branch allows) the previously recorded
oGMemory accuracy, and record token usage plus per-stage wall time.

This branch dropped ``fact_index`` and has a different ``qa_context`` schema
than the frozen source that produced the paper 57.2%, so this is an
*approximate* re-run on the current code, not a byte-identical reproduction.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile
import time

import yaml

ROOT = Path(os.environ.get('M3_EVAL_ROOT', Path(__file__).resolve().parent.parent))
NAME = 'm3_parallel4_paper8_20261113'
RUN, EXP = ROOT / 'runs' / NAME, ROOT / 'experiments' / NAME
SOURCE = RUN / '_source/oGMemory'
CONFIG = EXP / 'private/config.yaml'
PYTHON = ROOT / 'oGMemory/.venv/bin/python'
VIDEOS = ['study_09', 'kitchen_01', 'bedroom_05', 'study_23',
          'kitchen_23', 'study_05', 'gym_03', 'bedroom_04']
ACCOUNT = 'm3-parallel4-paper8-20261113'
DB = 'ogmem_m3_parallel4_paper8_20261113'
TOKENIZER = ROOT / 'runs/context_assets/qwen-omni-tokenizer.json'
KEY_SOURCE = ROOT / 'experiments/m3_reduce_face_qa_budget_20261113/private/config.yaml'

spec = importlib.util.spec_from_file_location('helper', ROOT / 'scripts/run_m3_ten_video.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
for key in ('ROOT', 'NAME', 'RUN', 'EXP', 'SOURCE', 'CONFIG', 'PYTHON', 'ACCOUNT', 'DB'):
    setattr(h, key, globals()[key])


def real_api_key():
    key = os.environ.get('MINIMAX_API_KEY', '').strip()
    if key and not key.startswith('${'):
        return key
    cfg = yaml.safe_load(KEY_SOURCE.read_text())
    key = (cfg.get('llm', {}).get('api_key') or '')
    if not key or key.startswith('${'):
        raise RuntimeError('No real MINIMAX_API_KEY available')
    return key


def db_settings():
    pw = os.environ.get('M3_DB_PASSWORD')
    if not pw:
        pw = (ROOT / 'runs/.m3p4_pgpass_20261113').read_text().strip()
    port = os.environ.get('M3_DB_PORT', '31877')
    return {'host': '127.0.0.1', 'port': port, 'user': 'ogmem', 'dbname': DB, 'password': pw}


def qa_context(max_input_tokens=24000):
    """Current-branch ContextBudget schema only (fields it actually accepts)."""
    return {
        'enabled': True,
        'context_window': 32768,
        'max_input_tokens': max_input_tokens,
        'safety_tokens': 2048,
        'max_top_k': 16,
        'max_candidates': 32,
        'max_result_nodes': 24,
        'max_neighbor_clips': 64,
        'max_search_calls': 8,
        'max_selected_nodes': 32,
        'max_passages_per_node': 12,
        'tokenizer_path': str(TOKENIZER),
        'tokenizer_multiplier': 1.15,
    }


def prepare(max_input_tokens=24000):
    from psycopg2.extensions import make_dsn
    if RUN.exists() or EXP.exists():
        raise FileExistsError('Experiment already exists; refusing overwrite')
    key = real_api_key()
    branch = subprocess.check_output(['git', '-C', str(ROOT / 'oGMemory'), 'branch',
                                      '--show-current'], text=True).strip()
    if branch != 'rychkov/m3-parallel':
        raise RuntimeError(f'Unexpected branch {branch!r}; expected rychkov/m3-parallel')
    rev = subprocess.check_output(['git', '-C', str(ROOT / 'oGMemory'), 'rev-parse', 'HEAD'],
                                  text=True).strip()
    RUN.mkdir(parents=True)
    (EXP / 'private').mkdir(parents=True, mode=0o700)
    SOURCE.mkdir(parents=True)
    with (RUN / 'source.tar').open('wb') as fh:
        subprocess.run(['git', '-C', str(ROOT / 'oGMemory'), 'archive', rev], stdout=fh, check=True)
    with tarfile.open(RUN / 'source.tar') as bundle:
        bundle.extractall(SOURCE, filter='data')
    patch = subprocess.check_output(['git', '-C', str(ROOT / 'oGMemory'), 'diff', '--binary', 'HEAD'])
    (RUN / 'source.patch').write_bytes(patch)
    if patch.strip():
        subprocess.run(['patch', '--batch', '-p1', '-i', str(RUN / 'source.patch')], cwd=SOURCE, check=True)
    for rel in ('m3/context_budget.py', 'm3/qa.py', 'm3/episodic_memory.py',
                'm3/llm_factory.py', 'providers/llm/minimax_llm.py', 'm3/pipeline/stages.py'):
        if (SOURCE / rel).read_bytes() != (ROOT / 'oGMemory' / rel).read_bytes():
            raise RuntimeError(f'Frozen source mismatch: {rel}')
    shutil.copytree(ROOT / 'scripts/instrumentation', RUN / '_instrumentation',
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(ROOT / 'scripts/run_context_experiment.py', RUN / 'quota_api.py')
    shutil.copyfile(Path(__file__), RUN / 'driver.py')

    config = yaml.safe_load(KEY_SOURCE.read_text())
    dsn = make_dsn(**db_settings())
    for role in ('storage', 'vector_db'):
        config[role]['connection_string'] = dsn
    config['vector_db']['type'] = 'pgvector'
    config.setdefault('m3', {})['vector_index'] = {
        'backend': 'opengauss', 'pool_size': 8, 'table_name': 'm3_episodic_vectors'}
    config['storage']['pool_size'] = 8
    config['vector_db']['pool_size'] = 8
    config['identity']['account_id'] = ACCOUNT
    config['identity']['agent_id'] = 'main'
    config['identity']['user_id'] = 'u-alice'
    for role in ('episodic_llm', 'qa_llm', 'judge_llm'):
        config['m3'][role]['api_key'] = key
        config['m3'][role]['model'] = 'MiniMax-M3.1-Flash'
        config['m3'][role]['max_tokens'] = 2048 if role == 'judge_llm' else 8192
        config['m3'][role]['reasoning_effort'] = 'medium'
        config['m3'][role]['extra_body'] = {'thinking': {'type': 'adaptive'}}
    config['llm']['api_key'] = key
    config['m3']['embedding'] = {
        'provider': 'openai', 'api_key': 'abc-123', 'base_url': 'http://127.0.0.1:8001/v1',
        'model': 'BAAI/bge-m3', 'dimension': 1024}
    config['m3']['qa_context'] = qa_context(max_input_tokens)
    h.private_yaml(CONFIG, config)
    (EXP / 'config.redacted.yaml').write_text(
        yaml.safe_dump(h.redact(config), allow_unicode=True))

    sys.path.insert(0, str(SOURCE))
    from m3.qa import load_questions
    qs = [q for q in load_questions(h.ANNOTATIONS) if q['video_id'] in VIDEOS]
    counts = {v: sum(q['video_id'] == v for q in qs) for v in VIDEOS}
    assert len(qs) == 97, len(qs)

    video_meta = {}
    for v in VIDEOS:
        p = Path('/data/m3_data/videos/robot') / (v + '.mp4')
        if not p.is_file():
            raise FileNotFoundError(p)
        probe = json.loads(subprocess.check_output(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(p)], text=True))
        video_meta[v] = {'bytes': p.stat().st_size,
                         'duration_seconds': float(probe['format']['duration'])}
    h.save(EXP / 'manifest.json', {
        'experiment': NAME, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'source_commit': rev, 'source_branch': branch,
        'patch_sha256': h.digest(RUN / 'source.patch'),
        'driver_sha256': h.digest(RUN / 'driver.py'),
        'config_sha256': h.digest(CONFIG),
        'annotations_sha256': h.digest(h.ANNOTATIONS),
        'tokenizer_sha256': h.digest(TOKENIZER),
        'videos': VIDEOS, 'question_counts': counts, 'questions': len(qs),
        'video_meta': video_meta,
        'account': ACCOUNT, 'database': DB,
        'db': {'host': '127.0.0.1', 'port': db_settings()['port'], 'user': 'ogmem', 'dbname': DB},
        'model': 'MiniMax-M3.1-Flash', 'thinking': 'adaptive', 'reasoning_effort': 'medium',
        'qa_output_tokens': 8192, 'judge_output_tokens': 2048, 'max_iterations': 6,
        'timeout_seconds': 180, 'temporal_policy': 'current_clip',
        'qa_context': {k: v for k, v in qa_context(max_input_tokens).items() if k != 'tokenizer_path'},
        'tokenizer_path': '<local pinned tokenizer>',
        'workers': {'video': 1, 'stage_max_workers': 4, 'face_workers': 4,
                    'frames': '30 extraction / 16 model'},
        'pipeline': 'pipeline -> qa (qa stage disabled in pipeline; standalone React QA)',
        'judge_seed': 20261113,
        'design': ('Approximate re-run on rychkov/m3-parallel of the paper eight-video cohort; '
                   'fresh perception+memory into a new pgvector DB; branch has no fact_index and '
                   'a reduced qa_context schema, so not byte-identical to the 57.2% paper baseline'),
        'safety_stop_at_unix': time.time() + 60 * 3600,
    })
    h.save(EXP / 'status.json', {'status': 'prepared', 'questions': len(qs), 'counts': counts})
    print(json.dumps({'prepared': NAME, 'questions': len(qs), 'counts': counts,
                      'source_commit': rev}, ensure_ascii=False))


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
            raise RuntimeError(f'{role} model request configuration mismatch: {params}')


def _recover_key():
    try:
        cfg = yaml.safe_load(CONFIG.read_text())
        key = cfg.get('llm', {}).get('api_key') or cfg['m3']['qa_llm'].get('api_key')
        if key and not key.startswith('${'):
            os.environ.setdefault('OGMEM_MINIMAX_API_KEY', key)
            os.environ.setdefault('MINIMAX_API_KEY', key)
    except Exception:
        pass


def worker(phase, video):
    sys.path.insert(0, str(SOURCE))
    _recover_key()
    verify_model_settings()
    gate = h.QuotaGate()
    gate.wait()
    h.instrument(gate, memory=phase == 'pipeline')
    work = RUN / video
    work.mkdir(exist_ok=True)
    if phase == 'pipeline':
        from m3.run_video_pipeline import main
        return main([
            '--video', f'/data/m3_data/videos/robot/{video}.mp4', '--video-id', video,
            '--seconds', '0', '--work-dir', str(work), '--config', str(CONFIG),
            '--pipeline', str(SOURCE / 'config/m3/pipelines/video_episodic_full.yaml'),
            '--graph-store', 'sql', '--account-id', ACCOUNT, '--no-smart-resume',
            '--max-frames', '30', '--episodic-max-frames', '16', '--disable-stage', 'qa',
            '--set', 'voice_asr.params.device=cuda',
            '--set', 'speaker_diarization.params.device=cuda',
            '--set', 'speaker_embedding.params.device=cuda:0'])
    if phase == 'qa':
        from m3.qa import main_react_qa
        from m3.control_adapter import OGMemoryControlAdapter
        from m3.config import load_m3_runtime_config
        os.environ['OGMEM_GRAPH_CONNECTION_STRING'] = \
            load_m3_runtime_config(str(CONFIG)).graph_connection_string
        original = OGMemoryControlAdapter.run

        def guarded(self, ctx, task):
            gate.wait()
            return original(self, ctx, task)

        OGMemoryControlAdapter.run = guarded
        output = RUN / 'og_qa' / video / 'm3_qa_results.jsonl'
        output.parent.mkdir(parents=True, exist_ok=True)
        # Resumable: on retries keep already-answered questions and only
        # re-run the (video_id, question_id) pairs that are missing or
        # failed. ``completed_keys`` treats protocol-scaffolding / errored
        # answers as incomplete, so a bad first answer is re-asked rather
        # than frozen as the final prediction.
        return main_react_qa([
            '--config', str(CONFIG), '--annotations', str(h.ANNOTATIONS), '--video', video,
            '--graph-store', 'sql', '--account-id', ACCOUNT, '--output-jsonl', str(output),
            '--resume',
            '--temperature', '0', '--max-iterations', '6', '--timeout-seconds', '180',
            '--top-k', '8', '--candidate-k', '20', '--transcript-top-k', '20',
            '--transcript-fusion', 'rrf', '--context-k', '8', '--include-neighbors', '1',
            '--temporal-policy', 'current_clip'])
    raise ValueError(phase)


def judge():
    sys.path.insert(0, str(SOURCE))
    from m3.qa import load_questions
    from m3.eval import build_judge_llm, judge_answer
    _recover_key()
    verify_model_settings()
    gate = h.QuotaGate()
    h.instrument(gate)
    llm = build_judge_llm(config_path=str(CONFIG))
    qs = [q for q in load_questions(h.ANNOTATIONS) if q['video_id'] in VIDEOS]
    tasks = []
    for video in VIDEOS:
        answers = {r['question_id']: r for r in h.rows(RUN / 'og_qa' / video / 'm3_qa_results.jsonl')}
        for q in (x for x in qs if x['video_id'] == video):
            r = answers.get(q['question_id'], {})
            tasks.append({'video_id': video, 'question_id': q['question_id'], 'question': q['question'],
                          'expected_answer': q['answer'],
                          'predicted_answer': r.get('predicted_answer', ''), 'error': r.get('error', '')})
    random.Random(20261113).shuffle(tasks)
    h.save(EXP / 'judging_inputs.json', tasks)
    path = EXP / 'judgments.jsonl'
    existing = h.rows(path)
    done = {(x['video_id'], x['question_id']) for x in existing}
    with path.open('a') as out:
        for t in tasks:
            if (t['video_id'], t['question_id']) in done:
                continue
            if not t['predicted_answer'] or t['error']:
                score, reason, error = 0., 'Missing or failed answer', False
            else:
                gate.wait()
                score, reason, error = judge_answer(judge_llm=llm, question=t['question'],
                                                    expected=t['expected_answer'],
                                                    predicted=t['predicted_answer'])
            out.write(json.dumps({**t, 'score': score, 'reason': reason,
                                  'judge_error': bool(error)}, ensure_ascii=False) + '\n')
            out.flush()
            os.fsync(out.fileno())
            if error:
                raise RuntimeError('Judge error, preserved for review')
    return 0


def environment(phase, video):
    env = {**os.environ, 'M3_EVAL_ROOT': str(ROOT),
           'CUDA_VISIBLE_DEVICES': os.environ.get('M3_GPU', '3'),
           'PYTHONHASHSEED': '0',
           'PYTHONPATH': f"{RUN / '_instrumentation'}:{SOURCE}",
           'M3_USAGE_LOG': str(EXP / f'usage-{phase}-{video}.jsonl'),
           'OGMEM_M3_MAX_WORKERS': '4', 'OGMEM_M3_FACE_WORKERS': '4',
           'OGMEM_GRAPH_POOL_SIZE': '8', 'OGMEM_EMBED_MAX_CONCURRENCY': '4',
           'PYTHONUNBUFFERED': '1'}
    env['HUGGINGFACE_TOKEN'] = env.get('HUGGINGFACE_TOKEN') or env.get('HF_TOKEN', '')
    lib = ROOT / 'oGMemory/.venv/lib/python3.12/site-packages/nvidia'
    env['LD_LIBRARY_PATH'] = f"{lib / 'cublas/lib'}:{lib / 'cudnn/lib'}:" + env.get('LD_LIBRARY_PATH', '')
    return env


def run():
    manifest = json.loads((EXP / 'manifest.json').read_text())
    assert h.digest(CONFIG) == manifest['config_sha256']
    assert h.digest(h.ANNOTATIONS) == manifest['annotations_sha256']
    assert h.digest(RUN / 'driver.py') == manifest['driver_sha256']
    history = []
    for v in VIDEOS:
        for phase in ('pipeline', 'qa'):
            receipt = EXP / 'stages' / f'{v}-{phase}.json'
            if receipt.exists():
                r = json.loads(receipt.read_text())
                if r.get('exit_code'):
                    raise RuntimeError(f'Prior failed stage needs review: {r}')
            else:
                if time.time() >= manifest['safety_stop_at_unix']:
                    h.save(EXP / 'status.json', {'status': 'safety_stopped', 'history': history})
                    return 74
                h.save(EXP / 'status.json', {'status': 'running', 'pid': os.getpid(),
                                             'video': v, 'phase': phase, 'history': history})
                start = time.time()
                with (EXP / 'private' / f'{v}-{phase}.log').open('a') as log:
                    p = subprocess.run([str(PYTHON), str(RUN / 'driver.py'), 'worker', phase, v],
                                       cwd=SOURCE, env=environment(phase, v),
                                       stdout=log, stderr=subprocess.STDOUT)
                r = {'video': v, 'phase': phase, 'exit_code': p.returncode,
                     'status': 'completed' if p.returncode == 0 else 'failed',
                     'started_at_unix': start, 'seconds': time.time() - start}
                if phase == 'qa' and p.returncode != 0:
                    # QA returns 2 when individual questions fail (e.g. the
                    # model emits [Search]/[Answer] scaffolding). As long as
                    # the video produced real answers we keep the cohort
                    # moving and let the judge score what is present
                    # (missing -> 0). One bad question must not abort the
                    # remaining videos, mirroring the pipeline-stage
                    # convention.
                    out = RUN / 'og_qa' / v / 'm3_qa_results.jsonl'
                    n_ok = 0
                    if out.exists():
                        for line in out.read_text(encoding='utf-8').splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                row = json.loads(line)
                            except ValueError:
                                continue
                            if str(row.get('predicted_answer') or '').strip():
                                n_ok += 1
                    if n_ok > 0:
                        r['status'] = 'degraded'
                        r['partial'] = True
                        r['answered'] = n_ok
                h.save(receipt, r)
                if r['status'] == 'failed':
                    h.save(EXP / 'status.json', {'status': 'needs_review', 'failed': r,
                                                 'history': history})
                    return p.returncode
            history.append(r)
    h.save(EXP / 'status.json', {'status': 'judging', 'history': history})
    with (EXP / 'private' / 'judge.log').open('a') as log:
        p = subprocess.run([str(PYTHON), str(RUN / 'driver.py'), 'judge'], cwd=SOURCE,
                           env=environment('judge', 'all'), stdout=log, stderr=subprocess.STDOUT)
    h.save(EXP / 'status.json', {'status': 'finished' if p.returncode == 0 else 'judge_failed',
                                 'history': history, 'judge_exit_code': p.returncode})
    return p.returncode


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'worker':
        ret = worker(sys.argv[2], sys.argv[3])
    elif action == 'prepare':
        prepare(int(sys.argv[2]) if len(sys.argv) > 2 else 24000)
        ret = 0
    else:
        ret = globals()[action]()
    raise SystemExit(ret or 0)
