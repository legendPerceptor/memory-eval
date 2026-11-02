#!/usr/bin/env python3
"""Isolated ten-video full-interval study; never reuse historical predictions.

prepare fixes cohort/config/source before execution. run starts the frozen driver.
Workers generate both methods from shared fresh perception, then a blinded common
judge scores immutable answers. Raw logs/configs remain private or under runs/.
"""
import argparse
import copy
import datetime as dt
import fcntl
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
import threading
import time

import yaml

ROOT = Path(os.environ.get('M3_EVAL_ROOT', Path(__file__).resolve().parent.parent))
NAME = 'm3_ten_fullinterval_20261102'
ACCOUNT = 'm3-ten-fullinterval-20261102'
DB = 'ogmem_m3_ten_fullinterval_20261102'
EXP = ROOT / 'experiments' / NAME
RUN = ROOT / 'runs' / NAME
SOURCE = RUN / '_source/oGMemory'
PYTHON = ROOT / 'oGMemory/.venv/bin/python'
ANNOTATIONS = Path('/data/m3_data/annotations/robot.json')
CONFIG = EXP / 'private/config.yaml'


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()] if Path(path).exists() else []


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_cohort(candidates, original, seed=42):
    candidates = set(candidates)
    if len(original) != 5 or len(set(original)) != 5 or not set(original) <= set(candidates):
        raise ValueError('Invalid original cohort')
    new = random.Random(seed).sample(sorted(set(candidates) - set(original)), 5)
    return new


def redact(value):
    if isinstance(value, dict):
        return {k: '<REDACTED>' if k in ('api_key', 'connection_string', 'password', 'hf_token')
                else redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def private_yaml(path, config):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('x') as out:
        os.chmod(path, 0o600)
        yaml.safe_dump(config, out, allow_unicode=True)


def archive(repo, target):
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise ValueError(f'{repo.name} must be clean before freezing')
    target.mkdir(parents=True)
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    tarpath = target.parent / (target.name + '.tar')
    with tarpath.open('wb') as out:
        subprocess.run(['git', '-C', str(repo), 'archive', 'HEAD'], stdout=out, check=True)
    with tarfile.open(tarpath) as bundle:
        bundle.extractall(target, filter='data')
    return {'commit': revision, 'archive_sha256': digest(tarpath)}


def prepare():
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import parse_dsn, make_dsn
    if not os.environ.get('MINIMAX_API_KEY'):
        raise ValueError('Source ~/.bashrc first; API key is not printed')
    if EXP.exists() or RUN.exists():
        raise FileExistsError('Use the existing manifest/run; preparation will not overwrite anything')
    sys.path.insert(0, str(ROOT / 'oGMemory'))
    from m3.qa import load_questions
    questions = load_questions(ANNOTATIONS)
    old_manifest = json.loads((ROOT / 'experiments/m3_flash_medium_20261031/sample_manifest.json').read_text())
    canonical_annotations = hashlib.sha256(json.dumps(
        json.loads(ANNOTATIONS.read_text()), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert canonical_annotations == old_manifest['annotations_sha256']
    original = old_manifest['selected_video_ids']
    candidates = sorted({q['video_id'] for q in questions})
    added = select_cohort(candidates, original)
    videos = original + added
    counts = {v: sum(q['video_id'] == v for q in questions) for v in videos}
    video_meta = {}
    for v in videos:
        p = Path('/data/m3_data/videos/robot') / (v + '.mp4')
        assert p.is_file(), v
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                 'format=duration', '-of', 'json', str(p)], text=True))
        video_meta[v] = {'bytes': p.stat().st_size, 'duration_seconds': float(probe['format']['duration']),
                         'sha256': digest(p)}
    EXP.mkdir(parents=True)
    (EXP / 'private').mkdir(mode=0o700)
    RUN.mkdir(parents=True)
    source = archive(ROOT / 'oGMemory', SOURCE)
    reference = archive(ROOT / 'm3-agent', RUN / '_source/m3-agent')
    shutil.copytree(ROOT / 'comparisons/m3_agent_matched', RUN / '_adapter',
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT / 'scripts/instrumentation', RUN / '_instrumentation',
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(Path(__file__), RUN / 'driver.py')
    shutil.copyfile(ROOT / 'scripts/run_context_experiment.py', RUN / 'quota_api.py')
    base = yaml.safe_load((ROOT / 'experiments/m3_flash_medium_20261031/private/config.yaml').read_text())
    for role in ('storage', 'vector_db'):
        params = parse_dsn(base[role]['connection_string'])
        conn = psycopg2.connect(**params); conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute('SELECT 1 FROM pg_database WHERE datname=%s', (DB,))
            if not cur.fetchone():
                cur.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(DB)))
        conn.close(); params['dbname'] = DB
        base[role]['connection_string'] = make_dsn(**params)
    with psycopg2.connect(base['storage']['connection_string']) as conn:
        with conn.cursor() as cur:
            cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
            cur.execute("SELECT to_regclass('graph_nodes')")
            if cur.fetchone()[0] is not None:
                raise ValueError('Target database already has graph tables; refusing reuse')
    model = dict(provider='minimax', api_key='${MINIMAX_API_KEY}', base_url='https://api.minimax.cn/v1',
                 model='MiniMax-M3.1-Flash', max_tokens=8192, temperature=0, request_timeout_seconds=300,
                 is_reasoning_model=True, reasoning_effort='medium', extra_body={'thinking': {'type': 'adaptive'}})
    base['llm'] = copy.deepcopy(model)
    for role in ('episodic_llm', 'qa_llm', 'judge_llm'):
        base['m3'][role] = {**copy.deepcopy(model), 'max_tokens': 2048 if role == 'judge_llm' else 8192}
    old_qa = yaml.safe_load((ROOT / 'experiments/m3_strict_incremental_20261102/private/study_05.yaml').read_text())
    qa_context = copy.deepcopy(old_qa['m3']['qa_context'])
    for k in ('fact_index_path', 'fact_index_sha256'):
        qa_context.pop(k, None)
    # Per-video QA configs receive their immutable fact-index path after generation.
    base['m3'].pop('qa_context', None)
    base['identity']['account_id'] = ACCOUNT
    private_yaml(CONFIG, base)
    (EXP / 'config.redacted.yaml').write_text(yaml.safe_dump(redact(base), allow_unicode=True))
    private_yaml(EXP / 'private/qa_template.yaml', {'qa_context': qa_context})
    manifest = {
        'experiment': NAME, 'created_at': dt.datetime.now().astimezone().isoformat(),
        'original_video_ids': original, 'new_video_ids': added, 'selected_video_ids': videos,
        'execution_order': added + original, 'selection_seed': 42,
        'selection': 'sorted available annotation IDs minus original five; random.sample without replacement',
        'candidate_ids': candidates, 'question_counts': counts, 'total_questions': sum(counts.values()),
        'annotations_sha256': digest(ANNOTATIONS),
        'annotations_canonical_sha256': canonical_annotations, 'videos': video_meta,
        'source': source, 'm3_reference': reference, 'account': ACCOUNT, 'database': DB,
        'config_sha256': digest(CONFIG), 'qa_template_sha256': digest(EXP / 'private/qa_template.yaml'),
        'tokenizer_sha256': digest(qa_context['tokenizer_path']),
        'source_files': {str(p.relative_to(RUN)): digest(p) for folder in ('_adapter', '_instrumentation')
                         for p in sorted((RUN / folder).glob('*.py'))},
        'driver_sha256': digest(RUN / 'driver.py'), 'quota_api_sha256': digest(RUN / 'quota_api.py'),
        'frames': {'clip_seconds': 30, 'fps': 1, 'extracted_cap': 30, 'model_cap': 16,
                   'model_indices': 'floor(i * n / k); same selector in both methods',
                   'face_detection': 'all collected frames; M3 additionally sends up to 8 face crops'},
        'methods': ['ogmemory_strict_incremental', 'm3_reference_text'],
        'model': 'MiniMax-M3.1-Flash', 'thinking': 'adaptive', 'reasoning_effort': 'medium',
        'qa_context': {**qa_context, 'tokenizer_path': '<local pinned tokenizer>'},
        'qa_output_tokens': 8192, 'judge_output_tokens': 2048, 'max_iterations': 6,
        'timeout_seconds': 180, 'temporal_policy': 'current_clip', 'judge_seed': 1102,
        'historical_predictions_reused': False,
        'perception': 'fresh once per video, shared exactly between two independently generated memories',
        'safety_stop_at_unix': time.time() + 24 * 3600,
        'quota_policy': 'wait before new memory calls/questions below 12% interval; stop below 5% weekly; 24h safety stop',
    }
    save(EXP / 'manifest.json', manifest)
    save(EXP / 'status.json', {'status': 'prepared', 'total_questions': manifest['total_questions']})
    (EXP / 'PLAN.md').write_text(f'''# 十视频全片段采样实验（预登记）

- 原五视频：{', '.join(original)}。
- 从其余视频排序后 seed=42 无放回抽取新增五视频：{', '.join(added)}。
- 固定总题数：{manifest['total_questions']}；分组分别报告，不能将旧采样结果拼进新总分。
- 先处理新增五视频，再处理原五视频；全部重新生成记忆，复用的仅是两方法共享的本轮感知。
- 每30秒以1fps收集最多30帧；两方法均以同一规则从中选至多16张基础帧。
  当前均匀规则为floor(i*n/k)，标准片段模型时间约0至28秒，并不保证所有瞬时动作可见。
- 音频完整提取；人脸检测覆盖本轮收集帧；M3仍可额外使用至多8张人物裁剪。
- 模型 MiniMax-M3.1-Flash / adaptive / medium；QA 8192、评委2048。
- oGMemory固定此前E严格增量3500方案；M3保留参考Action文本控制器。不是官方权重复现。
- 当前片段末尾时间协议；6轮、180秒软预算；不做提分参数搜索，不择优重试答案。
- 统一盲评按seed1102打乱(method,question)，评分0/0.5/1；失败计零，评委异常另列。
- 阶段失败保留日志和状态，继续其他视频；缺失答案明确计为失败，不缩小分母。
- 额度不足等待下一窗口，不切模型；周额度低于5%或运行达到24小时安全截止时暂停工作并保留状态。
- 历史五视频分数和论文不覆盖；新成绩完成后写入本目录REPORT.md / comparison.json。
''')
    print(json.dumps({'new_videos': added, 'total_questions': manifest['total_questions'],
                      'counts': counts, 'source_commit': source['commit']}, ensure_ascii=False))


class QuotaGate:
    def __init__(self):
        spec = importlib.util.spec_from_file_location('quota_api', RUN / 'quota_api.py')
        self.api = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.api)
        self.lock = threading.Lock(); self.checked = 0.; self.observation = None
        self.deadline = json.loads((EXP / 'manifest.json').read_text())['safety_stop_at_unix']

    def wait(self):
        with self.lock:
            while True:
                if time.time() >= self.deadline:
                    raise SystemExit(74)  # escape per-clip Exception handlers
                if time.time() - self.checked < 20 and self.observation:
                    q = self.observation
                else:
                    try:
                        q = self.api.read_quota_observation()
                    except Exception:
                        raise SystemExit(76) from None
                    self.checked = time.time(); self.observation = q
                    with (EXP / 'quota_observations.jsonl').open('a') as out:
                        out.write(json.dumps(q) + '\n')
                if q['weekly_remaining_percent'] < 5:
                    raise SystemExit(74)
                if q['interval_remaining_percent'] > 12:
                    return
                save(EXP / 'quota_status.json', {'status': 'waiting_interval_reset', **q})
                time.sleep(min(60, max(5, q['window_end_unix'] - time.time() + 5)))
                self.checked = 0


def instrument(gate, memory=False):
    from providers.llm.openai_llm import OpenAILLM
    original = OpenAILLM._create_completion
    if not getattr(original, '_m3_usage_audit', False):
        raise RuntimeError('Usage instrumentation absent; no model work allowed')
    # Only memory calls can wait inside a request boundary. QA/judge wait before
    # their timers so quota waiting cannot consume the reasoning time budget.
    if memory:
        def guarded(self, **kwargs):
            if kwargs.get('model') == 'MiniMax-M3.1-Flash':
                gate.wait()
            return original(self, **kwargs)
        OpenAILLM._create_completion = guarded


def qa_config(video):
    path = EXP / 'private' / f'{video}.yaml'
    index = RUN / video / 'fact_index/manifest.json'
    settings = yaml.safe_load(CONFIG.read_text())
    context = yaml.safe_load((EXP / 'private/qa_template.yaml').read_text())['qa_context']
    settings['m3']['qa_context'] = {**context, 'fact_index_path': str(index), 'fact_index_sha256': digest(index)}
    if path.exists():
        if yaml.safe_load(path.read_text()) != settings:
            raise ValueError('QA config changed; cannot resume')
    else:
        private_yaml(path, settings)
    save(EXP / 'qa_configs' / f'{video}.json', {'config_sha256': digest(path), 'fact_index_sha256': digest(index),
                                              'settings': redact(settings['m3']['qa_context'])})
    return path


def worker(phase, video):
    sys.path.insert(0, str(SOURCE))
    gate = QuotaGate(); gate.wait(); instrument(gate, memory=phase in ('pipeline', 'm3_memory'))
    work = RUN / video
    work.mkdir(exist_ok=True)
    if phase == 'pipeline':
        from m3.run_video_pipeline import main
        rc = main(['--video', f'/data/m3_data/videos/robot/{video}.mp4', '--video-id', video,
                   '--seconds', '0', '--work-dir', str(work), '--config', str(CONFIG),
                   '--pipeline', str(SOURCE / 'config/m3/pipelines/video_episodic_full.yaml'),
                   '--graph-store', 'sql', '--account-id', ACCOUNT, '--no-smart-resume',
                   '--max-frames', '30', '--episodic-max-frames', '16', '--disable-stage', 'qa',
                   '--set', 'episodic_memory.params.workers=4', '--set', 'voice_asr.params.device=cuda',
                   '--set', 'speaker_diarization.params.device=cuda', '--set', 'speaker_embedding.params.device=cuda:0'])
        if rc:
            return rc
        (work / 'perception_exit_code.txt').write_text('0\n')
        return audit_frames(video)
    if phase == 'index':
        from m3.build_fact_index import main
        return main(['--config', str(CONFIG), '--account-id', ACCOUNT, '--video', video,
                     '--output', str(work / 'fact_index')])
    if phase == 'og_qa':
        from m3.qa import main_react_qa
        from m3.control_adapter import OGMemoryControlAdapter
        from m3.config import load_m3_runtime_config
        # Standalone QA does not infer the graph DSN from --config. Bind the
        # isolated graph explicitly, as the earlier fixed-memory worker did.
        os.environ['OGMEM_GRAPH_CONNECTION_STRING'] = load_m3_runtime_config(
            str(CONFIG)).graph_connection_string
        original = OGMemoryControlAdapter.run
        state = {'index': 0}
        def guarded(self, ctx, task):
            gate.wait()
            os.environ['M3_AUDIT_VIDEO'] = video
            os.environ['M3_AUDIT_QUESTION_INDEX'] = str(state['index']); state['index'] += 1
            return original(self, ctx, task)
        OGMemoryControlAdapter.run = guarded
        output = RUN / 'og_qa' / video / 'm3_qa_results.jsonl'
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError('Refusing to replace first QA answers')
        return main_react_qa(['--config', str(qa_config(video)), '--annotations', str(ANNOTATIONS),
                    '--video', video, '--graph-store', 'sql', '--account-id', ACCOUNT,
                    '--output-jsonl', str(output), '--temperature', '0', '--max-iterations', '6',
                    '--timeout-seconds', '180', '--top-k', '8', '--candidate-k', '20',
                    '--transcript-top-k', '20', '--transcript-fusion', 'rrf', '--context-k', '8',
                    '--include-neighbors', '1', '--temporal-policy', 'current_clip'])
    sys.path.insert(0, str(RUN / '_adapter'))
    if phase == 'm3_memory':
        import runner as adapter
        coverage = adapter.run_video(video, CONFIG, skip_qa=True)
        if coverage['completed_clips'] / coverage['expected_clips'] < .95:
            raise RuntimeError('M3 memory coverage below preregistered 95%')
        # Build the embedding cache without executing the native-tool diagnostic
        # controller. Only the upstream Action text controller is evaluated.
        from reference import CachedEmbeddings, load_reference
        from m3.index_episodic import build_embedder
        embedder = CachedEmbeddings(build_embedder(None, config_path=str(CONFIG)))
        ref = load_reference(RUN / '_source/m3-agent', embedder)
        updates = json.loads((work / 'identity_updates.json').read_text())
        memories = {int(p.stem): json.loads(p.read_text())['memory']
                    for p in (work / 'm3_memories').glob('*.json') if '.failure.' not in p.name}
        count = 1 + max((u['id'] for c in updates for u in c['updates']), default=-1)
        adapter.build_prefix(ref, updates, memories, max(memories), count)
        save(work / 'embedding_cache.json', embedder.cache)
        frame_audit = json.loads((EXP / 'frame_audits' / f'{video}.json').read_text())
        clips = json.loads((work / 'clip_inputs.json').read_text())
        for clip in clips:
            assert clip['base_frame_node_ids'] == frame_audit['clips'][str(clip['clip_id'])]['selected_node_ids']
        return 0
    if phase == 'm3_qa':
        import reference_text as ref
        original = ref.build_prefix
        state = {'index': 0}
        def guarded(*args, **kwargs):
            gate.wait()
            os.environ['M3_AUDIT_VIDEO'] = video
            os.environ['M3_AUDIT_QUESTION_INDEX'] = str(state['index']); state['index'] += 1
            return original(*args, **kwargs)
        ref.build_prefix = guarded
        ref.run_video(video)
        return 0
    raise ValueError(phase)


def audit_frames(video):
    from core.models import RequestContext
    from m3.config import load_m3_runtime_config
    from m3.load_clips import build_graph_service
    from m3.episodic_memory import _index_frame_records
    from extraction.operators.builtin.episodic_memory_generator import _evenly_spaced_indices
    store, _ = build_graph_service('sql', load_m3_runtime_config(str(CONFIG)).graph_connection_string)
    ctx = RequestContext(account_id=ACCOUNT, user_id='m3bench-user', agent_id='m3bench-agent',
                         session_id=video, trace_id='frame-audit')
    index = _index_frame_records(RUN / video / 'frames.jsonl')
    clips = {}
    for key, ids in index.items():
        nodes = [store.get_raw_node(n, ctx) for n in ids]
        timestamps = [float(n.metadata['ts_sec']) for n in nodes]
        selected = _evenly_spaced_indices(len(nodes), 16)
        clip = key.split('|')[-1]
        clips[clip] = {'extracted_count': len(nodes), 'timestamps': timestamps,
                      'selected_timestamps': [timestamps[i] for i in selected],
                      'selected_node_ids': [str(nodes[i].node_id) for i in selected]}
        assert len(nodes) <= 30 and len(selected) <= 16
        assert timestamps == sorted(timestamps)
        if len(nodes) == 30:
            assert timestamps[-1] >= 28 and timestamps[selected[-1]] >= 27
    assert clips
    save(EXP / 'frame_audits' / f'{video}.json', {'video': video, 'clips': clips})
    return 0


def environment(phase, video):
    env = {**os.environ, 'M3_EVAL_ROOT': str(ROOT), 'M3_MATCHED_NAME': NAME,
           'M3_REFERENCE_NAME': NAME, 'M3_MATCHED_ACCOUNT': ACCOUNT,
           'M3_REFERENCE_OUTPUT': str(RUN / 'm3_qa'), 'OGMEM_MINIMAX_MODEL': 'MiniMax-M3.1-Flash',
           'CUDA_VISIBLE_DEVICES': '2', 'PYTHONHASHSEED': '0',
           'PYTHONPATH': f"{RUN / '_instrumentation'}:{SOURCE}:{RUN / '_adapter'}",
           'M3_USAGE_LOG': str(EXP / f'usage-{phase}-{video}.jsonl')}
    env['HUGGINGFACE_TOKEN'] = env.get('HUGGINGFACE_TOKEN') or env.get('HF_TOKEN', '')
    lib = ROOT / 'oGMemory/.venv/lib/python3.12/site-packages/nvidia'
    env['LD_LIBRARY_PATH'] = f"{lib / 'cublas/lib'}:{lib / 'cudnn/lib'}:" + env.get('LD_LIBRARY_PATH', '')
    for k in ('M3_AUDIT_VIDEO', 'M3_AUDIT_QUESTION_INDEX'):
        env.pop(k, None)
    return env


def fixed_tasks(manifest):
    sys.path.insert(0, str(SOURCE))
    from m3.qa import load_questions
    questions = [q for q in load_questions(ANNOTATIONS) if q['video_id'] in manifest['selected_video_ids']]
    assert len(questions) == manifest['total_questions']
    tasks, hashes = [], {}
    expected = {(q['video_id'], q['question_id']) for q in questions}
    for method, folder in [('ogmemory_strict_incremental', 'og_qa'), ('m3_reference_text', 'm3_qa')]:
        data = {}
        for video in manifest['selected_video_ids']:
            p = RUN / folder / video / 'm3_qa_results.jsonl'
            if p.exists():
                hashes[str(p.relative_to(RUN))] = digest(p)
                for row in rows(p):
                    key = (row['video_id'], row['question_id'])
                    if key in data or key not in expected:
                        raise ValueError('Duplicate or out-of-cohort prediction')
                    data[key] = row
        for q in questions:
            row = data.get((q['video_id'], q['question_id']))
            if row:
                assert row['question'] == q['question'] and row['expected_answer'] == q['answer']
            tasks.append({'method': method, 'video_id': q['video_id'], 'question_id': q['question_id'],
                          'question_index': q['question_index'], 'question': q['question'],
                          'expected_answer': q['answer'], 'predicted_answer': row.get('predicted_answer', '') if row else '',
                          'error': row.get('error', '') if row else 'Missing prediction: upstream stage failed',
                          'finish_reason': row.get('finish_reason') if row else 'upstream_failure'})
    random.Random(manifest['judge_seed']).shuffle(tasks)
    return tasks, hashes


def judge():
    sys.path.insert(0, str(SOURCE))
    from m3.eval import build_judge_llm, judge_answer
    from core.operation_deadline import operation_deadline
    gate = QuotaGate(); instrument(gate)
    manifest = json.loads((EXP / 'manifest.json').read_text())
    tasks, hashes = fixed_tasks(manifest)
    target = EXP / 'joint_questions.jsonl'
    if target.exists():
        raise FileExistsError('Existing judge results; do not selectively overwrite')
    save(EXP / 'judging_inputs.json', {'prediction_hashes': hashes, 'tasks': tasks})
    llm = build_judge_llm(config_path=str(CONFIG))
    with target.open('x') as out:
        for index, task in enumerate(tasks):
            os.environ['M3_AUDIT_VIDEO'] = task['video_id']
            os.environ['M3_AUDIT_QUESTION_INDEX'] = str(task['question_index'])
            if task['error'] or not task['predicted_answer']:
                score, reason, error = 0., 'Invalid or missing QA answer; no judge call.', False
            else:
                gate.wait()
                with operation_deadline(120):
                    score, reason, error = judge_answer(judge_llm=llm, question=task['question'],
                        expected=task['expected_answer'], predicted=task['predicted_answer'])
            out.write(json.dumps({**task, 'judge_score': score, 'judge_reasoning': reason,
                                  'judge_error': error}, ensure_ascii=False) + '\n'); out.flush()
            save(EXP / 'judge_status.json', {'completed': index + 1, 'total': len(tasks)})
    for p, h in hashes.items():
        assert digest(RUN / p) == h, 'Predictions changed during judging'
    return 0


def summarize():
    manifest = json.loads((EXP / 'manifest.json').read_text())
    grades = rows(EXP / 'joint_questions.jsonl')
    assert len(grades) == 2 * manifest['total_questions']
    assert len({(r['method'], r['video_id'], r['question_id']) for r in grades}) == len(grades)
    result = {'cohort': manifest, 'methods': {}}
    for method in manifest['methods']:
        group = [r for r in grades if r['method'] == method]
        def metrics(rs):
            errors = sum(bool(r['judge_error']) for r in rs)
            return {'questions': len(rs), 'valid_answers': sum(not r['error'] and bool(r['predicted_answer']) for r in rs),
                    'judge_errors': errors, 'mean_score': None if errors else sum(r['judge_score'] for r in rs) / len(rs)}
        result['methods'][method] = {**metrics(group), 'by_video': {
            v: metrics([r for r in group if r['video_id'] == v]) for v in manifest['selected_video_ids']},
            'original_five': metrics([r for r in group if r['video_id'] in manifest['original_video_ids']]),
            'new_five': metrics([r for r in group if r['video_id'] in manifest['new_video_ids']])}
        phase = 'og_qa' if method == 'ogmemory_strict_incremental' else 'm3_qa'
        calls = [r for p in EXP.glob(f'usage-{phase}-*.jsonl') for r in rows(p)
                 if r.get('requested_model') == 'MiniMax-M3.1-Flash']
        known = [r for r in calls if r.get('usage')]
        result['methods'][method]['qa_usage'] = {
            'calls': len(calls), 'unknown_usage_calls': len(calls) - len(known),
            'prompt_tokens': sum(r['usage']['prompt_tokens'] for r in known),
            'cached_prompt_tokens': sum((r['usage'].get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0 for r in known),
            'completion_tokens': sum(r['usage']['completion_tokens'] for r in known)}
    save(EXP / 'comparison.json', result)
    def fmt(value):
        return '待恢复评委错误' if value is None else f'{value:.4f}'
    lines = ['# 十视频全片段采样：共同评估结果', '',
             '固定16张基础模型帧，原五和新增五全部重新生成记忆；非官方M3权重复现。', '',
             '| 视频 | 题数 | oGMemory E | M3参考适配 |', '|---|---:|---:|---:|']
    a, b = [result['methods'][m] for m in manifest['methods']]
    for v in manifest['selected_video_ids']:
        lines.append(f"| {v} | {manifest['question_counts'][v]} | {fmt(a['by_video'][v]['mean_score'])} | {fmt(b['by_video'][v]['mean_score'])} |")
    for label, key in [('原五视频', 'original_five'), ('新增五视频', 'new_five')]:
        lines.append(f"| {label} | {a[key]['questions']} | {fmt(a[key]['mean_score'])} | {fmt(b[key]['mean_score'])} |")
    lines += [f"| 全部十视频 | {manifest['total_questions']} | {fmt(a['mean_score'])} | {fmt(b['mean_score'])} |", '',
              '自动0/0.5/1均分，不是人工准确率；按固定全集计分，失败计零，评委异常不伪装为错答。',
              '同一MiniMax生成及评分；current_clip非严格因果在线。新增五视频未经本轮按成绩调参，原五仍为开发视频。',
              '两方法共享本轮原始感知，但记忆生成、身份组织及控制协议不同；M3可额外使用人脸裁剪。',
              'QA用量不含记忆生成、感知或评委；旧前缀采样结果独立保留，不与本轮拼接。']
    (EXP / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return result


def run():
    manifest = json.loads((EXP / 'manifest.json').read_text())
    assert digest(CONFIG) == manifest['config_sha256']
    assert digest(ANNOTATIONS) == manifest['annotations_sha256']
    assert digest(RUN / 'driver.py') == manifest['driver_sha256']
    lock = (RUN / 'driver.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    history = []
    for video in manifest['execution_order']:
        failed = set()
        dependencies = {'pipeline': [], 'index': ['pipeline'], 'og_qa': ['index'],
                        'm3_memory': ['pipeline'], 'm3_qa': ['m3_memory']}
        for phase in dependencies:
            receipt = EXP / 'stages' / f'{video}-{phase}.json'
            if receipt.exists():
                result = json.loads(receipt.read_text())
            elif any(d in failed for d in dependencies[phase]):
                result = {'video': video, 'phase': phase, 'status': 'skipped_upstream_failure', 'exit_code': 2}
                save(receipt, result)
            else:
                if time.time() >= manifest['safety_stop_at_unix']:
                    save(EXP / 'status.json', {'status': 'safety_stopped', 'history': history}); return 74
                save(EXP / 'status.json', {'status': 'running', 'pid': os.getpid(), 'video': video,
                                          'phase': phase, 'history': history})
                started = time.time()
                log = EXP / 'private' / f'{video}-{phase}.log'
                with log.open('a') as out:
                    child = subprocess.run([str(PYTHON), str(RUN / 'driver.py'), 'worker',
                                            '--phase', phase, '--video', video], cwd=SOURCE,
                                           env=environment(phase, video), stdout=out, stderr=subprocess.STDOUT)
                result = {'video': video, 'phase': phase, 'exit_code': child.returncode,
                          'status': 'completed' if child.returncode == 0 else 'failed',
                          'started_at_unix': started, 'seconds': time.time() - started}
                if child.returncode in (74, 76):
                    save(EXP / 'status.json', {'status': 'quota_or_safety_stopped',
                                              'history': history, 'interrupted': result})
                    return child.returncode
                save(receipt, result)
            history.append(result)
            if result['exit_code']:
                failed.add(phase)
    save(EXP / 'status.json', {'status': 'judging', 'history': history})
    with (EXP / 'private/judge.log').open('a') as log:
        child = subprocess.run([str(PYTHON), str(RUN / 'driver.py'), 'judge'], cwd=SOURCE,
                               env=environment('judge', 'all'), stdout=log, stderr=subprocess.STDOUT)
    if child.returncode:
        save(EXP / 'status.json', {'status': 'needs_judge_review', 'history': history, 'exit_code': child.returncode})
        return child.returncode
    result = summarize()
    failures = any(x['exit_code'] for x in history) or any(m['judge_errors'] for m in result['methods'].values())
    save(EXP / 'status.json', {'status': 'finished_with_failures' if failures else 'finished',
                              'history': history, 'report': 'REPORT.md'})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'worker', 'judge', 'summarize'])
    parser.add_argument('--phase'); parser.add_argument('--video')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(); return 0
    if args.action == 'run':
        return run()
    if args.action == 'worker':
        return worker(args.phase, args.video)
    if args.action == 'judge':
        return judge()
    summarize(); return 0


if __name__ == '__main__':
    raise SystemExit(main())
