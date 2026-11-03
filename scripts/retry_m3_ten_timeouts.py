#!/usr/bin/env python3
"""One independent rerun of the two original kitchen_01 timeout questions.

Uses frozen ten-video source, unchanged config/indices/time budget. Original
answers and main-study judging inputs are never overwritten. Recovery grades
are diagnostic, not the ten-video common judging pass.
"""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(os.environ.get('M3_EVAL_ROOT', Path(__file__).resolve().parent.parent))
BASE = 'm3_ten_fullinterval_20261102'
NAME = 'm3_ten_timeout_recovery_20261103'
EXP, RUN = ROOT / 'experiments' / NAME, ROOT / 'runs' / NAME
SOURCE = ROOT / 'runs' / BASE / '_source/oGMemory'
IDS = ('kitchen_01_Q03', 'kitchen_01_Q13')
ACCOUNT = 'm3-ten-fullinterval-20261102'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_timeouts(questions, initial, ids):
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate targets')
    old = {r['question_id']: r for r in initial}
    if len(old) != len(initial):
        raise ValueError('Duplicate original answers')
    selected = [q for q in questions if q['question_id'] in ids]
    if len(selected) != len(ids):
        raise ValueError('Missing or duplicate target annotations')
    for q in selected:
        r = old[q['question_id']]
        if r.get('finish_reason') != 'timeout' or not r.get('error'):
            raise ValueError('Recovery is restricted to original timeouts')
        if any(r[k] != q[k] for k in ('video_id', 'question_id', 'question_index', 'question', 'timestamp', 'before_clip')):
            raise ValueError('Original question identity changed')
        if r['expected_answer'] != q['answer']:
            raise ValueError('Reference answer changed')
    return selected


def source_audit(config, index):
    import psycopg2
    from m3.config import load_m3_runtime_config
    from m3.fact_index import content_hash
    records = json.loads(index.read_text())['records']
    expected = {r['node_id']: r['content_sha256'] for r in records}
    with psycopg2.connect(load_m3_runtime_config(str(config)).graph_connection_string) as db:
        db.set_session(readonly=True, isolation_level='REPEATABLE READ')
        with db.cursor() as cur:
            cur.execute("""SELECT node_id,extracted_content FROM graph_nodes
                WHERE account_id=%s AND extractor_type='episodic_memory_generator'
                AND extracted_content->>'source_video_id'=%s ORDER BY node_id""", (ACCOUNT, 'kitchen_01'))
            actual = {str(node): content_hash(body) for node, body in cur.fetchall()}
    if len(actual) != 79 or actual != expected:
        raise ValueError('Source memories changed relative to the pinned fact index')
    return actual


def main():
    if EXP.exists() or RUN.exists():
        raise FileExistsError('Recovery already exists; never repeat until a better answer appears')
    sys.path.insert(0, str(SOURCE))
    from providers.llm.openai_llm import OpenAILLM
    if not getattr(OpenAILLM._create_completion, '_m3_usage_audit', False):
        raise RuntimeError('Usage instrumentation is required before paid calls')
    from m3 import qa
    from m3.config import load_m3_runtime_config
    from m3.control_adapter import OGMemoryControlAdapter
    from m3.eval import build_judge_llm, judge_answer
    from core.operation_deadline import operation_deadline

    base_exp, base_run = ROOT / 'experiments' / BASE, ROOT / 'runs' / BASE
    original = base_run / 'og_qa/kitchen_01/m3_qa_results.jsonl'
    source_config = base_exp / 'private/kitchen_01.yaml'
    annotation = Path('/data/m3_data/annotations/robot.json')
    index = base_run / 'kitchen_01/fact_index/manifest.json'
    settings = json.loads((base_exp / 'qa_configs/kitchen_01.json').read_text())
    assert digest(source_config) == settings['config_sha256']
    assert digest(index) == settings['fact_index_sha256']
    initial = rows(original)
    selected = select_timeouts(qa.load_questions(annotation), initial, IDS)
    pinned = [original, source_config, annotation, index, index.parent / 'vectors.npy',
              SOURCE / 'm3/qa.py', SOURCE / 'm3/control_adapter.py', SOURCE / 'm3/context_budget.py',
              base_run / 'kitchen_01/voice_asr.jsonl', base_run / 'kitchen_01/episodic_index.jsonl']
    hashes = {str(p): digest(p) for p in pinned}
    before = source_audit(source_config, index)
    RUN.mkdir(parents=True); EXP.mkdir(parents=True); (EXP / 'private').mkdir(mode=0o700)
    config = EXP / 'private/config.yaml'; shutil.copyfile(source_config, config); config.chmod(0o600)
    shutil.copyfile(Path(__file__), RUN / 'runner.py')
    spec = importlib.util.spec_from_file_location('frozen_ten_driver', base_run / 'driver.py')
    driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    gate = driver.QuotaGate()
    save(EXP / 'manifest.json', {
        'created_at': dt.datetime.now().astimezone().isoformat(), 'origin': BASE,
        'targets': [{'question_id': q['question_id'], 'question_index': q['question_index']} for q in selected],
        'source_commit': json.loads((base_exp / 'manifest.json').read_text())['source']['commit'],
        'config_sha256': digest(config), 'runner_sha256': digest(RUN / 'runner.py'),
        'source_hashes_before': hashes, 'source_node_hashes': before,
        'model': 'MiniMax-M3.1-Flash', 'reasoning_effort': 'medium', 'thinking': 'adaptive',
        'qa_output_tokens': 8192, 'judge_output_tokens': 2048, 'max_iterations': 6,
        'soft_timeout_seconds': 180, 'temporal_policy': 'current_clip',
        'retry_policy': 'One independent rerun per timeout; no answer selection; original predictions unchanged.',
        'grading': 'Separate diagnostic judging only; not main-study common grading.'})
    # Filter after annotation normalization so original question_index values
    # (2 and 12) and all original fields survive intact.
    qa.load_questions = lambda path: selected
    os.environ['OGMEM_GRAPH_CONNECTION_STRING'] = load_m3_runtime_config(str(config)).graph_connection_string
    original_run = OGMemoryControlAdapter.run
    position = 0
    def measured(self, ctx, task):
        nonlocal position
        q = selected[position]; position += 1
        gate.wait()
        os.environ['M3_AUDIT_VIDEO'] = q['video_id']
        os.environ['M3_AUDIT_QUESTION_INDEX'] = str(q['question_index'])
        save(EXP / 'status.json', {'status': 'running_qa', 'question_id': q['question_id']})
        started = time.monotonic()
        try:
            return original_run(self, ctx, task)
        finally:
            task.search_stats['wall_seconds'] = time.monotonic() - started
    OGMemoryControlAdapter.run = measured
    # sitecustomize captures its path at startup; preserve that single log and
    # partition QA/judge calls by their explicit request budgets.
    rc = qa.main_react_qa(['--config', str(config), '--annotations', str(annotation),
        '--video', 'kitchen_01', '--graph-store', 'sql', '--account-id', ACCOUNT,
        '--output-jsonl', str(RUN / 'm3_qa_results.jsonl'), '--temperature', '0',
        '--max-iterations', '6', '--timeout-seconds', '180', '--top-k', '8', '--candidate-k', '20',
        '--transcript-top-k', '20', '--transcript-fusion', 'rrf', '--context-k', '8',
        '--include-neighbors', '1', '--temporal-policy', 'current_clip'])
    answers = rows(RUN / 'm3_qa_results.jsonl')
    assert len(answers) == len({r['question_id'] for r in answers}) == 2
    old = {r['question_id']: r for r in initial}
    for r in answers:
        assert r['temporal_scope'] == old[r['question_id']]['temporal_scope']
    assert source_audit(config, index) == before
    assert all(digest(Path(p)) == h for p, h in hashes.items())
    save(EXP / 'status.json', {'status': 'diagnostic_judging', 'qa_exit_code': rc})
    judge = build_judge_llm(config_path=str(config)); graded = []
    for r in answers:
        os.environ['M3_AUDIT_QUESTION_INDEX'] = str(r['question_index'])
        if r.get('error') or not r.get('predicted_answer'):
            score, reason, error = 0., 'QA did not complete; no judge call.', False
        else:
            gate.wait()
            with operation_deadline(120):
                score, reason, error = judge_answer(judge_llm=judge, question=r['question'],
                    expected=r['expected_answer'], predicted=r['predicted_answer'])
        graded.append({**r, 'judge_score': score, 'judge_reasoning': reason, 'judge_error': error})
    (EXP / 'questions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in graded))
    stats = {'questions': 2, 'valid_answers': sum(not r.get('error') and bool(r.get('predicted_answer')) for r in answers),
             'qa_exit_code': rc, 'judge_errors': sum(bool(r['judge_error']) for r in graded),
             'original_predictions_unchanged': digest(original) == hashes[str(original)],
             'source_memories_unchanged': source_audit(config, index) == before,
             'diagnostic_score_mean': None if any(r['judge_error'] for r in graded) else sum(r['judge_score'] for r in graded)/2}
    save(EXP / 'metrics.json', stats)
    save(EXP / 'status.json', {'status': 'finished' if not stats['judge_errors'] else 'finished_with_judge_errors', **stats})
    lines=['# 两题超时恢复实验', '', '保持原模型、来源记忆、检索配置与180秒软预算；每题只重跑一次。',
           '原始十视频实验答案未覆盖。以下分数为单独诊断评分，不是共同盲评。', '',
           '| 题目 | 首次状态 | 恢复状态 | 本次QA秒数 | 诊断评分 |', '|---|---|---|---:|---:|']
    for r in graded:
        lines.append(f"| {r['question_id']} | timeout | {r['finish_reason']} | {r['search_stats'].get('wall_seconds',0):.2f} | {'评委错误' if r['judge_error'] else r['judge_score']} |")
    (EXP / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(stats, ensure_ascii=False))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
