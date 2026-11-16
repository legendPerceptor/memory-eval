#!/usr/bin/env python3
"""Read-only, ten-minute experiment monitor; does not restart or stop work."""
import datetime as dt
import json
import os
from pathlib import Path
import time

ROOT = Path('/home/yuanjian/Development/memory-eval')
EXP = ROOT / 'experiments/m3_reduce_face_one_20261112'
NAME = 'm3_reduce_face_one_20261112'

def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}

def inspect():
    now = time.time()
    status = read_json(EXP / 'status.json')
    state = status.get('status', 'unknown')
    pid = status.get('pid')
    try:
        command = Path(f'/proc/{int(pid)}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        alive = NAME in command and 'driver.py run' in command
    except (OSError, TypeError, ValueError):
        alive = False
    alerts = []
    if (EXP / 'paused_due_to_face_clustering.txt').exists():
        alerts.append('experiment_paused_face_clustering_degraded')
    if state in ('running', 'judging') and not alive:
        # Judging status has no pid; find the matching driver instead.
        if state != 'judging' or not any(NAME.encode() in p.read_bytes() and b'driver.py\x00run' in p.read_bytes() for p in Path('/proc').glob('[0-9]*/cmdline') if p.exists() and os.access(p, os.R_OK)):
            alerts.append('driver_not_running')
    if state in ('needs_review', 'judge_failed'):
        alerts.append('experiment_' + state)
    if state == 'unknown':
        alerts.append('status_unreadable')
    calls = errors = truncated = mismatch = input_tokens = output_tokens = reasoning = 0
    for path in sorted(EXP.glob('usage-*.jsonl')):
        try:
            with path.open() as stream:
                for line in stream:
                    try: row = json.loads(line)
                    except ValueError: continue
                    calls += 1
                    errors += row.get('status') == 'error'
                    truncated += 'length' in (row.get('finish_reasons') or [])
                    if row.get('requested_model') == 'MiniMax-M3.1-Flash' and (row.get('reasoning_effort') != 'medium' or row.get('thinking') != {'type':'adaptive'}):
                        mismatch += 1
                    u = row.get('usage') or {}
                    input_tokens += u.get('prompt_tokens') or 0
                    output_tokens += u.get('completion_tokens') or 0
                    reasoning += (u.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0
        except OSError:
            alerts.append('usage_log_unreadable')
    if errors: alerts.append('model_request_errors')
    if truncated: alerts.append('output_truncated')
    if mismatch: alerts.append('model_configuration_mismatch')
    # A zero-exit pipeline may still report a degraded face-clustering stage.
    for summary in (ROOT / 'runs' / NAME).glob('*/pipeline_summary.json'):
        data = read_json(summary)
        stages = data.get('stages', {})
        face = stages.get('face_clustering', {}) if isinstance(stages, dict) else {}
        if face.get('status') == 'degraded':
            alerts.append('face_clustering_degraded:' + summary.parent.name)
    video, phase = status.get('video'), status.get('phase')
    log = EXP / 'private' / (f'{video}-{phase}.log' if video and phase else 'judge.log')
    stage = None
    try:
        for line in log.read_text(errors='replace').splitlines():
            if 'Running stage ' in line: stage = line.split('Running stage ', 1)[1]
        idle = int(now - log.stat().st_mtime)
        if state in ('running','judging') and idle > 5400: alerts.append('log_inactive_over_90_minutes')
    except OSError:
        idle = None
    return {'at':dt.datetime.now().astimezone().isoformat(timespec='seconds'), 'status':state,
            'video':video,'phase':phase,'stage':stage,'driver_alive':alive if state == 'running' else None,
            'completed_stages':len(status.get('history', [])), 'model_calls':calls,'model_errors':errors,
            'truncated_calls':truncated,'config_mismatches':mismatch,'input_tokens':input_tokens,
            'output_tokens':output_tokens,'reasoning_tokens':reasoning,'log_idle_seconds':idle,'alerts':alerts}

def main():
    previous = None
    while True:
        try:
            result = inspect()
        except Exception as exc:
            result = {'at':dt.datetime.now().astimezone().isoformat(timespec='seconds'), 'alerts':['monitor_error'], 'error_type':type(exc).__name__}
        with (EXP/'monitor.jsonl').open('a') as f:
            f.write(json.dumps(result, ensure_ascii=False)+'\n')
        tmp = EXP/'monitor_latest.tmp'; tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n'); tmp.replace(EXP/'monitor_latest.json')
        signature = (tuple(result.get('alerts', [])), result.get('status'))
        if (result.get('alerts') or result.get('status') == 'finished') and signature != previous:
            with (EXP/'monitor_alerts.jsonl').open('a') as f: f.write(json.dumps(result, ensure_ascii=False)+'\n')
        previous = signature
        if result.get('status') in ('finished', 'needs_review', 'judge_failed'): break
        time.sleep(600)

if __name__ == '__main__': main()
