import datetime as dt
import math

import pytest

from plot_m3_stage_times import (EXPECTED_STAGES, SHARED_GROUPS, finite_seconds,
                                method_times, stage_durations, collect)


def fixture():
    timezone = dt.timezone(dt.timedelta(hours=8))
    start = dt.datetime(2026, 11, 2, 10, tzinfo=timezone).timestamp()
    lines = []
    for i, name in enumerate(EXPECTED_STAGES):
        stamp = dt.datetime.fromtimestamp(start + 2 + i * 10, timezone).strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        lines.append(f'{stamp} INFO m3.pipeline.runner: Running stage {name} (test)')
    receipt = {'started_at_unix': start, 'seconds': 2 + len(lines) * 10}
    return '\n'.join(lines), receipt, timezone


def test_stage_boundaries_reconcile_to_receipt():
    log, receipt, timezone = fixture()
    stages = stage_durations(log, receipt, timezone)
    assert stages['runner_startup'] == 2
    assert stages['episodic_memory'] == 10
    assert sum(stages.values()) == receipt['seconds']


def test_shared_perception_not_double_counted_for_actual_experiment():
    log, receipt, timezone = fixture()
    stages = stage_durations(log, receipt, timezone)
    receipts = {'pipeline': receipt, 'index': {'seconds': 5}, 'og_qa': {'seconds': 20},
                'm3_memory': {'seconds': 30}, 'm3_qa': {'seconds': 10}}
    methods, actual = method_times(stages, receipts)
    shared = sum(stages[s] for names in SHARED_GROUPS.values() for s in names)
    assert methods['ogmemory']['shared_perception'] == methods['m3_agent']['shared_perception'] == shared
    assert sum(m['allocated_total'] for m in methods.values()) - shared == actual
    for m in methods.values():
        assert math.isclose(sum(m[k] / m['allocated_total'] for k in ('shared_perception', 'memory_and_index', 'qa')), 1)


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'outside', 'order'])
def test_ambiguous_stage_logs_fail_closed(mutation):
    log, receipt, timezone = fixture()
    if mutation == 'duplicate': log += '\n' + log.splitlines()[0]
    if mutation == 'missing': log = '\n'.join(log.splitlines()[1:])
    if mutation == 'outside': receipt['seconds'] = 10
    if mutation == 'order': log = '\n'.join(reversed(log.splitlines()))
    with pytest.raises(ValueError): stage_durations(log, receipt, timezone)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, True, '2'])
def test_invalid_seconds(value):
    with pytest.raises(ValueError): finite_seconds(value)


def test_no_completed_pair_does_not_generate_zero_filled_results(tmp_path):
    import json
    exp = tmp_path / 'experiments/demo'; exp.mkdir(parents=True)
    (exp / 'manifest.json').write_text(json.dumps({'created_at': '2026-11-02T10:00:00+08:00',
                                                  'selected_video_ids': ['v'], 'question_counts': {'v': 1}}))
    with pytest.raises(ValueError, match='No paired completed'):
        collect(tmp_path, 'demo')
