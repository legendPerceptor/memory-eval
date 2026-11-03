#!/usr/bin/env python3
"""Plot paired completed video stage wall times, with explicit shared-input accounting.

Read-only with respect to experiments; never launches inference or judges. The
100% bars are ratios of summed durations, not averages of video percentages.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
PHASES = ('pipeline', 'index', 'og_qa', 'm3_memory', 'm3_qa')
START = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) INFO m3\.pipeline\.runner: Running stage (\w+) \(', re.M)
EXPECTED_STAGES = (
    'trim', 'annotate', 'preprocess', 'load_clips', 'extract_audio', 'extract_frames',
    'detect_faces', 'voice_asr', 'speaker_diarization', 'speaker_embedding',
    'episodic_memory', 'create_memory_nodes', 'build_index', 'face_clustering',
    'voice_segmentation', 'voice_clustering', 'semantic_memory_consolidator',
)
SHARED_GROUPS = {
    'Setup / slicing / loading': ('runner_startup', 'trim', 'annotate', 'preprocess', 'load_clips'),
    'Audio extraction': ('extract_audio',),
    'Frame extraction': ('extract_frames',),
    'Face detection / features': ('detect_faces',),
    'ASR': ('voice_asr',),
    'Speaker diarization': ('speaker_diarization',),
    'Speaker embeddings': ('speaker_embedding',),
}
CATEGORIES = ('shared_perception', 'memory_and_index', 'qa')
METHODS = ('ogmemory', 'm3_agent')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_seconds(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError('Invalid elapsed duration')
    return float(value)


def stage_durations(log, receipt, timezone):
    """Sequential stage intervals; final interval includes worker finalization."""
    start = finite_seconds(receipt['started_at_unix'])
    duration = finite_seconds(receipt['seconds'])
    end = start + duration
    markers = [(name, dt.datetime.strptime(stamp, '%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=timezone).timestamp())
               for stamp, name in START.findall(log)]
    if tuple(n for n, _ in markers) != EXPECTED_STAGES:
        raise ValueError('Missing, duplicated, reordered, or unexpected stages; do not infer timings')
    times = [start, *[t for _, t in markers], end]
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError('Log timestamps do not fit the phase receipt')
    result = {'runner_startup': markers[0][1] - start}
    for i, (name, stamp) in enumerate(markers):
        result[name] = (markers[i + 1][1] if i + 1 < len(markers) else end) - stamp
    if not math.isclose(sum(result.values()), duration, abs_tol=.002):
        raise ValueError('Stage durations do not reconcile')
    return result


def method_times(durations, receipts):
    shared = sum(durations[s] for keys in SHARED_GROUPS.values() for s in keys)
    og_memory = sum(durations.values()) - shared + finite_seconds(receipts['index']['seconds'])
    methods = {
        'ogmemory': dict(shared_perception=shared, memory_and_index=og_memory,
                        qa=finite_seconds(receipts['og_qa']['seconds'])),
        'm3_agent': dict(shared_perception=shared, memory_and_index=finite_seconds(receipts['m3_memory']['seconds']),
                        qa=finite_seconds(receipts['m3_qa']['seconds'])),
    }
    for method in methods.values():
        method['allocated_total'] = sum(method.values())
    actual = sum(finite_seconds(receipts[p]['seconds']) for p in PHASES)
    if not math.isclose(sum(m['allocated_total'] for m in methods.values()) - shared, actual, abs_tol=.002):
        raise ValueError('Shared perception must be counted only once for experiment elapsed work')
    return methods, actual


def phase_finished(receipt, phase, answers=None, expected_count=None, video=None):
    """QA exit 2 with a full ledger is finished work, not an unfinished video.

    Include failed questions and their elapsed time rather than selecting only
    successful QA runs. Other failed phases do not establish comparable input.
    """
    if phase not in ('og_qa', 'm3_qa'):
        return receipt.get('exit_code') == 0 and receipt.get('status') == 'completed'
    if receipt.get('exit_code') not in (0, 2):
        return False
    if not answers or len(answers) != expected_count:
        return False
    keys = {(r['video_id'], r['question_id']) for r in answers}
    return len(keys) == expected_count and all(r['video_id'] == video for r in answers)


def collect(root, name):
    exp, run = root / 'experiments' / name, root / 'runs' / name
    manifest_path = exp / 'manifest.json'
    manifest = read(manifest_path)
    timezone = dt.datetime.fromisoformat(manifest['created_at']).tzinfo
    if timezone is None:
        raise ValueError('Manifest timezone required for log timestamps')
    videos, excluded, source_hashes = [], [], {str(manifest_path.relative_to(root)): digest(manifest_path)}
    for video in manifest['selected_video_ids']:
        paths = {p: exp / 'stages' / f'{video}-{p}.json' for p in PHASES}
        missing = [p for p, path in paths.items() if not path.exists()]
        if missing:
            excluded.append({'video': video, 'reason': 'incomplete', 'missing_phases': missing})
            continue
        receipts = {p: read(path) for p, path in paths.items()}
        qa_rows = {}
        for phase in ('og_qa', 'm3_qa'):
            path = run / phase / video / 'm3_qa_results.jsonl'
            qa_rows[phase] = [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
            if path.exists():
                source_hashes[str(path.relative_to(root))] = digest(path)
        bad = [p for p, r in receipts.items() if not phase_finished(
            r, p, qa_rows.get(p), manifest['question_counts'][video], video)]
        if bad:
            excluded.append({'video': video, 'reason': 'unsuccessful_phase', 'phases': bad})
            continue
        log = exp / 'private' / f'{video}-pipeline.log'
        summary = run / video / 'pipeline_summary.json'
        status = read(summary)
        if any(not status['stages'][s]['ok'] for s in EXPECTED_STAGES):
            raise ValueError('Completed receipt conflicts with pipeline stage status')
        durations = stage_durations(log.read_text(), receipts['pipeline'], timezone)
        methods, actual = method_times(durations, receipts)
        videos.append({'video': video, 'questions': manifest['question_counts'][video],
                       'stage_seconds': durations, 'methods': methods,
                       'qa_failed_answers': {p: sum(bool(r.get('error')) or not r.get('predicted_answer')
                                                   for r in rs) for p, rs in qa_rows.items()},
                       'actual_experiment_phase_seconds': actual})
        for path in [*paths.values(), log, summary]:
            source_hashes[str(path.relative_to(root))] = digest(path)
    if not videos:
        raise ValueError('No paired completed videos; no zero-filled plot will be produced')
    totals = {m: {k: sum(v['methods'][m][k] for v in videos) for k in (*CATEGORIES, 'allocated_total')}
              for m in METHODS}
    detail = {label: sum(v['stage_seconds'][s] for v in videos for s in stages)
              for label, stages in SHARED_GROUPS.items()}
    return {'experiment': name, 'generated_at': dt.datetime.now().astimezone().isoformat(),
            'status': 'complete' if len(videos) == len(manifest['selected_video_ids']) else 'interim',
            'selected_videos': [v['video'] for v in videos], 'planned_videos': len(manifest['selected_video_ids']),
            'questions_per_method': sum(v['questions'] for v in videos), 'excluded': excluded,
            'videos': videos, 'totals_seconds': totals, 'shared_breakdown_seconds': detail,
            'actual_experiment_phase_seconds': sum(v['actual_experiment_phase_seconds'] for v in videos),
            'accounting': 'Shared perception physically runs once; assign identical measured duration to each standalone-method total. Do not sum the two allocated totals as experiment wall time.',
            'scope': 'Stage wall-clock intervals, including local work, remote waits/retries and any within-stage quota waits. Excludes final common judge and between-phase driver gaps. Not API CPU/GPU compute time.',
            'granularity': 'M3 memory stage bundles input preparation, identities, generation and embedding; oGMemory memory/graph/index work is grouped to match that scope.',
            'source_sha256': source_hashes}


def plot(data, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    labels = ['Shared perception + setup', 'Memory / graph / index', 'QA']
    colors = ['#9aa5b1', '#3b78a8', '#e69f43']
    names = ['oGMemory', 'M3-Agent\nreference adaptation']
    count = len(data['selected_videos'])
    scope = f"{data['status'].upper()}: {count}/{data['planned_videos']} paired-completed videos | {data['questions_per_method']} questions / method"
    if count <= 2:
        scope += '\n' + ', '.join(data['selected_videos'])
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.8), gridspec_kw={'height_ratios': [1, 1]}, layout='constrained')
    max_total = max(data['totals_seconds'][m]['allocated_total'] / 60 for m in METHODS)
    for ax, percent in zip(axes, (False, True)):
        for row, method in enumerate(METHODS):
            values = data['totals_seconds'][method]
            total = values['allocated_total']
            left = 0.
            for key, color in zip(CATEGORIES, colors):
                width = values[key] / total * 100 if percent else values[key] / 60
                ax.barh(row, width, left=left, color=color, height=.52, edgecolor='white', linewidth=.7)
                text = f'{width:.1f}%' if percent else f'{width:.2f}'
                ax.text(left + width / 2, row, text, ha='center', va='center', fontsize=10,
                        color='white' if key == 'memory_and_index' else '#15202b')
                left += width
            if not percent:
                ax.text(left + max_total * .015, row, f'{left:.2f} min', va='center', fontweight='bold')
        ax.set_yticks([0, 1], names)
        ax.set_ylim(1.6, -.65)
        ax.set_xlim(0, 100 if percent else max_total * 1.20)
        ax.set_xlabel('Share of allocated per-method time (%)' if percent else 'Allocated processing time (minutes)')
        ax.set_title('(b) Stage shares' if percent else '(a) Absolute stage durations', loc='left', fontsize=11)
        ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
        ax.spines[['top', 'right', 'left']].set_visible(False)
    fig.suptitle('Video processing time: stage breakdown\n' + scope, fontsize=12)
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(colors, labels)],
               loc='lower center', bbox_to_anchor=(.5, .055), ncols=3, frameon=False, fontsize=9)
    fig.get_layout_engine().set(rect=(0, .14, 1, .86))
    fig.text(.5, .015, 'Shared perception executed once, allocated to each path. Final judging excluded.',
             ha='center', fontsize=8, color='#444444')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(prefix.with_suffix('.' + ext), dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.3, 4.6), layout='constrained')
    details = data['shared_breakdown_seconds']; total = sum(details.values())
    ax.barh(list(details), list(details.values()), color='#738ca0', height=.62)
    for i, value in enumerate(details.values()):
        ax.text(value + max(details.values())*.025, i, f'{value:.1f} s  ({100*value/total:.1f}%)', va='center', fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, max(details.values()) * 1.46)
    ax.set_xlabel('Stage wall time (seconds)'); ax.set_title('Shared perception (executed once for both methods)\n' + scope, fontsize=11)
    ax.spines[['top', 'right', 'left']].set_visible(False); ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    detail_prefix = prefix.with_name(prefix.name + '_shared')
    for ext in ('pdf', 'png'):
        fig.savefig(detail_prefix.with_suffix('.' + ext), dpi=180, bbox_inches='tight')
    plt.close(fig)
    return [p.with_suffix('.' + ext) for p in (prefix, detail_prefix) for ext in ('pdf', 'png')]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default='m3_ten_fullinterval_20261102')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.experiment):
        raise ValueError('Experiment must be a single safe component')
    data = collect(ROOT, args.experiment)
    folder = ROOT / 'experiments' / args.experiment / 'timing'
    folder.mkdir(exist_ok=True)
    prefix = ROOT / 'latex/figures' / f'{args.experiment}_stage_times'
    outputs = plot(data, prefix)
    (folder / 'stage_times.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    with (folder / 'stage_times.csv').open('w') as out:
        writer = csv.writer(out); writer.writerow(['video', 'method', 'stage', 'seconds', 'share_percent'])
        for video in data['videos']:
            for method, values in video['methods'].items():
                for stage in CATEGORIES:
                    writer.writerow([video['video'], method, stage, values[stage], 100*values[stage]/values['allocated_total']])
    og, m3 = [data['totals_seconds'][m] for m in METHODS]
    lines = ['# 视频处理耗时：阶段分解', '',
             f"状态：**{data['status']}**。仅包括两方法阶段均完成的 {len(data['selected_videos'])}/{data['planned_videos']} 个视频：" + ', '.join(data['selected_videos']) + '。',
             f"每方法 {data['questions_per_method']} 题；不包含最终评委。未完成视频未补零。", '',
             '| 阶段 | oGMemory 分钟（占比） | M3-Agent参考适配 分钟（占比） |', '|---|---:|---:|']
    for label, key in zip(['共享感知与启动', '记忆构建、图谱及索引', 'QA'], CATEGORIES):
        cells = [f"{v[key]/60:.2f}（{100*v[key]/v['allocated_total']:.1f}%）" for v in (og, m3)]
        lines.append(f'| {label} | {cells[0]} | {cells[1]} |')
    lines += [f"| 各自计入共享感知后的单方法总时间 | {og['allocated_total']/60:.2f} | {m3['allocated_total']/60:.2f} |", '',
              f"共享感知实际上只执行一次；两方法的阶段时间合计（去除重复分摊）为 **{data['actual_experiment_phase_seconds']/60:.2f} 分钟**。", '',
              '主图为可比的单方法核算：将同一次共享感知的实测时间分别计入两条方法路径，不表示又运行了一遍感知。',
              'M3的记忆阶段包含输入准备、身份组织、生成与embedding，当前无法可靠拆开；oGMemory按同口径合并。',
              '时间来自阶段日志边界和进程回执，含该阶段内等待/重试/初始化，不是纯计算时间；不含阶段间调度间隙。',
              'QA若完整执行但包含失败题，其耗时仍纳入，不能只选全成功视频。没有用并发请求耗时相加代替墙钟时间；占比按累计时间计算。',
              '只有一个样本时不能据此宣称稳定加速比；样本、远端服务负载和模型控制策略均影响结果。', '',
              '## 共享感知细分', '', '| 子阶段 | 秒 | 感知时间内占比 |', '|---|---:|---:|']
    for label, value in data['shared_breakdown_seconds'].items():
        lines.append(f"| {label} | {value:.2f} | {100*value/og['shared_perception']:.1f}% |")
    lines += ['', '## 刷新命令', '', '```bash', f'oGMemory/.venv/bin/python scripts/plot_m3_stage_times.py --experiment {args.experiment}', '```', '',
              '刷新只读取已完成记录，不调用模型、不修改实验结果。完整十视频完成后运行同一命令即可。',
              '主论文尚未插入此阶段性图，以免与此前五视频的质量结果混用。']
    (folder / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    rel = prefix.relative_to(ROOT / 'latex')
    snippet = ('% Generated timing figure; not automatically included in report.tex.\n'
               '% Requires \\usepackage{graphicx} in the main document preamble.\n'
               '\\begin{figure}[htbp]\n\\centering\n'
               f'\\includegraphics[width=\\linewidth]{{{rel}.pdf}}\n'
               '\\caption{视频处理时间及阶段占比。'
               f"本图为{'完整结果' if data['status']=='complete' else '阶段性结果'}，仅覆盖两方法均完成的"
               f"{len(data['selected_videos'])}/{data['planned_videos']}个视频、每方法{data['questions_per_method']}题。"
               '共享感知实际执行一次，为单方法对比在两条路径中分别计入同一份耗时；记忆阶段包含图谱及索引，'
               '不含最终评委，不代表纯模型计算耗时。}\n\\label{fig:video-stage-times}\n\\end{figure}\n')
    snippet_path = ROOT / 'latex/video_stage_times.tex'; snippet_path.write_text(snippet)
    receipt = {'script_sha256': digest(Path(__file__)), 'data_sha256': digest(folder / 'stage_times.json'),
               'artifacts': {str(p.relative_to(ROOT)): digest(p) for p in [*outputs, snippet_path]},
               'planned_videos': data['planned_videos'], 'paired_completed': len(data['selected_videos']),
               'status': data['status'], 'inference_calls': 0}
    (folder / 'plot_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('\n'.join(lines[:12])); print('Figures:', prefix)


if __name__ == '__main__':
    main()
