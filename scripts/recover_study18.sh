#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="$ROOT/experiments/m3_study18_recovery_20261101"
RUN="$ROOT/runs/m3_study18_recovery_20261101"
: "${MINIMAX_API_KEY:?Source ~/.bashrc first}"
cd "$ROOT/oGMemory"
export OGMEM_MINIMAX_MODEL=MiniMax-M3.1-Flash UV_NO_SYNC=1 CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="$ROOT/scripts/instrumentation:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export M3_USAGE_LOG="$EXP/model-usage.jsonl"
disabled=()
for stage in trim annotate preprocess load_clips extract_audio extract_frames detect_faces voice_asr speaker_diarization speaker_embedding; do
  disabled+=(--disable-stage "$stage")
done
set +e
.venv/bin/python -m m3.run_video_pipeline \
  --video /data/m3_data/videos/robot/study_18.mp4 --video-id study_18 --seconds 0 \
  --work-dir "$RUN/study_18" --config "$EXP/private/config.yaml" \
  --pipeline config/m3/pipelines/video_episodic_full.yaml \
  --graph-store sql --account-id m3-flash-medium-20261031 --no-smart-resume \
  "${disabled[@]}" \
  --set episodic_memory.params.retry_failed=true --set episodic_memory.params.workers=4 \
  --set qa.params.annotations=/data/m3_data/annotations/robot.json \
  --set qa.params.video=study_18 --set qa.params.max_iterations=6 \
  --set qa.params.timeout_seconds=180 --set qa.params.temperature=0 \
  --set qa.params.top_k=8 --set qa.params.candidate_k=20 \
  --set qa.params.transcript_top_k=20 --set qa.params.transcript_fusion=rrf \
  --set qa.params.context_k=8 --set qa.params.include_neighbors=1 \
  --set qa.params.temporal_policy=current_clip \
  > "$RUN/study_18/pipeline.log" 2>&1
rc=$?
echo "$rc" > "$EXP/pipeline_exit_code.txt"
if [[ -s "$RUN/study_18/m3_qa_results.jsonl" ]]; then
  .venv/bin/python -m m3.eval --config "$EXP/private/config.yaml" \
    --annotations /data/m3_data/annotations/robot.json --video study_18 \
    --input-qa-results "$RUN/study_18/m3_qa_results.jsonl" --output-dir "$RUN/_aggregate" \
    > "$EXP/private/judge.log" 2>&1
  echo "$?" > "$EXP/judge_exit_code.txt"
fi
echo "$rc" > "$EXP/exit_code.txt"
exit "$rc"
