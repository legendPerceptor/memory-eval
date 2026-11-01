#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="$ROOT/experiments/m3_agent_matched_20261101"
RUN="$ROOT/runs/m3_agent_matched_20261101"
PY="$ROOT/oGMemory/.venv/bin/python"
cd "$RUN/_source/oGMemory"
export CUDA_VISIBLE_DEVICES=2
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
export LD_LIBRARY_PATH="$ROOT/oGMemory/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:$ROOT/oGMemory/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
# Do not let editable installs or another experiment's PYTHONPATH override snapshot code.
export PYTHONPATH="$PWD"
for video in study_05 gym_03 bedroom_04 study_18 kitchen_20; do
  mkdir -p "$RUN/$video"
  if [[ -e "$RUN/$video/perception_exit_code.txt" ]]; then
    echo "Refusing to overwrite $video perception receipt" >&2; exit 2
  fi
  set +e
  "$PY" -m m3.run_video_pipeline \
    --video "/data/m3_data/videos/robot/$video.mp4" --video-id "$video" --seconds 0 \
    --work-dir "$RUN/$video" --config "$EXP/private/config.yaml" \
    --pipeline "$EXP/perception.yaml" --graph-store sql \
    --account-id m3-agent-matched-20261101 --no-smart-resume \
    --set voice_asr.params.device=cuda --set voice_asr.params.force=true \
    --set speaker_diarization.params.device=cuda --set speaker_diarization.params.force=true \
    --set speaker_embedding.params.device=cuda:0 --set speaker_embedding.params.force=true \
    --set detect_faces.params.force=true \
    > "$RUN/$video/perception.log" 2>&1
  rc=$?
  set -e
  echo "$rc" > "$RUN/$video/perception_exit_code.txt"
  if ((rc != 0)); then echo "Perception failed for $video" >&2; exit "$rc"; fi
done
echo 0 > "$EXP/perception_exit_code.txt"
