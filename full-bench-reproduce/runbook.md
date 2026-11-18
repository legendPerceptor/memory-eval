# M3-Bench Experiment Runbook — full configuration & commands

For colleagues reproducing our M3-Bench experiments on this host.
Everything below is copy-pasteable. Secrets are included since this stays
on the shared host — rotate before sharing outside the team.

---

## 1. Stack topology (all containers, host network)

| Container | Purpose | Port / URL |
|---|---|---|
| `rychkov-ogmem-postgres` | pgvector postgres (graph + vector store) | `127.0.0.1:15432`, db `ogmem`, user `ogmem`, pw `<set via PG_PASSWORD / .pgpass>` |
| `rychkov-embedding` | dedicated CPU embedder (Qwen3-Embedding-0.6B, 1024 dims) | `http://127.0.0.1:18000/v1`, bearer `abc-123` |
| `ogmem-rychkov-cv-1` | face detection sidecar (insightface) | `http://127.0.0.1:18081` |
| `ogmem-m3runner` | M3 pipeline runner (`sleep infinity`, drive via `docker exec`) | — |

Bring-up (from the repo's `deploy/` dir; `.env` holds the secrets):

```bash
cd /home/rychkov/projects/oGMemory/deploy
sg docker -c "docker compose -p ogmem-rychkov --profile with-pg --profile with-cv \
  --profile with-m3runner up -d --build"
```

`sg docker -c "..."` wraps commands for the docker group (needed in agent
shells; a normal login shell with the docker group can use plain docker).

The m3runner container mounts:
- `/data/m3_data` → `/data_m3` (M3-Bench videos/annotations, read-only)
- `/data/hf_cache/rychkov` → `/hf-cache` (HF model cache, `HF_HOME=/hf-cache`)
- `/data/m3_work_rychkov` → `/work` (work dirs, scripts, logs — host-visible)
- repo `deploy/ogmemory.m3.yaml` → `/etc/ogmem/config.yaml`

## 2. Models (all three roles on distinct models)

| Role | Model | Provider |
|---|---|---|
| Extraction | `deepseek-v4.1-flash` | Huawei MaaS `https://api.modelarts-maas.com/v2` (OpenAI-compatible) |
| QA | `glm-5.3` | same MaaS endpoint — **10 req/min cap → use 2 video workers** |
| Judge | `deepseek-v4.1-flash` | same as all our baselines — never change it mid-comparison |
| Embedder | `Qwen/Qwen3-Embedding-0.6B` | local `:18000` (1024 dims = `vector_db.dimension`) |
| ASR | `faster-whisper large-v3-turbo` | local CPU, cached in `/hf-cache` |
| Diarization | `pyannote/speaker-diarization-3.1` | local CPU (full-stages runs only) |
| Speaker emb. | `speechbrain/spkrec-ecapa-voxceleb` | local CPU (full-stages runs only) |

MaaS API key (`LLM_API_KEY`):
set via the `HUAWEI_API_KEY` environment variable (never commit the real key)

These live in the container env (`LLM_PROVIDER=openai`, `LLM_BASE_URL`,
`LLM_MODEL`, `LLM_API_KEY`) and are consumed by the runtime YAML via
`${LLM_*}` placeholder refs.

## 3. Runtime config

Two config variants in `/data/m3_work_rychkov/` (= `/work` in the container):

- **`ogmemory.m3-ds.yaml`** — standard: LLM `request_timeout_seconds: 300`
- **`ogmemory.m3-ds600.yaml`** — identical but `request_timeout_seconds: 600`
  (use for QA retry passes; glm-5.3 occasionally hangs past 300 s)

Role wiring inside the YAML (extraction resolves `${LLM_MODEL}` = env):

```yaml
llm:      { provider: "${LLM_PROVIDER}", base_url: "${LLM_BASE_URL}", model: "${LLM_MODEL}", max_tokens: 32768 }
embedding: { provider: openai, model: "Qwen/Qwen3-Embedding-0.6B", base_url: "http://127.0.0.1:18000/v1" }
storage:   { connection_string: "${M3_STORAGE_DSN}" }
vector_db: { connection_string: "${M3_VECTOR_DSN}" }
m3:
  extraction_llm: { model: "deepseek-v4.1-flash" }   # via ${LLM_*}
  qa_llm:         { model: "glm-5.3" }
  judge_llm:      { model: "deepseek-v4.1-flash" }
```

`max_tokens: 32768` is required — the reasoning model burns ~7k tokens on
reasoning per call; 8192 truncates tool calls (`finish_reason=length`).

## 4. The wrapper scripts (in `/work`, i.e. `/data/m3_work_rychkov/`)

**`run_maas.sh`** — reduced-stages baseline (6 stages disabled). Env knobs:

| Env | Meaning | Our value |
|---|---|---|
| `M3_VIDEOS` | comma-separated video ids (empty = all 100) | per experiment |
| `M3_ACCOUNT` | graph account | `m3bench` |
| `M3_WORK_ROOT` | work-dir root (container path) | `/work/<run-name>` |
| `M3_CONFIG` | runtime YAML | `/work/ogmemory.m3-ds.yaml` (or `-ds600`) |
| `M3_JUDGE` | `1` = live per-video judge | `1` |
| `M3_VIDEO_WORKERS` | video-level parallelism | `2` (glm-5.3 QA cap) |
| `OGMEM_M3_MAX_WORKERS` | per-stage workers | `4` (MaaS limits) |
| `M3_QA_TIMEOUT` | per-question ReAct budget (s) | `300` (retry passes: `600`) |
| `M3_STORAGE_DSN` / `M3_VECTOR_DSN` / `OGMEM_GRAPH_CONNECTION_STRING` | DB DSNs | point at a **fresh DB per experiment** |

Fixed flags inside the wrapper: `--seconds 0` (full length),
`--disable-stage speaker_diarization speaker_embedding voice_segmentation
voice_clustering face_clustering semantic_memory_consolidator`,
`extract_frames.params.max_frames=4`, `episodic_memory.params.max_frames=4`,
`episodic_memory.params.no_audio=true` (no audio *attachments* — ASR still runs),
`--qa-top-k 8`, `qa.params.transcript_top_k=20`, `qa.params.resume=true`,
`qa.params.max_iterations=6`, `--judge`.

**`run_maas_full.sh`** — same but with **zero** `--disable-stage` flags
(all stages on; needs the speaker-stack setup in §6).

### 4.1 Stage matrix — baseline vs full-stages

The pipeline (`config/m3/pipelines/video_episodic_full.yaml`) defines 18
stages; the two wrappers differ only in which are disabled:

| Stage | Baseline (`run_maas.sh`) | Full (`run_maas_full.sh`) | Why disabled in baseline |
|---|---|---|---|
| trim / annotate / preprocess / load_clips | ✅ | ✅ | — |
| extract_audio | ✅ | ✅ | — |
| extract_frames (max_frames=4) | ✅ | ✅ | — |
| detect_faces (cv_service) | ✅ | ✅ | — |
| **voice_asr** (faster-whisper) | ✅ **always on** | ✅ | — transcripts drive QA retrieval (`transcript_top_k=20`) |
| speaker_diarization (pyannote) | ❌ | ✅ | speaker stack not in the runner image (torch tax, gated HF model); needs §6 setup |
| speaker_embedding (ECAPA) | ❌ | ✅ | same |
| episodic_memory (max_frames=4, no_audio) | ✅ | ✅ | — |
| create_memory_nodes | ✅ | ✅ | — |
| build_index | ✅ | ✅ | — |
| face_clustering | ❌ | ✅ | output not consumed by transcript-first QA retrieval — pure cost |
| voice_segmentation (energy VAD) | ❌ | ✅ | only feeds voice_clustering |
| voice_clustering (ECAPA / spectral) | ❌ | ✅ | only feeds speaker-aware retrieval, not used by QA |
| semantic_memory_consolidator | ❌ | ✅ | expensive LLM pass, not on the episodic QA path |
| qa | ✅ | ✅ | — |

Note: `episodic_memory.params.no_audio=true` disables audio *attachments*
to episodic nodes (storage bloat), never ASR. All our scored baselines
(0.267 → 0.514 full-100 → 0.479 sub8) used the baseline flag set — keep it
unchanged for comparability.

### 4.2 `run_maas.sh` verbatim (baseline wrapper)

```sh
#!/bin/sh
# Generalized M3 bench driver wrapper (runs inside ogmem-m3runner).
# Env knobs:
#   M3_VIDEOS        comma-separated video_ids; empty = ALL (robot.json)
#   M3_ACCOUNT       graph account (default m3bench)
#   M3_WORK_ROOT     work-dir root (default /work/maas-run)
#   M3_CONFIG        runtime config path (default /etc/ogmem/config.yaml)
#   M3_JUDGE         1 = per-video live judge on
#   M3_VIDEO_WORKERS video-level parallelism (default 1)
#   OGMEM_M3_MAX_WORKERS  per-stage workers (MaaS limits: use 4)
#   M3_QA_TIMEOUT    per-question ReAct budget (default 300)
VIDS=""
[ -n "$M3_VIDEOS" ] && VIDS="--videos $M3_VIDEOS"
JUDGE=""
[ "$M3_JUDGE" = "1" ] && JUDGE="--judge"
exec python3 /opt/ogmem/scripts/run_m3_bench_subset.py \
  $VIDS \
  --annotations /data_m3/annotations/robot.json \
  --videos-root /data_m3/videos \
  --work-dir-root "${M3_WORK_ROOT:-/work/maas-run}" \
  --config "${M3_CONFIG:-/etc/ogmem/config.yaml}" \
  --pipeline /opt/ogmem/config/m3/pipelines/video_episodic_full.yaml \
  --graph-store sql \
  --connection-string "$OGMEM_GRAPH_CONNECTION_STRING" \
  --account-id "${M3_ACCOUNT:-m3bench}" \
  --seconds 0 \
  --disable-stage speaker_diarization \
  --disable-stage speaker_embedding \
  --disable-stage voice_segmentation \
  --disable-stage voice_clustering \
  --disable-stage face_clustering \
  --disable-stage semantic_memory_consolidator \
  --set extract_frames.params.max_frames=4 \
  --set episodic_memory.params.max_frames=4 \
  --set episodic_memory.params.no_audio=true \
  --qa-top-k 8 \
  --set qa.params.transcript_top_k=20 \
  --qa-timeout "${M3_QA_TIMEOUT:-300}" \
  --set qa.params.resume=true \
  --video-workers "${M3_VIDEO_WORKERS:-1}" \
  $JUDGE \
  --python python3
```

`run_maas_full.sh` is byte-identical minus the six `--disable-stage` lines
(and its default work root is `/work/sub8-run-full`).

## 5. Running an experiment (baseline, reduced stages)

**Always create a fresh database first** — M3 node IDs are deterministic and
NOT account-scoped, so re-running the same videos against an existing DB
collides with old nodes:

```bash
sg docker -c "docker exec rychkov-ogmem-postgres psql -U ogmem -d ogmem \
  -c 'CREATE DATABASE \"ogmem-<run-name>\" OWNER ogmem;'"
```

Launch (detached, logs to `/work/<run-name>.log` on the host):

```bash
VIDS="study_09,kitchen_01,bedroom_05,study_23,kitchen_23,study_05,gym_03,bedroom_04"
RUN=sub8-run            # work root name AND db suffix
DSN="postgresql://ogmem:${PG_PASSWORD:-<password>}@127.0.0.1:15432/ogmem-$RUN"

sg docker -c "docker exec -d ogmem-m3runner sh -c '
  M3_VIDEOS=$VIDS \
  M3_ACCOUNT=m3bench \
  M3_WORK_ROOT=/work/$RUN \
  M3_CONFIG=/work/ogmemory.m3-ds.yaml \
  M3_JUDGE=1 \
  M3_VIDEO_WORKERS=2 \
  OGMEM_M3_MAX_WORKERS=4 \
  M3_QA_TIMEOUT=300 \
  M3_STORAGE_DSN=\"$DSN\" \
  M3_VECTOR_DSN=\"$DSN\" \
  OGMEM_GRAPH_CONNECTION_STRING=\"$DSN\" \
  sh /work/run_maas.sh > /work/$RUN.log 2>&1'"
```

Monitor: `tail -f /data/m3_work_rychkov/$RUN.log` (driver) and
`/data/m3_work_rychkov/$RUN/<vid>/pipeline.log` (per-video).

### 5.1 Running the WHOLE dataset (all 100 robot videos)

Same commands — just leave `M3_VIDEOS` **empty** (the wrapper then passes no
`--videos` filter and the driver loads all 100 entries from `robot.json`).
For a dataset-scale run, wrap the driver in a resume loop so transient
failures (LLM timeouts, episodic single-clip aborts) heal themselves —
this is `full_run_loop.sh` in `/work` (host `/data/m3_work_rychkov/`):

```sh
#!/bin/sh
# Loop the bench driver (with per-video judge) until every video in
# robot.json has complete QA. Resume-safe: each pass skips completed
# stages/questions via the smart-resume probe layer.
LOG=/work/<run-name>-loop.log
PASS=0
while :; do
    PASS=$((PASS+1))
    echo "=== pass $PASS $(date -u +%FT%TZ) ===" >> "$LOG"
    M3_JUDGE=1 sh /work/run_maas.sh >> "$LOG" 2>&1   # + the env from §5
    if python3 /work/check_complete.py /data_m3/annotations/robot.json \
        "${M3_WORK_ROOT:-/work/maas-run}" >> "$LOG" 2>&1; then
        echo "=== ALL COMPLETE after $PASS passes ===" >> "$LOG"
        break
    fi
    echo "--- sleeping 120s before resume pass ---" >> "$LOG"
    sleep 120
done
```

`check_complete.py <annotations.json> <work_dir_root>` (also in `/work`)
exit-0s only when every video's `m3_qa_results.jsonl` has all questions
answered with no error rows — it's the loop's termination gate.

Full-100 reference numbers (our run, same models/knobs as §5):
- **0.514 weighted mean**, 1269/1276 answered, ~25 h wall at 2 video
  workers (ingest ≈ 16 h, QA ≈ 6 h, judge ≈ 14 min)
- ~15.3M LLM tokens ingest + ~40.7M QA; per-video pipeline ≈ 5–8 min
  (ASR-bound), QA ≈ 5–10 min
- Expect a handful of resume passes — budget ~10–15% extra wall time

**Resume pass** (retry failed/unanswered questions — completed videos and
stages are skipped; use the 600 s config for stubborn glm-5.3 timeouts):

```bash
# same command with:
#   M3_VIDEOS=<only the incomplete videos>
#   M3_CONFIG=/work/ogmemory.m3-ds600.yaml
#   M3_QA_TIMEOUT=600
```

**Manual per-video judge** (needed on the *current container image only* —
its driver predates the digest-bound judge caching on our PR branch; after
an image rebuild the resume pass re-judges automatically):

```bash
sg docker -c "docker exec ogmem-m3runner sh -c 'python3 -m m3.eval \
  --annotations /data_m3/annotations/robot.json \
  --input-qa-results /work/$RUN/<vid>/m3_qa_results.jsonl \
  --output-dir /work/$RUN/<vid> \
  --config /work/ogmemory.m3-ds.yaml \
  --graph-store sql --account-id m3bench \
  --connection-string \"$DSN\" --video <vid>'"
```

**Final aggregate re-merge** (after all videos complete — rerun the launch
command with the full `M3_VIDEOS` list; everything is skipped and only the
`_aggregate/` merge refreshes):

```
/data/m3_work_rychkov/$RUN/_aggregate/m3_eval_summary.json   # question-level mean
```

## 6. Full-stages run (speaker diarization + ECAPA + clustering + consolidator)

One-time setup inside the container (the image deliberately ships without
the speaker stack; **version pairing matters** — mismatched torch/torchcodec
crashes natively):

```bash
# pip (as root; ~2 GB)
sg docker -c "docker exec -u 0 ogmem-m3runner sh -c '
  pip install --no-cache-dir torch==2.15.0 torchaudio==2.11.1 torchcodec==0.17.0 \
    --index-url https://download.pytorch.org/whl/cpu && \
  pip install --no-cache-dir \"pyannote.audio>=4,<5\" speechbrain==1.1.2'"
```

Models (HF token required; `pyannote/speaker-diarization-3.1` is **gated** —
accept the EULA once at huggingface.co/pyannote/speaker-diarization-3.1 with
the token's account; token: `hf_...` (set it via the `HF_TOKEN` environment variable; never commit a real token):

```bash
sg docker -c "docker exec ogmem-m3runner sh -c 'export HF_TOKEN=hf_...; python3 - <<EOF
from huggingface_hub import snapshot_download
for repo in [\"pyannote/speaker-diarization-3.1\", \"pyannote/segmentation-3.0\",
            \"speechbrain/spkrec-ecapa-voxceleb\"]:
    snapshot_download(repo)
EOF'"
```

Launch — same command as §5 but with the full-stages wrapper and three extra
env vars (`HOME=/hf-cache` matters: the container's default `HOME=/` breaks
the speechbrain cache dir — fixed on our PR branch, but the current image
needs the override; `MPLCONFIGDIR` silences fontconfig noise):

```bash
sg docker -c "docker exec -d ogmem-m3runner sh -c '
  export HOME=/hf-cache HUGGINGFACE_TOKEN=hf_... HF_TOKEN=hf_... MPLCONFIGDIR=/tmp/mpl
  M3_VIDEOS=$VIDS M3_ACCOUNT=m3bench M3_WORK_ROOT=/work/$RUN-full \
  M3_CONFIG=/work/ogmemory.m3-ds.yaml M3_JUDGE=1 M3_VIDEO_WORKERS=2 \
  OGMEM_M3_MAX_WORKERS=4 M3_QA_TIMEOUT=300 \
  M3_STORAGE_DSN=\"$DSN\" M3_VECTOR_DSN=\"$DSN\" OGMEM_GRAPH_CONNECTION_STRING=\"$DSN\" \
  sh /work/run_maas_full.sh > /work/$RUN-full.log 2>&1'"
```

Runtime shape (8 videos, 2 workers): baseline ≈ 70 min; full-stages ≈ 2.5 h —
diarization dominates (~1× realtime on CPU, ~15–17 min per 25–35 min video);
ECAPA ~25 s, face clustering ~13 s, voice clustering ~6 s,
semantic consolidator ~1 min per video.

## 7. Gotchas (all hit live — read before debugging)

1. **Fresh DB per experiment** (deterministic node IDs — see §5).
2. **glm-5.3 QA cap** → 2 video workers max; a single glm-5.3 call can hang
   past even 600 s on rare prompts (we saw 1/97 questions fail 4/4 attempts).
3. **`no_audio=true` disables audio *attachments*, not ASR** — transcripts
   always flow into episodic memory and QA retrieval.
4. **Episodic stage aborts the video on one bad clip** (malformed LLM output
   after retries → "Stage episodic_memory failed; stopping pipeline").
   Just re-run the resume command — only the failed clip re-attempts.
5. **Old-image judge staleness**: resume passes on the current image don't
   re-judge changed predictions (pre-review-item-9 driver) — run the manual
   judge (§5). Fixed on PR branch; rebuild the image to make it go away.
6. **Duplicate QA records on resume** (retry appends, doesn't replace) —
   scores stay correct (judge takes latest) but merged counts inflate;
   fixed on PR branch.
7. Embedder `base_url` must include `/v1`; embedder dims (1024) must equal
   `vector_db.dimension`.
8. `sg docker -c` wrapper needed in shells started before the docker-group
   change; plain `docker` works in fresh login shells.

## 8. Artifacts per run

```
/data/m3_work_rychkov/<run>/<vid>/m3_eval_summary.json     # per-video judge summary
/data/m3_work_rychkov/<run>/<vid>/m3_eval_results.jsonl    # per-question judged rows
/data/m3_work_rychkov/<run>/<vid>/m3_qa_results.jsonl      # raw QA predictions
/data/m3_work_rychkov/<run>/<vid>/pipeline.log             # full pipeline log
/data/m3_work_rychkov/<run>/_aggregate/m3_eval_summary.json  # question-level aggregate
/data/m3_work_rychkov/<run>.log                            # driver log
```

Reference results (8 videos, 97 questions): baseline 0.4794 / full-stages
0.4948 (question-level mean); reports:
`REPORT-sub8.md`, `REPORT-sub8-full.md` in `/data/m3_work_rychkov/`.
