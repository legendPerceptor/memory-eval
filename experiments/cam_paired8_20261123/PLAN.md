# CAM on M3-Bench paired-8 (same MiniMax-M3.1-Flash model)

## Goal
Run CAM (https://github.com/Jake-Tian/CAM, commit 8b98c0b) on the same eight
M3-Bench robot videos / 97 questions as `m3_paper_paired8_20261103`, with the
same model (`MiniMax-M3.1-Flash`, reasoning medium), re-scored with the oGMemory
0/0.5/1 judge so it is directly comparable to oGMemory (0.5722) and
m3_agent (0.4381). CAM is a *related work* baseline here — we run it as-is and
report its true performance; we do NOT tune it.

## Cohort (97 questions)
study_09(12), kitchen_01(13), bedroom_05(12), study_23(14),
kitchen_23(12), study_05(12), gym_03(11), bedroom_04(11)

## Environment / adaptations (CAM logic unchanged; only I/O adapters)
- Python venv `/tmp/camvenv` (3.10): openai, pydantic, langchain*, langgraph,
  opencv-python-headless==4.10.0.84, pytest.
- LLM adapters `utils/llm_gpt.py` / `utils/mllm_gpt.py`: OpenAI-compatible Chat
  Completions (MiniMax). Structured output via `json_object` + schema + pydantic
  validate/retry (MiniMax ignores `json_schema`). `reasoning_effort=medium`,
  `thinking adaptive`, timeout/max_tokens configurable.
- Embedding `utils/embedding.py`: local bge-m3 (127.0.0.1:8001, 1024-d), batched
  <=8 (server rejects >8), + transport reliability (retry w/ backoff, mild pace).
- Reasoning agent `reasoning/agent.py`: Chat Completions tool calling.
- `reasoning/tools.py`: accept 3-tuples from MiniMax as 6-tuples (weight 1.0).
- Videos: local mp4 symlinked into `CAM/data/videos/`; subtitles from HF
  `ByteDance-Seed/M3-Bench` into `CAM/data/subtitles/robot/`.

## Durability / resume (the whole run auto-resumes across quota + restarts)
- `process_full_video.py`: per-clip checkpoint `data/graphs/<video>_checkpoint.pkl`
  (graph + appearance + episodic + stage_usage + done_clips). On provider
  quota (429) or two consecutive MLLM failures it aborts WITHOUT writing a
  truncated graph and KEEPS the checkpoint. Removed on clean finish.
- `reason.py`: resumes per-question from partial `data/reasoning/<video>.json`;
  quota error aborts with partial results kept; writes `data/reasoning/<video>.done`
  only when all questions answered.
- `utils/repair_embeddings.py`: before reasoning, fills any `None`
  edge/node/scene embeddings (deterministic recompute). Safety net for the TEI
  bge-m3 server's occasional queue-overflow restarts, so lost embeddings during
  construction cannot silently degrade CAM's retrieval (fair to CAM).
- `CAM/orchestrator.sh`: single-instance (flock) durable supervisor. Keeps the
  `run_full_job.sh` worker pool (MAX_PARALLEL=4) alive; revives the embedding
  container if it dies; once all 8 `.done` exist it auto-runs the judge and the
  comparison builder. Re-launch is idempotent.
- `run_full_job.sh`: true worker pool (a long 79-clip video never blocks others);
  per video MEMO (build graph) then REASON, each auto-retried after quota.

## Pipeline
1. extract frames (1 fps, 30-frame clips)  [DONE all 8]
2. process_full_video.py (per-clip MLLM episodic + triples, then abstraction +
   embeddings)  [IN PROGRESS — resuming from per-clip checkpoints]
3. reason.py (five-tool agent, budget 6) per video  [auto after each graph]
4. judge_cam.py (oGMemory 0/0.5/1 judge; resumable; aborts on quota) -> cam_summary.json
5. build_comparison.py -> CAM vs oGMemory vs m3_agent

## Status
- Memory building ~36% (200/547 clips) and advancing; 4-way parallel;
  ~350-450 s/clip/worker (single 30-frame visual MLLM call is 207-280 s, model-
  intrinsic). No account quota hit yet in this run.
- An earlier attempt lost progress at 336/547 because old code had no
  per-clip checkpoint; that is fixed and the current run resumes from checkpoints.

## Notes / limitations to report
- Adaptations are I/O only (MiniMax chat API, json_object structured output,
  local bge-m3 with batch<=8 and retries). CAM's prompts/algorithm are unchanged.
- CAM's built-in binary judge is NOT used for the headline score; we re-judge all
  answers with the exact oGMemory 0/0.5/1 prompt for a like-for-like comparison.
- Single run, greedy (temperature 0). Visual MLLM latency makes building slow
  (~10 h remaining for graphs, then reasoning).
