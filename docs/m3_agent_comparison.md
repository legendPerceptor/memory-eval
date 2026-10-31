# M3-Agent vs. the current oGMemory M3 pipeline

## Scope and evidence

This is a **code-level comparison, not a head-to-head benchmark result**.
The reference checkout is `m3-agent/`, commit
`0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c`, initially clean on `master`.
The oGMemory side is `dev` at `8c86c3ed` plus the experiment's recorded
reasoning-parameter and bounded-worker patches.

The reference `robot.json` is canonically identical to `/data/m3_data/annotations/robot.json`.
All five selected videos and all 65 QA records match. Hashes and checks are saved in
`docs/m3_agent_review.json`. This establishes dataset alignment, **not protocol alignment**.

No M3-Agent checkpoint was loaded, no pickle graph was deserialized, no reference
API was called, and no M3-Agent evaluation was run for this review. Its expected
local checkpoint directories are absent in this checkout. The reference repository
has not been modified, and its setup script was not run in our active GPU environment.
Published README charts are not imported as results of our five-video experiment.

## 1. How the reference implementation works

### 1.1 Two paths: memorization and control

The main entry points are:

1. `m3_agent/memorization_intermediate_outputs.py`: per-clip face/audio preprocessing.
2. `m3_agent/memorization_memory_graphs.py`: construct one `VideoGraph` per video,
   then serialize it to the annotation's `mem_path` with pickle.
3. `m3_agent/control.py`: load a memory graph, iteratively search/answer, then grade.

The README describes a trained memorization model and a trained control model,
with training code linked separately. In this checkout we can verify the inference
calls, not reproduce or independently validate the training procedure.

### 1.2 Perception and persistent identity anchors

The preprocessing helper `process_video_clip` samples the clip at **5 fps** by
default and extracts 16-kHz audio. These frames support face processing; this is
**not evidence that the memorization model consumes exactly 150 frames** for every
30-second clip. The Qwen path sends a video file to `process_mm_info` and the model
processor, which determine the model's actual video/audio sampling.

Face processing uses InsightFace `buffalo_l`, quality filtering, within-clip
clustering, and matching against graph face nodes. The checked-in face initialization
passes `ctx_id=-1`; a GPU deployment would need an explicitly recorded adaptation,
not an assumption that the reference defaults match our CV service.

Voice processing uses `gemini-1.5-pro-002` to obtain time-stamped dialogue/transcription
when cached intermediate outputs are absent, filters segments shorter than two seconds,
and computes 192-dimensional ERes2NetV2 speaker embeddings through `speakerlab` on CUDA.
It matches those embeddings to voice nodes in the video graph. It is **not the same**
ASR/diarization stack as our Whisper + pyannote + ECAPA pipeline.

Face and voice anchors retain bounded embedding sets: configuration defaults are
10 image embeddings and 20 voice embeddings, with matching thresholds 0.3 and 0.6.
Updates can randomly subsample embeddings, which is another reproducibility detail.

Sources: `mmagent/utils/video_processing.py::process_video_clip`,
`mmagent/face_processing.py::process_faces`,
`mmagent/voice_processing.py::process_voices`,
`mmagent/videograph.py::{search_img_nodes,search_voice_nodes,update_node}`,
`configs/memory_config.json`.

### 1.3 Joint episodic and semantic generation

The default Qwen memorization path supplies:

- the clip's video, with audio enabled in the multimodal processor;
- face crops tagged with identifiers such as `<face_17>`;
- speaker segments tagged with identifiers such as `<voice_8>`, timestamps and ASR.

The prompt requests atomic event descriptions and **inferred high-level conclusions**:
intentions, relationships, identities, and explicit
`Equivalence: <face_x>, <voice_y>` statements. Thus identity evidence is injected
into generation, not merely stored in a parallel CV branch.

`memory_processing_qwen.py` embeds individual memory strings with
`text-embedding-3-large`. Episodic strings become new text nodes connected to
referenced face/voice entities. Semantic strings either become new nodes or affect
related existing semantic nodes: similarity above 0.85 reinforces edges; similarity
below 0 weakens edges. These are **cosine-similarity heuristics**, not a verified
logical contradiction solver. In the negative branch, the current code suppresses
new-node creation too; comments should not be substituted for actual behavior.

`VideoGraph` has four main node types: `img`, `voice`, `episodic`, and `semantic`,
plus bidirectional weighted edges. `refresh_equivalences` uses equivalence memories,
conflict-selection heuristics, and union-find to map face/voice anchors into
`character_N` identities.

Sources: `mmagent/memory_processing_qwen.py::{generate_video_context,generate_all_memories,process_memories}`,
`mmagent/utils/chat_qwen.py::{get_response,generate_messages}`,
`mmagent/prompts.py::prompt_generate_memory_with_ids_sft`,
`mmagent/videograph.py::{add_text_node,reinforce_node,weaken_node,refresh_equivalences}`.

### 1.4 Retrieval uses the identity graph

In the reference control path, the model is encouraged to first resolve a name to
a `character_N`, then search using that ID.

`retrieve.py` expands character IDs back into face/voice IDs and can restrict text
candidates to episodic/semantic nodes connected to those entities. It embeds queries
with `text-embedding-3-large`, scores memory nodes, aggregates scores by clip
(default max pooling), and returns the text memories of selected clips.

The main controller uses **top-k=2 clips**, threshold 0.5, and avoids returning clips
already present in `current_clips`. A special query containing the literal string
`"character id"` uses memory-wise retrieval with top-k=20. Returned face/voice
references are translated into character IDs, and explicit equivalence strings are
removed from the displayed memory list after contributing to identity mapping.

This is entity-conditioned vector retrieval plus iterative control. It should not
be described as unrestricted multi-hop graph traversal merely because a graph exists.
Likewise, the generic GPT-style `answer_with_retrieval` helper offers other settings,
but it is **not** the active default controller in `m3_agent/control.py`.

Sources: `mmagent/retrieve.py::{translate,back_translate,get_related_nodes,retrieve_from_videograph,search}`,
`mmagent/videograph.py::search_text_nodes`, `m3_agent/control.py::consumer`.

### 1.5 Control and evaluation defaults

`m3_agent/control.py` loads `models/M3-Agent-Control` through vLLM with tensor
parallel size 2. The checked-in settings use batch size 64, five rounds,
temperature 0.6, top-p 0.95, top-k 20, and at most 1024 newly generated tokens.
Actions are parsed from textual `Action: [Search]` / `Action: [Answer]`, rather
than native API function calls. On the final round, the prompt explicitly requires
an answer and permits reasonable guesses.

The default QA input is textual memory; it does not reopen raw video for each
question. Our current oGMemory QA also consumes retrieved textual memory, despite
both systems using audiovisual input during memorization.

The reference judge is Azure `gpt-4o-2024-11-20`, with a **binary Yes/No**
entailment-style prompt. Empty answers and judge exceptions both become `False`
in `eval_answer`; a comparison harness should separately audit those failures.

Sources: `m3_agent/control.py`,
`configs/processing_config.json`, `mmagent/prompts.py::prompt_agent_verify_answer_referencing`.

## 2. The key design differences

| Dimension | Reference M3-Agent path | Current oGMemory experiment |
|---|---|---|
| Core organization | Entity-centric per-video multimodal graph | Provenance-oriented Raw → Extracted → Memory layers in SQL |
| Identity during generation | Tagged face crops, voice segments and explicit equivalence facts | Frames + raw audio + diarized ASR; no equivalent face-ID/voice-ID prompt contract |
| Semantic memory | LLM-generated high-level claims, reinforcement/weakening, identity equivalences | Deterministic per-video concatenation and set union, description capped at 50,000 UTF-8 bytes |
| Actual QA targets | Episodic and semantic text, optionally filtered by linked entities | Default extractor filter selects `episodic_memory_generator` evidence |
| Identity during retrieval | Character ↔ face/voice expansion changes candidate selection | Face/voice clustering is executed, but default Search does not explicitly use that identity mapping |
| Retrieval unit | Memory-node similarity → clip score → all memories in selected clips | Clip-level episodic evidence, transcript retrieval, RRF and exact neighboring clips |
| Text embeddings | `text-embedding-3-large` | Local GPU BGE-M3, 1024 dimensions |
| Audio stack | Gemini dialogue segmentation + ERes2NetV2 | Whisper large-v3-turbo + pyannote + ECAPA |
| Visual input | Preprocessing at 5 fps; video processor controls memorization inputs | 1 fps, first up to 16 frames per 30-second clip |
| Controller | Dedicated checkpoint, textual action parser, 5 rounds, final-round guessing | MiniMax-M3.1-Flash, native tools, 6 rounds, 180-second loop budget |
| Persistence and scope | Python `VideoGraph` + per-video pickle | SQL graph/vector index, account/video filters, independent experiment database |
| Diagnostics | Entry-point-local outputs and retries | Stage summaries, failure receipts, fixed-cohort manifest, model-usage and SQL audits |
| Current grade | Binary GPT-4o judgement | Same-model MiniMax judgement in `{0, 0.5, 1}` |

**Interpretation:** oGMemory is not simply “M3-Agent with SQL” and is not currently
a complete implementation of M3-Agent's identity-aware semantic-memory mechanism.
Our strongest implemented differences are persistence, provenance, explicit
failure handling and the transcript-enhanced retrieval stack. M3-Agent more
directly connects identity grounding and inferred semantic facts to the answering
path. Neither observation establishes a quality winner without a controlled run.

oGMemory source anchors: `config/m3/pipelines/video_episodic_full.yaml`,
`extraction/operators/builtin/episodic_memory_generator.py`,
`m3/semantic_memory_consolidator.py::consolidate`,
`m3/control_adapter.py::{SearchConfig,Search}`,
`m3/temporal.py::resolve_question_scope`, `m3/eval.py::judge_answer`.

Engineering qualifications: our pipeline is not one atomic transaction; failed
stages can leave recoverable partial writes. Legacy clip IDs are not fully
account-scoped, so an independent database—not merely another account—was used
for this experiment. The sample manifest fixes the cohort, not every runtime
setting; source/config fingerprints and separate run roots remain necessary.

## 3. Why the current scores are not yet comparable

1. **Different scoring rules.** Our soft mean is not M3-Agent's binary accuracy.
   A half-credit answer must not silently become a binary success. Prefer reporting
   a common binary judge for both, plus the same secondary soft score for both.
2. **Different models and compute budgets.** Dedicated checkpoints versus MiniMax,
   different embeddings, five versus six rounds, and different output budgets
   confound an architecture-only interpretation.
3. **Different visibility.** The reference control script reads `before_clip`;
   it does not use the annotation timestamp as an absolute-second cutoff.
   Our `current_clip` policy uses the timestamp and includes the current clip's end.
   With nominal 30-second, zero-indexed clips, `study_05_Q01` at 03:45 and
   `before_clip=6` illustrates a roughly 210-second versus 240-second endpoint.
   Exact limits must be computed from source intervals, not assumed from this example.
4. **Different audiovisual evidence.** Matching clip duration does not match frame
   coverage, face crops, ASR, or speaker segmentation.
5. **Different coverage behavior.** Forced final guesses versus timeout/no-answer
   changes answer coverage and apparent accuracy. Report both and retain failures.
6. **Precomputed memory is a different workload.** Running the reference controller
   on supplied `.pkl` graphs is QA-only, not an end-to-end memory-building baseline.
   Do not use its timing as if it included perception and memorization.

For full-video prebuilt graphs, there is an additional **static-review temporal risk**:
later clips can strengthen earlier semantic edges and update identity anchors.
Deleting future text nodes does not by itself prove that earlier graph state has
been restored. Audit this or build prefix snapshots for a strict-causal comparison.
This is a review finding, not a measured leakage rate.

## 4. Proposed evaluation matrix

### Track A: reference-system baseline

- Freeze the checked-out revision and use the same five videos/65 questions.
- First run **reference graphs + official controller** as a clearly labelled QA-only baseline.
- Optionally run fresh reference memorization + official controller as a separate
  end-to-end experiment; account for its preprocessing and model requirements.
- Record every adaptation needed for this machine and retain the original defaults.
- Re-grade both systems' saved predictions with the same independent judge and
  prompt; retain original grades as separate diagnostic columns.
- Publish both native visibility policies and a separate matched-cutoff run.
  Native runs with different visibility are a system comparison, not a controlled
  claim that one memory design is better.

### Track B: model-matched architecture ablation

Use the same memorization/answer backbone, embedder, judge, audiovisual budget,
time cutoff, answer policy and compute budget. Preserve the architectural difference:
entity/equivalence-aware memories and retrieval versus clip-level episodic/transcript
retrieval. Such a port must be called **M3-Agent-style/adapted**, not the released
M3-Agent checkpoint result.

Match delivered evidence size, not just the numeric top-k: two clips containing
many atomic memories are not equivalent to eight episodic nodes plus neighbors.
When replacing an embedder, rebuild all relevant memory embeddings; equal vector
dimensions alone would not make old and new embedding spaces compatible.

### Useful isolated ablations

1. Identity equivalence in generation and retrieval, on/off.
2. Generated high-level semantic facts versus our deterministic video aggregation.
3. Same number of prefix frames versus whole-clip coverage.
4. Frozen memory, unbounded versus bounded QA context.
5. Current-clip versus strict matched temporal cutoffs.

Report fixed-denominator scores, answer/judge coverage, error types, per-video and
per-label results, model usage, actual input peaks, stage wall time, and memory
coverage. Distinguish interrupted attempts, valid-but-wrong answers, no-answer
timeouts, judge failures and infrastructure failures.

## 5. Preflight issues in this checkout

These are visible code inconsistencies or deployment requirements, **not claims
that the paper's reported experiment failed**:

- README sample data uses `intermediate_path`; both memorization entry points read
  `sample["intermediate_outputs"]`. Normalize the input contract explicitly.
- The SFT prompt demonstrates `video_description` (singular), while
  `generate_all_memories` reads `video_descriptions` (plural). The JSON fixer does
  not rename that key. A model following the prompt literally could raise a KeyError;
  test the actual checkpoint/API response before scaling out.
- Clip traversal uses unsorted `glob.glob`, while temporal truncation relies on
  insertion/node-ID ordering. Enforce and test numeric clip order in any reproduction
  patch, rather than relying on filesystem enumeration order.
- `generate_all_memories` can fall back to empty lists after parsing retries;
  record this as missing memory, not a successful full-coverage video.
- The README asks for different memorization/control dependency setups.
  `setup.sh` contains global `sudo pip` installs; do **not** run it against the
  existing oGMemory GPU environment. Use separate environments/containers.
- Verify the declared FlashAttention backend, official checkpoints, speakerlab
  weights, embedding/Gemini/Azure interfaces and credentials on this machine.
  API names in source code do not prove present-day availability.
- Existing `/data/m3_data/memory_graphs/` archives are available, but were not
  unpacked or executed in this review. Verify selected graph paths, compatibility,
  provenance and embedding model before loading trusted pickle files.

## Recommended next step

Start with a pinned, separately recorded **QA-only reference baseline on the same
65 questions**, once the official control checkpoint and graph prerequisites are
ready. In parallel, freeze our existing memories for the bounded-context QA
ablation. Do not rebuild all audiovisual memories merely to change the answer
controller, and do not change the current five-video run mid-experiment.
