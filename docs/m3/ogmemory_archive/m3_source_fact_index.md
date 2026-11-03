# Source-grounded fact indexing and incremental M3 QA

Experimental and opt-in. These features do not change your existing QA config.
Do not infer quality gains from the existence of the implementation: compare
complete fixed-cohort runs before selecting a profile.

## Build a new immutable index

Run from the oGMemory repository root (or an environment where its modules are installed).

```bash
python -m m3.build_fact_index \
  --config /path/to/private-config.yaml \
  --account-id YOUR_ACCOUNT \
  --video VIDEO_A --video VIDEO_B \
  --output /path/to/new-fact-index \
  --dry-run
```

Remove `--dry-run` to call the configured **embedding** service and write the
index. No chat model is called; no graph memory or audiovisual input is
regenerated or overwritten. The output directory must not exist, including
after a prior failed build. JSON and NumPy artifacts are mode0600 in a new
mode0700 directory. Never deserialize a pickle index.

To source one selected video from another store configuration, add
`--source-config VIDEO_B=/path/to/other-private-config.yaml`. Each selected video
has one authoritative store; all configs must declare the same embedding model
and dimensions. The CLI reads an existing SQL store in a read-only transaction,
binds the tenant using the normal RLS helper, and pushes account/type/video
filters into SQL. It does not instantiate the schema-bootstrapping store.
The collector checks video, node type and account again.

The source-node cap defaults to10000 per requested account/video subset and store.
Exceeding it fails rather than silently omitting older clips. This is a bounded
prototype; large requested subsets need deliberate partitioning rather than an unbounded scan.
The receipt counts indexable source nodes, not guaranteed full-video coverage;
check the original pipeline's coverage report separately.

Copy the returned manifest path and SHA256 into a **new** private QA config:

```yaml
m3:
  qa_context:
    enabled: true
    # Preserve the tested tokenizer, operational window and output reservation.
    fact_memory: true
    identity_aware: true
    incremental_delivery: true
    evidence_format: grouped
    max_passages_per_node: 12
    incremental_tokens: 5000
    answer_reserve_seconds: 45
    fact_index_path: /path/to/new-fact-index/manifest.json
    fact_index_sha256: REPLACE_WITH_THE_RETURNED_SHA256
```

Merge these keys with an existing validated context configuration; this snippet
does not replace model, tokenizer or connection settings. The lowercase SHA256
must be the actual64-character digest, not the placeholder above. Use the same
embedding model for QA queries as for index construction. Use a new QA output
directory, never resume old answers under a different retrieval configuration.

## What is and is not implemented

- Facts are original description excerpts, adjacent ASR windows and complete
  event/character/entity values, not new LLM semantic conclusions.
- Each proposed node is hydrated through scoped graph access and its complete
  original content hash is checked. A stale index is rejected at the node level;
  rebuild it after source memory changes.
- Names/roles are source claims. Matching them is a soft retrieval preference,
  not a verified mapping between a name, face and voice.
- Incremental delivery caches question-local evidence projections and tracks
  actually visible source positions. Capacity rollover rebuilds a verified
  context epoch; removed content is not treated as still visible.
- In incremental mode, `max_selected_nodes` and `max_passages_per_node` cap
  each new admission projection, not the entire retained history. The global
  input-token budget still covers all retained messages. Do not treat a
  32-node admission cap as a global 32-visible-node guarantee.
- `fact_scene_anchors` optionally preserves initial scene excerpts next to
  isolated facts. `force_answer_tool` optionally requests `required` when only
  Answer is available; use only a provider that supports this protocol.
  A provider may still return ordinary text. The existing valid-text fallback
  remains enabled; this option is not a guarantee of native-tool enforcement.
- Default new flags are false. `incremental_delivery` can also be tested alone
  with the existing narrative/passages representation and32-passage limit.
- The dense NumPy matrix is not a scalable ANN store or a live-updating semantic
  graph. The current facts have conservative availability at source-clip end.

The benchmark's existing index was built by its frozen experiment script.
This reusable command must not overwrite that artifact. Different batching or
ordering can produce a different index digest; register a new artifact and
experiment instead of silently changing the inputs of a previous run.
