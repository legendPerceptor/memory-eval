# Validation log

## First implementation: oGMemory 9eb782e4

- `tests/unit/m3/test_answer_reserve.py` + `test_context_budget.py`:
  **35 passed** (`unit-context-tests.log`).
- `tests/unit/m3 tests/unit/providers/llm tests/unit/core/test_react_loop.py`:
  **1023 passed** (`unit-regression-tests.log`).
- Expanded M3 integration/provider/core run before the last additional reserve
  test: **1369 passed, 108 skipped, 8 failed, 1 deselected**.
  This is not an all-green repository test run (`regression-tests.log`).
- All eight failures were independently reproduced on frozen pre-change
  **c992d45b** using the corresponding three integration test files:
  **8 failed, 11 passed** (`frozen-baseline-failures.log`).
  They concern legacy config resolution, embedding/index MVP assumptions,
  unresolved local deployment placeholders, and duplicate-node resume behavior.
  They do not become fixed merely by excluding them.
- The one explicitly deselected extract-audio duplicate-node test was already
  independently reproduced before this work; see the study_18 recovery logs.
- `git diff --check` and Python compilation of the new experiment scripts passed.

New tests verify:

1. Reserve opt-in and invalid configuration rejection.
2. Observed previous-call latency causes earlier answer-only mode.
3. Final round removes Search from schema **and** blocks a model ignoring schema.
4. Reserve-disabled mode keeps legacy tools.
5. An overlong request is still a timeout, with no hidden extra round.
6. Per-question state reset.
7. Packing retains no-new-evidence guidance without adding facts or mutating raw input.

## Telemetry follow-up

- **1024 passed** for the same M3/provider/core unit selection
  (`telemetry-regression-tests.log`).
- Added visible-passage novelty counters, distinct from retrieved node IDs.
  They are diagnostic only and do not change no-gain decisions.
- Fixed generic ReAct trace tool names to accept both `name` and `tool`, matching
  actual dispatch. Previous OpenAI-compatible traces had empty tool names.
- A/B frozen sources are unchanged. C includes these diagnostic-only additions.

## Database snapshots

`database_early.json` is a read-only snapshot taken **after A had started** and
before B. It covers row contents of graph_nodes, graph_edges, and
m3_episodic_vectors in both source databases. Final comparison can establish
unchanged data from that time onward, not falsely prove an earlier snapshot existed.

## Grouped evidence and quota protection

- Grouped format + budget + answer-reserve tests: **45 passed**.
- M3/provider/core unit selection: **1033 passed**.
- Experiment reporting, interleaved-judge input checks, exact-source replay,
  and quota circuit breaker: **14 passed**.
- Quota tests use stubs only: first provider failure is persisted, subsequent
  calls stop, including when the first quota error occurs on the last question.
- Grouped format tests preserve all admitted transcript/event/character/entity
  values, nested metadata, temporal provenance, content-version distinctions
  and valid references. A constrained test leaves optional content omitted;
  it does not assert that arbitrarily small budgets can fit an entire clip.
- Real-data offline replay verifies every selected source by SHA256 before
  reconstructing excerpts. No model or embedding calls are made.
- D has **not** been quality-evaluated online: provider quota was exhausted
  before its launch. Offline format savings must not be reported as QA gains.

## Committed-head expanded check and full-trajectory replay

- On09f3cf30: **1380 passed,108 skipped,8 failed,1 deselected**;
  all8 failing test IDs exactly match the frozen-baseline reproduction.
- With only those8 plus the previously reproduced audio case excluded:
  **1380 passed,108 skipped,9 deselected**.
  This is not a claim that the excluded issues were fixed.
- Root analysis/recovery/reporting tools: **25 passed**.
- The initial offline format audit used prepared views from the interrupted A
  run, including one view whose API call did not return usage.
  The stronger full65 replay now uses the recovered fixed trajectory, with
  all call views paired to received usage:294 evidence-bearing calls,
  332 source versions,7268 source occurrences,43086 passages, zero mismatches.
  Evidence-only proxy tokens4,181,181 ->3,338,182 (20.1617% reduction).
  Both audits remain stored; neither measures D's answer quality.

## Configuration utility and current release

- `m3.configure_qa_context` creates a new mode0600 configuration copy, keeps
  model/storage/auth fields unchanged, refuses overwrite, and validates a local
  tokenizer without network/model/storage calls. It never publishes a
  credential-bearing config hash.
- Balanced/compact/grouped configuration tests plus the context suite:
  **52 passed** before the additional same-path tokenizer-pin test.
- Current-file tokenizer-pin regression: **7 passed** in the utility suite.
- Release unit selection before that extra pin test: **1040 passed**;
  expanded selection with the9 known exclusions: **1387 passed,108 skipped,9 deselected**.
  A final committed-head pass will be recorded before delivery.
- Root reporting/recovery/integrity tests currently **28 passed**; the later
  B-repeat summary test will be included in the final run.
- The real local configuration-copy validation produced only a private
  example file and a nonsecret receipt (`config-tool-validation.json`), no model
  or database calls. It must not be applied retroactively to frozen runs.

## Final committed code check (83e51d98)

- M3/provider/core unit selection: **1041 passed**.
- Expanded selection with the same9 previously reproduced exclusions:
  **1388 passed,108 skipped,9 deselected**.
- Exact command and revision: `release_validation_revision.json`;
  logs: `final-head-unit-tests.log`, `final-head-known-exclusions-tests.log`.
- Root analysis/recovery/integrity suite: **29 passed**, including preservation
  of both B runs rather than selecting the higher run and detection of changed
  prediction files or judge task fields.

## Input and artifact checks before D

- All root test files together: **39 passed**, including the legacy summarizer.
- Canonical annotation JSON matches the original sampling manifest. The manifest
  hashes `json.dumps(parsed, sort_keys=True, ensure_ascii=False)` with default
  separators, not raw file bytes. An initial raw-versus-canonical comparison was
  corrected and recorded; no annotation content change was found.
- Both databases' row-content fingerprints still match the early snapshot.
- The older M3 adaptation unit tests were independently rerun: **9 passed**.
- Archived prior CSVs only had record terminators normalized toLF; parsed field
  values, including embedded whitespace, were verified unchanged.

## Final artifacts

- All root tests: **48 passed**, including joint-judge recovery, renderer,
  input-integrity checks and actual-call ledger accounting.
- Joint evaluation:455 rows,448 real judge API calls and7 automatic zeros for
  invalid QA; zero judge errors, all reported models/request settings match,
  all calls returnedusage and none terminated at output length.
- Final frozen-input audit verifies455 task records and35 prediction-file hashes.
  All5 new-arm source snapshots match their recorded Git archives.
- Final database row-content fingerprints match the early snapshot; canonical
  annotations match the original sample manifest.
- PDF successfully built12 pages; final log has no undefined reference,
  missing-character or overfull warnings. Pages8–10 were visually inspected.
- `release_audit.json` records artifact hashes and the technical completion
  checks. User-facing delivery remains scheduled for23:30.
