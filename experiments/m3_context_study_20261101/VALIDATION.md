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
