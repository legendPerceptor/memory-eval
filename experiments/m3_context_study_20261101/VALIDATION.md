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

## Database snapshots

`database_early.json` is a read-only snapshot taken **after A had started** and
before B. It covers row contents of graph_nodes, graph_edges, and
m3_episodic_vectors in both source databases. Final comparison can establish
unchanged data from that time onward, not falsely prove an earlier snapshot existed.
