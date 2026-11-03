# PR #300 selective integration and oGMemory-only rerun

## Branches

- Preserve `oGMemory/m3_dev_1031`: `c237bfef22e4474cd3d16c85d9182b15f2f7d613`.
- New branch (user-requested spelling): `intergation_1103`.
- **One integration commit**: `eceb96ff2958e72b373a7c3ea280d46827981351`.
- Reviewed PR head: `edcdcf1388a7224467d021e2972a05f06958012e`.
- No push or PR submission performed.

See `oGMemory/docs/m3_parallel_integration.md` for scope, attribution and knobs.
The existing main-branch LaTeX edits and prior experiment outputs are untouched.

## Validation

Executed on November 3, 2026:

| Checks | Result |
|---|---|
| M3 + extraction + graph/provider + dedicated SQL person/cache regressions | 1,109 passed |
| SQL pool/vector/embedding unit regressions | 119 passed, 12 skipped |
| Root experiment scripts | 113 passed |
| Final focused parallel/episodic worker tests (overlap above) | 22 passed |
| Final runner/instrumentation tests (overlap above) | 4 passed |
| Ruff on new parallel files, tests and runner | Passed |
| Core `git diff --check` | Passed |
| Local CV and embedding health | HTTP 200 |
| Real embedding request using configured BGE-M3 | 1 vector, 1,024 dimensions |

SQL tests used `ogmem_person_test_20261103`, not an experiment database.
Tests cover bounded pool acquisition and slot release on failure, copied
deadlines/context variables, per-thread adapters, immediate QA journaling,
resume without repeating completed outcomes, out-of-order judge pairing,
and concurrent usage attribution. Skipped provider tests are not claimed
as passing live-service tests.

## Experiment

- Name: `m3_parallel_ogonly_20261103`.
- Started: **2026-11-03 14:11 Asia/Shanghai**, detached coordinator PID 204927.
- Frozen source, driver, helper and usage instrumentation.
- Fresh database: `ogmem_m3_parallel_ogonly_20261103`.
- Fresh account: `m3-parallel-ogonly-20261103`.
- Same ten video IDs and 128 questions, in the original execution order:
  `study_09`, `kitchen_01`, `bedroom_05`, `study_23`, `kitchen_23`,
  `study_05`, `gym_03`, `bedroom_04`, `study_18`, `kitchen_20`.
- Same MiniMax-M3.1-Flash / adaptive / medium, QA generation cap 8,192,
  judge cap 2,048, six QA iterations, 180-second soft QA budget.
- Same full-interval extraction: 30-second clips, up to 30 extracted frames,
  16 evenly spaced base frames supplied to memory generation.
- Current base's person grounding, up to eight face crops, and scoped person
  retrieval are enabled. Single-video clustering is retained.
- Memory workers 8; QA 4; CPU extraction 4; HTTP CV 2; embedding requests 4;
  graph pool 16; vector pool 8; video concurrency 1; local GPU audio serial.
- GPU 2 is shared with existing CV and recovered embedding services. No
  unrelated services or LoCoMo jobs are stopped or reconfigured.
- Interval quota gate at 12%, weekly floor 5%, new 48-hour safety window.
  This does **not** inherit the previous experiment's November 3 14:32 deadline.
- QA gating is outside the adapter's soft timer. Usage labels are ContextVar
  values rather than mutable process-wide question counters.
- Failures in pipeline/index stop the coordinator for review. Completed QA
  outcomes with failures are retained and counted, not selectively retried.

**M3-Agent memory generation and QA are not rerun.** Its original answers,
grades and resource logs are frozen in the experiment baseline. New oGMemory
answers use the same judge implementation/model; M3 grades are reused.

## Measurement caveats

The previous ten-video job ended with failures:

- oGMemory: 109/128 valid answers; `kitchen_01` has two QA timeouts and
  `kitchen_20` has 17 missing answers after an embedding connection failure.
- M3 reference: 100/128 valid answers; `study_18` has 11 connection failures
  and `kitchen_20` has no predictions.
- Thus a lower old full-cohort token bill can reflect unexecuted work,
  **not** better efficiency.

Report both the full 128-question cohort and the matching first-eight-video
97-question subset. Retain failed outcomes in the denominator. Separate memory
generation/indexing from QA, count all attempts and explicitly count unknown
usage. Record per-video and per-stage wall time, including quota waiting.

The old source was `8d61b772`, before current person grounding. The new
experiment is an integrated-system comparison, **not a pure concurrency
ablation**. Person images can increase memory input tokens even if QA becomes
more efficient. Pure parallelism is expected to overlap waiting, not to reduce
logical prompt token requirements.

Old episodic generation already used four workers; moving to eight does not
offer an eightfold speedup over that baseline. Serial GPU/audio work, endpoint
rate limits, retries and quota windows limit end-to-end gains. Do not publish
a numerical speedup or token reduction before measurements exist.

## Files and operation

Public progress: `experiments/m3_parallel_ogonly_20261103/status.json`.
Per-phase receipts: `experiments/m3_parallel_ogonly_20261103/stages/`.
Private logs/configuration: experiment `private/` (Git-ignored).
Artifacts/frozen source: `runs/m3_parallel_ogonly_20261103/` (Git-ignored).
After successful completion, the coordinator writes `REPORT.md` and
`comparison.json` with per-video scores and grouped resource comparisons.

Check without accessing secrets:

```bash
cat experiments/m3_parallel_ogonly_20261103/status.json
ps -p 204927 -o pid,etime,stat,cmd
```

To resume the frozen coordinator **only after reviewing a stopped run**:

```bash
source ~/.bashrc
export MINIMAX_API_KEY
export M3_EVAL_ROOT=/home/yuanjian/Development/memory-eval
oGMemory/.venv/bin/python runs/m3_parallel_ogonly_20261103/driver.py run
```

The coordinator uses an exclusive lock. Failed stage receipts deliberately
block automatic retry; do not delete them or overwrite first QA answers
without recording the recovery protocol. A safety/quota stop may require a
separately recorded continuation rather than changing the frozen manifest.
