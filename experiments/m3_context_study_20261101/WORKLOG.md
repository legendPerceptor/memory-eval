# Current work state / handoff

Latest user instruction: work autonomously **until 2026-11-01 23:30 Asia/Shanghai**,
then report and let the user decide the next phase. This supersedes the original
8-hour wording still present in the goal metadata. Do not push.

## Commits

oGMemory branch `m3_dev_1031`, clean:

- `9eb782e4`: optional answer reserve / final-round Answer-only, guidance retention.
- `68e1993c`: visible-passage diagnostics and actual tool names in ReAct traces.
- `09f3cf30`: grouped presentation of admitted original evidence, opt-in.
- `76879efa`: clarify grouped list positions are local, not persistent entity IDs.
- `3d5a007d` + `8a56a9be`: safe private config-copy CLI and fresh-byte tokenizer pin validation.
- `83e51d98`: clarify ReAct versus legacy single-pass applicability in CLI notice.

Root repository `main` has checkpoints through `1f56f45`.
Some earlier M3-Agent/report artifacts predate this phase and were already
uncommitted when this work began; preserve them.
Newer checkpoints: `2b03f3b` (A/B/C and provisional report), `d367375`
(archive completed prior M3/study18 artifacts; no prediction changes).

## Experiments and scheduling

All five videos / 65 questions, same stored memories, same model/settings:
MiniMax-M3.1-Flash, adaptive/medium, QA output8192, judge2048, 6 rounds/180s,
current_clip. No memory regeneration or perception changes.

The completed coordinator was the frozen
`runs/context_assets/coordinator_repeat_advance_20261101.py`,
launched through exec session **66342** (now exited0; do not restart).
Its status is `experiments/m3_context_study_20261101/coordinator_status.json`.
It launches each frozen `runs/<name>/runner.py`, checks read-only MiniMax quota,
and writes logs under `private/`. It will not purchase/upgrade or switch models.

Current planned logical arms:

| Logical arm | Run / experiment name | Source | State / timing |
|---|---|---|---|
| A | m3_context_budget_recovered_20261101 | c992d45b | finished |
| B | m3_context_finalization_recovered_20261101 | 9eb782e4 | finished |
| C | m3_context_compact12k_guarded_20261101 | 68e1993c | finished:64/65, initial score0.523077 |
| D | m3_context_grouped12k_v2_20261101 | 76879efa | finished:65/65, independent score0.530769 |
| B repeat | m3_context_finalization_repeat_20261101 | 9eb782e4 | finished65/65, independent0.5615, joint0.5769 |

B repeat is a **full independent repeat**, not best-of or answer reuse.
It references B's exact frozen source/config, but its manifest explicitly marks
the reference as an independent replication, not failed-request overhead.

C's one failed answer is bedroom_04_Q05: rejected protocol scaffolding despite
finish_reason=done, not a quota failure. Do not rerun it to improve the score.
C received2,721,838 input tokens, peak9102,366 calls.
Use metrics.json, not this note, for final tables.

After these, coordinator ran `scripts/judge_context_comparison.py`
(freezes a copy at execution) for **455 shuffled (method, question) pairs**:
historical oGMemory, historical M3 reference text, A/B/C/D, and B repeat.
Method names were not sent to the judge. No joint judging had started at
the schedule amendment; it is now fully completed. Grade API errors may be repaired separately, but never
regrade successful rows selectively or change predictions.

Old coordinator session68378 was stopped while idle before C. Its first schedule
is preserved in `execution_schedule_initial.json`.
Old joint queue session14129 was also stopped before any judging.
Session59388 was stopped while idle before D to freeze the local-index
clarification. D original had zero calls/answers/grades and is retained.

## Quota incident and recovery

- A original `m3_context_budget_only_20261101`: 55 valid answers, last10 QA failed
  with provider429/2056; 7 valid kitchen answers lacked successful judging.
- B original `m3_context_finalization_20261101`: stopped with28 saved failed QA
  rows, all provider quota errors, no valid answer. An in-flight request may not
  have a final usage record; do not equate missing usage with zero billing.
- C original `m3_context_compact12k_20261101` never started; new guarded wrapper
  clones its exact source/config.
- Official read-only quota endpoint reported a fresh10:00–15:00 window.
  A209-token health probe at10:12 returned200. No purchase, plan change or model
  substitution occurred.
- A recovery preserves **55 successful QA rows and48 successful grade rows
  bit-for-bit as parsed JSON**, hashes verified. Only quota failures/missing
  grades were filled. Original failed files remain untouched.
- B restarted independently on all65 because no original answers were valid.
- `source_archive_sha256` and private-config semantic equality verified for
  A/B/C clones; no credential-bearing config hashes are published.

## Completed results before joint regrading

These are independent/retained grades, **not final joint-judge scores**:

| Condition | Valid | Score | QA calls | Received input | Peak |
|---|---:|---:|---:|---:|---:|
| Old fixed oGMemory |59/65|0.5462|319|19,905,749|156,689|
| Old M3 reference text adaptation |65/65|0.4615|310|569,796|4,740|
| A recovered |65/65|0.553846|359|4,722,001|16,904|
| B |65/65|0.600000|365|4,804,362|16,922|
| C |64/65|0.523077|366|2,721,838|9,102|
| D |65/65|0.530769|366|2,760,479|9,251|

A by-video: .4583 / .6364 / .8182 / .3214 / .5882.
B by-video: .5833 / .7273 / .8182 / .3214 / .6176.
B had54 last-iteration finalizations, **zero time-reserve triggers**;
max question wall time79.35s. Do not attribute its score change to the45s reserve.
D finished atabout15:50 with no QA/judge errors and no loop-time overruns.
It had56 last-round finalizations, no time-reserve triggers. Actual current
quota remaining76%; its initial repeat schedule was20:05, later advanced as below.
Do not add extra trial arms to fill the waiting period.

At the same12k budget, D admitted more original spans/entries than C
(mean final-view78.83 vs66.66), but independent grades remain close. These are
not counts of distinct facts and queries differ; don't treat this as a causal
information-gain estimate. See compact_admission_comparison.json.
Update16:36:read-only quota still76%. Existing B repeat moved to16:40 with a
70% early-launch threshold, leaving more recovery time before23:30. Same frozen
run, no extra condition or answer reuse. Old coordinator stopped only while
idle; schedule and reason archived in repeat_schedule_advance.json.

The original6 timeout questions all got answers in A;5 scored1 and scissors-count
study_05_Q12 still scored0. A versus old independent grades:48 equal,8 higher,9 lower.
See `docs/m3_context_case_review.md` for cautious qualitative interpretation.

## Important cache / lossless-format findings

- Original cached input12,987,771 (65.2%), uncached6,917,978.
- A cached724,155 (15.3%), uncached3,997,846.
- A total input reduction76.3%, **uncached reduction42.2%**, not a billing claim.
- Per-round repacking changes earlier tool payloads; this plausibly reduces
  prefix stability, but provider cache internals were not controlled.
- Offline same-admission grouped-format replay:
  251 calls with evidence,330 unique source versions,6201 verified source
  occurrences,37323 verified passages; all content hashes matched.
  Evidence-only proxy tokens3,600,977 ->2,875,088 (**20.16% lower**).
  Zero model/embedding calls; not a whole-request or QA-quality result.
- Existing semantic consolidator concatenates descriptions up to50KB and unions
  lists; it is not a compact, time-safe M3-style semantic fact/identity index.
  Do not simply put full-video consolidated memory into earlier-question QA.

## Validation

- Latest oGMemory selected M3/provider/core units: **1033 passed**.
- Grouped/context/reserve subset: **45 passed**.
- Root tooling tests: **24 passed** after adding independent-repeat accounting.
- Expanded integration attempt:1369 passed,108 skipped,8 failed,1 deselected.
  All8 failures reproduced on frozen pre-change c992d45b (logs retained).
  The deselected extract-audio duplicate-node case was previously reproduced.
  Do not say the entire repo is all green.
- Early database snapshot was taken after A started; later snapshot matches it
  exactly for graph_nodes, graph_edges and vectors in both databases.
  Take a final snapshot after all runs. Do not invent a pre-A snapshot.
- Staged-value scan found no actual MiniMax key or known DB passwords in the
  checkpoint; private configs/logs remain ignored. No push.

Latest expanded run at09f3cf30:1380 passed,108 skipped,8 known failures,
1 known deselection. Excluding exactly the9 reproduced pre-existing cases gives
1380 passed,108 skipped,9 deselected. Root tooling now25 passed.
Full65 grouped replay verifies7268 occurrences/43086 passages from332 versions,
294 evidence-bearing calls,20.1617% evidence-only proxy-token savings.
The earlier6201-occurrence audit included a quota-rejected prepared view; retain
it as a preliminary preparation audit, not as proof all views reached the model.
Latest76879efa unit selection:1034 passed; context subset46 passed.
Frozen core source files for all5 new runs were independently compared against
their recorded Git archives:passed. Joint input verification is prepared, not
yet executed because joint grading is pending.
Configuration utility adds6 initial tests plus a pin regression. Final
committed83e51d98 checks:1041 unit passed;1388 passed/108 skipped/9 known
exclusions in the expanded selection. Root tools29 passed.
Utility invocation and output path are documented in
oGMemory/docs/m3_bounded_context.md.
All root tests together now39 passed; older reference adapter recheck9 passed.
Before-D DB fingerprints match early snapshot, and annotations match the
original canonical-JSON hash. Do not compare raw bytes against that canonical
manifest hash; both hashes and the correction are recorded.

Additional evidence:
- `compact_protocol_failure.json` records C's final API input8607/output622,
  normalstop and0 tool calls; it was not input/output overflow or quota.
- `visual_review_study18_q07.json` links original last sampled435s frame and
  private raw445–449s review frames. These are after the434s question but inside
  current_clip allowance; no new frame was fed to any experiment.
- Do not add new trial arms opportunistically. The plan remains A/B/C/D+B repeat.

## Final status and delivery

Update18:15: ALL QA and455 joint grades are complete, zero judge errors.
Coordinator session66342 has exited0. No further paid model requests are needed.
Joint scores: old .5384615, M3 .4615385, A .5538462, B .6076923,
C .5230769, D .5384615, B repeat .5769231. B descriptive mean .5923077.
All five video-cluster intervals include0; don't claim significant gains.
Final data/judge audit: DB contents unchanged since early snapshot;448 judge
API calls plus7 automatic zero scores, all actual model/settings verified,
all usage returned, no length terminations. Full input audit verifies455 tasks
and35 source-prediction files unchanged. Recovery script was unit-tested but
not used (no final judge errors).
Final PDF rebuilt12 pages at18:19 and visually reviewed pages8–10. Root tests48
passed. `release_audit.json` at18:24 records exact PDF/source/data hashes, all
input/model/database checks, and the remaining timed user-facing delivery.
No additional paid model calls are needed or planned.

Final shared-judge results:

| Method | Valid | Joint score |
|---|---:|---:|
| Original fixed oGMemory |59/65|0.5385|
| M3 reference text adaptation |65/65|0.4615|
| A |65/65|0.5538|
| B first |65/65|0.6077|
| B repeat |65/65|0.5769|
| C |64/65|0.5231|
| D |65/65|0.5385|

B two-run descriptive mean0.5923; don't pick0.6077 alone.
Five new trajectories have325 records,324 valid answers,0 timeouts; maximum
observed question wall112.53 seconds. Time reserve never triggered.
Score decomposition: on original59 answered questions, B first loses1.5 points
and B repeat loses2.5; old6 timeout questions gain6 and5 points respectively.
Overall gains thus primarily accompany recovered answer coverage, not universal
per-question quality improvement. All five cluster intervals include0.

Recorded call ledger:2635 logged calls,2590 withusage,45 unknown; received
20,201,770 input and496,900 output tokens, plus separate209-token probe.
This includes original failed attempts, not only fixed trajectories; not a bill.

Remaining required action: commit final reviewed artifacts, then honor the user's
23:30 reporting time. Use a real local timer if work finishes early; don't claim a
future notification without keeping the goal active and observing a live handle.
At delivery recheck artifact hashes/branches, mark goal complete and report.

### Completed execution checklist (historical procedure)

1. Monitor coordinator and each status/usage file; no unnecessary model probes.
2. For completed arms, run `scripts/summarize_context_experiment.py <name>`.
   It validates reused rows, temporal scopes and actual model settings, and
   separately reports prior quota overhead.
3. After455 joint grades with no unresolved judge errors:
   - `python3 scripts/analyze_context_cache.py`
   - `python3 scripts/analyze_context_cases.py`
   - `oGMemory/.venv/bin/python scripts/plot_context_usage.py`
   - `python3 scripts/render_context_comparison.py`
4. Add generated `context_optimization_results.tex` to `latex/report.tex`;
   update abstract to distinguish prior and new grading batches, not overwrite
   historical results. An earlier checkpoint had an11-page provisional report;
   current PDF is the12-page final report. Prior9-page PDF archived as
   `report-before-context.pdf`.
5. `bash scripts/build_report.sh`; inspect pdfinfo/pdftotext and final LaTeX log.
   Existing fontspec Fandol warnings are nonfatal; check undefined refs/overfulls.
6. Add B-repeat interpretation, cache caveats, failed-attempt overhead,
   limits and recommendation. Do not select the higher B run or merge best answers.
7. Verify DB unchanged, source branches, credential exclusions; commit new
   results/docs on main. Preserve unrelated pre-existing changes.
8. At23:30 report exact final results, validations, commits, limitations,
   and paths. User decides next phase. Do not keep launching work past deadline.
