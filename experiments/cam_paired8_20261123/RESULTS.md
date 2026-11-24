# CAM on M3-Bench paired-8 — Final Results (vs latest paper oGMemory)

Related-work baseline. CAM (github Jake-Tian/CAM, commit 8b98c0b) run as-is on the
same 8 M3-Bench robot videos / 97 questions / same model (MiniMax-M3.1-Flash,
reasoning medium). Answers re-scored with the exact oGMemory 0/0.5/1 judge
(MiniMax, temp 0). CAM prompts/algorithm NOT tuned; only I/O adapters + durability added.

Baselines (latest, as in the paper / latex/paired8_results.tex):
- oGMemory = 4-way parallel re-run, `experiments/m3_parallel4_paper8_20261113`
  (judgments.jsonl, 97/97, mean 62.0/97 = 63.92%).
- m3_agent = frozen reference, `experiments/m3_paper_paired8_20261103/comparison.json` (43.81%).

## Overall (题目加权, 97 questions)

| method            | n  | valid | mean acc | full-credit |
|-------------------|----|-------|----------|-------------|
| CAM (this run)    | 97 | 97    | 53.61%   | 46.39%      |
| oGMemory (latest) | 97 | 97    | 63.92%   | 54.64%      |
| m3_agent (frozen) | 97 | 97    | 43.81%   | 37.11%      |

CAM sits between m3_agent and oGMemory: +9.80pp over m3_agent, -10.31pp vs oGMemory.

## Per-video mean accuracy (%)

| video       | 题数 | CAM    | oGMemory | m3_agent | Δ(CAM-oGMem) |
|-------------|------|--------|----------|----------|--------------|
| study_09    | 12   | 62.50  | 62.50    | 50.00    | +0.00        |
| kitchen_01  | 13   | 23.08  | 34.62    | 19.23    | -11.54       |
| bedroom_05  | 12   | 33.33  | 50.00    | 37.50    | -16.67       |
| study_23    | 14   | 46.43  | 75.00    | 32.14    | -28.57       |
| kitchen_23  | 12   | 58.33  | 79.17    | 62.50    | -20.83       |
| study_05    | 12   | 54.17  | 62.50    | 37.50    | -8.33        |
| gym_03      | 11   | 72.73  | 68.18    | 72.73    | +4.55        |
| bedroom_04  | 11   | 86.36  | 81.82    | 45.45    | +4.55        |
| **weighted**| 97   | **53.61** | **63.92** | **43.81** | **-10.31** |

CAM wins/edges oGMemory on gym_03, bedroom_04, ties study_09; loses notably on
study_23 (-28.6), kitchen_23 (-20.8), bedroom_05 (-16.7). Its weakest video
(kitchen_01) is "where should the robot get X" spatial grounding — confident but
wrong shelf/basket. No answer ends in "insufficient evidence" (0/97): low scores
are genuine wrong answers, not tool/agent failures.

## Token usage (all MiniMax-M3.1-Flash, recorded memory+QA)

| method    | memory tokens | qa tokens | total tokens |
|-----------|---------------|-----------|--------------|
| CAM       | 10,197,974    | 4,474,275 | 14,672,249   |
| oGMemory  | 3,295,082     | 6,718,348 | 10,013,430   |
| m3_agent  | ~3,620,387    | ~778,834  | 4,399,221    |

CAM ≈ 1.47x oGMemory / 3.3x m3_agent tokens; heavy per-clip 30-frame visual MLLM
(~11-14k tokens/clip, ~200-280 s/call) dominates CAM's memory cost.

## Timing

- CAM final continuous stretch (8 per-video workers): memory+reasoning 21:19:46 ->
  02:25:24 (~5h05m wall), judge 02:29 (~3 min). Full memory build spanned earlier
  sessions with an account-quota gap; resumed from checkpoints (no data lost).
- Baseline `time_seconds` are "allocated" stage sums (not wall clock under 8-way
  parallelism), so cross-method time is not directly comparable; tokens are.

## Method / adaptations (I/O only, CAM logic unchanged)

- MiniMax Chat Completions + json_object/schema/pydantic (reasoning_effort=medium,
  thinking adaptive); local bge-m3 embeddings (batch<=8, retry+backoff, repair of
  any None vectors before reasoning); per-clip graph checkpoints, per-question
  reasoning resume, resumable judge; orchestrator auto-retries after quota and
  auto-runs judge+comparison once all 8 videos finish.
- Headline score uses the oGMemory 0/0.5/1 judge (CAM's built-in binary judge unused).

## Limitations

- Single run, greedy (temp 0). Only I/O adapters added; CAM prompts/algorithm untouched.
- Minor: one internal conversation-summary extraction failed on bedroom_04 (did not affect answers).

## Artifacts

- comparison_cam.txt / comparison_cam.json, cam_summary.json, judge_grades.jsonl, cam_usage.json
