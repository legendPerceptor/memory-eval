# Five-video QA optimization: interim record

As of 2026-11-09 21:05 China time, the completed comparisons cover the same
61 questions in `bedroom_04`, `kitchen_01`, `bedroom_05`, `gym_03`, and
`study_23`. They reuse existing memories and do not measure memory-construction
cost. QA token counts exclude the separate blind-judging calls.

## Four-way blind re-evaluation

All four sets of answers were frozen, randomly mixed, and judged in one batch.
Scores are means of 0, 0.5, and 1 over questions, not exact-match accuracies.

| QA run | Mean score | QA tokens | Failed answers |
| --- | ---: | ---: | ---: |
| Historical compact baseline | 55.74% | 2,288,538 | 0 |
| Same baseline, new run | 50.82% | 2,218,620 | 0 |
| M3 Agent | 39.34% | 454,617 | 0 |
| 2,500-token incremental budget | 51.64% | 2,145,583 | 0 |

The candidate used 3.29% fewer recorded QA tokens than the new baseline run,
and 6.25% fewer than the historical baseline. The historical baseline's
`bedroom_04` usage includes a retry, making that comparison less controlled.
The candidate remained above M3 Agent in this sample but used approximately
4.72 times as many QA tokens.

The candidate was 0.82 percentage points above the new baseline but 4.10
points below the historical answers. The sample is too small and the model
and judge vary across runs; neither quality preservation nor statistical
superiority over M3 Agent is established. An exploratory paired question-level
bootstrap for the candidate versus the new baseline gave approximately
[-9.0, +10.7] percentage points. Questions from the same video are correlated,
so that interval is not a video-level generalization guarantee.

| Video | Historical | New baseline | M3 Agent | 2,500 increment |
| --- | ---: | ---: | ---: | ---: |
| bedroom_04 | 81.82% | 72.73% | 45.45% | 86.36% |
| kitchen_01 | 42.31% | 26.92% | 19.23% | 23.08% |
| bedroom_05 | 33.33% | 37.50% | 41.67% | 33.33% |
| gym_03 | 68.18% | 63.64% | 68.18% | 72.73% |
| study_23 | 57.14% | 57.14% | 28.57% | 50.00% |

The frozen inputs, per-question judgments, and summary are under
`experiments/m3_four_way_20261109/`. Other comparisons have their own
randomized judging batches and should not be treated as if their absolute
scores came from the same judge pass.

## Other completed runs

- Columnar evidence: 55.74% versus 54.10% historical in its batch, but
  2,303,299 versus 2,288,538 QA tokens; not an efficiency improvement.
- Person information on demand: 56.56% versus 54.10% historical, but
  2,328,872 QA tokens; not an efficiency improvement.
- Six-search cap: 49.18% versus 54.92% historical, with 2,278,158 QA tokens;
  insufficient savings and lower score.
- Early-answer prompt: 55.74% versus 53.28% historical and 40.16% M3 Agent
  in its batch. At least 2,246,834 QA tokens were recorded, above the new
  baseline's 2,218,620; nine failed calls and an interrupted request have
  unknown usage. Failed questions were retried and all 61 final answers exist.
- Ambiguity guard plus 2,500-token increment: 53.28% versus 54.92%
  historical and 40.16% M3 Agent in its batch. At least 2,110,939 QA tokens
  were recorded, with 25 failed calls of unknown usage. It answered
  `study_23_Q12` with Paul, unlike the 2,500-only run; one trial cannot
  establish that the guard caused the correction or preserved overall quality.

These are single runs, not causal proofs. The `study_23_Q12` candidate
answered with a car model although the question asked who Sophia's model
was; both baseline runs answered Paul. Its search trajectory pursued the car
interpretation rather than the role-model interpretation. Conversely, the
judge assigned different scores to nearly equivalent answers for
`kitchen_01_Q10`, so the apparent change there cannot safely be attributed to
the policy.

## Interrupted and pending work

The early-answer and ambiguity-guard experiments encountered API 429 errors,
including both rate and plan-usage limits. Failed QA questions were requeued
and incomplete judgments archived. Both retries and fresh judging have
completed. Request failures and interrupted requests may have unknown usage,
so their token counts are lower bounds rather than complete cost measurements.

## Adaptive incremental results

The adaptive candidate retains a 3,500-token initial evidence increment and
uses 2,000 for later increments. Its first five-video run completed all 61
questions without failed or unknown-usage calls, using 2,105,323 QA tokens.
Relative to the same-configuration baseline rerun, that is 5.11% fewer
recorded tokens; relative to the historical run, it is 8.01% fewer, although
the historical run included a bedroom_04 retry. These comparisons exclude
memory construction, embeddings, judging, and other experiment costs.

The first randomized blind-judging batch scored adaptive at 56.56%, historical
baseline at 54.10%, and M3 Agent at 41.80%. A separate randomized four-arm
blind-judging batch scored the same frozen answers as follows:

| Answer set | Score | QA tokens |
| --- | ---: | ---: |
| Historical compact baseline | 52.46% | 2,288,538 |
| Same baseline, new run | 49.18% | 2,218,620 |
| M3 Agent | 38.52% | 454,617 |
| Adaptive incremental | 55.74% | 2,105,323 |

These are means of 0/0.5/1 blind-judge scores, not exact-match accuracy.
The same answers received different scores in independent judging batches.
Despite promising sample-level results, five videos and two candidate runs
cannot establish non-inferiority or consistent superiority, and the
candidate still uses about 4.63 times the M3 Agent QA tokens. In the four-arm
batch, adaptive scored 30.77% on kitchen_01 versus the historical baseline's
38.46%; for example, its cotton-rope answer omitted the specific material-box
location. On bedroom_04_Q05 it selected Korean rather than vintage style, and
on study_23_Q13 it counted two car movies rather than three. These examples
identify errors but do not prove the budget change caused them. Another
near-equivalent answer received a different partial score, illustrating
judge noise. The first batch failed the predeclared eligibility rule, including
its minimum 10% token reduction and maximum 10-point per-video drop.

Frozen four-way inputs, judgments, and per-video scores are in
`experiments/m3_adaptive_four_way_20261109/`.

## Independent adaptive repeat

An independent run of the same adaptive policy completed all 61 questions
with no failed or unknown-usage calls and used 2,087,395 QA tokens: 5.91%
less than the original-policy rerun and 8.79% less than the historical run.
Its own randomized blind-judging batch produced:

| Answer set | Score | QA tokens |
| --- | ---: | ---: |
| Historical compact baseline | 52.46% | 2,288,538 |
| M3 Agent | 40.98% | 454,617 |
| Adaptive repeat | 59.84% | 2,087,395 |

The adaptive repeat's per-video scores for bedroom_04, kitchen_01, bedroom_05,
gym_03, and study_23 were 90.91%, 30.77%, 41.67%, 72.73%, and 67.86%.
The corresponding historical baseline scores in the same judging batch were
81.82%, 38.46%, 29.17%, 63.64%, and 53.57%. Thus kitchen_01 remains weaker
than the historical answers in both adaptive runs. The repeat correctly
answered the clothing-style question that the first adaptive run missed, but
misinterpreted `study_23_Q12` as asking for a car model rather than a person;
the first run answered that question correctly. Both runs show meaningful
question-level variability and neither establishes the cause of its errors.

The repeat did not meet the predeclared 10% token-saving criterion; neither
adaptive run qualifies for automatic selection. QA stage elapsed times were
2,112 seconds and 2,301 seconds for the first and repeat runs respectively,
without a controlled runtime comparison to other runs. The repeat's frozen
answers, logs, usage records and judgments are under
`experiments/m3_adaptive_repeat_20261109/` and the corresponding `runs/`
directory. No candidate has been enabled by default.
