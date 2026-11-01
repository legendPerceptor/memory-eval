# Preparation-only failure

The first attempt to prepare the context-budget ablation failed while copying
the already cached tokenizer from Docker (`docker cp` hit a container bind-mount
error). No QA, judge, or memory-generation model calls were made.

The local private preparation log and frozen source tree are retained. The
successful preparation used read-only `docker exec cat` plus a pinned SHA256
check and is recorded separately as `m3_context_budget_only_20261101`.
