# AetherSearch SFT model evaluation inventory

Audit date: 2026-08-28

## Scope and headline

There is currently **no trained Search-SFT-2000 model and therefore no
Search-SFT-2000 evaluation result**. The 2,000-record release was created after
the existing SFT checkpoints and explicitly states that its performance has not
been measured.

The newest complete pre-SFT-2000 checkpoint that has recoverable local
evaluation evidence is designated for release through
[muradil211/AetherSearch-SFT](https://huggingface.co/muradil211/AetherSearch-SFT).

It is the older V3 multi-search SFT model followed by a 300-step V3.1 repair
stage. It is not the result of training on
[`final_sft_2000.jsonl`](https://huggingface.co/datasets/muradil211/AetherSearch_SFT/blob/main/final_sft_2000.jsonl).

## What the evaluation actually measures

The historical local evaluator was named `eval_multisearch_sft.py`; that legacy
evaluator and its raw logs are not included in this repository release. For
each gold assistant turn, it constructed a prefix from the clean system/user
prompt plus all earlier **gold** trajectory events, then greedily generated the
next search or answer. Consequently, this is teacher-forced, turn-level
generation evaluation. It is not an end-to-end online agent rollout and does
not measure retrieval success after feeding the model's own earlier search
decisions back into the agent.

Metric definitions:

- `has_think`: generated turn contains a complete think block.
- `has_search`: a gold search turn contains a generated search block.
- `search_exact`: normalized generated query is string-equal to the gold query.
- `answer_early_on_search_turn`: the model emits an answer on a gold search turn.
- `has_answer`: the final turn contains a complete answer block.
- `answer_relaxed_match`: normalized prediction equals or contains/is contained
  by one of the gold answers.
- `sample_all_search_exact_and_answer_ok`: every search query is exact and the
  final answer passes relaxed matching.

`search_exact` can reject a semantically valid alternative query, while the
substring branch of `answer_relaxed_match` can accept some overly broad
answers. These are format/behavior diagnostics, not a complete QA benchmark.

## Main available result: V3.1 repair model, 300 samples

Historical artifact identity:
`v31_split/multisearch_v31_eval_raw.jsonl`, SHA-256
`8036321a5354846d47f259847276a45eef210deb4fb67a990d17d1859467e7d3`.

The file contains 962 unique questions. The log evaluates a deterministic
random sample of 300 rows.

| Metric | Count | Rate |
|---|---:|---:|
| complete think | 1,021 / 1,021 turns | 100.0% |
| search block present | 715 / 721 search turns | 99.2% |
| search query exact | 266 / 721 search turns | 36.9% |
| premature answer on search turn | 6 / 721 search turns | 0.8% |
| final answer block present | 299 / 300 questions | 99.7% |
| relaxed final-answer match | 260 / 300 questions | 86.7% |
| every search exact | 20 / 300 questions | 6.7% |
| every search exact and answer correct | 20 / 300 questions | 6.7% |

Answer-match breakdown: 233 normalized exact matches, 27 substring matches,
and 40 failures.

### By gold search depth

| Search depth | Questions | Search present | Search exact | Early answer | Final answer match | All-search exact + answer |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 196 | 392/392 (100.0%) | 134/392 (34.2%) | 0/392 (0.0%) | 168/196 (85.7%) | 16/196 (8.2%) |
| 3 | 87 | 256/261 (98.1%) | 109/261 (41.8%) | 5/261 (1.9%) | 76/87 (87.4%) | 4/87 (4.6%) |
| 4 | 17 | 67/68 (98.5%) | 23/68 (33.8%) | 1/68 (1.5%) | 16/17 (94.1%) | 0/17 (0.0%) |

## Intended paired diagnostic: before and after V3.1 repair

Both 100-row logs use the same eval file and have identical depth totals. The
evaluator defaults to seed 42, although the logs do not print the seed, so this
is the intended paired comparison rather than independently sampled scores.

| Metric | V3 before repair | V3.1 repair | Change |
|---|---:|---:|---:|
| complete think | 83.9% | 100.0% | +16.1 pp |
| search block present | 99.6% | 99.6% | 0.0 pp |
| search query exact | 35.5% | 35.5% | 0.0 pp |
| premature answer | 5.4% | 0.4% | -5.0 pp |
| final answer block present | 44.0% | 100.0% | +56.0 pp |
| relaxed final-answer match | 40.0% | 88.0% | +48.0 pp |
| every search exact | 8.0% | 6.0% | -2.0 pp |
| every search exact and answer correct | 4.0% | 6.0% | +2.0 pp |

The repair stage clearly fixed final-answer emission and reduced early answers,
but it did not improve strict search-query matching on this 100-row diagnostic.

## Critical leakage/holdout qualification

The V3.1 split contains 8,661 repair-train questions and 962 repair-eval
questions, with zero normalized-question overlap between those two files.
However, exact stable-row hashing confirms that their union is the complete
9,623-row dataset previously used to train the V3 base checkpoint:

`search_sft_hybrid_v1_multisearch_10000_clean.jsonl`, SHA-256
`e573822e740e2b6cf5b5dcae35e268b6ec34f3522d26648aa4ff03d5b4a61f5a`.

Therefore all 962 V3.1 eval rows had already been seen during the earlier V3
training stage. This is a valid holdout only for measuring the incremental
repair stage, not a globally unseen test set for the final model. The 86.7%
answer score must not be presented as clean out-of-distribution generalization.

## Older 50-row diagnostics

These use the old 9,623-row multi-search data itself and are historical training-
set diagnostics, not held-out benchmarks.

| Model | Think | Search present | Search exact | Early answer | Answer present | Relaxed answer |
|---|---:|---:|---:|---:|---:|---:|
| V2 query-rewrite SFT | 100.0% | 42.0% | 0.0% | 60.5% | 100.0% | 54.0% |
| V3 multi-search SFT (`minnew8`) | 79.9% | 100.0% | 26.1% | 14.3% | 32.0% | 28.0% |

The V3.1 repair training log records 25,983 turn-level examples, 300 optimizer
steps, train loss 0.0611189539, runtime 11,517.5 seconds, and epoch 0.28. Train
loss is an optimization statistic and is not an evaluation score.
