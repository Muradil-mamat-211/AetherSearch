# Search-SFT 2000

> **Complete dataset:** [AetherSearch SFT on Hugging Face](https://huggingface.co/datasets/muradil211/aethersearch_sft)
>
> This directory contains the release documentation and integrity metadata
> only. The JSONL payloads are intentionally hosted on Hugging Face rather than
> committed to this GitHub repository.

## Release at a glance

| Item | Details |
|---|---|
| Complete data | [Download from Hugging Face](https://huggingface.co/datasets/muradil211/aethersearch_sft) |
| Records | 2,000 validated trajectories |
| Training unit | Full trajectory |
| License metadata | `unknown` |
| GitHub contents | Documentation, attribution, manifest, and checksums |

## Dataset Overview

This release contains 2,000 validated full agent trajectories for Qwen2.5-3B
Agentic Search format cold start. Every trajectory ends with the Qwen assistant
termination token <|im_end|>.

## Composition

| Trajectory type | Records | Share |
|---|---:|---:|
| single_search | 1,025 | 51.25% |
| multi_search | 975 | 48.75% |
| Total | 2,000 | 100.00% |

Search-depth distribution:

| Search depth | Records | Share |
|---:|---:|---:|
| 1 | 1,025 | 51.25% |
| 2 | 667 | 33.35% |
| 3 | 265 | 13.25% |
| 4 | 43 | 2.15% |

## Public Schema

Every public training record contains exactly these five fields, in this order:

1. id
2. question
3. trajectory_type
4. search_count
5. full_trajectory_text

The full_trajectory_text field is the only training text. The frozen record
order has been globally shuffled with a deterministic seed of 42, then IDs were
assigned in that shuffled order from 000001 through 002000.

## Assistant Termination

Each complete Qwen assistant trajectory ends exactly as:

</answer><|im_end|>

The final <|im_end|> is the assistant EOT/EOS token and is included in the
assistant supervision target. It is not duplicated and no endoftext token is
added.

## Training Semantics

The dataset semantic contract is:

- system/user/question text is not supervised;
- the complete <information>...</information> span is not supervised;
- assistant <think>...</think> is supervised;
- assistant <search>...</search> is supervised;
- assistant <answer>...</answer> is supervised;
- the final assistant <|im_end|> is supervised as assistant EOT/EOS.

The public JSONL does not contain token-level masks. A downstream trainer must
construct token-level masking from this contract. No token-level mask is stored
in this dataset artifact.

## Full-Trajectory Unit

The training unit is:

training_unit = full_trajectory

Single-search trajectories preserve one search/information turn through the
final answer. Multi-search trajectories preserve every sequential
search/information turn through the final answer.

## Provenance and Audit

provenance_manifest.jsonl is audit-only. Each new public id maps to the
pre-shuffle public id, legacy source identifiers, source hashes, the pre-EOT
full-trajectory hash, the post-EOT full-trajectory hash, and the deterministic
shuffle key.

## Limitations

This dataset artifact does not claim that SFT performance has been measured.
It defines the full-trajectory data contract; token-level masking belongs to
the downstream trainer.

## Checksums

Download `final_sft_2000.jsonl` and `provenance_manifest.jsonl` from the
Hugging Face dataset linked above into this directory, then verify the complete
release with:

sha256sum -c checksums.sha256
