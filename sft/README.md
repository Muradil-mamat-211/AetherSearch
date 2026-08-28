# AetherSearch SFT

[![SFT Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-AetherSearch--SFT-yellow)](https://huggingface.co/muradil211/AetherSearch-SFT)
[![SFT Data](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-AetherSearch__SFT-yellow)](https://huggingface.co/datasets/muradil211/AetherSearch_SFT)
[![Checksums](https://img.shields.io/badge/checksums-sha256-blue)](checksums.sha256)

> **Complete dataset:** [AetherSearch SFT on Hugging Face](https://huggingface.co/datasets/muradil211/AetherSearch_SFT)
>
> This directory is the complete public boundary for the SFT stage: release
> metadata, the strict SFT-2000 trainer, DeepSpeed configuration, and
> dependency pins. The JSONL payloads remain on
> Hugging Face Datasets. Model files are managed
> separately by the maintainer through the linked Hugging Face model repository.

## Release at a glance

| Item | Details |
|---|---|
| Complete data | [muradil211/AetherSearch_SFT](https://huggingface.co/datasets/muradil211/AetherSearch_SFT) |
| SFT-2000 model output repository | [muradil211/AetherSearch-SFT](https://huggingface.co/muradil211/AetherSearch-SFT) |
| Reproduction entrypoint | [`scripts/run_train_sft_2000_zero3.sh`](scripts/run_train_sft_2000_zero3.sh) |
| Records | 2,000 validated trajectories |
| Training unit | Full trajectory |
| License metadata | `unknown` |
| GitHub contents | Data metadata, training code, configuration, and tests |

## Dataset Overview

This release contains 2,000 validated full agent trajectories for Qwen2.5-3B
Agentic Search format cold start. Every trajectory ends with the Qwen assistant
termination token `<|im_end|>`.

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

1. `id`
2. `question`
3. `trajectory_type`
4. `search_count`
5. `full_trajectory_text`

The `full_trajectory_text` field is the only training text. The frozen record
order has been globally shuffled with a deterministic seed of 42, then IDs
were assigned in that shuffled order from `000001` through `002000`.

## Assistant Termination

Each complete Qwen assistant trajectory ends exactly as:

```text
</answer><|im_end|>
```

The final `<|im_end|>` is the assistant EOT/EOS token and is included in the
assistant supervision target. It is not duplicated and no endoftext token is
added.

## Training Semantics

The dataset semantic contract is:

- system/user/question text is not supervised;
- the complete `<information>...</information>` span is not supervised;
- assistant `<think>...</think>` is supervised;
- assistant `<search>...</search>` is supervised;
- assistant `<answer>...</answer>` is supervised;
- the final assistant `<|im_end|>` is supervised as assistant EOT/EOS.

The public JSONL does not contain token-level masks. A downstream trainer must
construct token-level masking from this contract. No token-level mask is stored
in this dataset artifact.

## Full-Trajectory Unit

The training unit is `training_unit = full_trajectory`.

Single-search trajectories preserve one search/information turn through the
final answer. Multi-search trajectories preserve every sequential
search/information turn through the final answer.

## Provenance and Audit

`provenance_manifest.jsonl` is audit-only. Each new public ID maps to the
pre-shuffle public id, legacy source identifiers, source hashes, the pre-EOT
full-trajectory hash, the post-EOT full-trajectory hash, and the deterministic
shuffle key.

## Reproduce SFT-2000

The public trainer implements the loss contract above directly from
`full_trajectory_text`. It validates every record before model loading and
never truncates a full trajectory in the supported recipe.

Install a CUDA-compatible PyTorch build for the target host, then install the
SFT dependencies:

```bash
python -m pip install -r sft/requirements.txt
```

Download the frozen data into the default location and verify it:

```bash
hf download muradil211/AetherSearch_SFT \
  final_sft_2000.jsonl provenance_manifest.jsonl \
  --repo-type dataset \
  --local-dir sft
sha256sum -c sft/checksums.sha256
```

Run the strict data and loss-mask preflight without starting training:

```bash
python sft/scripts/train_sft_2000.py \
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --model_revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --train_file sft/final_sft_2000.jsonl \
  --output_dir outputs/sft/sft_2000_preflight \
  --expected_num_samples 2000 \
  --expected_sha256 fec609652d3832c7a6c0ee2861c6f946b6cf7c3d3d40fc5d9be9b75df6325dcb \
  --check_data_only \
  --audit_report_path outputs/sft/sft_2000_preflight/data_audit.json
```

The canonical preflight result is 2,000 accepted records, 2,119,664 segmented
input tokens, a maximum length of 2,901 tokens, 326,451 prompt-masked tokens,
1,625,435 information-masked tokens, and 167,778 supervised tokens. All 2,000
segmented sequences decode back to the tokenizer-normalized source.

Start the single-node BF16 ZeRO-3 recipe on all visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
NPROC_PER_NODE=3 \
bash sft/scripts/run_train_sft_2000_zero3.sh
```

The launcher defaults to the immutable Qwen base revision above, one epoch,
sequence length 4,096, learning rate `2e-6`, per-device batch size 1, gradient
accumulation 8, cosine scheduling, BF16, TF32, gradient checkpointing,
length-grouped dynamic padding, and no intermediate checkpoint. It runs the
full preflight first, refuses to overwrite existing output, validates BF16
support and free disk, and exports `final_model/` through an atomic directory
rename. Paths and hyperparameters are configurable through the environment
variables declared at the top of the launcher.

The trainer has passed the complete CPU-side data/mask/collator preflight and
source tests.

## Model release contract

The canonical public SFT procedure is one supervised fine-tuning stage from the
pinned Qwen base model over the frozen 2,000-record `final_sft_2000.jsonl`.
The [AetherSearch-SFT repository](https://huggingface.co/muradil211/AetherSearch-SFT)
is reserved for the `final_model/` checkpoint produced by this exact recipe.

## Limitations

The 2,000-record release defines the full-trajectory data contract; the public
trainer constructs its token-level masks. A model release must record the exact
dataset checksum and training configuration used to produce its weights.

## Checksums

Download `final_sft_2000.jsonl` and `provenance_manifest.jsonl` from the
Hugging Face dataset linked above into this directory, then verify the complete
release with:

```bash
sha256sum -c checksums.sha256
```
