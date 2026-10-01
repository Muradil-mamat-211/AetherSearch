# RL Recipe

This directory contains the public entry configuration for the verified
AetherSearch RL topology. It does not contain SFT or DPO training recipes. The
independent entrypoints are documented in [`../../sft/`](../../sft/) and
[`../../dpo/`](../../dpo/).

The entry recipe composes four independent concerns:

| Concern | Source |
|---|---|
| experiment and schedule | inherited algorithm configuration |
| external artifacts | `AETHERSEARCH_ASSET_MANIFEST` and local path variables |
| hardware mapping | `configs/hardware/4x48gb_3rl.yaml` |
| runtime mapping | `configs/runtime/verl_fsdp2_vllm_4x48_reference.yaml` |
| verified-machine contract | `configs/qualification/official_4x48gb_v1.yaml` |

The current reference profile uses four 48 GiB GPUs: one dedicated role hosts
the hybrid retriever and asynchronous worker, while three RL roles host the
three-rank vLLM/FSDP2 runtime. The exact physical mapping is expressed in the
topology block and enforced only by the official qualification profile.

Every evaluation scheduled at a 20-update checkpoint uses all 1,400 rows of
[`muradil211/AetherSearch_Eval_1400`](https://huggingface.co/datasets/muradil211/AetherSearch_Eval_1400),
in original `eval_1400.jsonl` row order. The manifest locks the file SHA-256,
total row count, seven source counts, and
all row identities.

The evaluator constructs the existing Search-R1 user prompt from `question`
and keeps `answers` exclusively for scoring. Diagnostic evidence is not loaded.

Download the pinned validation JSONL from Hugging Face:

```bash
hf download muradil211/AetherSearch_Eval_1400 eval_1400.jsonl \
  --repo-type dataset --revision db4f3c87a2f1cf1d14117954d69756cb246bc2d2 \
  --local-dir /path/to/eval-data
```

Set `AETHERSEARCH_VALIDATION_DATA=/path/to/eval-data/eval_1400.jsonl` in the
local environment file. Training data, the 20-update cadence, greedy decoding,
and scoring remain unchanged.

Download the upstream Search-R1 training parquet; it is not duplicated in this
repository:

```bash
hf download PeterJinGo/nq_hotpotqa_train train.parquet \
  --repo-type dataset --local-dir /path/to/train-data
```

The recipe's train-data contract is the upstream file SHA-256 recorded in
`configs/base.yaml` and `EXTERNAL_ASSETS.md`.

Prepare a local environment file without committing machine paths:

```bash
cp environment/env.template.sh environment/env.local.sh
# Edit environment/env.local.sh, then:
source environment/env.local.sh
```

Resolve and validate the recipe without starting services:

```bash
bash scripts/train_rl.sh --dry-run
```

Start the verified RL run:

```bash
bash scripts/train_rl.sh
```

Resume only from a checkpoint that has passed the repository's fresh-runtime
distributed restore validation:

```bash
bash scripts/resume_rl.sh \
  --config recipes/rl/train_4x48gb.yaml \
  --checkpoint /path/to/checkpoint \
  --restore-validation /path/to/restore_validation.json \
  --source-run /path/to/source_run
```

By default, the launcher creates a timestamped run below
`AETHERSEARCH_RUNTIME_ROOT`. Use `--run-dir PATH` to select an exact run
directory. The launcher materializes `configs/resolved_config.yaml` there,
runs the formal preflight, then starts the Retriever, asynchronous
training-time evaluation worker, and three-rank RL runtime from the same
resolved config.

`train_4x48gb.yaml` is the only formally qualified training profile. Its prompt
filter ranks groups by raw terminal-outcome sample variance and retains the
configured cumulative variance mass before retrieval utility is scored.

A different server uses a user-owned hardware/runtime YAML and
`qualification.mode: portable`; algorithm Python code is unchanged. Generic
non-reference layouts are covered only by CPU configuration-planning tests.
That synthetic coverage does not establish GPU-memory fit, runtime
compatibility, training stability, throughput, or production qualification.
