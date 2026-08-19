<div align="center">

# AetherSearch

**Search-augmented post-training with SFT, DPO, and reinforcement learning.**

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-AetherSearch-yellow)](https://huggingface.co/muradil211/AetherSearch)
[![SFT Data](https://img.shields.io/badge/%F0%9F%A4%97%20Data-AetherSearch%20SFT-yellow)](https://huggingface.co/datasets/muradil211/aethersearch_sft)
[![Code](https://img.shields.io/badge/GitHub-Code-181717?logo=github)](https://github.com/Muradil-mamat-211/AetherSearch)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](pyproject.toml)

🤗 [AetherSearch Model](https://huggingface.co/muradil211/AetherSearch) |
🤗 [AetherSearch SFT Data](https://huggingface.co/datasets/muradil211/aethersearch_sft)

</div>

## Latest News

- **2026-08:** Full training code released for the AetherSearch SFT + DPO + RL
  pipeline.
- **2026-08:** AetherSearch model weights are available on
  [Hugging Face](https://huggingface.co/muradil211/AetherSearch).
- **2026-08:** The public SFT data release is hosted directly on
  [Hugging Face Datasets](https://huggingface.co/datasets/muradil211/aethersearch_sft).

## Open Resources

| Resource | Link | Contents |
|---|---|---|
| 🤗 Model | [muradil211/AetherSearch](https://huggingface.co/muradil211/AetherSearch) | model weights, tokenizer, config, and model card |
| 🤗 SFT data | [muradil211/aethersearch_sft](https://huggingface.co/datasets/muradil211/aethersearch_sft) | full JSONL payload, provenance manifest, checksums, and dataset card |
| Code | this repository | SFT build scripts, DPO/RL training code, configs, runtime assets, and tests |

## Table of Contents

- [Overview](#overview)
- [Training Pipeline](#training-pipeline)
- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [Release Boundary](#release-boundary)

## Overview

AetherSearch is a search-augmented post-training project organized around three
stages: supervised fine-tuning, preference optimization, and reinforcement
learning. The repository preserves the training code and operational assets
needed to reproduce the training flow, while large data and model artifacts are
hosted on Hugging Face.

## Training Pipeline

| Stage | Purpose | Primary locations |
|---|---|---|
| SFT | cold-start full-trajectory supervision | `sft_data/`, `sft_data/scripts/` |
| DPO | preference-style alignment stage | `src/agentic_rl/`, `configs/`, `scripts/` |
| RL | search-augmented rollout and policy optimization | `src/agentic_rl/`, `scripts/`, `configs/formal_resolved/` |

## Quick Start

Install the observed core environment and the local package:

```bash
python -m pip install -r environment/requirements-core-observed.txt
python -m pip install -e .
```

Run the static validation checks:

```bash
bash scripts/validate_static.sh
```

The production RL entrypoint is:

```bash
bash scripts/_run_runtime_job.sh FORMAL <resolved_config.yaml> <run_dir> [resume_checkpoint]
```

For the preserved formal configs and launch commands, see
`configs/formal_resolved/` and `TRAINING_REPRODUCTION.md`.

## Repository Layout

- `sft_data/`: SFT data release metadata and build scripts. The full SFT JSONL
  payload is hosted on [Hugging Face Datasets](https://huggingface.co/datasets/muradil211/aethersearch_sft).
- `src/agentic_rl/`: RL training, rollout, advantage, policy loss, retriever,
  checkpoint, and runtime adapter code.
- `scripts/`: launch, preflight, resume, validation, and operational scripts for
  the RL training stage.
- `configs/`: base, formal, hardware, retriever, and executed resolved configs.
- `runtime_assets/`: local runtime assets required by the training launcher.
- `tests/`: unit and integration checks for the training code.
- `environment/`: observed package versions and environment template.

## Release Boundary

Large model weights, optimizer-state checkpoints, eval result bundles, report
archives, and runtime snapshots are intentionally not committed to this GitHub
repository. Public model and data artifacts are linked from Hugging Face above.
