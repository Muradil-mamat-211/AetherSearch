<div align="center">

# AetherSearch

**Search-augmented post-training with SFT assets and reinforcement learning.**

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-AetherSearch-yellow)](https://huggingface.co/muradil211/AetherSearch)
[![SFT Data](https://img.shields.io/badge/%F0%9F%A4%97%20Data-AetherSearch%20SFT-yellow)](https://huggingface.co/datasets/muradil211/aethersearch_sft)
[![Eval Data](https://img.shields.io/badge/%F0%9F%A4%97%20Eval-Search--R1%20Full-yellow)](https://huggingface.co/datasets/muradil211/AetherSearch-Eval)
[![Code](https://img.shields.io/badge/GitHub-Code-181717?logo=github)](https://github.com/Muradil-mamat-211/AetherSearch)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](pyproject.toml)

🤗 [AetherSearch Model](https://huggingface.co/muradil211/AetherSearch) |
🤗 [AetherSearch SFT Data](https://huggingface.co/datasets/muradil211/aethersearch_sft) |
🤗 [Full Eval Data](https://huggingface.co/datasets/muradil211/AetherSearch-Eval)

</div>

## Latest News

- **2026-08:** SFT release metadata and the full AetherSearch RL training code
  were released. The RL recipe consumes an externally hosted DPO warm-start
  model.
- **2026-08:** AetherSearch model weights are available on
  [Hugging Face](https://huggingface.co/muradil211/AetherSearch).
- **2026-08:** The public SFT data release is hosted directly on
  [Hugging Face Datasets](https://huggingface.co/datasets/muradil211/aethersearch_sft).

## Open Resources

| Resource | Link | Contents |
|---|---|---|
| 🤗 Model | [muradil211/AetherSearch](https://huggingface.co/muradil211/AetherSearch) | model weights, tokenizer, config, and model card |
| 🤗 SFT data | [muradil211/aethersearch_sft](https://huggingface.co/datasets/muradil211/aethersearch_sft) | full JSONL payload, provenance manifest, checksums, and dataset card |
| 🤗 Search-R1 train data | [PeterJinGo/nq_hotpotqa_train](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train) | upstream `train.parquet`, pinned by checksum in `EXTERNAL_ASSETS.md` |
| 🤗 Full eval data | [muradil211/AetherSearch-Eval](https://huggingface.co/datasets/muradil211/AetherSearch-Eval) | complete 51,713-row Search-R1 `test.parquet`, provenance, and checksums |
| Code | this repository | SFT build scripts, RL training code, configs, runtime assets, and tests |

## Table of Contents

- [Overview](#overview)
- [Training Pipeline](#training-pipeline)
- [Agentic RL Method](#agentic-rl-method)
- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [Release Boundary](#release-boundary)

## Overview

AetherSearch is a search-augmented post-training project whose public code
currently covers SFT data construction metadata and the downstream RL stage.
The released RL recipe starts from an externally hosted DPO warm-start model;
the DPO trainer and DPO data-generation pipeline are not included in this
repository. Large data and model artifacts are hosted on Hugging Face.

## Training Pipeline

| Stage | Purpose | Primary locations |
|---|---|---|
| SFT | cold-start full-trajectory supervision | `sft_data/`, `sft_data/scripts/` |
| DPO warm start | externally produced actor/reference initialization | `AETHERSEARCH_ACTOR_MODEL`, `AETHERSEARCH_REFERENCE_MODEL` |
| RL | search-augmented rollout and policy optimization | `src/agentic_rl/`, `scripts/`, `recipes/rl/` |

## Agentic RL Method

> **Method status.** This section describes the final Agentic-RL credit-assignment path used by AetherSearch:
> `answer_only_ragen2_mica_ig_v1_singleton_outcome`.
>
> The method is **MICA-inspired**, but it is not a verbatim implementation of the original MICA algorithm. It combines:
> 1. answer-outcome variance filtering inspired by RAGEN-2,
> 2. exact information-gain rewards derived from answer likelihood,
> 3. mixed immediate-and-return credit assignment,
> 4. the existing turn-level clipped policy optimizer and reference-model KL regularization.

---

### 1. Overview

For a prompt $p$, the current policy samples a group of trajectories
$$
\{\tau_{p,i}\}_{i=1}^{G}.
$$

A trajectory may contain multiple search actions:
$$
\tau_{p,i}=(s_{p,i,1},o_{p,i,1},\ldots,s_{p,i,T_i},o_{p,i,T_i},a_{p,i}),
$$
where $s_{p,i,t}$ is the $t$-th model-generated Search turn,
$o_{p,i,t}$ is the retrieved observation, and $a_{p,i}$ is the final Answer turn.

The final pipeline is
$$
\boxed{
\text{Rollout}\rightarrow
\text{terminal outcome}\rightarrow
\text{Answer-only RAGEN-2}\rightarrow
\text{selected-only Exact IG}\rightarrow
\text{MICA-IG credit}\rightarrow
\text{policy update}
}
$$

Exact-IG scoring is deferred until **after prompt selection**, so non-selected prompt groups do not incur the expensive Exact-IG model forward.

---

### 2. Answer-only RAGEN-2 prompt selection

Prompt filtering uses only the dispersion of terminal task outcomes. Exact IG does **not** enter prompt selection.

For prompt $p$, let $O_{p,i}$ be the terminal task outcome of trajectory $i$. With $G$ rollouts,
$$
\bar O_p=\frac{1}{G}\sum_{i=1}^{G}O_{p,i},
$$
and the prompt score is the sample variance
$$
V_p^{O}
=
\frac{1}{G-1}
\sum_{i=1}^{G}
\left(O_{p,i}-\bar O_p\right)^2.
$$

Prompts are sorted in descending order of $V_p^{O}$. With variance-mass threshold $\rho$, the selected set is the shortest prefix satisfying
$$
\sum_{j=1}^{K^\star}V_{\sigma(j)}^{O}
\ge
\rho\sum_p V_p^{O}.
$$

In our training configuration,
$$
\boxed{\rho=0.9}.
$$

Filtering is performed at the **prompt-group level**: if a prompt is selected, its rollout group is retained for the subsequent credit-assignment stage.

---

### 3. Exact information gain

For each selected trajectory, Exact IG evaluates how much a retrieved observation changes the model's confidence in the canonical answer.

Let:
- $h^-_{p,i,t}$: trajectory prefix immediately before the $t$-th retrieved observation;
- $h^+_{p,i,t}$: trajectory prefix immediately after that observation;
- $a_p^\star$: the canonical answer.

We construct the fixed target
$$
y_p=
\texttt{<think>The retrieved evidence now supports the answer.</think><answer>}
+a_p^\star+
\texttt{</answer>}.
$$

The scaffold and answer tags are teacher-forced context, while the score is averaged only over the token positions belonging to the **answer body**.

For a prefix $h$,
$$
\Phi_p(h)
=
\frac{1}{|B_p|}
\sum_{j\in B_p}
\log\pi_{\theta_{\mathrm{snap}}}
\left(y_{p,j}\mid h,y_{p,<j}\right),
$$
where $B_p$ is the answer-body token span and $\theta_{\mathrm{snap}}$ is the rollout-start policy snapshot.

The immediate process reward is
$$
\boxed{
r^{IG}_{p,i,t}
=
\Phi_p(h^+_{p,i,t})
-
\Phi_p(h^-_{p,i,t})
}.
$$

There is **no exponentiation**. Exact IG is a detached reward signal; gradients do not flow through this scoring forward. A Search turn without a valid pre/post retrieval state is not Exact-IG eligible.

---

### 4. Raw suffix return

For each valid Search position, MICA-IG forms a raw future-return channel. Let
$$
\mathcal V_{p,i}=\{t:\text{Search }t\text{ has valid Exact IG}\}.
$$

Then
$$
G^{IG}_{p,i,t}
=
\sum_{\substack{k\ge t\\k\in\mathcal V_{p,i}}}
\gamma^{k-t}r^{IG}_{p,i,k}.
$$

The final V1 configuration fixes
$$
\boxed{\gamma=1},
$$
hence
$$
\boxed{
G^{IG}_{p,i,t}
=
\sum_{\substack{k\ge t\\k\in\mathcal V_{p,i}}}
r^{IG}_{p,i,k}.
}
$$

This is a **raw suffix sum**. The MICA branch does not apply the older $1/\sqrt n$ future-credit rescaling. Missing/invalid Search positions are omitted from the suffix return rather than inserted as zero-reward positions.

---

### 5. MICA-IG credit assignment

For each selected prompt group, raw Exact-IG values and their suffix returns are normalized independently by Search depth across the prompt group. For a Search turn with at least two eligible peers, the final Search advantage is
$$
A^{search}_{p,i,t}
=
\alpha A^{return}_{p,i,t}
+(1-\alpha)A^{local}_{p,i,t},
$$
with $\alpha=0.5$ in V1.

If a Search depth has only one eligible peer, the implementation uses the normalized terminal outcome as the singleton fallback. A policy-credit-eligible Search turn without a valid Exact-IG reward receives zero MICA Search advantage.

The rollout, Exact-IG, selection, credit-assignment, KL, and policy-update behavior is defined by the code and formal recipe in `src/agentic_rl/`, `configs/`, and `recipes/rl/`.

## Quick Start

Install the local package inside an RL environment that already contains the
compatible PyTorch, veRL, vLLM, Ray, and FlashAttention stack:

```bash
python -m pip install -e .
```

Create the machine-local environment configuration:

```bash
cp environment/env.template.sh environment/env.local.sh
# Edit environment/env.local.sh, then:
source environment/env.local.sh
```

Run the lightweight source, shell, and configuration checks:

```bash
bash scripts/validate_static.sh
```

Validate the only released hardware recipe without starting services:

```bash
bash scripts/train_rl.sh --dry-run
```

Start RL training on the validated four-GPU topology:

```bash
bash scripts/train_rl.sh
```

The included recipe assigns physical GPU 0 to retrieval and asynchronous
training-time evaluation, and physical GPUs 1-3 to the three-rank vLLM/FSDP2
runtime. Every 20-update evaluation uses the complete 51,713-row Search-R1
`test.parquet`. See `recipes/rl/README.md` for the configuration boundary.
The resolved configuration is materialized inside each new run directory.

## Repository Layout

- `sft_data/`: SFT data release metadata and build scripts. The full SFT JSONL
  payload is hosted on [Hugging Face Datasets](https://huggingface.co/datasets/muradil211/aethersearch_sft).
- `src/agentic_rl/`: RL training, rollout, advantage, policy loss, retriever,
  checkpoint, and runtime adapter code.
- `scripts/`: launch, preflight, resume, validation, and operational scripts for
  the RL training stage; see `scripts/README.md` for public versus historical
  entrypoints.
- `recipes/rl/`: the single validated public RL recipe and its usage boundary.
- `configs/`: base, formal, hardware, retriever, stage configs, and the Exact-IG
  runtime gate;
  see `configs/README.md` for their portability boundary.
- `runtime_assets/`: local runtime assets required by the training launcher.
- `tests/`: unit and integration checks for the training code.
- `environment/`: observed package versions and environment template.

## Release Boundary

Large model weights, optimizer-state checkpoints, eval result bundles, report
archives, and runtime snapshots are intentionally not committed to this GitHub
repository. Public model and data artifacts are linked from Hugging Face above.
