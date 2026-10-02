<a id="aethersearch-dpo"></a>

# 🎯 AetherSearch DPO

[![DPO Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-AetherSearch__DPO-yellow)](https://huggingface.co/muradil211/AetherSearch_DPO)
[![DPO Data](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-AetherSearch__DPO-yellow)](https://huggingface.co/datasets/muradil211/AetherSearch_DPO)
[![Checksums](https://img.shields.io/badge/checksums-sha256-blue)](checksums.sha256)

> 📚 **Complete dataset:** [AetherSearch DPO on Hugging Face](https://huggingface.co/datasets/muradil211/AetherSearch_DPO)
>
> This directory is the complete public boundary for the DPO stage: strict
> preference-data validation, token-level loss masking, the DPO objective,
> the hardware-independent launcher, DeepSpeed configuration, dependency
> pins, release metadata, and source checks.

## 🧭 Quick navigation

- [Dataset overview](#dataset-overview)
- [Data construction workflow](#dpo-data-construction-workflow)
- [Preference loss and token masks](#preference-loss-contract)
- [Training recipe](#training-recipe)
- [Reproduce the DPO stage](#reproduce-the-dpo-stage)
- [DPO evaluation](#dpo-evaluation)
- [Files](#files) · [Checksums](#checksums)

<a id="release-at-a-glance"></a>

## 📦 Release at a glance

| Item | Details |
|---|---|
| Complete data | [muradil211/AetherSearch_DPO](https://huggingface.co/datasets/muradil211/AetherSearch_DPO) |
| Base checkpoint | [muradil211/AetherSearch_SFT](https://huggingface.co/muradil211/AetherSearch_SFT) |
| DPO model output | [muradil211/AetherSearch_DPO](https://huggingface.co/muradil211/AetherSearch_DPO) |
| Reproduction entrypoint | [`scripts/run_train_dpo_zero3.sh`](scripts/run_train_dpo_zero3.sh) |
| Preference pairs | 2,126 |
| Training unit | Shared prompt with chosen/rejected continuations |
| License metadata | `unknown` |
| GitHub contents | Data metadata, training code, configuration, and source checks |

<a id="dataset-overview"></a>

## 🧩 Dataset overview

The stage uses the complete 2,126-pair `train.jsonl` release. Every normalized
question is unique and every row contains one shared `prompt_text`, one
preferred continuation, and one non-preferred continuation. The exact data
identity is:

```text
c42adcb0f194cff3126134b37afd85e4b89aa9917e5c98dda4b09904509f61e9
```

Source composition:

| Source | Pairs |
|---|---:|
| TriviaQA | 1,445 |
| MuSiQue | 410 |
| Natural Questions | 130 |
| WebQuestions | 87 |
| 2WikiMultiHopQA | 54 |
| **Total** | **2,126** |

Preference composition:

| Pair type | Pairs |
|---|---:|
| `answer_hard_negative` | 1,166 |
| `query_hard_negative` | 289 |
| `true_full_trajectory_preference` | 235 |
| `insufficient_information_continue_search` | 173 |
| `evidence_misread_negative` | 94 |
| `premature_answer_negative` | 93 |
| `multi_hop_decomposition_negative` | 30 |
| `regression_protection_pair` | 28 |
| `query_refinement_negative` | 18 |
| **Total** | **2,126** |

<a id="dpo-data-construction-workflow"></a>

## 🛠️ DPO Data Construction Workflow

The workflow described by the project maintainer starts with **5,000 isolated
candidate questions** and targets approximately **2,126 high-quality preference
pairs**. It uses the failure taxonomy established through
[SFT rollout analysis](../sft/README.md#sft-evaluation-and-failure-analysis)
to identify concrete decision errors and construct preferences at the same
agent state. The exact published count and source composition are recorded
in [Dataset overview](#dataset-overview).

### 1. Build the candidate question pool

Sample 5,000 candidate questions by source from datasets such as NQ, TriviaQA,
PopQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, and Bamboogle, as well as other
eligible QA sources. Strictly exclude questions that overlap with **SFT
training data, evaluation data, or RL training data**.

Use the same question normalization as Eval-1400: Unicode NFKC, casefold,
whitespace collapse, strip, and removal of trailing ASCII/full-width question
marks. Exclude normalized exact matches and confirmed high-confidence
near-duplicates before sampling.

Stratify across sources with eligible questions; eligibility determines which
sources can contribute. In particular, the complete 125-question Bamboogle
release is already included in
[frozen Eval-1400](https://huggingface.co/datasets/muradil211/AetherSearch_Eval_1400/tree/frozen-v1),
so it contributes **zero eligible DPO candidates** under strict evaluation
isolation. Select the 5,000 candidates from the remaining eligible sources.
If the eligible pool is too small, stop without relaxing the exclusions.

### 2. Run four real SFT rollouts per question

Use the [SFT model](https://huggingface.co/muradil211/AetherSearch_SFT)
to sample **$K=4$ complete trajectories per question**: 20,000 initial
rollouts for a 5,000-question pool. Every `<search>` action must execute the
real retriever, and the resulting observations become part of the trajectory.
Retain the actual actions, retrieval results, and interaction history.

### 3. Locate the first actionable failure

Codex reviews all **format-valid trajectories for each question**, using the
known failure taxonomy. Across those trajectories, identify the **first
actionable failure**: the earliest genuinely incorrect decision for which a
correction can be made and verified at the same state.

Record the shared state **$x$ immediately before that decision**, including the
question, conversation prefix, previous agent actions, and real retrieval
observations. The target is a decision failure in a valid trajectory.

### 4. Use the real SFT error as rejected

Use the incorrect continuation that SFT actually generated from that state:

```math
y_l = \text{actual incorrect SFT continuation generated from } x
```

Preserve the observed error and its rollout provenance. The rejected side is
grounded in real SFT behavior at the recorded prefix.

### 5. Find chosen from the same prefix

First, fix **$x$** and resample **four continuations with the SFT model**,
executing any subsequent searches through the real retriever. If SFT produces
a correct behavior that passes verification, use:

```math
y_w = \text{verified good SFT continuation generated from } x
```

If none of the four resamples succeeds, ask a stronger teacher or Codex for a
**minimal correction** from that same prefix. Keep the question, prior actions,
and existing observations in $x$ unchanged; correct the faulty decision with
as little change to the continuation as possible.

### 6. Verify that chosen is better

Validate the correction with observable evidence:

- **Answer:** check the answer against the original gold answer or accepted
  aliases.
- **Search:** execute the proposed search through the real retriever and
  inspect whether it obtains more useful new evidence for the question.

An LLM judge saying that a continuation is better is not sufficient for
acceptance. Keep only corrections supported by answer verification or actual
retrieval evidence.

### 7. Form the candidate preference pair

The preference unit is:

```math
\boxed{(x,\ y_w,\ y_l)}
```

Map it to the public training fields:

| Preference component | Public field | Meaning |
|---|---|---|
| $x$ | `prompt_text` | Exact shared state before the faulty decision |
| $y_w$ | `chosen` | Verified better continuation from that state |
| $y_l$ | `rejected` | Actual incorrect SFT continuation from that state |

Attach the question, source, gold aliases, and corresponding `pair_type`.
Examples include `premature_answer_negative`, `query_hard_negative`, and
`evidence_misread_negative`, following the
[SFT failure-to-pair mapping](../sft/README.md#sft-evaluation-and-failure-analysis).
Both sides share the exact same `prompt_text`; neither continuation duplicates
the prefix.

### 8. Keep one highest-quality pair per question

Keep **one pair per normalized question**. Prefer the pair whose failure is
**earliest, clearest, most directly verifiable**, and whose chosen/rejected
continuations differ by the **smallest correction** needed to fix the decision.
Discard ambiguous, unverified, or lower-quality alternatives.

### 9. Deduplicate and audit the final release

Check question uniqueness, format validity, data leakage against SFT/Eval/RL,
shared-prefix consistency, non-empty and distinct chosen/rejected
continuations, valid trajectory structure, and the evidence supporting each
preference. Review a human sample of the retained pairs before release.

The target is approximately **2,126 high-quality DPO pairs**; the current
canonical release contains **exactly 2,126**. Quality and isolation checks
determine acceptance, and must not be weakened to meet the target count.

<a id="public-schema"></a>

## 📋 Public schema

Each canonical JSONL row contains exactly these fields, in this order:

1. `id`
2. `question`
3. `source_dataset`
4. `answers`
5. `pair_type`
6. `prompt_text`
7. `chosen`
8. `rejected`

The preference unit is:

```text
(prompt_text, chosen, rejected)
```

The trainer rejects duplicate IDs, duplicate normalized questions, malformed
ChatML prompts, malformed trajectory tags, empty continuations, identical
preference pairs, checksum drift, record-count drift, and unsafe sequence
truncation before allocating model weights.

<a id="preference-loss-contract"></a>

## 🧮 Preference-loss contract

For policy model $`\pi_\theta`$, frozen SFT reference $`\pi_{\mathrm{ref}}`$,
chosen continuation $`y_w`$, rejected continuation $`y_l`$, and shared prompt
$`x`$, the implementation uses the summed-token sigmoid DPO objective:

```math
\mathcal{L}_{\mathrm{DPO}} =
-\log \sigma\!\left(
\beta\left[
\log \frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\log \frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right]
\right).
```

The token contract is exact:

- every `prompt_text` token is masked on both sides;
- every environment-provided `<information>...</information>` span inside a
  continuation is masked, including its boundary tags;
- all other continuation tokens are scored;
- an answer-terminal continuation supervises one final `<|im_end|>` token;
- a search-terminal continuation does not append `<|im_end|>`, because the
  retrieval runtime must provide the next information span;
- chosen and rejected log probabilities are sums over scored continuation
  tokens, matching the sigmoid DPO objective;
- policy and reference models begin from the same pinned SFT checkpoint, and
  reference parameters remain frozen.

Segments are tokenized independently at every mask boundary. The trainer then
decodes each reconstructed sequence and requires an exact match with the
tokenizer-normalized source, preventing a BPE token from crossing between
masked and scored regions.

<a id="canonical-data-preflight"></a>

## ✅ Canonical data preflight

The full tokenizer-level preflight accepts all 2,126 pairs and filters none.
Across both sides, the longest complete sequence is 2,361 tokens, safely below
the fixed 4,096-token ceiling. It verifies 4,252 decoded round trips, 447
chosen-side information blocks, 259 rejected-side information blocks, and no
all-masked continuation.

<a id="training-recipe"></a>

## ⚙️ Training recipe

| Setting | Value |
|---|---:|
| Policy start | `muradil211/AetherSearch_SFT` |
| Frozen reference | Same SFT checkpoint |
| SFT revision | `437aca474d3966e57e82af565db95d0ad64aa24d` |
| Preference pairs | 2,126 |
| Epochs | 1 |
| Learning rate | `5e-7` |
| DPO beta | `0.1` |
| Scheduler | Cosine |
| Warmup ratio | `0.03` |
| Weight decay | `0.0` |
| Effective global batch | 12 pairs |
| Per-device batch | 1 pair |
| Maximum sequence length | 4,096 |
| Precision | BF16 with optional TF32 matrix math |
| Distributed optimizer | DeepSpeed ZeRO-3 |
| Gradient checkpointing | Enabled |
| Seed | 42 |
| Intermediate saves | Disabled by default |

The canonical forward mode is `sequential`, minimizing peak activation memory
by scoring chosen and rejected sequences separately. Hosts with additional
memory may set `FORWARD_MODE=concatenated`; this changes batching strategy,
not the mask or objective.

When ZeRO-3 is active, both policy and frozen reference weights are sharded.
The reference receives no optimizer and no gradients. Final model export uses
an incomplete directory followed by an atomic rename, so a failed export is
never presented as `final_model/`.

<a id="reproduce-the-dpo-stage"></a>

## 🚀 Reproduce the DPO stage

Install a CUDA-compatible PyTorch build for the target host, then install the
stage dependencies:

```bash
python -m pip install -r dpo/requirements.txt
```

Download the canonical data and metadata without replacing this README:

```bash
hf download muradil211/AetherSearch_DPO \
  train.jsonl dataset_manifest.json ATTRIBUTION.md \
  --repo-type dataset \
  --local-dir dpo
sha256sum -c dpo/checksums.sha256
```

Run the strict CPU-side data and mask preflight:

```bash
python dpo/scripts/train_dpo.py \
  --model_name_or_path muradil211/AetherSearch_SFT \
  --model_revision 437aca474d3966e57e82af565db95d0ad64aa24d \
  --ref_model_name_or_path muradil211/AetherSearch_SFT \
  --ref_model_revision 437aca474d3966e57e82af565db95d0ad64aa24d \
  --train_file dpo/train.jsonl \
  --output_dir outputs/dpo/preflight \
  --expected_num_samples 2126 \
  --expected_sha256 c42adcb0f194cff3126134b37afd85e4b89aa9917e5c98dda4b09904509f61e9 \
  --check_data_only \
  --audit_report_path outputs/dpo/preflight/data_audit.json
```

Start the canonical BF16 ZeRO-3 recipe:

```bash
bash dpo/scripts/run_train_dpo_zero3.sh
```

The launcher uses every CUDA device already visible to the process and derives
gradient accumulation from `NPROC_PER_NODE`, per-device batch size, and global
batch 12. For one, two, three, four, six, or twelve workers, the default
accumulation resolves to 12, 6, 4, 3, 2, or 1. A topology that cannot preserve
the configured global batch exactly is rejected.

Machine-local controls such as `PYTHON_BIN`, `DATA_FILE`, `MODEL_NAME_OR_PATH`,
`REFERENCE_MODEL_NAME_OR_PATH`, `OUTPUT_DIR`, `DEEPSPEED_CONFIG`,
`DATALOADER_NUM_WORKERS`, and `MINIMUM_FREE_KB` are environment inputs. The
launcher does not set physical GPU IDs, node addresses, NCCL fabric policy,
CUDA allocator tuning, CPU thread counts, or server-specific absolute paths.
Device visibility and cluster orchestration belong to the surrounding runtime.

<a id="released-checkpoint"></a>

## 🤗 Released checkpoint

The [AetherSearch DPO checkpoint](https://huggingface.co/muradil211/AetherSearch_DPO)
was trained in one DPO stage from the pinned AetherSearch SFT checkpoint over
all 2,126 pairs in the canonical `train.jsonl`, using the code and recipe in
this directory. Training was performed on a separate server. The public model
repository contains the final model artifacts; this GitHub boundary contains
the corresponding training implementation and does not include server-local
run logs or optimizer state.

<a id="dpo-evaluation"></a>

## 📊 DPO Evaluation

DPO is evaluated through two complementary parts:

```math
\boxed{\textbf{DPO Eval} = \text{End-to-End Eval} + \text{Preference Eval}}
```

The first measures the deployed Search Agent's behavior; the second measures
how the model ranks good and bad continuations from the same decision state.
Both compare the [SFT model](https://huggingface.co/muradil211/AetherSearch_SFT)
with the [DPO model](https://huggingface.co/muradil211/AetherSearch_DPO):

```math
\boxed{\text{SFT model} \rightarrow \text{DPO model}}
```

### 1. End-to-End Eval: real Search-Agent rollouts on frozen Eval-1400

Use exactly the same frozen
[Eval-1400](https://huggingface.co/datasets/muradil211/AetherSearch_Eval_1400/tree/frozen-v1)
as [SFT evaluation](../sft/README.md#sft-evaluation-and-failure-analysis).
Keep all 1,400 questions unchanged. For each checkpoint, run a complete
Search-Agent rollout: execute the model's searches against the real
retriever, feed the returned observations into subsequent turns, and score
the final answer.

Keep the evaluator, retriever assets, decoding settings, interaction budgets,
and metric definitions fixed between the SFT and DPO runs. Continue reporting
the same four metrics:

```math
\boxed{EM,\quad F1,\quad FTFA,\quad AvgSearch}
```

| Metric | What the SFT-to-DPO comparison checks |
|---|---|
| EM | Exact-match accuracy of the final answer |
| F1 | Token-level quality of the final answer |
| FTFA | Preservation of the format and tool-call schema ability established by SFT |
| AvgSearch | Average number of executed searches per question, interpreted together with answer quality |

The objective is to verify that DPO improves final answers and search behavior
while preserving format ability. Interpret search counts alongside EM and F1:
an agent that searches less by answering prematurely has not demonstrated
better search behavior. A lower preference-training loss alone does not
establish these improvements.

### 2. Preference Eval: held-out chosen/rejected ranking

Use a separate held-out preference set with the same failure taxonomy and
pair types as DPO training. Its questions and preference pairs must never
participate in DPO training; also exclude overlap with SFT training, RL
training, and Eval-1400 using the same question-normalization and duplicate
checks described in [data construction](#dpo-data-construction-workflow).
The canonical 2,126-pair release remains train-only; the held-out set is a
separate evaluation artifact.

For each fixed pair $(x,y_w,y_l)$, compute both chosen and rejected sequence
scores under each checkpoint, using teacher-forced scoring in evaluation mode
with gradients disabled. For model $\pi$, define:

```math
S_{\pi}(y\mid x)
=
\sum_{t:\,m_t=1}
\log \pi(y_t\mid x,y_{<t}),
```

where $m_t$ selects the scored continuation tokens. Reuse the exact
[preference-loss token contract](#preference-loss-contract) and
[sequence scoring implementation](scripts/train_dpo.py): mask the prompt,
retrieved-information spans, and padding; preserve causal next-token
alignment, mask-boundary tokenization, and terminal-token handling. Scores
are sums of token log probabilities, with the same tokenization and masking
for both checkpoints.

Preference accuracy measures each model's own chosen/rejected ranking:

```math
\boxed{
\mathrm{PrefAcc}_{\pi}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
\left[
S_{\pi}(y_{w,i}\mid x_i)
>
S_{\pi}(y_{l,i}\mid x_i)
\right]
}
```

A tie does not count as a correct preference. Evaluate SFT and DPO on the
same held-out pairs and test whether:

```math
\boxed{\mathrm{PrefAcc}_{\mathrm{DPO}} > \mathrm{PrefAcc}_{\mathrm{SFT}}}
```

Report **Overall PrefAcc and PrefAcc for every pair type**, including the
number of evaluated pairs and the SFT-to-DPO change for each type. This shows
which decision boundaries improved, such as continuing search versus
answering prematurely, selecting a useful query, or interpreting retrieved
evidence correctly, and whether any pair type regressed.

### Joint interpretation

> 💡 **Eval-1400 asks whether the real Search Agent becomes stronger. Preference
> Eval asks whether DPO has learned to rank good behavior ahead of bad behavior.**

The two evaluations together determine whether the learned preferences
translate into better agent behavior. Report the paired EM, F1, FTFA, and
AvgSearch comparison alongside overall and per-type PrefAcc. The inequality
above is an evaluation target; measured results are needed before claiming
that the released DPO checkpoint satisfies it.

<a id="files"></a>

## 🗂️ Files

| File | Purpose |
|---|---|
| `scripts/train_dpo.py` | Strict dataset audit, mask construction, DPO objective, and trainer |
| `scripts/run_train_dpo_zero3.sh` | Hardware-independent single-node launcher |
| `configs/ds_zero3_bf16.json` | BF16 ZeRO-3 configuration |
| `requirements.txt` | Stage dependency pins excluding host-specific PyTorch |
| `dataset_manifest.json` | Public data schema, distribution, and integrity metadata |
| `ATTRIBUTION.md` | Source attribution and rights status |
| `checksums.sha256` | Release integrity checksums |

<a id="limitations"></a>

## 📝 Limitations

Preference labels include curated hard negatives and trajectory corrections;
they are not human preference votes for every pair. Retrieved information can
be incomplete or incorrect. Training code reproduces the released objective
and data boundary, but users remain responsible for hardware capacity,
retriever behavior, downstream safety, and applicable source terms.

<a id="checksums"></a>

## 🔍 Checksums

After downloading `train.jsonl` from the linked dataset repository into this
directory, verify the complete stage boundary with:

```bash
sha256sum -c dpo/checksums.sha256
```
