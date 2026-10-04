# AetherSearch SFT-2600

[![SFT Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-AetherSearch__SFT-yellow)](https://huggingface.co/muradil211/AetherSearch_SFT)
[![SFT Data](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-AetherSearch__SFT-yellow)](https://huggingface.co/datasets/muradil211/AetherSearch_SFT)
[![Checksums](https://img.shields.io/badge/checksums-sha256-blue)](checksums.sha256)

This directory is the public SFT boundary: SFT-2600 metadata, complete data
construction code, the strict full-trajectory trainer, the BF16 ZeRO-3 launcher,
configuration, tests, and dependency pins. The full JSONL and provenance
payloads are hosted in
[`muradil211/AetherSearch_SFT`](https://huggingface.co/datasets/muradil211/AetherSearch_SFT).

## Where to Start

Run all commands below from the repository root.

| Your goal | Start here | What you need |
|---|---|---|
| Train on the published 2,600 trajectories | [Reproduce Training](#reproduce-training) | Published SFT JSONL, base model, and training environment |
| Generate trajectories with DeepSeek | [Data Construction](#data-construction) | Teacher API access, source questions, and assets for the selected generation branch |
| Understand the training records | [Public Schema](#public-schema) | Field definitions and loss-mask rules |

Training on the published dataset does not require DeepSeek API access, a
running retriever, DPO data, or RL data. Those inputs belong to data generation.

## Release

| Item | Value |
|---|---|
| Records | 2,600 validated trajectories |
| Training unit | Full trajectory |
| Data file | `final_sft_2600.jsonl` |
| Data SHA-256 | `5619896ccc30bfb9d39c2676ec058cb59a0295c31082645153318102da0a7ec8` |
| Trainer | [`scripts/train_sft_2600.py`](scripts/train_sft_2600.py) |
| Launcher | [`scripts/run_train_sft_2600_zero3.sh`](scripts/run_train_sft_2600_zero3.sh) |
| Model repository | [`muradil211/AetherSearch_SFT`](https://huggingface.co/muradil211/AetherSearch_SFT) |
| License metadata | `unknown` |

## Composition

| Trajectory type | Records | Share |
|---|---:|---:|
| `direct_answer` | 600 | 23.08% |
| `single_search` | 1,025 | 39.42% |
| `multi_search` | 975 | 37.50% |
| **Total** | **2,600** | **100.00%** |

### Qwen behavior verification

The question groups reflect Qwen experiments reported by the project
maintainer. Qwen was tested on the `direct_answer` questions and could answer
them correctly without retrieval. On the questions used for `single_search`
and `multi_search` trajectories, Qwen chose to search before answering.

These experiments establish the Qwen behavior observed on the selected
questions. DeepSeek supplies the visible teacher actions used for SFT; the
released `search_count` records the exported training trajectory's search
count, not necessarily the number of searches in the Qwen experiment.

| Search count | Records | Share |
|---:|---:|---:|
| 0 | 600 | 23.08% |
| 1 | 1,025 | 39.42% |
| 2 | 667 | 25.65% |
| 3 | 265 | 10.19% |
| 4 | 43 | 1.65% |

All records were globally shuffled with deterministic seed 42 and assigned IDs
`000001` through `002600` in the shuffled order.

## Public Schema

Each JSONL record contains exactly five fields, in this order:

1. `id`
2. `question`
3. `trajectory_type`
4. `search_count`
5. `full_trajectory_text`

`full_trajectory_text` is the training unit. Every trajectory contains one
system message, one user message, and one assistant trajectory, and ends exactly
with `</answer><|im_end|>`.

The loss contract is:

- mask system, user, and question tokens;
- mask every complete `<information>...</information>` observation;
- supervise assistant `<think>`, `<search>`, and `<answer>` actions;
- supervise the final assistant `<|im_end|>` token;
- never truncate a full trajectory in the canonical recipe.

For `direct_answer`, `search_count=0` and no `<search>` or `<information>` span
is present. `single_search` has exactly one search/information turn.
`multi_search` retains every sequential search/information turn.

### Final-Answer Schema and RL Alignment

Final answer turns use fixed, retrieval-state-dependent summaries:

```text
direct_answer:
<think>Reliable prior knowledge is sufficient to answer.</think><answer>{answer}</answer>

single_search / multi_search:
<think>The retrieved evidence now supports the answer.</think><answer>{answer}</answer>
```

These are the same scaffolds used by RL Exact-IG teacher-forced scoring.
The no-retrieval baseline (including a direct-answer trajectory and the prefix
before the first search) uses the prior-knowledge scaffold. Prefixes after a
retrieved observation use the retrieved-evidence scaffold. SFT teaches this
consistent output format so scoring context and learned answer format align;
it does not guarantee exact formatting on every generated answer.

RL scores only the canonical-answer covering token span, not the fixed think
text itself. With these state-dependent scaffolds, the first IG difference
includes both the evidence change and the scaffold change; it is not a
fixed-scaffold estimate of evidence gain alone. Later retrieval increments use
the same retrieved-evidence scaffold. Zero-search trajectories have a baseline
score but no retrieval IG increment. See [RL information gain](../README.md#3-retrieval-information-gain).

## SFT Evaluation and Failure Analysis

We use the frozen [AetherSearch Eval-1400](https://huggingface.co/datasets/muradil211/AetherSearch_Eval_1400/tree/frozen-v1)
benchmark of **1,400 evaluation questions** during SFT training and after
training completes. The model runs a complete, real Search-Agent trajectory
for each question, including tool execution and multi-turn interaction.
We measure **EM, F1, FTFA, and the average number of searches**.

### Eval-1400 results

The project maintainer reports the following results for the
[released SFT checkpoint](https://huggingface.co/muradil211/AetherSearch_SFT)
on all **1,400 questions**:

| Metric | SFT |
|---|---:|
| EM (%) | **25.0** |
| F1 (%) | **33.0** |
| FTFA (%) | **97.5** |
| AvgSearch (searches/question) | **2.5** |

EM, F1, and FTFA are expressed as percentages. AvgSearch is the average
number of executed searches per question.

The **97.5% FTFA** result reflects SFT's role in stabilizing output format,
tool calls, and multi-turn interaction: FTFA measures the rate of calls that
follow the required schemas. These results form the baseline for the
[DPO end-to-end comparison](../dpo/README.md#eval-1400-results).

### Failure analysis

We also look beyond final answer accuracy: the same incorrect answer can
result from very different failures. We use **stratified sampling** of the
rollout trajectories, selecting **20 examples from each of the seven sources**
(NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, and Bamboogle),
for **140 examples in total**. Human review first identifies an initial
**failure taxonomy**. We then explain those failure modes to Codex using
concrete examples from the real rollouts, and have Codex review the remaining
trajectories. Codex can assign existing categories and also **flag new failure
modes** for review.

This analysis determines which failures to target with **Direct Preference
Optimization (DPO)** and the corresponding preference-pair types:

| DPO pair type | SFT failure exposed |
|---|---|
| `answer_hard_negative` | The final answer sounds plausible but is factually wrong. |
| `query_hard_negative` | The query follows the required format but searches in an unhelpful direction. |
| `insufficient_information_continue_search` | The agent fails to continue searching when the available information is insufficient. |
| `premature_answer_negative` | The agent answers before gathering enough evidence. |
| `evidence_misread_negative` | The agent obtains the correct evidence but misinterprets it. |
| `multi_hop_decomposition_negative` | The agent decomposes a multi-hop question in an unsuitable order. |
| `query_refinement_negative` | The agent fails to refine the second-hop query using information already obtained. |
| `true_full_trajectory_preference` | Individual steps may not be clearly wrong, but the full trajectory is substantially worse. |

The evaluation analysis informs the failure categories. DPO preference pairs
use separate training questions that exhibit those failures, so the frozen
Eval-1400 questions remain held out.

The [DPO construction specification](../dpo/README.md#dpo-data-construction-workflow)
uses four initial complete SFT rollouts to locate an actionable decision
error, then four complete continuations from the exact prefix before that
error. A candidate must pass terminal protocol validation and alias-aware
EM = 1, followed by verification of its next Search or Answer action and its
advantage over the actual rejected action. Codex proposes a minimal correction
only when those sampled candidates fail; corrected actions undergo the same
full-rollout and local checks. For these decision-level pairs, full rollouts
provide verification evidence and the verified next actions supply the
chosen/rejected training targets.

## Reproduce Training

This workflow uses the published `final_sft_2600.jsonl` directly. Data generation
is a separate workflow and is not a prerequisite for this training run.

Install a CUDA-compatible PyTorch build and the SFT dependencies:

```bash
python -m pip install -r sft/requirements.txt
```

Download and verify the frozen release:

```bash
hf download muradil211/AetherSearch_SFT \
  final_sft_2600.jsonl provenance_manifest.jsonl \
  --repo-type dataset \
  --local-dir sft
(cd sft && sha256sum -c checksums.sha256)
```

Run the strict structure, tokenizer, and loss-mask preflight:

```bash
python sft/scripts/train_sft_2600.py \
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --model_revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --train_file sft/final_sft_2600.jsonl \
  --output_dir outputs/sft/sft_2600_preflight \
  --expected_num_samples 2600 \
  --expected_sha256 5619896ccc30bfb9d39c2676ec058cb59a0295c31082645153318102da0a7ec8 \
  --check_data_only \
  --audit_report_path outputs/sft/sft_2600_preflight/data_audit.json
```

Start the canonical BF16 ZeRO-3 recipe:

```bash
bash sft/scripts/run_train_sft_2600_zero3.sh
```

The launcher pins the data count and SHA, Qwen base revision, one epoch,
sequence length 4096, learning rate `2e-6`, effective global batch size 24,
cosine scheduling, BF16, TF32, gradient checkpointing, grouped dynamic padding,
and strict no-truncation behavior. It discovers the visible local GPUs, derives
gradient accumulation from the worker count, runs preflight before model
allocation, refuses to overwrite an existing final model, and exports
`final_model/` atomically.

Machine-local paths and topology are supplied through `PYTHON_BIN`,
`DATA_FILE`, `OUTPUT_DIR`, `DEEPSPEED_CONFIG`, `CUDA_VISIBLE_DEVICES`, and the
other environment variables declared by the launcher. The launcher contains no
server-specific absolute path or physical GPU assignment.

## Data Construction

Use this section when you want to generate teacher trajectories. There are two
branches: retrieval trajectories and direct-answer trajectories. Both export
the same five-field public schema described above.

Paths beginning with `/absolute/path/to/` are placeholders: replace them with
actual paths on your server. Reference answers stay in the controller for
validation; they are not included in the teacher's question prompt.

The code under
[`data_generation/search_sft_teacher/`](data_generation/search_sft_teacher/)
constructs the complete SFT-2600 trajectory set by distilling visible DeepSeek
actions into the public AetherSearch format. Private provider
`reasoning_content` is not a training target.

The source question set is exactly the question set published in
[`muradil211/AetherSearch_SFT`](https://huggingface.co/datasets/muradil211/AetherSearch_SFT).

### Retrieval trajectories

[`deepseek_rollout.py`](data_generation/search_sft_teacher/deepseek_rollout.py)
runs adaptive multi-turn teacher trajectories. DeepSeek decides whether another
search is needed and emits each visible search or final-answer action. The
controller executes search actions against the local Hybrid-RAG service and
inserts only real retrieved passages as `<information>` observations.

The retrieval policy is wiki18 BM25 top-20 plus E5-base-v2 FAISS `IndexFlatIP`
top-20, fused with RRF (`k=60`) to top-3. There is no synthetic evidence and no
silent dense-only fallback. Up to `--concurrency` trajectories run in parallel;
the async gateway batches dense requests without changing per-trajectory turn
ordering.

Run from the repository root after configuring the external assets and a
verified QA JSONL containing `id`, `question`, `golden_answers`, `data_source`,
and `split=train`:

```bash
export AETHERSEARCH_SFT_WORKSPACE=/absolute/path/to/runtime-assets
export QUESTIONS_FILE=/absolute/path/to/verified_train_qa.jsonl
bash sft/data_generation/run_teacher_rollout.sh
```

The controller persists API receipts, retrieval traces, candidate status, retry
history, and review state in SQLite. Only explicitly approved, structurally
valid trajectories can be exported.

### Direct-answer trajectories

[`generate_direct_answer_sft.py`](data_generation/search_sft_teacher/generate_direct_answer_sft.py)
constructs `search_count=0` trajectories. It registers no tools, runs DeepSeek
with thinking disabled, requires an exact
`<think>...</think><answer>...</answer>` response, validates the normalized
minimal answer against isolated aliases, enforces the student action budget,
and writes separate public and audit artifacts. The public trajectory contains
no golden-answer field or API receipt.

Candidate questions come from the NQ and WebQuestions training Arrow files.
The generator selects 300 accepted questions from each source. Before calling
DeepSeek, it excludes questions that overlap with the retrieval-trajectory
input, DPO questions, or the audited RL training selection.

**`--dpo-input` is a question-exclusion input.** The generator reads its
`question` fields to skip duplicate candidates. DPO `chosen` and `rejected`
responses are not sent to DeepSeek or used as SFT training targets. Similarly,
`--rl-train` supplies questions for overlap checks, not SFT training examples.

| Argument | Purpose |
|---|---|
| `--workspace` | Base directory for source data and generated artifacts; set `AETHERSEARCH_SFT_WORKSPACE` to the same directory |
| `--nq-arrow`, `--webq-arrow` | Source questions and reference answers; default to the respective `data/raw/flashrag/.../train/data-00000-of-00001.arrow` files under the workspace |
| `--retrieval-input` | Validated retrieval trajectories used for question exclusion and the unshuffled full-dataset output |
| `--dpo-input` | Canonical DPO file used only to exclude overlapping questions |
| `--rl-train` | Canonical RL source file used to reconstruct the audited training selection and exclude overlapping questions |
| `--concurrency 16` | Process up to 16 candidate questions concurrently |
| `--max-attempts 2200` | Try at most 2,200 candidates to reach the 600 accepted direct-answer target |
| `--seed 42` | Fix deterministic candidate ordering |

The exclusion inputs are checksum-pinned. Supply the canonical files expected
by the script; an arbitrary file with the same name will not pass preflight.

```bash
AETHERSEARCH_SFT_WORKSPACE=/absolute/path/to/runtime-assets \
python sft/data_generation/search_sft_teacher/generate_direct_answer_sft.py \
  --workspace /absolute/path/to/runtime-assets \
  --retrieval-input /absolute/path/to/retrieval_trajectories.jsonl \
  --dpo-input /absolute/path/to/aethersearch_dpo_2126.jsonl \
  --rl-train /absolute/path/to/nq_hotpotqa_train.parquet \
  --concurrency 16 \
  --max-attempts 2200 \
  --seed 42
```

Validate this branch with
[`validate_direct_answer_sft.py`](data_generation/search_sft_teacher/validate_direct_answer_sft.py).

### Final release

[`build_sft_2600_release.py`](data_generation/search_sft_teacher/build_sft_2600_release.py)
checks fixed input identities, validates all public rows and direct-answer audit
pairings, verifies question uniqueness, performs the deterministic global
shuffle, reassigns contiguous IDs, and writes:

- `final_sft_2600.jsonl`;
- `provenance_manifest.jsonl`;
- `dataset_manifest.json`.

The detailed construction guide documents the API protocol, continuation
history, token budgets, retrieval service, checkpoint behavior, review gates,
and release validation:
[`data_generation/search_sft_teacher/README.md`](data_generation/search_sft_teacher/README.md).

## Audit and Provenance

[`dataset_manifest.json`](dataset_manifest.json) records the release schema,
composition, shuffle policy, fixed input identities, and integrity checks.
`provenance_manifest.jsonl` is hosted with the dataset and maps every public ID
to source identifiers and content hashes. [`ATTRIBUTION.md`](ATTRIBUTION.md)
records source and redistribution status.

The trainer validates schema, IDs, normalized-question uniqueness, trajectory
structure, search/information adjacency, tokenization round trips, EOT
supervision, loss-mask accounting, length limits, row count, and the canonical
data SHA before training.

## Limitations

The dataset defines the full-trajectory training contract; it does not by itself
establish downstream model quality. Any model release must record the exact
dataset checksum and training configuration used for its weights. Redistribution
rights remain unresolved as documented in `ATTRIBUTION.md`.
