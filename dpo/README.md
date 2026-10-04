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
| Original candidate question pool | 5,000 |
| Preference pairs | 2,126 |
| Training unit | Shared prompt with chosen/rejected continuations |
| License metadata | `unknown` |
| GitHub contents | Data metadata, training code, configuration, and source checks |

<a id="dataset-overview"></a>

## 🧩 Dataset overview

The stage uses the complete 2,126-pair `train.jsonl` release, constructed from
SFT rollouts on the original **5,000-question candidate pool** and retained
after preference-pair construction and filtering. Every normalized question
is unique and every row contains one shared `prompt_text`, one preferred
continuation, and one non-preferred continuation. The exact data identity is:

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

The published dataset was constructed from **5,000 isolated candidate
questions**. SFT rollouts on this pool, preference-pair construction, and
filtering produced **exactly 2,126 retained preference pairs**. These are the
observed input and output counts of the released dataset.

The decision-level construction specification below follows the central
rule: **roll out to the final answer for verification, then train on the
verified next action at the shared prefix**. SFT-sampled and Codex-corrected
candidates must pass the same acceptance gates.

The [published release](#dataset-overview) contains exactly 2,126 pairs,
including 235 `true_full_trajectory_preference` pairs whose training units are
whole continuations. That release keeps its recorded counts and data checksum.
Historical rows require their original rollout and audit records before they
can be certified against this specification; a documentation update alone
does not establish that certification.

Human review establishes the failure taxonomy from the stratified 140-example
[SFT analysis](../sft/README.md#sft-evaluation-and-failure-analysis). Codex then
locates decision errors, applies the taxonomy, proposes minimal corrections,
and judges actions using actual evidence. The rollout controller executes
retrieval and checks protocol validity and answer EM. Humans review new
failure categories, unresolved judgments, and a sample of accepted pairs.
Evaluation examples define the taxonomy; separate training questions supply
the DPO pairs.

### 1. Isolate the question pool and freeze the construction settings

Sample 5,000 candidate questions, stratified by eligible source, from QA
datasets such as NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, and MuSiQue.
Strictly exclude overlap with **SFT training, evaluation, and RL training
questions**. Use the Eval-1400 question normalization: Unicode NFKC, casefold,
whitespace collapse, strip, and removal of trailing ASCII/full-width question
marks. Exclude normalized exact matches and confirmed high-confidence
near-duplicates before sampling.

The complete 125-question Bamboogle release is already included in
[frozen Eval-1400](https://huggingface.co/datasets/muradil211/AetherSearch_Eval_1400/tree/frozen-v1),
so it contributes **zero eligible DPO candidates** under strict isolation.
If the eligible pool is too small, stop without relaxing the exclusions.

Pin the SFT checkpoint, retriever and corpus revisions, retrieval settings,
decoding settings, rollout budgets, protocol parser, answer scorer, and
accepted gold aliases before generation. Record the Codex model, prompts,
taxonomy version, and teacher-retry limit as well. Gold answers and aliases
are verification inputs and must stay outside policy and correction prompts.

### 2. Generate four initial complete trajectories per question

Use the frozen [SFT policy](https://huggingface.co/muradil211/AetherSearch_SFT)
to sample **$`K=4`$ trajectories per question**: 20,000 initial rollout attempts
for a 5,000-question pool. Each attempt continues to a terminal `<answer>` or
the fixed execution limit. Every `<search>` calls the **real retriever**;
retain the actual actions, returned observations, document identifiers,
settings, and termination reason.

The intended sampling unit is a complete trajectory. A timeout, exhausted
budget, or malformed output is recorded as a failed attempt rather than
treated as a successful completion.

### 3. Locate the first actionable error and freeze its prefix

Codex reviews the format-valid initial trajectories. Within each trace,
identify the **earliest genuinely faulty decision** that can be corrected and
verified at the same state. A wrong final answer alone does not prove that
an earlier search was wrong; a correct final answer can still contain an
unnecessary or unhelpful search. Discard cases with no clear, verifiable
decision error.

For a selected error, freeze **$`x`$ immediately before the faulty action**:
the question, exact conversation prefix, earlier actions, and existing real
retrieval observations. Preserve the faulty **next action** as **$`a_l`$
(`rejected`)**, including its `<think>` block and complete `<search>` or
`<answer>` block. Keep this action unchanged and link it to the original SFT
trace. Neither future observations nor later actions belong in $`x`$.

### 4. Sample four complete continuations from that exact prefix

From the same frozen **$`x`$**, sample **four new continuations with the same
SFT policy**. Execute every subsequent search and continue each attempt to
its final answer or execution limit. These are a second set of four complete
rollout attempts, additional to the initial four in step 2.

For candidate $`i`$, retain both its first next action **$`a_i`$** and its full
continuation **$`\tau_i`$**. The action is the proposed training target; the
complete continuation supplies the terminal-verification evidence. A
next-action-only sample cannot establish that a Search leads to a correct
final answer. An immediate Answer is already a terminal continuation.

### 5. Apply the terminal gate first

A candidate can supply `chosen` only if its complete rollout:

- finishes with exactly one terminal `<answer>` under the pinned protocol;
- has valid action ordering, tags, final-answer schema, and real retrieval
  observations throughout;
- achieves **normalized, alias-aware EM = 1** on the extracted final answer.

Use
[`max_alias_exact_match`](../src/agentic_rl/outcome/token_f1.py)
with the approved aliases and the pinned production normalization: lowercase,
ASCII punctuation replaced by spaces, and whitespace normalization. Question
deduplication and answer scoring have distinct normalization contracts.
Validate the complete protocol before extracting the terminal answer; the
answer scorer alone is not a trajectory-format validator.

**F1 is recorded for analysis only. Neither F1 > 0.75 nor any other partial
overlap threshold automatically accepts a candidate.** Incomplete,
format-invalid, or EM-failing candidates supply no chosen action to this
main dataset. A suspected missing alias needs independent evidence and
review; version the approved alias change and rerun verification instead of
adding an alias merely to accept a candidate.

### 6. Verify the candidate's next action and relative improvement

Only terminal-passing candidates proceed to the local action gate. Evaluate
**$`a_i`$ at $`x`$ against the actual rejected action $`a_l`$**:

| Next action | Required acceptance conditions |
|---|---|
| **Search** | Codex checks the real retrieval results: searching is necessary at this state, the returned evidence fills or clarifies an information gap relevant to the question, and this action has a clear advantage over `rejected`. Useful intermediate-hop or bridge facts count; redundant results or merely different wording do not. |
| **Answer** | The answer and format pass the terminal checks. For a premature-answer failure, Codex must also verify that the current state provides sufficient grounds to stop, under the task's allowed evidence and prior-knowledge rules. |

For a Search judgment, provide the question, $`x`$, both competing actions,
the candidate's actual returned information, and the rejected search's actual
results when applicable. Record the specific supporting snippets or document
IDs, the identified information gap, and the reason for the improvement.
A necessary query that returns no useful evidence fails this gate. When
the state already supports an answer, another search fails the necessity
check.

The local judge assesses evidence available at this decision, including the
newly returned Search results. Keep future continuation outcomes, gold
answers, and candidate origin out of this judgment to reduce hindsight and
teacher bias. A correct final answer does not by itself validate the earlier
Search. For an immediate Answer, the terminal gate already checks its answer
EM; the additional local check concerns whether ending now is justified.
Direct answers may use reliable prior knowledge where the task permits it.

Require a **clear preference**, with no unresolved tie or uncertainty.
Uncertain judgments need review or exclusion. A Codex assertion without
supporting retrieval evidence or answer verification is insufficient.
Chosen and rejected may have different action types, such as Search replacing
a premature Answer or Answer replacing an unnecessary Search.

### 7. If no sampled candidate passes, propose a minimal teacher correction

If none of the four continuations passes **both** gates, Codex or a stronger
teacher proposes a minimal correction of the next action from the **same
$`x`$**. Preserve the prefix and original rejected action. Correct the decision
and any affected `<think>` text together; the correction may change the query,
the answer, or the action type.

| Observed failure | Minimal next-action correction |
|---|---|
| Wrong entity, relation, or multi-hop query | Repair the Search query using the question and information already present in $`x`$. |
| Answer issued before the needed information is available | Replace Answer with a Search targeting the missing fact. |
| Evidence already available but misread | Correct Answer and its associated reasoning from that evidence. |
| Search issued despite sufficient information | Replace the unnecessary Search with a supported Answer. |

The teacher uses only the question and state available at the decision.
Future observations and gold answers must not be used to design the repaired
action, and `<information>` blocks must never be invented.

For a corrected **Search**, execute the new query through the real retriever,
append its actual observation, and return control to the **same frozen SFT
policy** to generate the remaining continuation through the final answer.
Regenerate the dependent suffix; results and actions belonging to the old
query cannot be reused as though they followed the new query. A corrected
**Answer** ends the rollout immediately.

Then apply **the same terminal gate in step 5 and the same action gate in
step 6**. Teacher generation and acceptance judging are separate passes;
the proposal is never accepted merely because its author endorses it.
Search corrections require both a successful final-answer rollout and a
verified useful, necessary search. Answer corrections require correct valid
answers and the applicable stopping-evidence check. If the predeclared retry
limit is exhausted without a passing candidate, discard the case.

### 8. Export the verified next-action preference pair

Among passing candidates, prefer the clearest verified improvement requiring
the smallest correction. Set **$`a_w`$ (`chosen`)** to that candidate's next
action. The decision-level training unit is:

```math
\boxed{(x,\ a_w,\ a_l)}
```

| Component | Public field | Exported content |
|---|---|---|
| $`x`$ | `prompt_text` | Exact shared prefix immediately before the target action |
| $`a_w`$ | `chosen` | Verified next `<think>` + `<search>` or `<think>` + `<answer>` action |
| $`a_l`$ | `rejected` | Actual faulty next action from the original SFT trace |

Both sides use the same `prompt_text` and omit the repeated prefix. A local
Search target ends at `</search>`; its new retrieval observation and later
actions belong in the audit record, outside the local training target.
Retain the existing [eight-field public schema](#public-schema), including
the question, source, gold aliases, and matching failure `pair_type`.

Store full original and candidate trajectories, retrieval provenance,
EM/F1 results, gate judgments, correction origin, and construction versions
in a linked **audit sidecar**. This preserves the distinction between the
full rollout used to verify a pair and the next action used to train it.
Whole-continuation `true_full_trajectory_preference` pairs in the published
release keep their separate training scope; a local action pair must not be
labeled as a full-trajectory pair.

### 9. Deduplicate, audit, and release

Keep **one highest-quality pair per normalized question**. Across candidate
errors, prefer the earliest clear, directly verifiable decision failure with
a passing correction; discard ambiguous or lower-quality alternatives.
Check unique IDs and questions, SFT/Eval/RL isolation, exact shared prefixes,
valid non-empty and distinct actions, terminal EM = 1, real retrieval
provenance, and the documented local preference. Human review samples accepted
pairs across sources, failure categories, and candidate origins, and resolves
new categories before release.

Trainer preflight checks schema, tokens, masks, and data identity. It cannot
recover a local Search target's final answer or certify its usefulness from
the exported action alone; the construction audit records supply that
evidence. The actual construction result was **2,126 retained preference pairs
from the original 5,000 candidate questions**, as recorded in the
[published release](#dataset-overview). Acceptance depends on the fixed
verification and audit requirements. A regenerated dataset needs its own
version, actual counts, checksums, and audit records.

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
\boxed{\mathrm{EM},\quad \mathrm{F1},\quad \mathrm{FTFA},\quad \mathrm{AvgSearch}}
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

#### Eval-1400 results

The project maintainer reports the following end-to-end results on the
**same frozen 1,400 questions**:

| Metric | SFT | DPO | Change (DPO − SFT) |
|---|---:|---:|---:|
| EM (%) | 25.0 | **31.5** | **+6.5 pp** |
| F1 (%) | 33.0 | **38.7** | **+5.7 pp** |
| FTFA (%) | 97.5 | **99.2** | **+1.7 pp** |
| AvgSearch (searches/question) | 2.5 | **1.5** | **−1.0 searches/question** |

EM, F1, and FTFA are expressed as percentages; **pp** means percentage
points. AvgSearch is the average number of executed searches per question.

DPO improves both final-answer metrics and format compliance, while
AvgSearch falls from **2.5 to 1.5**, a **40% reduction** in executed searches
per question. Taken together, the reported results show better answer
quality with fewer searches, while preserving and improving the format
ability established by SFT.

This table reports **End-to-End Eval**. Held-out **Preference Eval** measures
chosen/rejected ranking separately, using the protocol below.

### 2. Preference Eval: held-out chosen/rejected ranking

Use a separate held-out preference set with the same failure taxonomy and
pair types as DPO training. Its questions and preference pairs must never
participate in DPO training; also exclude overlap with SFT training, RL
training, and Eval-1400 using the same question-normalization and duplicate
checks described in [data construction](#dpo-data-construction-workflow).
The canonical 2,126-pair release remains train-only; the held-out set is a
separate evaluation artifact.

For each fixed pair $`(x,y_w,y_l)`$, compute both chosen and rejected sequence
scores under each checkpoint, using teacher-forced scoring in evaluation mode
with gradients disabled. For model $`\pi`$, define:

```math
S_{\pi}(y\mid x) = \sum_{t:\,m_t=1} \log \pi\left(y_t \mid x,y_{\lt t}\right).
```

Here $`y_{\lt t}`$ denotes the continuation tokens before position $`t`$,
and $`m_t`$ selects the scored continuation tokens. Reuse the exact
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
\gt
S_{\pi}(y_{l,i}\mid x_i)
\right]
}
```

A tie does not count as a correct preference. Evaluate SFT and DPO on the
same held-out pairs and test whether:

```math
\boxed{\mathrm{PrefAcc}_{\mathrm{DPO}} \gt \mathrm{PrefAcc}_{\mathrm{SFT}}}
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
translate into better agent behavior. Use the Eval-1400 comparison above
together with overall and per-type PrefAcc to assess the checkpoint.
The PrefAcc inequality above remains an evaluation target until held-out
preference results are reported.

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
