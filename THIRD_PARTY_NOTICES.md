# Third-Party Notices

Code-audit date: 2026-07-25 UTC
Provenance update: 2026-08-24 UTC

This notice distinguishes two evidence levels:

- **Code-audited provenance:** IGPO and A²TGPO, with pinned revisions and
  reviewed source hashes recorded below.
- **Paper-level method provenance:** RAGEN-2 and MICA, with verified paper
  metadata but no claim of source-code parity.

Paper-level entries do not claim a vendored implementation, audited code
revision, source-hash parity, or upstream license for the referenced method.

## IGPO

- Repository: https://github.com/GuoqingWang1/IGPO
- Pinned commit: `64165e2741ed8801f977948c8128080ce87b4101`
- License: Apache License 2.0
- License SHA-256:
  `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`

Reviewed official files:

| Official source | SHA-256 |
|---|---|
| `verl/utils/reward_score/info_gain.py` | `af0c5e180925982c83a56cde425264cceeda2acbe7c43b31cd6fc8bab96d4b67` |
| `scrl/llm_agent/vectorized_gt_logprob.py` | `a00da4b594238baa9b2fef911fb5d0a418c5c258d5097559fff3fc6389689f9d` |
| `scrl/llm_agent/prealigned_vectorized.py` | `636edca70e84408a988bfb2ff6c7ea0747d3f2bdbc5591d0ad34e3813c963cc9` |
| `scrl/llm_agent/generation.py` | `7019243992a3b70fe4d74d3fff6808cb3d7e25fa0b9ec461a4effa41681333b0` |

Project mapping:

| Official concept | Independent project implementation |
|---|---|
| `check_tags_balance`, `preprocess_text`, `deal_multi_labels`, `compute_f1` | `src/agentic_rl/outcome/token_f1.py` |
| appended ground-truth copies | `src/agentic_rl/exact_ig/task_builder.py` |
| structural attention | `src/agentic_rl/exact_ig/masks.py` |
| logical positions | `src/agentic_rl/exact_ig/position_ids.py` |
| shifted target scoring | `src/agentic_rl/exact_ig/vectorized_scorer.py` |
| alias maximum | `src/agentic_rl/exact_ig/alias_reduce.py` |

Release note: this upload package includes a minimal IGPO official source
snapshot under
`third_party/igpo_official_64165e2741ed8801f977948c8128080ce87b4101/` for audit
and parity reference. The training runtime does not import that snapshot. The
project implementation is independent; its active scoring path uses the
documented no-anchor causal alignment rather than importing the vendored
snapshot at runtime.

The outcome compatibility implementation is mechanically audited against the
pinned `info_gain.py`: lowercase handling, ASCII punctuation-to-space
normalization, whitespace collapse, set-token F1, `<|answer_split|>` alias
serialization, special Factbench/politifact/liar2 handling, tag-balance checks,
and first-answer extraction are retained. The stricter project protocol parser
is an outer gate and does not replace or weaken the official-compatible
function.

Per the task boundary, no model was loaded and no Fast Path versus Oracle
comparison, score error measurement, sign-agreement test, selected-set parity
test, or throughput benchmark was run. Those remain deployment gates rather
than claims made by this notice.

## RAGEN-2

- Provenance level: paper-level method provenance
- Paper: **RAGEN-2: Reasoning Collapse in Agentic RL**
- arXiv: `2604.06268`
- URL: https://arxiv.org/abs/2604.06268

This work informed the use of within-prompt terminal-outcome sample variance
and cumulative raw variance-mass prompt filtering. AetherSearch maintains its
public implementation independently.

Paper-level provenance only; no RAGEN-2 source-code snapshot is vendored or
claimed as code-parity evidence by this notice.

## MICA

- Provenance level: paper-level method provenance
- Paper: **MICA: Multi-granularity Intertemporal Credit Assignment for
  Long-Horizon Emotional Support Dialogue**
- arXiv: `2603.06194`
- URL: https://arxiv.org/abs/2603.06194

This work informed the combination of an immediate retrieval signal and a
delayed future-return signal after separate peer normalization. AetherSearch
adapts that general credit-assignment pattern to its own Search credit.

AetherSearch is not a direct reproduction of the paper's emotional-support
task. The project independently defines its retrieval-information-gain signal,
same-prompt and same-Search-depth peer groups, outcome fallback, missing-score
fail-closed rule, terminal Answer credit, and policy objective.

Paper-level provenance only; this notice does not claim an audited official
code revision, source-hash parity, upstream license, or code equivalence.

## A²TGPO

- Repository: https://github.com/CuSO4-Chen/A-TGPO
- Pinned commit: `f3121f772b267e6d4980e2455e1956316c0ff997`
- Paper: arXiv `2605.06200`
- Reviewed implementation:
  `ATGPO/verl_atgpo/verl/trainer/ppo/core_algos.py`
- Reviewed launch configuration:
  `ATGPO/scripts/ATGPO_multihop_qwen3_4B.sh`

Reviewed source hashes:

| Official source | SHA-256 |
|---|---|
| `ATGPO/verl_atgpo/verl/trainer/ppo/core_algos.py` | `74ee43eb8ec8f6305c38f5d10cb30bf74384582f1b5c8f083b38de912f1af30c` |
| `ATGPO/scripts/ATGPO_multihop_qwen3_4B.sh` | `12edd2394ff482354e30d7ae7c496691f06a09091b98e677dbec81ea0380b8b7` |
| `ATGPO/verl_atgpo/LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |

Official code evidence at this commit:

- `core_algos.py:1340-1376` computes
  `1 + 0.3 * (2 * sigmoid(normalized_ig) - 1)`;
- the same block initializes scales to one and leaves the Outcome turn at
  neutral scale one;
- `ATGPO_multihop_qwen3_4B.sh:54-56` configures low/high base widths
  `3e-3`/`4e-3` and enables dynamic clipping.

The project independently implements this clipping mechanism in
`src/agentic_rl/policy/strict_onpolicy_loss.py`. Project-specific advantage,
selection, reduction, KL, and transaction semantics remain project-defined
and are not claimed as code-equivalent to the audited upstream implementation.

## Existing Search-R1 and Retriever

Search-R1 remains an external source dependency configured through
`AETHERSEARCH_SEARCH_R1_ROOT`. The Hybrid Retriever server used by the public
runtime is vendored at
`runtime_assets/retriever/hybrid_retrieval_server.py`. Public asset and source
provenance is recorded in `EXTERNAL_ASSETS.md` and the corresponding runtime
documentation.
