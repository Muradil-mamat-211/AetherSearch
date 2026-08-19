<div align="center">

<img src="assets/aethersearch-mark.svg" alt="AetherSearch monogram" width="130">

# AetherSearch

**✨ A 3B multi-turn search agent trained with full-trajectory SFT, DPO, and information-gain-guided Agentic RL.**

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

🔗 Models, datasets, and reproducibility inputs are linked directly below.

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
- [Evaluation Results](#evaluation-results)
- [Agentic RL Method](#agentic-rl-method)
- [Quick Start](#quick-start)
- [Configuration Boundary](#configuration-boundary)
- [Repository Layout](#repository-layout)
- [Release Boundary](#release-boundary)

## Overview

AetherSearch is a search-augmented post-training project whose public code
currently covers SFT data construction metadata and the downstream RL stage.
The released RL recipe starts from an externally hosted DPO warm-start model;
the DPO trainer and DPO data-generation pipeline are not included in this
repository. Large data and model artifacts are hosted on Hugging Face.

## Training Pipeline

🧭 The complete training lineage is SFT → DPO → Agentic RL.

| Stage | Purpose | Primary locations |
|---|---|---|
| SFT | cold-start full-trajectory supervision | `sft_data/`, `sft_data/scripts/` |
| DPO warm start | externally produced actor/reference initialization | `AETHERSEARCH_ACTOR_MODEL`, `AETHERSEARCH_REFERENCE_MODEL` |
| RL | search-augmented rollout and policy optimization | `src/agentic_rl/`, `scripts/`, `recipes/rl/` |

## Evaluation Results

📊 Exact-match results across the released Search-R1 evaluation suites.

| Model | NQ | TriviaQA | PopQA | HotpotQA | 2WikiMultiHopQA | Musique | Bamboogle | Overall / Avg. EM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Search-R1 (Qwen2.5-3B Base, PPO) | **0.406** | 0.587 | **0.435** | 0.284 | 0.273 | 0.049 | 0.088 | 0.303 |
| Search-R1 (Qwen2.5-3B Instruct, PPO) | 0.341 | 0.545 | 0.378 | 0.324 | 0.319 | 0.103 | **0.264** | 0.325 |
| AetherSearch | 0.3977 | **0.5877** | 0.4229 | **0.3333** | **0.3985** | **0.1200** | 0.2320 | **0.3560** |

## Agentic RL Method

> 🧠 **Method scope.** This section documents the public AetherSearch
> Agentic-RL recipe and the code path selected by
> `answer_only_ragen2_mica_ig_v1_singleton_outcome`. The method is
> **MICA-inspired**, but it is a project-specific adaptation rather than a
> claim of verbatim reproduction of the original MICA algorithm.
>
> AetherSearch uses paper-style RAGEN-2 raw terminal-outcome sample variance
> followed by cumulative raw-variance-mass Top-p filtering. Exact IG is
> computed only after prompt selection and is used by MICA-IG for Search-turn
> credit assignment, not for selecting prompts.

### 1. Overview

For a prompt `p`, the rollout policy samples a fixed group of trajectories:

```math
\{\tau_{p,i}\}_{i=1}^{G}.
```

A trajectory may contain several model-generated Search turns followed by a
terminal Answer turn:

```math
\tau_{p,i}
=
(s_{p,i,1},o_{p,i,1},\ldots,s_{p,i,T_i},o_{p,i,T_i},a_{p,i}),
```

where `s` is a Search action, `o` is the retrieved environment
observation, and `a` is the terminal Answer action.

The complete release-recipe path is:

```math
\boxed{
\text{On-policy rollout}
\rightarrow
\text{terminal outcome scoring}
\rightarrow
\text{Answer-only prompt selection}
\rightarrow
\text{selected-only Exact IG}
\rightarrow
\text{MICA-IG Search credit}
\rightarrow
\text{Answer credit}
\rightarrow
\text{turn-level policy optimization}
\rightarrow
\text{reference KL}
}
```

Exact-IG is intentionally **selected-only**. Candidate prompt groups are
scored for terminal outcome first; non-selected groups do not incur the
expensive Exact-IG model forward.

### 2. Terminal task outcome

The terminal task outcome and the prompt-selection eligibility mask are
different concepts. Define the outcome-eligible set:

```math
\mathcal E_p^O
=
\left\{
i:
\texttt{terminal\_answer\_valid}_{p,i}
\land
\texttt{trajectory\_system\_valid}_{p,i}
\right\}.
```

Eligibility is exactly `terminal_answer_valid == true` and
`trajectory_system_valid == true`; only a trajectory in `\mathcal E_p^O` enters the outcome-reward
statistics. For an eligible trajectory, the production scorer computes the
maximum IGPO-compatible token-level F1 over all ground-truth aliases:

```math
O_{p,i}
=
\max_{a^\star\in\mathcal A_p}
\mathrm{TokenF1}
\left(
\hat a_{p,i},a^\star
\right).
```

If the terminal answer is not valid or the trajectory system is invalid:

```math
O_{p,i}=0.
```

That zero must not be confused with a valid answer whose F1 happens to be
zero. In particular:

- `raw outcome = 0` is a numeric reward value;
- `outcome_reward_eligible = false` means the trajectory is excluded
  from outcome peer normalization and outcome selection statistics.

The production scorer is not an exact-match-only scorer: the ordinary terminal
task reward is alias-aware token F1. Exact match is used by the separate
deterministic sufficiency-probe path, which is disabled in this MICA V1
configuration.

### 3. Answer-only RAGEN-2 prompt filtering

The final recipe uses:

```text
selection.mode   = answer_outcome_only_ragen2_paper_variance_top_p
selection.signal = answer_outcome_only
top_p_mass       = 0.9
```

For every candidate prompt, AetherSearch samples `G` on-policy trajectories.
Only outcome-eligible trajectories in `\mathcal E_p^O` enter the score. Let
`N_p = |\mathcal E_p^O|`:

```math
\bar O_p
=
\frac{1}{N_p}
\sum_{i\in\mathcal E_p^O}O_{p,i},
```

```math
\boxed{
V_p^O
=
\frac{1}{N_p-1}
\sum_{i\in\mathcal E_p^O}
\left(O_{p,i}-\bar O_p\right)^2
}.
```

This is the within-prompt **sample variance** (`ddof = 1`), not standard
deviation or population variance. The implementation returns zero when fewer
than two eligible outcomes are available. High variance identifies prompts on
which the current policy sometimes succeeds and sometimes fails, providing a
lightweight group-relative signal-to-noise proxy.

For intuition only:

```text
Prompt A outcomes: [1, 1, 1, 1]  -> variance = 0
Prompt B outcomes: [0, 0, 0, 0]  -> variance = 0
Prompt C outcomes: [1, 0, 1, 0]  -> variance > 0
```

Prompt C is prioritized because its within-group outcomes are distinguishable.
The production outcome remains alias-aware token F1 and is not restricted to
binary values.

Prompts are ranked by descending raw variance:

```math
V_{\sigma(1)}^O
\ge
V_{\sigma(2)}^O
\ge
\cdots
\ge
V_{\sigma(P)}^O.
```

The selector keeps the smallest prefix whose cumulative raw-variance mass
reaches `\rho` of the total mass:

```math
\boxed{
K^\star
=
\min
\left\{
K:
\sum_{j=1}^{K}V_{\sigma(j)}^O
\ge
\rho\sum_pV_p^O
\right\},
\qquad
\rho=0.9
}.
```

```math
\mathcal P_{\mathrm{selected}}
=
\left\{\sigma(1),\ldots,\sigma(K^\star)\right\}.
```

This means 90% of total **raw variance mass**, not the top 90% of prompts.
Minimum/maximum prompt counts, refill rounds, and capacity truncation remain
separate runtime policies.

The final selector uses raw terminal-outcome variance only. It does not use
Exact IG, MICA credit, standard deviation, noise-floor subtraction,
channel-scale or median normalization, EMA scaling, health gating, or
dual-channel weighting. Exact IG is computed only after selection for the
retained prompt groups.

### 4. Exact information gain

Exact IG measures how the retrieved observation changes the rollout-start
policy's likelihood of a fixed canonical answer. The canonical-answer policy
is:

```text
CANONICAL_ALIAS_POLICY = first
canonical_answer = aliases[0]
```

It is not alias-max. The fixed teacher-forced target, denoted by `y_p`, is:

```text
<think>The retrieved evidence now supports the answer.</think><answer>{canonical_answer}</answer>
```

The target is tokenized once as the complete rendered string with
`add_special_tokens=False` and `offset_mapping=True`. The
offset-based answer-covering span is the only scored span:

- scaffold text and `<think>`, `<answer>`,
  `</answer>` tags provide teacher-forced context;
- only answer-body tokens belong to `B_p`;
- one complete-target tokenization is used, rather than tokenizing the
  scaffold and answer independently;
- the causal logit for answer token `j` is the preceding position's
  logit.

For a prefix `h`, the answer-body mean log-probability is:

```math
\Phi_p(h)
=
\frac{1}{|B_p|}
\sum_{j\in B_p}
\log \pi_{\theta_{\mathrm{snap}}}
(y_{p,j}\mid h,y_{p,1:j-1}),
```

where `\theta_{\mathrm{snap}}` is the rollout-start policy snapshot and
`y_{p,1:j-1}` denotes the preceding target-token context.
For the pre- and post-observation prefixes:

```math
r^{IG}_{p,i,t}
=
\Phi_p(h^+_{p,i,t})
-
\Phi_p(h^-_{p,i,t}).
```

This is a log-probability difference:

```math
r^{IG}
\neq
\exp\left(\Phi_{\mathrm{post}}\right)
-
\exp\left(\Phi_{\mathrm{pre}}\right).
```

There is no exponentiation. The Exact-IG forward is detached/no-grad and
uses the rollout-start reward snapshot. The production path is the
independent FP32 `fp32_exact_ig` scorer with causal preceding-logit
alignment. An invalid or missing pre/post observation makes that Search
Exact-IG-ineligible. MICA V1 has no raw-IG fallback.

### 5. Raw suffix return

Exact IG is accumulated before any peer normalization. For trajectory
`(p,i)`, define the valid Search positions:

```math
\mathcal V_{p,i}
=
\left\{
t:
t\text{ has a valid Exact IG}
\right\}.
```

The raw suffix return is:

```math
G^{IG}_{p,i,t}
=
\sum_{\substack{k\ge t\\k\in\mathcal V_{p,i}}}
\gamma^{k-t}r^{IG}_{p,i,k}.
```

MICA-IG V1 locks:

```math
\boxed{\gamma=1}.
```

Therefore:

```math
\boxed{
G^{IG}_{p,i,t}
=
\sum_{\substack{k\ge t\\k\in\mathcal V_{p,i}}}
r^{IG}_{p,i,k}.
}
```

The order is important:

```text
raw Exact IG
→ raw suffix return
→ normalize local and return channels separately
→ mix the normalized channels
```

Missing or invalid Search positions are omitted from the valid suffix index
set; an artificial zero reward is not inserted. This branch also has no
legacy `1/sqrt(n)` return rescaling.

### 6. Same-prompt, same-depth normalization

MICA compares only peer Searches at the same prompt and the same Search
depth. For an immediate-IG peer vector, define:

```math
\mathcal I_{p,t}
=
\left\{
i:
r^{IG}_{p,i,t}
\text{ exists and is IG-eligible}
\right\},
\qquad
n_{p,t}=|\mathcal I_{p,t}|.
```

There is:

```text
NO cross-prompt normalization
NO cross-depth normalization
```

For any peer value vector `x` (immediate IG or raw suffix return), the
population statistics are:

```math
\mu_{p,t}(x)
=
\frac{1}{n_{p,t}}
\sum_{i\in\mathcal I_{p,t}}x_i,
```

```math
\sigma_{p,t}(x)
=
\sqrt{
\frac{1}{n_{p,t}}
\sum_{i\in\mathcal I_{p,t}}
\left(x_i-\mu_{p,t}(x)\right)^2
}.
```

The implementation uses population standard deviation, `ddof = 0`,
not sample standard deviation. With `epsilon = 1e-6`:

```math
Z_{p,t}(x_i)
=
\frac{x_i-\mu_{p,t}(x)}
{\sigma_{p,t}(x)+\epsilon}.
```

If `\sigma^2 \le 1e-12`, the corresponding channel advantage is
exactly zero.

### 7. MICA-IG local and delayed channels

The local channel compares the current immediate information gain with
same-prompt/same-depth peers:

```math
\boxed{
A^{loc}_{p,i,t}
=
Z_{p,t}
\left(
r^{IG}_{p,i,t}
\right).
}
```

The delayed channel independently compares the raw suffix returns with the
same peer scope:

```math
\boxed{
A^{ret}_{p,i,t}
=
Z_{p,t}
\left(
G^{IG}_{p,i,t}
\right).
}
```

The local and return channels have independent means and standard deviations:

```text
Local normalization statistics
and
Return normalization statistics
are independent.
```

They are mixed only after both channels have been normalized:

```math
A^{search}_{p,i,t}
=
\alpha A^{ret}_{p,i,t}
+
(1-\alpha)A^{loc}_{p,i,t},
\qquad
\boxed{\alpha=0.5}.
```

If one channel is zero because of zero variance, its fixed weight is still
zero contribution; the other channel is not reweighted from `0.5` to
`1.0`.

### 8. Singleton and missing-IG rules

The singleton fallback uses only the normalized terminal outcome. First
normalize outcomes within the same prompt over eligible trajectories:

```math
\mu_p^O
=
\frac{1}{N_p^O}
\sum_{i\in\mathcal E_p^O}O_{p,i},
```

```math
\sigma_p^O
=
\sqrt{
\frac{1}{N_p^O}
\sum_{i\in\mathcal E_p^O}
\left(O_{p,i}-\mu_p^O\right)^2
},
```

```math
Z^O_{p,i}
=
\frac{O_{p,i}-\mu_p^O}
{\sigma_p^O+\epsilon}.
```

This outcome normalization is also population normalization. If the outcome
is ineligible, or if `(\sigma_p^O)^2 \le 1e-12`, then
`Z^O_{p,i}=0`.

The rules are intentionally different:

- an IG-eligible Search with `n_{p,t}=1` cannot define a relative
  Search-vs-Search z-score, so it receives `Z^O_{p,i}`;
- a policy-credit-eligible Search with unavailable/invalid Exact IG receives
  `0`, not the singleton fallback;
- a policy-credit-ineligible Search is excluded from actor optimization.

The singleton fallback does not add the format advantage.

### 9. Final Search advantage

Putting the cases together, for an action that is eligible for policy credit:

```math
\boxed{
A^{search}_{p,i,t}
=
\begin{cases}
\frac12 A^{ret}_{p,i,t}
+
\frac12 A^{loc}_{p,i,t},
&
n_{p,t}\ge2,
\\[8pt]
Z^O_{p,i},
&
n_{p,t}=1,
\\[6pt]
0,
&
\text{Exact IG unavailable or invalid}.
\end{cases}
}
```

The third case is not a singleton peer group. It is a fail-closed missing-IG
case. No advantage is emitted for policy-ineligible actions, so they do not
enter the actor loss at all.

### 10. Answer advantage

The terminal format indicator is binary:

```math
F_{p,i}\in\{0,1\}.
```

Unlike MICA's local and return channels, the format channel is centered but
not divided by a standard deviation:

```math
A^{format}_{p,i}
=
F_{p,i}
-
\frac{1}{G}\sum_{j=1}^{G}F_{p,j}.
```

The terminal Answer advantage is:

```math
\boxed{
A^{answer}_{p,i}
=
\lambda_O Z^O_{p,i}
+
\lambda_F A^{format}_{p,i}.
}
```

The frozen configuration is:

```math
\lambda_O=1,
\qquad
\lambda_F=1,
\qquad
A^{answer}=Z^O+A^{format}.
```

MICA-IG changes Search credit only. It does not route `r_IG`,
`A_loc`, `A_ret`, or `A_search` into the terminal
Answer advantage.

### 11. Turn-to-token credit assignment

At the credit-assignment layer, a Search turn is one action. Every
policy-credit-eligible model-generated token in that Search turn receives the
same `A_search` value. The model-generated terminal Answer turn
receives the same `A_answer` value:

```math
\text{Search-turn tokens}
\mapsto
A^{search}_{p,i,t},
\qquad
\text{terminal Answer tokens}
\mapsto
A^{answer}_{p,i}.
```

The retriever returns an environment observation such as:

```text
<information> ... </information>
```

Those environment tokens are context, not actor actions:

```math
\boxed{
m^{policy}_{j}=0
\quad
\text{for environment-observation tokens}.
}
```

System tokens, user/prompt tokens, padding, and non-model fallback/code
tokens likewise do not receive actor policy credit. Only real
policy-credit-eligible model spans enter the policy mask.

### 12. Strict on-policy turn ratio

For one eligible turn `t`, let `\mathcal A_t` be its eligible
model-generated action-token set. The current-policy log-probabilities retain
gradients; old-policy log-probabilities are detached:

```math
\Delta_t
=
\frac{1}{|\mathcal A_t|}
\sum_{j\in\mathcal A_t}
\left[
\log\pi_\theta(x_j\mid x_{1:j-1})
-
\log\pi_{\mathrm{old}}(x_j\mid x_{1:j-1})
\right].
```

The turn ratio is:

```math
\boxed{
\rho_t=\exp(\Delta_t).
}
```

This is the geometric mean of the token likelihood ratios inside the turn.
It is a turn-level ratio, not an independently clipped PPO ratio for every
token. The release policy contract is:

```text
strict_on_policy = true
ratio_level = turn
ppo_epochs = 1
optimizer_mini_steps = 1
optimizer_steps_per_successful_update = 1
```

### 13. A²TGPO-style adaptive turn-level clipping

MICA-IG and adaptive clipping have different responsibilities. MICA defines
the Search credit; the policy layer applies the A²TGPO-style turn-level
surrogate.

For a Search turn, the clipping scale consumes normalized immediate IG,
written as `\widehat r^{IG}_t`. In the MICA path this is the supported
same-prompt/same-depth local normalization (`A_loc`); singleton and
missing-IG turns use neutral zero for clipping. The scale is:

```math
c_t
=
1+
\beta_c
\left(
2\sigma(\widehat r^{IG}_t)-1
\right),
\qquad
\sigma(x)=\frac{1}{1+e^{-x}}.
```

The frozen values and bounds are:

```math
\beta_c=0.3,
\qquad
\epsilon_{\mathrm{low}}=0.003,
\qquad
\epsilon_{\mathrm{high}}=0.004,
```

```math
l_t
=
1-c_t\epsilon_{\mathrm{low}},
\qquad
u_t
=
1+c_t\epsilon_{\mathrm{high}}.
```

For turn advantage `A_t`, the clipped surrogate is:

```math
J_t
=
\min
\left[
\rho_tA_t,
\mathrm{clip}
\left(\rho_t,l_t,u_t\right)A_t
\right].
```

Answer turns use the neutral scale `c_t=1`. The adaptive clip input is
normalized immediate IG, not the mixed `A_search`, and adaptive
clipping is not claimed as a contribution of MICA itself.

### 14. Frozen-reference full-vocabulary KL

The reference model is frozen, in evaluation mode, and has no trainable
parameters. KL is evaluated at the preceding causal states of eligible
policy tokens:

```math
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid s_j)
\;\middle\|\;
\pi_{\mathrm{ref}}(\cdot\mid s_j)
\right)
=
\sum_v
\pi_\theta(v\mid s_j)
\log
\frac{\pi_\theta(v\mid s_j)}
{\pi_{\mathrm{ref}}(v\mid s_j)}.
```

This is a **full-vocabulary** KL. It is not a sampled-token log-ratio proxy.
The reference logits are detached and the vocabulary sum is evaluated in
chunks only to control memory; chunking does not change the mathematical
sum.

The combined loss is:

```math
\boxed{
\mathcal L
=
-J_{\mathrm{task}}
+
\beta_{\mathrm{KL}}\mathcal L_{\mathrm{KL}},
\qquad
\beta_{\mathrm{KL}}=0.01.
}
```

### 15. Nested reduction

The task reduction is nested rather than a single global mean over all
action tokens. First average eligible action-token values within each
trajectory:

```math
J_{p,i}
=
\frac{1}{|\mathcal A_{p,i}|}
\sum_{j\in\mathcal A_{p,i}}J_{p,i,j}.
```

Then average trajectories within each prompt:

```math
J_p
=
\frac{1}{G}
\sum_{i=1}^{G}J_{p,i}.
```

Finally average prompt means globally:

```math
J_{\mathrm{task}}
=
\frac{1}{|\mathcal P|}
\sum_{p\in\mathcal P}J_p.
```

The same `token → trajectory → prompt` balancing applies to the KL
reduction. Thus a long trajectory does not automatically receive more weight
merely because it contains more eligible tokens:

```text
prompt
→ trajectory
→ eligible action-token balanced reduction
```

### 16. Compact algorithm

```text
Input:
    candidate prompts p
    G on-policy rollouts per prompt
    terminal outcomes O[p,i]

1. Score terminal outcomes.

2. Paper RAGEN-2 prompt filtering.
   Compute raw within-prompt terminal-outcome sample variance.
   Rank prompts by raw variance and retain the shortest prefix
   carrying at least 90% of total raw variance mass.

3. Selected-only Exact IG.
   For every selected valid Search:
       Phi_pre  = answer-body mean log-prob before observation
       Phi_post = answer-body mean log-prob after observation
       r_IG = Phi_post - Phi_pre

4. Raw suffix return.
       G_IG[t] = sum_{k >= t, valid} r_IG[k]
       gamma = 1

5. Same-prompt / same-depth peers.

   if peer_count >= 2:
       A_loc = zscore(raw IG)
       A_ret = zscore(raw suffix return)
       A_search = 0.5*A_ret + 0.5*A_loc

   elif peer_count == 1:
       A_search = Z_O

   elif Exact IG unavailable:
       A_search = 0

6. Answer:
       A_answer = Z_O + centered_format_advantage

7. Expand turn credit to eligible policy tokens.
   Environment/prompt tokens remain masked.

8. Compute the geometric-mean turn ratio.

9. Apply adaptive turn-level clipped surrogate.

10. Add frozen-reference full-vocabulary KL.

11. Reduce:
       action tokens → trajectory → prompt → global mean

12. Perform one strict on-policy optimizer step.
```

### 17. What MICA-IG changes and does not change

MICA-IG changes, in this project-specific V1 path:

- Search credit assignment;
- the local immediate-IG channel;
- the delayed raw-suffix-return channel;
- same-prompt/same-depth normalization;
- the singleton normalized-terminal-outcome fallback.

MICA-IG does not change:

- the terminal Answer scorer;
- the Answer advantage definition;
- the rollout engine or retriever;
- the old-policy turn-ratio definition;
- adaptive turn clipping;
- frozen-reference KL;
- nested task reduction;
- the optimizer;
- FSDP, vLLM, Ray, and hardware-topology infrastructure.

The current MICA V1 path also does not use `A_decision`,
`A_query`, routed outcome, sufficiency/novelty actor penalties, an
Answer-probe auxiliary actor loss, a role-localized gate loss, or raw-IG
fallback. The repository contains other experimental/qualification modes, but
those are not part of this release-recipe path.

### 18. Method lineage and project-specific adaptations

The naming boundary is:

```text
MICA-IG is a project-specific, MICA-inspired
information-gain credit-assignment method.
```

The design draws on several directions:

- IGPO: answer-likelihood information gain;
- RAGEN-2: high-signal prompt selection from reward dispersion;
- MICA: a mixture of immediate and delayed process credit;
- A²TGPO: turn-level normalization and adaptive clipped policy-optimization
  concepts.

The final AetherSearch formulas and boundaries are project-specific. In
particular, `gamma=1`, `alpha=0.5`, the Exact-IG scaffold,
selected-only Exact-IG, same-prompt/same-depth peers, and singleton outcome
fallback should not be presented as verbatim requirements of any one
original paper.

### 19. Interpretation boundary / known limitation

For `n_{p,t}\ge2`, the MICA peer set contains trajectories that:

- share the same prompt;
- reach the same Search depth;
- actually execute the Search;
- have a valid Exact-IG reward.

Therefore `A_search` primarily answers:

```text
Was this Search better than peer Searches at the same prompt and depth?
```

It does not directly estimate the same-state counterfactual:

```math
Q(s_t,\mathrm{Search})
-
Q(s_t,\mathrm{AnswerNow}).
```

Consequently, a positive `A_search` means that this Search
outperformed its eligible peer Searches. It does not strictly mean that
searching was better than stopping immediately. MICA-IG should not be
described as solving optimal stopping; this is an interpretation boundary,
not a runtime bug.

### 20. Implementation map

The source-of-truth mapping for the formulas above is:

| Contract | Implementation |
|---|---|
| terminal outcome eligibility and scoring | `src/agentic_rl/outcome/workers.py`, `src/agentic_rl/outcome/token_f1.py` |
| paper RAGEN-2 sample variance and raw-variance-mass Top-p | `src/agentic_rl/selection/paper_ragen2.py`, `src/agentic_rl/selection/candidate_pool.py`, `src/agentic_rl/selection/prompt_variance.py`, `src/agentic_rl/selection/top_p.py` |
| final release selection mode | `recipes/rl/train_4x48gb.yaml`, `configs/formal_train_answer_only_ragen2_paper_mica_ig_v1.yaml` |
| Exact-IG target, alias policy, offsets, score span | `src/agentic_rl/exact_ig/target_schema.py` |
| Exact-IG scorer, causal alignment, detached FP32 forward | `src/agentic_rl/exact_ig/vectorized_scorer.py`, `src/agentic_rl/exact_ig/precision_policy.py`, `src/agentic_rl/runtime/fsdp_worker.py` |
| raw suffix, local/return normalization, singleton and missing-IG rules | `src/agentic_rl/advantage/mica_ig.py`, `src/agentic_rl/advantage/a2tgpo.py` |
| format advantage and policy-token provenance | `src/agentic_rl/outcome/format_indicator.py`, `src/agentic_rl/rollout/trajectory_schema.py`, `src/agentic_rl/rollout/token_provenance.py` |
| turn ratio and adaptive clipping | `src/agentic_rl/policy/turn_ratio.py`, `src/agentic_rl/policy/strict_onpolicy_loss.py` |
| nested reduction and full-vocabulary reference KL | `src/agentic_rl/policy/reduction.py`, `src/agentic_rl/policy/reference_kl.py` |

The repository retains earlier scaled-selection machinery for historical and
experimental coverage, but it is not part of the final AetherSearch recipe.
The public recipe, not this documentation, is the executable selection
boundary. No GPU runtime test is implied by this README section.

## Quick Start

🚀 The public entrypoint resolves one algorithm recipe, one asset manifest,
and one explicit hardware/qualification profile.

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

The default recipe is an **Official Qualified** reproduction. It enables
`configs/qualification/official_4x48gb_v1.yaml`, which checks the published
4x48GB/48-CPU/360-GiB host contract and release asset checksums. Portable
user-defined profiles use resource minimums and generic topology invariants
instead of claiming official qualification.

Start RL training on the validated four-GPU topology:

```bash
bash scripts/train_rl.sh
```

The included recipe assigns physical GPU 0 to retrieval and asynchronous
training-time evaluation, and physical GPUs 1-3 to the three-rank vLLM/FSDP2
runtime. Every 20-update evaluation uses the complete 51,713-row Search-R1
`test.parquet`. See `recipes/rl/README.md` for the configuration boundary.
The resolved configuration is materialized inside each new run directory.

## Configuration Boundary

⚙️ Configuration answers five separate questions:

| Layer | Responsibility |
|---|---|
| experiment | what algorithm and schedule to run |
| assets | which model, data, tokenizer, and retriever artifacts to use |
| hardware | which physical resources and role placement are available |
| runtime | how Ray, veRL, FSDP2, and vLLM map onto those resources |
| qualification | whether the configuration exactly matches the official reference |

`environment/env.local.sh` supplies machine-local paths and Python
interpreters. Asset checksums live in `configs/assets/`; hardware roles and
Ray resources live in YAML. `TopologyPlan` is the runtime topology source of
truth and derives visible CUDA IDs, learner world size, Ray bundles, and
rollout DP/TP compatibility fields.

For another server, provide a user-owned hardware/runtime YAML, set
`qualification.mode: portable`, and keep the algorithm configuration
unchanged. Generic non-reference layouts are covered by CPU-only synthetic
configuration tests. That coverage does not establish GPU-memory fit, runtime
compatibility, training stability, throughput, or production qualification.

## Repository Layout

- `sft_data/`: SFT data release metadata and build scripts. The full SFT JSONL
  payload is hosted on [Hugging Face Datasets](https://huggingface.co/datasets/muradil211/aethersearch_sft).
- `src/agentic_rl/`: RL training, rollout, advantage, policy loss, retriever,
  checkpoint, and runtime adapter code.
- `scripts/`: launch, preflight, resume, validation, and operational scripts for
  the RL training stage; see `scripts/README.md` for public versus historical
  entrypoints.
- `recipes/rl/`: the single validated public RL recipe and its usage boundary.
- `configs/`: algorithm, hardware/topology, asset-manifest, qualification,
  retriever, stage configs, and the Exact-IG runtime gate;
  see `configs/README.md` for their portability boundary.
- `runtime_assets/`: local runtime assets required by the training launcher.
- `tests/`: unit and integration checks for the training code.
- `environment/`: observed package versions and environment template.

## Release Boundary

📦 The public repository covers SFT assets and metadata plus the complete
Agentic-RL training layer. The DPO warm start is an externally supplied model
checkpoint; the DPO trainer and preference-data generation pipeline are not
part of this release.

Large model weights, optimizer-state checkpoints, eval result bundles, report
archives, and runtime snapshots are intentionally not committed to this GitHub
repository. Public model and data artifacts are linked from Hugging Face above.
