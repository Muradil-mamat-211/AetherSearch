<div align="center">

# AetherSearch

**Search-augmented post-training with SFT assets and reinforcement learning.**

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-AetherSearch-yellow)](https://huggingface.co/muradil211/AetherSearch)
[![SFT Data](https://img.shields.io/badge/%F0%9F%A4%97%20Data-AetherSearch%20SFT-yellow)](https://huggingface.co/datasets/muradil211/aethersearch_sft)
[![Eval Data](https://img.shields.io/badge/%F0%9F%A4%97%20Eval-Search--R1%20Full-yellow)](https://huggingface.co/datasets/muradil211/AetherSearch-Eval)
[![Code](https://img.shields.io/badge/GitHub-Code-181717?logo=github)](https://github.com/Muradil-mamat-211/AetherSearch)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](pyproject.toml)

ðŸ¤— [AetherSearch Model](https://huggingface.co/muradil211/AetherSearch) |
ðŸ¤— [AetherSearch SFT Data](https://huggingface.co/datasets/muradil211/aethersearch_sft) |
ðŸ¤— [Full Eval Data](https://huggingface.co/datasets/muradil211/AetherSearch-Eval)

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
| ðŸ¤— Model | [muradil211/AetherSearch](https://huggingface.co/muradil211/AetherSearch) | model weights, tokenizer, config, and model card |
| ðŸ¤— SFT data | [muradil211/aethersearch_sft](https://huggingface.co/datasets/muradil211/aethersearch_sft) | full JSONL payload, provenance manifest, checksums, and dataset card |
| ðŸ¤— Search-R1 train data | [PeterJinGo/nq_hotpotqa_train](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train) | upstream `train.parquet`, pinned by checksum in `EXTERNAL_ASSETS.md` |
| ðŸ¤— Full eval data | [muradil211/AetherSearch-Eval](https://huggingface.co/datasets/muradil211/AetherSearch-Eval) | complete 51,713-row Search-R1 `test.parquet`, provenance, and checksums |
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

This is a **raw suffix sum**. The MICA branch does not apply the older $1/\sqrt n$ future-credit rescaling. Missing/invalid Search positions are omitted from the suffix return rather than ins²È="24Õ¹…Ù…¥±…‰±”½¥¹Ù…±¥‘ô¸)q•¹‘í…Í•Íô)ô(((´´´((ŒŒŒ€à¸¹ÍÝ•È…‘Ù…¹Ñ…”()Q¡”5%‰É…¹ ¡…¹•ÌM•…É É•‘¥Ð½¹±ä¸Q¡”•á¥ÍÑ¥¹œ¹ÍÝ•È…‘Ù…¹Ñ…”¥ÌÁÉ•Í•ÉÙ•è()q‰½á•‘ì)yí…¹ÍÝ•Éõ}íÀ±¥ô(ô)q±…µ‰‘…}<iy=}íÀ±¥ô(¬)q±…µ‰‘…}yí™½Éµ…Ñõ}íÀ±¥ô¸)ô(()%¸Ñ¡”…Ñ¥Ù”Á…Ñ °Ñ¡¥Ì¥ÌÑ¡”•á¥ÍÑ¥¹œ½ÕÑ½µ”µÁ±ÕÌµ™½Éµ…Ð¹ÍÝ•ÈÍ¥¹…°ì5%‘½•Ì¹½ÐÉ½ÕÑ”M•…É µÁÉ½•ÍÌÉ•Ý…É¥¹Ñ¼Ñ¡”™¥¹…°¹ÍÝ•È…‘Ù…¹Ñ…”¸((´´´((ŒŒŒ€ä¸QÕÉ¸µÑ¼µÑ½­•¸É•‘¥Ð…ÍÍ¥¹µ•¹Ð()M•…É ÑÕÉ¸¥ÌÑÉ•…Ñ•…Ì½¹”Á½±¥ä…Ñ¥½¸…ÐÑ¡”É•‘¥Ðµ…ÍÍ¥¹µ•¹Ð±•Ù•°¸Ù•Éäµ½‘•°µ•¹•É…Ñ•Ñ½­•¸‰•±½¹¥¹œÑ¼Ñ¡”Í…µ”M•…É ÑÕÉ¸É••¥Ù•ÌÑ¡”Í…µ”Í…±…È()yíÍ•…É¡õ}íÀ±¤±Ñô¸(()Q¡”™¥¹…°¹ÍÝ•ÈÑÕÉ¸É••¥Ù•Ì()yí…¹ÍÝ•Éõ}íÀ±¥ô¸(()I•ÑÉ¥•Ù••¹Ù¥É½¹µ•¹ÐÑ•áÐÍÕ …Ì)Ñ•áÐ(ñ¥¹™½Éµ…Ñ¥½¸ø€¸¸¸€ð½¥¹™½Éµ…Ñ¥½¸ø)€)¥Ì½¹Ñ•áÐ½¹±äè()q‰½á•‘íqÑ•áÑí•¹Ù¥É½¹µ•¹ÐÑ½­•¸Á½±¥äµ…Í­ôôÁô¸(()AÉ½µÁÐ½ÍåÍÑ•´½ÕÍ•ÈÑ½­•¹Ì…É”…±Í¼•á±Õ‘•™É½´…Ñ½ÈÉ•‘¥Ð¸()5%µ%¥¹ÑÉ½‘Õ•Ì¹¼Í•Á…É…Ñ”€‘}íqµ…Ñ¡Éµí‘•¥Í¥½¹õô°€‘}íqµ…Ñ¡ÉµíÅÕ•Éåõô°É½ÕÑ•½ÕÑ½µ”°ÍÕ™™¥¥•¹ä½¹½Ù•±ÑäÁ•¹…±Ñä°¹ÍÝ•ÈµÁÉ½‰”…Õá¥±¥…Éä±½ÍÌ°½ÈÉ½±”µ±½…°…Õá¥±¥…Éä½‰©•Ñ¥Ù”¸((´´´((ŒŒŒ€ÄÀ¸A½±¥ä½ÁÑ¥µ¥é…Ñ¥½¸()5%µ%¥Ì„€¨©É•‘¥Ðµ…ÍÍ¥¹µ•¹Ðµ½‘Õ±”¨¨°¹½Ð„É•Á±…•µ•¹Ð™½ÈÑ¡”Á½±¥ä½ÁÑ¥µ¥é•È¸()™Ñ•ÈM•…É …¹¹ÍÝ•È…‘Ù…¹Ñ…•Ì…É”½¹ÍÑÉÕÑ•°Ñ¡”ÁÉ½©•ÐÉ•ÕÍ•ÌÑ¡”•á¥ÍÑ¥¹œÍÑÉ¥Ð½¸µÁ½±¥ä±•…É¹•Èè(Ä¸É½±±½ÕÐµÍÑ…ÉÐ½±µÁ½±¥ä±½œµÁÉ½‰…‰¥±¥Ñ¥•Ì…É”‘•Ñ…¡•ì(È¸µ½‘•°µ•¹•É…Ñ•…Ñ¥½¸Ñ½­•¹Ì™½É´ÑÕÉ¸µ±•Ù•°±¥­•±¥¡½½É…Ñ¥½Ìì(Ì¸Ñ¡”•á¥ÍÑ¥¹œ±¥ÁÁ•ÍÕÉÉ½…Ñ”½‰©•Ñ¥Ù”¥Ì…ÁÁ±¥•ì(Ð¸Ñ…Í¬É•‘ÕÑ¥½¸É•µ…¥¹ÌÁÉ½µÁÐ½ÑÉ…©•Ñ½Éä½…Ñ¥½¸µÑ½­•¸‰…±…¹•ì(Ô¸„™É½é•¸I•™•É•¹”µ½‘•°ÁÉ½Ù¥‘•Ì™Õ±°µÙ½…‰Õ±…Éä-0É•Õ±…É¥é…Ñ¥½¸¸()M¡•µ…Ñ¥…±±ä°()qµ…Ñ¡…°0(ô(µqµ…Ñ¡…°)}íqµ…Ñ¡ÉµíÑ…Í­õô(¬)q‰•Ñ…}íqµ…Ñ¡Éµí-1õõqµ…Ñ¡…°1}íqµ…Ñ¡Éµí-1õô¸(()Q¡”5%‰É…¹ …‘‘Ì¹¼¥¹‘•Á•¹‘•¹Ð…Õá¥±¥…Éä…Ñ½È±½ÍÌ¸((´´´((ŒŒŒ€ÄÄ¸½µÁ…Ð…±½É¥Ñ¡´()Ñ•áÐ)%¹ÁÕÐè(€€€…¹‘¥‘…Ñ”ÁÉ½µÁÑÌÀ(€€€½¸µÁ½±¥äÉ½±±½ÕÑÌÁ•ÈÁÉ½µÁÐ(€€€Ñ•Éµ¥¹…°½ÕÑ½µ•Ì=mÀ±¥t((Ä¸¹ÍÝ•Èµ½¹±äI8´È(€€€Y}=mÁt€ôÍ…µÁ±•}Ù…É¥…¹•}¤¡=mÀ±¥t¤(€€€Í•±•Ð¡¥ µÙ…É¥…¹”ÁÉ½µÁÐÉ½ÕÁÌ‰äÑ½ÀµÀÙ…É¥…¹”µ…ÍÌ((È¸M•±•Ñ•µ½¹±äá…Ð%(€€€™½È•… Í•±•Ñ•ÑÉ…©•Ñ½Éä¤…¹Ù…±¥M•…É ‘•ÁÑ Ðè(€€€€€€€A¡¥}ÁÉ”€€ô…¹ÍÝ•Èµ‰½‘äµ•…¸±½œµÁÉ½ˆ‰•™½É”½‰Í•ÉÙ…Ñ¥½¸(€€€€€€€A¡¥}Á½ÍÐ€ô…¹ÍÝ•Èµ‰½‘äµ•…¸±½œµÁÉ½ˆ…™Ñ•È½‰Í•ÉÙ…Ñ¥½¸(€€€€€€€É}%mÀ±¤±Ñt€ôA¡¥}Á½ÍÐ€´A¡¥}ÁÉ”((Ì¸I…ÜÍÕ™™¥àÉ•ÑÕÉ¸(€€€}%mÀ±¤±Ñt€ôÍÕµ}í¬€øôÐ°Ù…±¥‘ôÉ}%mÀ±¤±­t(€€€…µµ„€ô€Ä((Ð¸M…µ”µÁÉ½µÁÐ½Í…µ”µ‘•ÁÑ 5%¹½Éµ…±¥é…Ñ¥½¸(€€€Á••ÉÌ€ôÑÉ…©•Ñ½É¥•ÌÝ¥Ñ Ù…±¥á…Ð%…Ð€¡À±Ð¤((€€€¥˜±•¸¡Á••ÉÌ¤€øô€Èè(€€€€€€€}±½Œ€ôéÍ½É•}Á••È¡É}%¤(€€€€€€€}É•Ð€ôéÍ½É•}Á••È¡}%¤(€€€€€€€}Í•…É €ô€À¸Ô€¨}±½Œ€¬€À¸Ô€¨}É•Ð((€€€•±¥˜±•¸¡Á••ÉÌ¤€ôô€Äè(€€€€€€€}Í•…É €ô¹½Éµ…±¥é•‘}Ñ•Éµ¥¹…±}½ÕÑ½µ”i}<((€€€•±Í”€¼á…Ð%Õ¹…Ù…¥±…‰±”è(€€€€€€€}Í•…É €ô€À((Ô¸¹ÍÝ•È(€€€}…¹ÍÝ•È€ô•á¥ÍÑ¥¹œ¹½Éµ…±¥é•½ÕÑ½µ”€¬•á¥ÍÑ¥¹œ™½Éµ…Ð…‘Ù…¹Ñ…”((Ø¸É•‘¥Ð(€€€…ÍÍ¥¸}Í•…É Ñ¼•Ù•Éäµ½‘•°Ñ½­•¸¥¸Ñ¡…ÐM•…É ÑÕÉ¸(€€€…ÍÍ¥¸}…¹ÍÝ•ÈÑ¼•Ù•Éäµ½‘•°Ñ½­•¸¥¸Ñ¡”¹ÍÝ•ÈÑÕÉ¸(€€€µ…Í¬ÁÉ½µÁÐ…¹É•ÑÉ¥•Ù•µ¥¹™½Éµ…Ñ¥½¸Ñ½­•¹Ì((Ü¸=ÁÑ¥µ¥é”(€€€É•ÕÍ”•á¥ÍÑ¥¹œÑÕÉ¸µ±•Ù•°±¥ÁÁ•½¸µÁ½±¥ä½‰©•Ñ¥Ù”(€€€€¬™É½é•¸µÉ•™•É•¹”™Õ±°µÙ½…‰Õ±…Éä-0)€((´´´((ŒŒŒ€ÄÈ¸]¡…Ð¥Ì5%µ±¥­”°…¹Ý¡…Ð¥ÌÁÉ½©•ÐµÍÁ•¥™¥Œü()Q¡”‘•Í¥¸™½±±½ÝÌÑ¡”5%µÍÑå±”ÁÉ¥¹¥Á±”½˜µ¥á¥¹œ()q‰½á•‘íqÑ•áÑí¥µµ•‘¥…Ñ”ÁÉ½•ÍÌÉ•‘¥Ñô­qÑ•áÑí‘•±…å•½É•ÑÕÉ¸É•‘¥Ñõô()…™Ñ•È¹½Éµ…±¥é…Ñ¥½¸¸()!½Ý•Ù•È°€¨©5%µ%¥Ì„ÁÉ½©•Ð…‘…ÁÑ…Ñ¥½¸¨¨è(´Ñ¡”¥µµ•‘¥…Ñ”É•Ý…É¥Ìá…Ð¥¹™½Éµ…Ñ¥½¸…¥¸µ•…ÍÕÉ•™É½´…¹ÍÝ•È±½œµ±¥­•±¥¡½½ì(´Ñ¡”‘•±…å•¡…¹¹•°¥ÌÑ¡”É…ÜÍÕ™™¥àÍÕ´½˜á…Ð%ì(´‰½Ñ ¡…¹¹•±Ì…É”¹½Éµ…±¥é•…ÐÑ¡”Í…µ”ÁÉ½µÁÐ…¹M•…É ‘•ÁÑ ì(´€‘q…µµ„ôÄì(´€‘q…±Á¡„ôÀ¸Ôì(´Í¥¹±•Ñ½¸‘•ÁÑ¡Ì™…±°‰…¬Ñ¼¹½Éµ…±¥é•Ñ•Éµ¥¹…°½ÕÑ½µ”ì(´ÁÉ½µÁÐÍ•±•Ñ¥½¸¥ÌÁ•É™½Éµ•Í•Á…É…Ñ•±ä‰ä…¹ÍÝ•Èµ½ÕÑ½µ”I8´Èì(´á…Ð%¥Ì½µÁÕÑ•½¹±ä™½ÈÍ•±•Ñ•ÁÉ½µÁÐÉ½ÕÁÌ¸()Q¡”¹…µ”€¨©5%µ%¨¨Í¡½Õ±Ñ¡•É•™½É”‰”É•……ÌƒŠq5%µ¥¹ÍÁ¥É•¥¹™½Éµ…Ñ¥½¸µ…¥¸É•‘¥Ð…ÍÍ¥¹µ•¹Ð³Št¹½Ð…Ì„±…¥´½˜É•ÁÉ½‘Õ¥¹œÑ¡”½É¥¥¹…°5%…±½É¥Ñ¡´•á…Ñ±ä¸((´´´((ŒŒŒ€ÄÌ¸•Í¥¸‰½Õ¹‘…Éä…¹­¹½Ý¸±¥µ¥Ñ…Ñ¥½¸()½È€‘¹}íÀ±Ñõq”È°5%µ%¥Ì™Õ¹‘…µ•¹Ñ…±±ä„€¨©É•±…Ñ¥Ù”M•…É µÙÌµM•…É •ÍÑ¥µ…Ñ½È¨¨è()yíÍ•…É¡õ}íÀ±¤±Ñô)qÅÕ…‘qÑ•áÑí½µÁ…É•ÌÑÉ…©•Ñ½É¥•ÌÑ¡…Ð…ÑÕ…±±äÍ•…É¡•…Ð‘•ÁÑ õÐ¸(()%Ð‘½•Ì¹½Ð‘¥É•Ñ±ä½¹ÍÑÉÕÐÑ¡”Í…µ”µÍÑ…Ñ”½Õ¹Ñ•É™…ÑÕ…°()D¡Í}Ð±qµ…Ñ¡ÉµíM•…É¡ô¤µD¡Í}Ð±qµ…Ñ¡Éµí¹ÍÝ•É9½Ýô¤¸(()Q¡•É•™½É”„Á½Í¥Ñ¥Ù”É•±…Ñ¥Ù”M•…É …‘Ù…¹Ñ…”µ•…¹Ì((øƒŠqÑ¡¥ÌM•…É Ý…Ì‰•ÑÑ•ÈÑ¡…¸Á••ÈM•…É¡•Ì…ÐÑ¡”Í…µ”‘•ÁÑ ³Št()¹½Ð¹••ÍÍ…É¥±ä((øƒŠqM•…É Ý…Ì‰•ÑÑ•ÈÑ¡…¸ÍÑ½ÁÁ¥¹œ…¹…¹ÍÝ•É¥¹œ»Št()Q¡¥Ì‘¥ÍÑ¥¹Ñ¥½¸µ…ÑÑ•ÉÌÝ¡•¸¥¹Ñ•ÉÁÉ•Ñ¥¹œM•…É µ‘•ÁÑ ½È½Ù•ÈµÍ•…É ‰•¡…Ù¥½È¸((´´´((ŒŒŒ€ÄÐ¸I•™•É•¹•Ì()Q¡”ÁÉ½©•Ðµ•Ñ¡½¥Ì‰Õ¥±Ð™É½´¥‘•…ÌÉ•±…Ñ•Ñ¼è(´€¨©%A<¨¨ƒŠP¥¹™½Éµ…Ñ¥½¸µ…¥¸ÁÉ½•ÍÌÉ•Ý…É‘Ì‰…Í•½¸¡…¹•Ì¥¸…¹ÍÝ•È±¥­•±¥¡½½¸(´€¨©I8´È¨¨ƒŠPÉ•Ý…ÉµÙ…É¥…¹”€¼M9Hµ…Ý…É”ÁÉ½µÁÐ™¥±Ñ•É¥¹œ¸(´€¨©5%¨¨ƒŠPµ¥á¥¹œ¥µµ•‘¥…Ñ”…¹‘•±…å•ÁÉ½•ÍÌÉ•‘¥Ð¸(´€¨©
ÉQA<¨¨ƒŠPÑÕÉ¸µ¥¹‘•àµ…Ý…É”¹½Éµ…±¥é…Ñ¥½¸…¹ÑÕÉ¸µ±•Ù•°…•¹Ñ¥ŒÁ½±¥äµ½ÁÑ¥µ¥é…Ñ¥½¸½¹•ÁÑÌ¸()Q¡”•ÅÕ…Ñ¥½¹Ì…‰½Ù”‘•ÍÉ¥‰”Ñ¡”€¨©•Ñ¡•ÉM•…É ¥µÁ±•µ•¹Ñ…Ñ¥½¸¨¨¸AÉ½©•ÐµÍÁ•¥™¥Œ‘•Ù¥…Ñ¥½¹ÌÍ¡½Õ±¹½Ð‰”…ÑÑÉ¥‰ÕÑ•Ù•É‰…Ñ¥´Ñ¼Ñ¡”½É¥¥¹…°Á…Á•ÉÌ¸((ŒŒEÕ¥¬MÑ…ÉÐ()%¹ÍÑ…±°Ñ¡”±½…°Á…­…”¥¹Í¥‘”…¸I0•¹Ù¥É½¹µ•¹ÐÑ¡…Ð…±É•…‘ä½¹Ñ…¥¹ÌÑ¡”)½µÁ…Ñ¥‰±”AåQ½É °Ù•I0°Ù114°I…ä°…¹±…Í¡ÑÑ•¹Ñ¥½¸ÍÑ…¬è()‰…Í )ÁåÑ¡½¸€µ´Á¥À¥¹ÍÑ…±°€µ”€¸)€()É•…Ñ”Ñ¡”µ…¡¥¹”µ±½…°•¹Ù¥É½¹µ•¹Ð½¹™¥ÕÉ…Ñ¥½¸è()‰…Í )À•¹Ù¥É½¹µ•¹Ð½•¹Ø¹Ñ•µÁ±…Ñ”¹Í •¹Ù¥É½¹µ•¹Ð½•¹Ø¹±½…°¹Í (Œ‘¥Ð•¹Ù¥É½¹µ•¹Ð½•¹Ø¹±½…°¹Í °Ñ¡•¸è)Í½ÕÉ”•¹Ù¥É½¹µ•¹Ð½•¹Ø¹±½…°¹Í )€()IÕ¸Ñ¡”±¥¡ÑÝ•¥¡ÐÍ½ÕÉ”°Í¡•±°°…¹½¹™¥ÕÉ…Ñ¥½¸¡•­Ìè()‰…Í )‰…Í ÍÉ¥ÁÑÌ½Ù…±¥‘…Ñ•}ÍÑ…Ñ¥Œ¹Í )€()Y…±¥‘…Ñ”Ñ¡”½¹±äÉ•±•…Í•¡…É‘Ý…É”É•¥Á”Ý¥Ñ¡½ÕÐÍÑ…ÉÑ¥¹œÍ•ÉÙ¥•Ìè()‰…Í )‰…Í ÍÉ¥ÁÑÌ½ÑÉ…¥¹}É°¹Í €´µ‘ÉäµÉÕ¸)€()MÑ…ÉÐI0ÑÉ…¥¹¥¹œ½¸Ñ¡”Ù…±¥‘…Ñ•™½ÕÈµATÑ½Á½±½äè()‰…Í )‰…Í ÍÉ¥ÁÑÌ½ÑÉ…¥¹}É°¹Í )€()Q¡”¥¹±Õ‘•É•¥Á”…ÍÍ¥¹ÌÁ¡åÍ¥…°AT€ÀÑ¼É•ÑÉ¥•Ù…°…¹…Íå¹¡É½¹½ÕÌ)ÑÉ…¥¹¥¹œµÑ¥µ”•Ù…±Õ…Ñ¥½¸°…¹Á¡åÍ¥…°AUÌ€Ä´ÌÑ¼Ñ¡”Ñ¡É•”µÉ…¹¬Ù114½M@È)ÉÕ¹Ñ¥µ”¸Ù•Éä€ÈÀµÕÁ‘…Ñ”•Ù…±Õ…Ñ¥½¸ÕÍ•ÌÑ¡”½µÁ±•Ñ”€ÔÄ°ÜÄÌµÉ½ÜM•…É µHÄ)Ñ•ÍÐ¹Á…ÉÅÕ•Ñ€¸M•”É•¥Á•Ì½É°½I5¹µ‘€™½ÈÑ¡”½¹™¥ÕÉ…Ñ¥½¸‰½Õ¹‘…Éä¸)Q¡”É•Í½±Ù•½¹™¥ÕÉ…Ñ¥½¸¥Ìµ…Ñ•É¥…±¥é•¥¹Í¥‘”•… ¹•ÜÉÕ¸‘¥É•Ñ½Éä¸((ŒŒI•Á½Í¥Ñ½Éä1…å½ÕÐ((´Í™Ñ}‘…Ñ„½€èMP‘…Ñ„É•±•…Í”µ•Ñ…‘…Ñ„…¹‰Õ¥±ÍÉ¥ÁÑÌ¸Q¡”™Õ±°MP)M=90(€Á…å±½…¥Ì¡½ÍÑ•½¸m!Õ¥¹œ…”…Ñ…Í•ÑÍt¡¡ÑÑÁÌè¼½¡Õ¥¹™…”¹¼½‘…Ñ…Í•ÑÌ½µÕÉ…‘¥°ÈÄÄ½…•Ñ¡•ÉÍ•…É¡}Í™Ð¤¸(´ÍÉŒ½…•¹Ñ¥}É°½€èI0ÑÉ…¥¹¥¹œ°É½±±½ÕÐ°…‘Ù…¹Ñ…”°Á½±¥ä±½ÍÌ°É•ÑÉ¥•Ù•È°(€¡•­Á½¥¹Ð°…¹ÉÕ¹Ñ¥µ”…‘…ÁÑ•È½‘”¸(´ÍÉ¥ÁÑÌ½€è±…Õ¹ °ÁÉ•™±¥¡Ð°É•ÍÕµ”°Ù…±¥‘…Ñ¥½¸°…¹½Á•É…Ñ¥½¹…°ÍÉ¥ÁÑÌ™½È(€Ñ¡”I0ÑÉ…¥¹¥¹œÍÑ…”ìÍ•”ÍÉ¥ÁÑÌ½I5¹µ‘€™½ÈÁÕ‰±¥ŒÙ•ÉÍÕÌ¡¥ÍÑ½É¥…°(€•¹ÑÉåÁ½¥¹ÑÌ¸(´É•¥Á•Ì½É°½€èÑ¡”Í¥¹±”Ù…±¥‘…Ñ•ÁÕ‰±¥ŒI0É•¥Á”…¹¥ÑÌÕÍ…”‰½Õ¹‘…Éä¸(´½¹™¥Ì½€è‰…Í”°™½Éµ…°°¡…É‘Ý…É”°É•ÑÉ¥•Ù•È°ÍÑ…”½¹™¥Ì°…¹Ñ¡”á…Ðµ%(€ÉÕ¹Ñ¥µ”…Ñ”ì(€Í•”½¹™¥Ì½I5¹µ‘€™½ÈÑ¡•¥ÈÁ½ÉÑ…‰¥±¥Ñä‰½Õ¹‘…Éä¸(´ÉÕ¹Ñ¥µ•}…ÍÍ•ÑÌ½€è±½…°ÉÕ¹Ñ¥µ”…ÍÍ•ÑÌÉ•ÅÕ¥É•‰äÑ¡”ÑÉ…¥¹¥¹œ±…Õ¹¡•È¸(´Ñ•ÍÑÌ½€èÕ¹¥Ð…¹¥¹Ñ•É…Ñ¥½¸¡•­Ì™½ÈÑ¡”ÑÉ…¥¹¥¹œ½‘”¸(´•¹Ù¥É½¹µ•¹Ð½€è½‰Í•ÉÙ•Á…­…”Ù•ÉÍ¥½¹Ì…¹•¹Ù¥É½¹µ•¹ÐÑ•µÁ±…Ñ”¸((ŒŒI•±•…Í”	½Õ¹‘…Éä()1…É”µ½‘•°Ý•¥¡ÑÌ°½ÁÑ¥µ¥é•ÈµÍÑ…Ñ”¡•­Á½¥¹ÑÌ°•Ù…°É•ÍÕ±Ð‰Õ¹‘±•Ì°É•Á½ÉÐ)…É¡¥Ù•Ì°…¹ÉÕ¹Ñ¥µ”Í¹…ÁÍ¡½ÑÌ…É”¥¹Ñ•¹Ñ¥½¹…±±ä¹½Ð½µµ¥ÑÑ•Ñ¼Ñ¡¥Ì¥Ñ!Õˆ)É•Á½Í¥Ñ½Éä¸AÕ‰±¥Œµ½‘•°…¹‘…Ñ„…ÉÑ¥™…ÑÌ…É”±¥¹­•™É½´!Õ¥¹œ…”…‰½Ù”¸(