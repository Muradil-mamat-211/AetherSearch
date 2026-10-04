# AetherSearch Improvement 2: Alternating RL and DPO

**Status: training-design proposal; the alternating pipeline is not yet
implemented or experimentally validated.** RL explores search and answering
strategies. Every 100 successful RL updates, a short DPO phase targets clear,
verified decision errors made by the current policy, then RL resumes:

$$
\pi_{100}\rightarrow\pi_{100}^{+}
\rightarrow\pi_{200}
\rightarrow\pi_{200}^{+}\rightarrow\cdots
$$

Each transition from $\pi_k$ to $\pi_k^+$ is one DPO epoch. Each transition
from $\pi_k^+$ to $\pi_{k+100}$ is 100 further successful RL updates.

This follows the [over-search improvement](over-search-improvement.md).
That proposal adjusts Search credit during RL; this proposal periodically
trains on better next actions at observed failure states. It can address
over-search, premature answers, poor queries, and other verifiable decision
errors without requiring a predefined failure taxonomy.

## 1. Initial configuration

| Setting | Proposed first version |
|---|---|
| RL interval | 100 **successful** RL optimizer updates |
| Initial sampling | 8 complete trajectory attempts per question |
| Resampling at one frozen prefix | 4 complete continuations; up to 4 more if none passes both gates |
| DPO data target | Up to 256 accepted pairs per phase |
| Question balance | At most one pair per normalized question per phase |
| DPO duration | One epoch over that phase's fresh accepted pairs |
| DPO actor and reference initialization | The same current RL checkpoint; reference remains frozen |

These are proposed settings, not measured optima. Data targets of 128, 256,
and 512 can be compared later. Fix candidate-question, execution-time, and
teacher-retry budgets before each phase. Stop at 256 pairs or the first
budget limit; accept fewer without relaxing verification. If no pair passes,
skip DPO and resume RL from the unchanged checkpoint.

Skipped or failed RL attempts do not advance the interval. DPO optimizer
steps have a separate counter and do not count as successful RL updates.

## 2. Freeze the current policy and collect real behavior

At boundary $k\in\{100,200,300,\ldots\}$, pause RL optimization and freeze
$\pi_k$. Initial trajectories, policy resampling, and continuations after
teacher repairs all use this checkpoint and the same pinned retriever,
corpus, decoding configuration, protocol, and execution budgets.

Use training-only questions and exclude evaluation questions and confirmed
near-duplicates. Unlike the separate historical offline DPO construction,
this online correction pool may draw from the RL training pool; record that
shared training exposure explicitly. Evaluation examples must not become
repair prompts or preference pairs.

For each question, sample eight trajectory attempts. Execute every Search
against the real retriever and continue to a terminal Answer or the fixed
execution limit. Preserve actions, observations, document provenance,
termination reason, protocol validity, and final EM/F1. Truncation or timeout
is a failed attempt, not a successful completed trajectory.

## 3. Discover an error and freeze its exact decision state

Codex reviews the eight trajectories and identifies the **earliest clear,
actionable error for which a correction can be verified**. Categories may
be assigned afterward for analysis; membership in a fixed taxonomy does not
determine which errors can be discovered or accepted.

Freeze $x$ as the exact policy-visible prefix immediately before the target
action: question, instructions, previous actions, and real observations.
Preserve the accompanying runtime state and remaining execution budget
alongside $x$, without inserting new metadata into the prompt. Keep the actual
faulty next action as $a_l$, the candidate `rejected` action. Do not rewrite it or invent a worse
negative. A wrong final answer alone does not establish an earlier decision
error; a correct final answer can still contain a wasteful action.

Discovery may inspect complete traces. Candidate generation must receive
only $x$, without future observations, gold answers, or hints copied from
successful continuations. Teacher repair uses a fresh context containing
$x$ and, if needed, a failure description justified by information at $x$.
This prevents the discovery stage from leaking future evidence into a repair.

## 4. Generate candidates and apply fixed verification gates

From the identical $x$, sample four complete continuations with $\pi_k$.
If none passes both gates below, sample up to four additional continuations.
For each candidate, retain its first action $a_j$ and its full continuation
$\tau'_j$. An immediate Answer is already terminal. A Search must execute
real retrieval and continue with $\pi_k$ to the final Answer or execution limit.
All continuations inherit the remaining budget at $x$; resampling does not
grant extra searches or tokens.

| Gate | Acceptance requirement |
|---|---|
| Terminal verification | Complete, protocol-valid continuation with normalized, alias-aware final-answer **EM = 1**. F1 is diagnostic, not a substitute for EM. |
| Local Search verification | The Search addresses an information need at $x$, its actual observation supplies relevant, useful evidence beyond what is already available, and the action is clearly better than $a_l$. A different query or new document ID alone is insufficient. |
| Local Answer verification | The answer passes the terminal check, stopping is justified at $x$ under the task's evidence and prior-knowledge rules, and the action is clearly better than $a_l$. |

Every chosen candidate must pass terminal verification and the local gate
for its action type. Reject ties, uncertainty, unsupported preferences, and
identical chosen/rejected actions. Local verification uses $x$, both actions,
and their immediate real retrieval results when applicable. It does not use
later outcomes, gold answers, or candidate origin to justify the local
preference. Record the information gap and supporting evidence.

**Codex proposes; the verifier accepts.** Acceptance belongs to a separate
verification stage with frozen rules, prompts, and versions. Codex cannot
waive a gate or accept its own proposal solely by assertion. Rule-based
protocol/EM checks and evidence-based local judgments retain separate audit
records; judgment uncertainty leads to exclusion or independent review.

If all policy samples fail, Codex may propose a minimal replacement action
from the same $x$, within the fixed teacher-retry budget. A repaired Search
must be executed by the real retriever and followed by a complete $\pi_k$
continuation. A repaired Answer terminates immediately. Both pass exactly
the same gates as policy-sampled candidates; teacher origin grants no
acceptance privilege.

## 5. Train on the verified next action

Among accepted candidates, prefer a clear decision improvement with the
smallest necessary correction. Keep at most one highest-quality pair per
normalized question:

$$
\mathcal D_k=\{(x_i,a_{w,i},a_{l,i})\}_{i=1}^{N_k},
\qquad 0\le N_k\le256.
$$

| Training field | Content |
|---|---|
| `prompt_text` | Exact shared prefix $x$ |
| `chosen` | Verified next action $a_w$ |
| `rejected` | Original policy-generated next action $a_l$ |

An action includes its associated `<think>` and complete `<search>` or
`<answer>` block. A Search target ends at `</search>`; its retrieved
observation and all subsequent actions belong in the linked audit record.
The full original and candidate trajectories verify the pair but are not
the preference-loss targets. Mask the shared prefix and preserve the
[existing action-token scoring contract](../dpo/README.md#preference-loss-contract).
Keep only pairs representable by that schema; do not silently edit an invalid
rejected action to make it trainable.

The audit record links checkpoint, question, exact prefix, actions, candidate
origin, full continuations, retrieval provenance, gate decisions, scores,
versions, and sampling budgets. Failure descriptions are descriptive metadata,
not a fixed list that blocks newly discovered error types.

## 6. Run one short DPO phase

For nonempty $\mathcal D_k$, initialize the actor and frozen reference from
the same checkpoint:

$$
\pi_{\theta}^{(0)}=\pi_k,
\qquad
\pi_{\mathrm{ref},k}^{\mathrm{DPO}}=\pi_k.
$$

The reference parameters remain frozen throughout the DPO phase and receive
no gradient updates.

First define the policy-to-reference log score of one next action:

$$
s_{\theta,k}(a,x)
=\log\pi_\theta(a\mid x)
-\log\pi_{\mathrm{ref},k}^{\mathrm{DPO}}(a\mid x).
$$

The preference margin is the chosen action's score minus the rejected
action's score:

$$
\Delta_{\theta,k}(x,a_w,a_l)
=s_{\theta,k}(a_w,x)-s_{\theta,k}(a_l,x).
$$

Use the sigmoid DPO loss over the accepted next-action pairs:

$$
\mathcal{L}_k
=-\mathbb{E}_{(x,a_w,a_l)\sim\mathcal{D}_k}
[\log\sigma(\beta_{\mathrm{DPO}}\Delta_{\theta,k}(x,a_w,a_l))].
$$

Here $\sigma$ is the sigmoid function. These three expressions are exactly
the log-ratio DPO objective: increasing the chosen action's score relative
to the rejected action's score decreases the loss.

Log probabilities sum over scored action tokens. Fix the DPO learning rate,
batching, and $\beta_{\mathrm{DPO}}>0$ before the phase; it is a separate coefficient
from the over-search penalty. Run one epoch over the fresh pairs to obtain
$\pi_k^+$. Refresh this DPO reference at each correction boundary; do not
silently replace the RL phase's separately configured KL reference.

## 7. Resume RL across an explicit policy boundary

After DPO, load $\pi_k^+$ into the RL actor and all rollout workers. Assign a
new policy revision even though the successful RL-update counter is still
$k$. Freeze fresh old-policy and reward-scoring snapshots from that revision
before sampling the next RL batch.

Invalidate pre-DPO rollouts, old log probabilities, advantages, and
policy-dependent scoring caches for future actor updates. Preserve them as
historical audit records. Start RL update $k+1$ using newly sampled
trajectories from $\pi_k^+$; do not reuse DPO-construction samples as an
on-policy RL batch.

DPO uses its own optimizer. The implementation must explicitly version its
RL optimizer-moment and learning-rate-schedule resume policy rather than
silently mixing optimizer state from the two phases. Keep the RL update
counter separate from DPO steps and policy revisions, and persist phase
completion so a resumed job does not repeat the same correction accidentally.

## 8. Expected benefit and validation

The proposal connects broad RL exploration with targeted corrections based
on the policy's current behavior. Full continuations check task completion;
the local gate reduces the risk of crediting an earlier action merely because
the final answer happened to be correct. Autonomous discovery can include
new failure types while fixed acceptance rules preserve a consistent bar.

A single successful continuation still does not prove the first action has
higher expected value. Record sampling and acceptance rates, verifier
uncertainty, and teacher-origin share. Compare continuous RL with alternating
training under matched question exposure and reported total compute, measuring
EM/F1, format validity, Search count, repeated searches, and correction cost.
Report multiple seeds and any post-DPO regression; lower DPO loss alone does
not establish a better search agent. The first over-search improvement can
be evaluated separately and jointly with this second proposal.

The [released DPO stage](../dpo/README.md) remains a separate historical
training recipe. This proposal requires an alternating-phase controller,
fresh-data manifests, verification orchestration, and checkpoint/optimizer
handoff support; it is not activated by the existing launcher.
