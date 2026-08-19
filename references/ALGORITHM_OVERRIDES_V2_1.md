# Algorithm Overrides V2.1

This file is the highest-priority project-local algorithm lock below the user
instruction that introduced V2.1. It supersedes `ALGORITHM_OVERRIDES.md` and
`FINAL_ALGORITHM_SPEC_V2.md`.

## 1. Effective-Position IG Variance Weighting

For prompt \(p\), only Search positions with at least two eligible trajectory
peers belong to \(T_+(p)\). The natural weight is

\[
\omega_{p,t}
=
\frac{n_{p,t}}{\sum_{k \in T_+(p)} n_{p,k}},
\qquad
t \in T_+(p).
\]

Positions with \(n_{p,t}<2\) have zero sample variance, zero weight, and do not
enter either the numerator or denominator of the natural-weight normalization.

## 2. Scale and Channel Activation Are Separate

At successful Update 1, the positive median \(m_1^c\) initializes
\(b_1^c=m_1^c\) even when the bootstrap activation gate is off. For Updates
2-10, bootstrap activation controls only current selection; it does not freeze
the post-commit log-EMA scale update. Once a channel has ten valid successful
health observations, an inactive absolute-health gate freezes its scale EMA.

No late initialization state machine, default scale, or epsilon substitute for
a missing positive median is allowed.

## 3. Unique Adaptive Turn-Level Clipping

The only active clipping algorithm is A-squared-TGPO adaptive turn-level
clipping:

\[
c_{p,i,t}
=
1 + 0.3\left(2\sigma(\widehat r^{IG}_{p,i,t})-1\right),
\]

\[
\operatorname{lower}_{p,i,t}=1-0.003c_{p,i,t},
\qquad
\operatorname{upper}_{p,i,t}=1+0.004c_{p,i,t}.
\]

The Search scale is stop-gradient. Answer/fallback turns use the neutral scale
\(c=1\), without a fabricated IG signal. Fixed DAPO Clip-Higher and its
`[0.8, 1.28]` bounds are not part of V2.1.

## 4. Exact IG and Search Reward

\(\Phi\) is the mean teacher-forced target log-likelihood. Exact IG is
\(r^{IG}_{p,i,t}=\Phi_{p,i,t}-\Phi_{p,i,t-1}\). It is never exponentiated or
converted to a probability difference. No additional Search reward, cost, or
penalty is permitted.

## 5. Strict One-Step Contract

Search advantage has exactly two terms:

\[
A_{\text{search}}=\overline D+z_O.
\]

Answer/fallback advantage has exactly two terms:

\[
A_{\text{answer}}=z_O+A_{\text{format}},
\qquad
A_{\text{format}}=F_{\text{ans}}-\operatorname{mean}(F_{\text{ans}}).
\]

`ppo_epochs=1`, `optimizer_mini_steps=1`, and every successful update performs
exactly one `optimizer.step()` and one `scheduler.step()`.
