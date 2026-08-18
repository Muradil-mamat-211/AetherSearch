from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


STOP_CONTINUE_CONSENSUS_MODE = "stop_continue_consensus"
NORMALIZED_OUTCOME_MODE = "normalized_outcome"
STOP_CONTINUE_ADVANTAGE_VERSION = (
    "stop_continue_consensus_search_advantage_zero_fallback_v2"
)


@dataclass(frozen=True)
class StopContinueRewardTriple:
    """Validated detached rewards for one real pre-Search state."""

    prompt_global_id: str
    trajectory_id: str
    search_index: int
    continue_reward: float
    stop_reward_1: float
    stop_reward_2: float
    continue_scorer_version: str
    stop_scorer_version_1: str
    stop_scorer_version_2: str
    candidate_rollout_policy_version: int
    exact_ig_policy_version: int
    stop_branch_policy_version: int
    old_logprob_policy_version: int
    prefix_provenance_valid: bool
    context_truncated: bool
    completion_count: int
    detached: bool = True

    @property
    def state_key(self) -> tuple[str, str, int]:
        return (
            str(self.prompt_global_id),
            str(self.trajectory_id),
            int(self.search_index),
        )

    @property
    def scorer_version(self) -> str:
        versions = {
            str(self.continue_scorer_version),
            str(self.stop_scorer_version_1),
            str(self.stop_scorer_version_2),
        }
        if len(versions) != 1:
            raise ValueError(
                f"{self.state_key}: Continue and Stop rewards use different scorers"
            )
        return next(iter(versions))

    @property
    def policy_version(self) -> int:
        versions = {
            int(self.candidate_rollout_policy_version),
            int(self.exact_ig_policy_version),
            int(self.stop_branch_policy_version),
            int(self.old_logprob_policy_version),
        }
        if len(versions) != 1:
            raise ValueError(
                f"{self.state_key}: Stop/Continue policy versions do not match"
            )
        return next(iter(versions))

    def validate(
        self,
        *,
        expected_policy_version: int | None = None,
        expected_scorer_version: str | None = None,
    ) -> None:
        if not self.prompt_global_id or not self.trajectory_id:
            raise ValueError("Stop/Continue state identifiers cannot be empty")
        if int(self.search_index) < 0:
            raise ValueError(f"{self.state_key}: search_index must be non-negative")
        rewards = np.asarray(
            [
                self.continue_reward,
                self.stop_reward_1,
                self.stop_reward_2,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(rewards)):
            raise ValueError(f"{self.state_key}: Stop/Continue rewards must be finite")
        if int(self.completion_count) != 2:
            raise ValueError(
                f"{self.state_key}: exactly two Stop completions are required"
            )
        if not self.prefix_provenance_valid:
            raise ValueError(f"{self.state_key}: Search-prefix provenance is invalid")
        if self.context_truncated:
            raise ValueError(f"{self.state_key}: Stop input was context-truncated")
        if not self.detached:
            raise ValueError(f"{self.state_key}: Stop rewards must be detached")
        observed_policy_version = self.policy_version
        if (
            expected_policy_version is not None
            and observed_policy_version != int(expected_policy_version)
        ):
            raise ValueError(
                f"{self.state_key}: expected policy version "
                f"{expected_policy_version}, got {observed_policy_version}"
            )
        observed_scorer_version = self.scorer_version
        if (
            expected_scorer_version is not None
            and observed_scorer_version != str(expected_scorer_version)
        ):
            raise ValueError(
                f"{self.state_key}: expected scorer {expected_scorer_version}, "
                f"got {observed_scorer_version}"
            )


@dataclass(frozen=True)
class StopContinueAdvantage:
    prompt_global_id: str
    trajectory_id: str
    search_index: int
    continue_reward: float
    stop_reward_1: float
    stop_reward_2: float
    delta_sc: float
    pooled_scale: float
    raw_advantage_sc: float
    advantage_sc: float
    sc_clear: bool
    clear_positive: bool
    clear_negative: bool
    normalized_outcome: float
    task_advantage: float
    clip_bound: float
    clipped: bool

    @property
    def state_key(self) -> tuple[str, str, int]:
        return (
            self.prompt_global_id,
            self.trajectory_id,
            self.search_index,
        )


@dataclass(frozen=True)
class StopContinueComputation:
    by_state: Mapping[tuple[str, str, int], StopContinueAdvantage]
    pooled_scale_by_prompt_search: Mapping[tuple[str, int], float]
    metrics: Mapping[str, float | int]


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def stop_continue_reward_from_mapping(
    value: Mapping[str, object],
) -> StopContinueRewardTriple:
    return StopContinueRewardTriple(
        prompt_global_id=str(value["prompt_global_id"]),
        trajectory_id=str(value["trajectory_id"]),
        search_index=int(value["search_index"]),
        continue_reward=float(value["continue_reward"]),
        stop_reward_1=float(value["stop_reward_1"]),
        stop_reward_2=float(value["stop_reward_2"]),
        continue_scorer_version=str(value["continue_scorer_version"]),
        stop_scorer_version_1=str(value["stop_scorer_version_1"]),
        stop_scorer_version_2=str(value["stop_scorer_version_2"]),
        candidate_rollout_policy_version=int(
            value["candidate_rollout_policy_version"]
        ),
        exact_ig_policy_version=int(value["exact_ig_policy_version"]),
        stop_branch_policy_version=int(value["stop_branch_policy_version"]),
        old_logprob_policy_version=int(value["old_logprob_policy_version"]),
        prefix_provenance_valid=bool(value["prefix_provenance_valid"]),
        context_truncated=bool(value["context_truncated"]),
        completion_count=int(value["completion_count"]),
        detached=bool(value.get("detached", True)),
    )


def compute_stop_continue_advantages(
    rewards: Sequence[StopContinueRewardTriple],
    *,
    normalized_outcome_by_trajectory: Mapping[tuple[str, str], float],
    expected_state_keys: Iterable[tuple[str, str, int]],
    group_size: int,
    reward_epsilon: float = 1.0e-6,
    scale_epsilon: float = 1.0e-8,
    pooled_scale_ddof: int = 0,
    expected_policy_version: int | None = None,
    expected_scorer_version: str | None = None,
) -> StopContinueComputation:
    """Compute selected-only Stop/Continue consensus Search task advantages."""

    if int(group_size) < 2:
        raise ValueError("Stop/Continue clipping requires group_size >= 2")
    if int(pooled_scale_ddof) != 0:
        raise ValueError("Stop/Continue pooled scale is locked to ddof=0")
    reward_epsilon = _finite_float(reward_epsilon, "reward_epsilon")
    scale_epsilon = _finite_float(scale_epsilon, "scale_epsilon")
    if reward_epsilon <= 0.0 or scale_epsilon <= 0.0:
        raise ValueError("Stop/Continue epsilons must be positive")

    expected = {
        (str(prompt_id), str(trajectory_id), int(search_index))
        for prompt_id, trajectory_id, search_index in expected_state_keys
    }
    if not expected:
        raise ValueError("Selected trajectories contain no trainable Search state")

    by_key: dict[tuple[str, str, int], StopContinueRewardTriple] = {}
    scorer_versions: set[str] = set()
    policy_versions: set[int] = set()
    for reward in rewards:
        reward.validate(
            expected_policy_version=expected_policy_version,
            expected_scorer_version=expected_scorer_version,
        )
        key = reward.state_key
        if key in by_key:
            raise ValueError(f"{key}: duplicate Stop/Continue reward triple")
        by_key[key] = reward
        scorer_versions.add(reward.scorer_version)
        policy_versions.add(reward.policy_version)
    if set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        unexpected = sorted(set(by_key) - expected)
        raise ValueError(
            "Stop/Continue reward-state coverage mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if len(scorer_versions) != 1:
        raise ValueError("Selected Stop/Continue states use different scorer versions")
    if len(policy_versions) != 1:
        raise ValueError("Selected Stop/Continue states use different policy versions")

    grouped: dict[tuple[str, int], list[StopContinueRewardTriple]] = {}
    for reward in by_key.values():
        grouped.setdefault(
            (reward.prompt_global_id, int(reward.search_index)),
            [],
        ).append(reward)

    pooled_scale: dict[tuple[str, int], float] = {}
    for prompt_search, values in grouped.items():
        flattened = np.asarray(
            [
                reward_value
                for reward in values
                for reward_value in (
                    reward.continue_reward,
                    reward.stop_reward_1,
                    reward.stop_reward_2,
                )
            ],
            dtype=np.float64,
        )
        if flattened.size != 3 * len(values):
            raise RuntimeError("Stop/Continue pooled reward cardinality is invalid")
        if not np.all(np.isfinite(flattened)):
            raise ValueError(f"{prompt_search}: pooled rewards must be finite")
        scale = float(np.std(flattened, ddof=0, dtype=np.float64))
        if not math.isfinite(scale):
            raise ValueError(f"{prompt_search}: pooled scale is non-finite")
        pooled_scale[prompt_search] = scale

    clip_bound = math.sqrt(int(group_size) - 1)
    results: dict[tuple[str, str, int], StopContinueAdvantage] = {}
    for key, reward in by_key.items():
        outcome_key = (reward.prompt_global_id, reward.trajectory_id)
        if outcome_key not in normalized_outcome_by_trajectory:
            raise ValueError(f"{key}: normalized Outcome is missing")
        normalized_outcome = _finite_float(
            normalized_outcome_by_trajectory[outcome_key],
            f"{key}.normalized_outcome",
        )
        delta = float(
            reward.continue_reward
            - 0.5 * (reward.stop_reward_1 + reward.stop_reward_2)
        )
        scale = pooled_scale[(reward.prompt_global_id, reward.search_index)]
        raw = float(delta / (scale + scale_epsilon))
        advantage_sc = float(np.clip(raw, -clip_bound, clip_bound))
        clear_positive = bool(
            reward.continue_reward
            > max(reward.stop_reward_1, reward.stop_reward_2) + reward_epsilon
        )
        clear_negative = bool(
            reward.continue_reward
            < min(reward.stop_reward_1, reward.stop_reward_2) - reward_epsilon
        )
        sc_clear = bool(clear_positive or clear_negative)
        # Outcome credit is terminal-only. An unclear Stop/Continue probe adds
        # no task term to a Search action.
        task_advantage = advantage_sc if sc_clear else 0.0
        values_to_check = (
            delta,
            scale,
            raw,
            advantage_sc,
            task_advantage,
        )
        if not all(math.isfinite(value) for value in values_to_check):
            raise ValueError(f"{key}: Stop/Continue advantage is non-finite")
        results[key] = StopContinueAdvantage(
            prompt_global_id=reward.prompt_global_id,
            trajectory_id=reward.trajectory_id,
            search_index=int(reward.search_index),
            continue_reward=float(reward.continue_reward),
            stop_reward_1=float(reward.stop_reward_1),
            stop_reward_2=float(reward.stop_reward_2),
            delta_sc=delta,
            pooled_scale=scale,
            raw_advantage_sc=raw,
            advantage_sc=advantage_sc,
            sc_clear=sc_clear,
            clear_positive=clear_positive,
            clear_negative=clear_negative,
            normalized_outcome=normalized_outcome,
            task_advantage=float(task_advantage),
            clip_bound=clip_bound,
            clipped=not math.isclose(
                raw,
                advantage_sc,
                rel_tol=0.0,
                abs_tol=0.0,
            ),
        )

    ordered = tuple(results[key] for key in sorted(results))
    state_count = len(ordered)
    clear_count = sum(value.sc_clear for value in ordered)
    clear_positive_count = sum(value.clear_positive for value in ordered)
    clear_negative_count = sum(value.clear_negative for value in ordered)
    fallback_count = state_count - clear_count
    deltas = np.asarray([value.delta_sc for value in ordered], dtype=np.float64)
    scales = np.asarray([value.pooled_scale for value in ordered], dtype=np.float64)
    raw_advantages = np.asarray(
        [value.raw_advantage_sc for value in ordered],
        dtype=np.float64,
    )
    advantages = np.asarray(
        [value.advantage_sc for value in ordered],
        dtype=np.float64,
    )
    metrics: dict[str, float | int] = {
        "sc/state_count": state_count,
        "sc/clear_count": clear_count,
        "sc/clear_rate": clear_count / state_count,
        "sc/fallback_count": fallback_count,
        "sc/fallback_rate": fallback_count / state_count,
        "sc/fallback_z_o_to_search_count": 0,
        "sc/clear_positive_count": clear_positive_count,
        "sc/clear_negative_count": clear_negative_count,
        "sc/delta_mean": float(np.mean(deltas, dtype=np.float64)),
        "sc/delta_std": float(np.std(deltas, ddof=0, dtype=np.float64)),
        "sc/scale_mean": float(np.mean(scales, dtype=np.float64)),
        "sc/raw_advantage_mean": float(
            np.mean(raw_advantages, dtype=np.float64)
        ),
        "sc/advantage_mean": float(np.mean(advantages, dtype=np.float64)),
        "sc/advantage_std": float(
            np.std(advantages, ddof=0, dtype=np.float64)
        ),
        "sc/clip_fraction": sum(value.clipped for value in ordered) / state_count,
        "sc/stop_reward_1_mean": float(
            np.mean(
                np.asarray(
                    [value.stop_reward_1 for value in ordered],
                    dtype=np.float64,
                ),
                dtype=np.float64,
            )
        ),
        "sc/stop_reward_2_mean": float(
            np.mean(
                np.asarray(
                    [value.stop_reward_2 for value in ordered],
                    dtype=np.float64,
                ),
                dtype=np.float64,
            )
        ),
        "sc/stop_reward_agreement_rate": (
            sum(
                math.isclose(
                    value.stop_reward_1,
                    value.stop_reward_2,
                    rel_tol=0.0,
                    abs_tol=reward_epsilon,
                )
                for value in ordered
            )
            / state_count
        ),
    }
    return StopContinueComputation(
        by_state=results,
        pooled_scale_by_prompt_search=pooled_scale,
        metrics=metrics,
    )
