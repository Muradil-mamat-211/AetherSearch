from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .attempt_state import SnapshotVersions


def parameter_version_checksum(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(str(parameter.dtype).encode("ascii"))
        digest.update(str(int(getattr(parameter, "_version", -1))).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenRolloutBoundary:
    versions: SnapshotVersions
    actor_parameter_version_checksum: str
    reference_model_fingerprint: str

    def validate_after_pre_step_work(self, actor: Any) -> None:
        self.versions.assert_rollout_boundary_parity()
        current = parameter_version_checksum(actor)
        if current != self.actor_parameter_version_checksum:
            raise RuntimeError(
                "Actor parameters changed between rollout start and optimizer step"
            )


class SnapshotCoordinator:
    def freeze(
        self,
        *,
        actor: Any,
        successful_update_step: int,
        synchronize_vllm: Callable[[int], int],
        materialize_old_policy: Callable[[int], int],
        materialize_reward_policy: Callable[[int], int],
        reference_model_fingerprint: str,
    ) -> FrozenRolloutBoundary:
        step = int(successful_update_step)
        checksum = parameter_version_checksum(actor)
        rollout_step = int(synchronize_vllm(step))
        old_step = int(materialize_old_policy(step))
        reward_step = int(materialize_reward_policy(step))
        versions = SnapshotVersions(step, rollout_step, old_step, reward_step)
        versions.assert_rollout_boundary_parity()
        return FrozenRolloutBoundary(
            versions=versions,
            actor_parameter_version_checksum=checksum,
            reference_model_fingerprint=str(reference_model_fingerprint),
        )


def assert_snapshot_logprob_parity(
    actor_logprobs: Any,
    old_policy_logprobs: Any,
    reward_policy_logprobs: Any,
    *,
    absolute_tolerance: float,
) -> float:
    """CPU/GPU-neutral parity gate for one fixed teacher-forced mini input."""
    import torch

    if not (
        actor_logprobs.shape
        == old_policy_logprobs.shape
        == reward_policy_logprobs.shape
    ):
        raise ValueError("Snapshot logprob tensors must have identical shapes")
    if absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be non-negative")
    tensors: Sequence[Any] = (
        actor_logprobs.detach().float(),
        old_policy_logprobs.detach().float(),
        reward_policy_logprobs.detach().float(),
    )
    if not all(bool(torch.isfinite(value).all().item()) for value in tensors):
        raise ValueError("Snapshot logprobs must be finite")
    maximum = max(
        float(torch.max(torch.abs(tensors[0] - candidate)).item())
        for candidate in tensors[1:]
    )
    if maximum > absolute_tolerance:
        raise RuntimeError(
            f"Snapshot logprob parity failed: max_abs_diff={maximum}, "
            f"tolerance={absolute_tolerance}"
        )
    return maximum
