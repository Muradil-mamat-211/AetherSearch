from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class ReplicaVersion:
    replica_id: int
    snapshot_step: int
    weight_checksum: str


class VersionedVLLMManager:
    def __init__(
        self,
        replicas: Sequence[Any],
        *,
        data_parallel_size: int,
        tensor_parallel_size: int,
    ) -> None:
        if len(replicas) != data_parallel_size:
            raise ValueError("There must be exactly one vLLM replica per DP rank")
        if tensor_parallel_size != 1:
            raise ValueError("This deployment locks vLLM tensor parallel size to 1")
        self.replicas = tuple(replicas)
        self.data_parallel_size = int(data_parallel_size)
        self.tensor_parallel_size = int(tensor_parallel_size)
        self._versions: tuple[ReplicaVersion, ...] = tuple()

    def assert_sleep_wake_api(self) -> None:
        for index, replica in enumerate(self.replicas):
            if not callable(getattr(replica, "sleep", None)):
                raise RuntimeError(f"vLLM replica {index} has no sleep() API")
            if not callable(getattr(replica, "wake_up", None)):
                raise RuntimeError(f"vLLM replica {index} has no wake_up() API")

    def wake_for_rollout(self) -> None:
        self.assert_sleep_wake_api()
        for replica in self.replicas:
            replica.wake_up()

    def sleep_after_rollout(self) -> None:
        self.assert_sleep_wake_api()
        for replica in self.replicas:
            replica.sleep()

    def synchronize_weights(
        self,
        *,
        snapshot_step: int,
        sync_one_replica: Callable[[int, Any], str],
    ) -> tuple[ReplicaVersion, ...]:
        versions = tuple(
            ReplicaVersion(
                replica_id=index,
                snapshot_step=int(snapshot_step),
                weight_checksum=str(sync_one_replica(index, replica)),
            )
            for index, replica in enumerate(self.replicas)
        )
        checksums = {version.weight_checksum for version in versions}
        if len(checksums) != 1:
            raise RuntimeError("vLLM replica weight checksums disagree")
        self._versions = versions
        return versions

    def assert_snapshot(self, expected_step: int) -> None:
        if len(self._versions) != self.data_parallel_size:
            raise RuntimeError("vLLM weights have not been synchronized")
        if any(version.snapshot_step != expected_step for version in self._versions):
            raise RuntimeError("vLLM replica snapshot version mismatch")
