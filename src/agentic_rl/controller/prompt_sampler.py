from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PromptCursorState:
    epoch: int
    cursor: int
    dataset_size: int
    shuffle_seed: int
    permutation_hash: str


class ImmutableDatasetPromptSampler:
    """In-memory index permutation; source parquet remains untouched."""

    def __init__(
        self,
        *,
        dataset_size: int,
        shuffle_seed: int,
        epoch: int = 0,
        cursor: int = 0,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        self.dataset_size = int(dataset_size)
        self.shuffle_seed = int(shuffle_seed)
        self.epoch = int(epoch)
        self.cursor = int(cursor)
        self._permutation = self._make_permutation(self.epoch)
        if not 0 <= self.cursor <= self.dataset_size:
            raise ValueError("cursor is outside the dataset")

    def _make_permutation(self, epoch: int) -> np.ndarray:
        generator = np.random.default_rng(self.shuffle_seed + int(epoch))
        return generator.permutation(self.dataset_size)

    @property
    def permutation_hash(self) -> str:
        return hashlib.sha256(
            self._permutation.astype(np.int64, copy=False).tobytes()
        ).hexdigest()

    def allocate(self, count: int) -> tuple[int, ...]:
        if count <= 0:
            raise ValueError("count must be positive")
        allocated: list[int] = []
        while len(allocated) < count:
            remaining = self.dataset_size - self.cursor
            take = min(count - len(allocated), remaining)
            allocated.extend(
                int(value)
                for value in self._permutation[self.cursor : self.cursor + take]
            )
            self.cursor += take
            if self.cursor == self.dataset_size and len(allocated) < count:
                self.epoch += 1
                self.cursor = 0
                self._permutation = self._make_permutation(self.epoch)
        return tuple(allocated)

    def state(self) -> PromptCursorState:
        return PromptCursorState(
            epoch=self.epoch,
            cursor=self.cursor,
            dataset_size=self.dataset_size,
            shuffle_seed=self.shuffle_seed,
            permutation_hash=self.permutation_hash,
        )

    @classmethod
    def restore(cls, state: PromptCursorState) -> "ImmutableDatasetPromptSampler":
        sampler = cls(
            dataset_size=state.dataset_size,
            shuffle_seed=state.shuffle_seed,
            epoch=state.epoch,
            cursor=state.cursor,
        )
        if sampler.permutation_hash != state.permutation_hash:
            raise RuntimeError("Dataset permutation hash changed during resume")
        return sampler
