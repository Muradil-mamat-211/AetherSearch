from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from agentic_rl.exact_ig.target_schema import select_canonical_answer


@dataclass(frozen=True)
class LogicalDatasetIdentity:
    source_path: str
    source_rows: int
    logical_rows: int
    nq_rows: int
    hotpotqa_rows: int
    selection_seed: int
    ordered_view_identity_sha256: str


def _as_text_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, Sequence):
        values = value
    else:
        values = (value,)
    return tuple(str(item) for item in values if str(item).strip())


def _ground_truth_aliases(row: Mapping[str, Any]) -> tuple[str, ...]:
    aliases = _as_text_list(row.get("golden_answers"))
    if aliases:
        return aliases
    reward_model = row.get("reward_model")
    if isinstance(reward_model, Mapping):
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, Mapping):
            aliases = _as_text_list(ground_truth.get("target"))
    if not aliases:
        raise ValueError("Training row has no non-empty answer alias")
    return aliases


def _canonical_ground_truth(row: Mapping[str, Any]) -> str:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, Mapping):
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, Mapping) and ground_truth.get("target") is not None:
            return select_canonical_answer(ground_truth["target"])
    if row.get("golden_answers") is not None:
        return select_canonical_answer(row["golden_answers"])
    raise ValueError("Training row has no Exact-IG canonical answer field")


def _prompt_messages(value: Any) -> tuple[dict[str, str], ...]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("Training row prompt must be a sequence of messages")
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("Every prompt message must be a mapping")
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", ""))
        if not role or not content:
            raise ValueError("Prompt messages require non-empty role and content")
        messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError("Training row has an empty prompt")
    return tuple(messages)


class DeterministicNQHotpotLogicalView:
    """Read-only 150,745-row logical view over the immutable source parquet.

    This reproduces the historically used 40% NQ / 60% HotpotQA construction:
    all HotpotQA rows, a deterministic NQ sample, then one deterministic global
    permutation. The source parquet is never rewritten or copied.
    """

    def __init__(
        self,
        source_path: str | Path,
        *,
        selection_seed: int = 20260708,
        expected_source_rows: int = 169615,
        expected_logical_rows: int = 150745,
        expected_nq_rows: int = 60298,
        expected_hotpotqa_rows: int = 90447,
        expected_identity_sha256: str | None = None,
    ) -> None:
        import pandas as pd

        self.source_path = Path(source_path).resolve()
        if not self.source_path.is_file():
            raise FileNotFoundError(self.source_path)
        self.selection_seed = int(selection_seed)
        self._frame = pd.read_parquet(self.source_path)
        if len(self._frame) != int(expected_source_rows):
            raise RuntimeError(
                f"Source row count changed: {len(self._frame)} != {expected_source_rows}"
            )
        sources = self._frame["data_source"].astype(str)
        hotpot_indices = self._frame.index[sources.eq("hotpotqa")]
        nq_indices = self._frame.index[sources.eq("nq")]
        if len(hotpot_indices) != int(expected_hotpotqa_rows):
            raise RuntimeError(
                "HotpotQA source count changed: "
                f"{len(hotpot_indices)} != {expected_hotpotqa_rows}"
            )
        if len(nq_indices) < int(expected_nq_rows):
            raise RuntimeError("Not enough NQ rows for the frozen logical view")

        sampled_nq = (
            self._frame.loc[nq_indices]
            .sample(n=int(expected_nq_rows), random_state=self.selection_seed)
            .index.to_numpy(dtype=np.int64, copy=True)
        )
        concatenated = np.concatenate(
            (
                hotpot_indices.to_numpy(dtype=np.int64, copy=True),
                sampled_nq,
            )
        )
        order = (
            pd.Series(np.arange(concatenated.size, dtype=np.int64))
            .sample(frac=1.0, random_state=self.selection_seed)
            .to_numpy(dtype=np.int64, copy=True)
        )
        self._source_indices = concatenated[order]
        if self._source_indices.size != int(expected_logical_rows):
            raise RuntimeError(
                "Logical row count changed: "
                f"{self._source_indices.size} != {expected_logical_rows}"
            )
        if np.unique(self._source_indices).size != self._source_indices.size:
            raise RuntimeError("Logical view contains duplicate source rows")

        identity_hash = hashlib.sha256()
        identity_rows = self._frame.loc[
            self._source_indices, ["id", "data_source"]
        ]
        for row_id, data_source in identity_rows.itertuples(index=False, name=None):
            identity_hash.update(f"{row_id}\0{data_source}\n".encode("utf-8"))
        self._identity_hash = identity_hash.hexdigest()
        if (
            expected_identity_sha256
            and self._identity_hash != str(expected_identity_sha256)
        ):
            raise RuntimeError(
                "Logical dataset identity changed: "
                f"{self._identity_hash} != {expected_identity_sha256}"
            )

        self.identity = LogicalDatasetIdentity(
            source_path=str(self.source_path),
            source_rows=len(self._frame),
            logical_rows=self._source_indices.size,
            nq_rows=int(expected_nq_rows),
            hotpotqa_rows=int(expected_hotpotqa_rows),
            selection_seed=self.selection_seed,
            ordered_view_identity_sha256=self._identity_hash,
        )

    def __len__(self) -> int:
        return int(self._source_indices.size)

    def source_index(self, logical_index: int) -> int:
        index = int(logical_index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return int(self._source_indices[index])

    def row(self, logical_index: int) -> dict[str, Any]:
        source_index = self.source_index(logical_index)
        raw = self._frame.loc[source_index].to_dict()
        data_source = str(raw["data_source"])
        row_id = str(raw["id"])
        aliases = _ground_truth_aliases(raw)
        canonical_answer = _canonical_ground_truth(raw)
        return {
            **raw,
            "logical_index": int(logical_index),
            "source_index": source_index,
            "prompt_global_id": f"{data_source}:{row_id}:{source_index}",
            "prompt_messages": _prompt_messages(raw["prompt"]),
            "gold_aliases": aliases,
            "canonical_answer": canonical_answer,
        }

    def rows(self, logical_indices: Sequence[int]) -> tuple[dict[str, Any], ...]:
        return tuple(self.row(index) for index in logical_indices)
