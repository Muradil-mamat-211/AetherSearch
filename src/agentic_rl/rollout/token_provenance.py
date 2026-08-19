from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .trajectory_schema import TokenSource


@dataclass(frozen=True)
class ProvenanceAssignment:
    token_sources: tuple[TokenSource, ...]
    turn_ids: tuple[int, ...]
    action_mask: tuple[int, ...]
    fallback_turn_index: int | None
    unmatched_model_token_count: int


def assign_model_turns_with_fallback(
    token_sources: Sequence[TokenSource],
    parsed_turn_ids: Sequence[int | None],
    *,
    next_fallback_turn_index: int,
) -> ProvenanceAssignment:
    if len(token_sources) != len(parsed_turn_ids):
        raise ValueError("token_sources and parsed_turn_ids must have equal length")

    unmatched = sum(
        source is TokenSource.MODEL and turn_id is None
        for source, turn_id in zip(token_sources, parsed_turn_ids)
    )
    fallback = next_fallback_turn_index if unmatched else None
    resolved: list[int] = []
    action_mask: list[int] = []
    for source, parsed_turn_id in zip(token_sources, parsed_turn_ids):
        if source is TokenSource.MODEL:
            resolved.append(
                int(parsed_turn_id)
                if parsed_turn_id is not None
                else int(next_fallback_turn_index)
            )
            action_mask.append(1)
        else:
            resolved.append(-1)
            action_mask.append(0)
    return ProvenanceAssignment(
        token_sources=tuple(token_sources),
        turn_ids=tuple(resolved),
        action_mask=tuple(action_mask),
        fallback_turn_index=fallback,
        unmatched_model_token_count=int(unmatched),
    )


def assert_environment_information_masked(
    token_sources: Sequence[TokenSource],
    action_mask: Sequence[int],
) -> None:
    if len(token_sources) != len(action_mask):
        raise ValueError("token_sources and action_mask must align")
    for source, mask in zip(token_sources, action_mask):
        expected = int(source is TokenSource.MODEL)
        if int(mask) != expected:
            raise ValueError(
                f"Invalid action mask for token source {source.value}: "
                f"expected {expected}, got {mask}"
            )


def build_policy_credit_mask(
    token_sources: Sequence[TokenSource],
    turn_ids: Sequence[int],
    policy_credit_eligible_by_turn: dict[int, bool],
    *,
    trajectory_system_valid: bool,
) -> tuple[int, ...]:
    """Combine immutable token provenance with explicit policy eligibility.

    Parser diagnostics are deliberately absent from this API: a malformed flag
    cannot implicitly add, subtract, or erase otherwise eligible policy credit.
    """
    if len(token_sources) != len(turn_ids):
        raise ValueError("token_sources and turn_ids must align")
    mask: list[int] = []
    for source, turn_id in zip(token_sources, turn_ids):
        if source is not TokenSource.MODEL:
            if turn_id != -1:
                raise ValueError("Non-model tokens must use turn_id=-1")
            mask.append(0)
            continue
        if turn_id < 0 or turn_id not in policy_credit_eligible_by_turn:
            raise ValueError("Every model token must map to a known policy turn")
        mask.append(
            int(
                trajectory_system_valid
                and policy_credit_eligible_by_turn[int(turn_id)]
            )
        )
    return tuple(mask)
