from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from .search_role_provenance import (
    ROLE_LOCALIZED_BRANCHES,
    ROLE_LOCALIZED_BRANCH_N_BUDGET,
    ROLE_LOCALIZED_BRANCH_N_INVALID,
    ROLE_LOCALIZED_BRANCH_N_SOFT,
    ROLE_LOCALIZED_BRANCH_NORMAL,
    ROLE_LOCALIZED_BRANCH_S_BEFORE,
)


class TokenSource(str, Enum):
    PROMPT = "prompt"
    MODEL = "model"
    ENVIRONMENT = "environment"
    PADDING = "padding"
    CODE_INSERTED = "code_inserted"


class TurnType(str, Enum):
    SEARCH = "search"
    ANSWER = "answer"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class TurnRecord:
    turn_index: int
    turn_type: TurnType
    model_text: str
    search_index: int | None = None
    query: str | None = None
    information_text: str | None = None
    parser_status: str = "valid"
    parser_error_type: str | None = None
    search_action_span_valid: bool = False
    search_prefix_valid: bool = False
    ig_reward_eligible: bool = False
    policy_credit_eligible: bool = True
    no_new_observation: bool | None = None
    exact_query_repeat: bool = False
    different_query_no_new_passage: bool = False
    current_passage_keys: tuple[str, ...] = field(default_factory=tuple)
    new_passage_keys: tuple[str, ...] = field(default_factory=tuple)
    role_localized_gate_enabled: bool = False
    retriever_executed: bool = False
    retrieval_budget_exhausted: bool = False
    model_search_invalid: bool = False
    main_credit_eligible: bool = False
    branch_type: str | None = None
    no_new_reason: str | None = None
    raw_query: str | None = None
    canonical_query: str | None = None
    new_passage_count: int = 0
    stable_passage_keys_before: tuple[str, ...] = field(default_factory=tuple)
    stable_passage_keys_after: tuple[str, ...] = field(default_factory=tuple)
    action_token_span: tuple[int, int] | None = None
    think_token_span: tuple[int, int] | None = None
    decision_token_span: tuple[int, int] | None = None
    query_token_span: tuple[int, int] | None = None
    observation_token_span: tuple[int, int] | None = None

    def validate(self, *, trajectory_system_valid: bool) -> None:
        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative")
        if self.turn_type is TurnType.SEARCH:
            if self.search_index is None or self.search_index < 0:
                raise ValueError("Search turns require a non-negative search_index")
            expected_ig_eligible = bool(
                self.search_action_span_valid
                and self.search_prefix_valid
                and trajectory_system_valid
            )
            if self.ig_reward_eligible != expected_ig_eligible:
                raise ValueError(
                    "ig_reward_eligible must equal action-span-valid AND "
                    "prefix-valid AND trajectory-system-valid"
                )
            if self.no_new_observation is False and not self.new_passage_keys:
                raise ValueError(
                    "N=0 requires at least one newly observed passage"
                )
            if self.no_new_observation is True and self.new_passage_keys:
                raise ValueError("N=1 cannot carry newly observed passages")
            if self.different_query_no_new_passage and (
                self.exact_query_repeat or self.no_new_observation is not True
            ):
                raise ValueError(
                    "different-query/no-new requires non-repeated query and N=1"
                )
        else:
            if self.search_index is not None:
                raise ValueError("Answer/fallback turns cannot have a search_index")
            if (
                self.search_action_span_valid
                or self.search_prefix_valid
                or self.ig_reward_eligible
            ):
                raise ValueError("Only Search turns can carry Exact-IG validity")
            if (
                self.no_new_observation is not None
                or self.exact_query_repeat
                or self.different_query_no_new_passage
                or self.current_passage_keys
                or self.new_passage_keys
            ):
                raise ValueError("Only Search turns can carry retrieval novelty")
        if not trajectory_system_valid and self.policy_credit_eligible:
            raise ValueError(
                "System-invalid trajectories cannot carry policy credit"
            )

    @staticmethod
    def _span_positions(
        span: tuple[int, int] | None,
        *,
        field_name: str,
        allow_empty: bool,
    ) -> set[int]:
        if span is None:
            return set()
        if len(span) != 2:
            raise ValueError(f"{field_name} must be a half-open pair")
        start, end = map(int, span)
        if start < 0 or end < start or (not allow_empty and end == start):
            raise ValueError(f"{field_name} is invalid")
        return set(range(start, end))

    def validate_role_localized_provenance(
        self,
        *,
        token_sources: Sequence[TokenSource],
        turn_ids: Sequence[int],
        policy_mask: Sequence[int],
    ) -> None:
        if not self.role_localized_gate_enabled:
            return
        if self.turn_type is not TurnType.SEARCH:
            raise ValueError("Role-localized provenance is Search-only")
        if self.branch_type not in ROLE_LOCALIZED_BRANCHES:
            raise ValueError("Role-localized Search branch_type is invalid")
        action = self._span_positions(
            self.action_token_span,
            field_name="action_token_span",
            allow_empty=False,
        )
        think = self._span_positions(
            self.think_token_span,
            field_name="think_token_span",
            allow_empty=False,
        )
        decision = self._span_positions(
            self.decision_token_span,
            field_name="decision_token_span",
            allow_empty=False,
        )
        query = self._span_positions(
            self.query_token_span,
            field_name="query_token_span",
            allow_empty=True,
        )
        observation = self._span_positions(
            self.observation_token_span,
            field_name="observation_token_span",
            allow_empty=False,
        )
        if not think and self.branch_type != ROLE_LOCALIZED_BRANCH_N_INVALID:
            raise ValueError("Role-localized Search is missing the H token span")
        if not decision.issubset(action) or not query.issubset(action):
            raise ValueError("Decision/Query spans must be inside the model action")
        if think and not think.issubset(action):
            raise ValueError("Think span must be inside the model action")
        if decision & query or decision & think or query & think:
            raise ValueError("H/D/Q token spans must be disjoint")
        token_count = len(token_sources)
        if any(position >= token_count for position in action | observation):
            raise ValueError("Role-localized token span is out of bounds")
        for position in action:
            if (
                token_sources[position] is not TokenSource.MODEL
                or int(turn_ids[position]) != int(self.turn_index)
            ):
                raise ValueError("Search action span contains non-model provenance")
        for position in observation:
            if (
                token_sources[position] is not TokenSource.ENVIRONMENT
                or int(turn_ids[position]) != -1
                or int(policy_mask[position]) != 0
            ):
                raise ValueError("Observation entered policy provenance")
        if self.retriever_executed != bool(observation):
            raise ValueError("Retriever execution and Observation span disagree")
        if self.new_passage_count != len(self.new_passage_keys):
            raise ValueError("new_passage_count does not match stable keys")
        before = set(self.stable_passage_keys_before)
        after = set(self.stable_passage_keys_after)
        if not before.issubset(after):
            raise ValueError("Stable passage history is not monotonic")
        if set(self.new_passage_keys) != after - before:
            raise ValueError("Persisted new passage keys do not match history delta")

        if self.branch_type == ROLE_LOCALIZED_BRANCH_N_BUDGET:
            if not (
                self.retrieval_budget_exhausted
                and not self.retriever_executed
                and not self.model_search_invalid
                and not self.main_credit_eligible
                and self.no_new_observation is True
            ):
                raise ValueError("N_budget facts are inconsistent")
        elif self.branch_type == ROLE_LOCALIZED_BRANCH_N_INVALID:
            if not (
                self.model_search_invalid
                and not self.retriever_executed
                and not self.main_credit_eligible
            ):
                raise ValueError("N_invalid facts are inconsistent")
        elif self.branch_type == ROLE_LOCALIZED_BRANCH_S_BEFORE:
            if self.main_credit_eligible:
                raise ValueError("S_before cannot carry Main credit")
        elif self.branch_type == ROLE_LOCALIZED_BRANCH_N_SOFT:
            if not (
                self.retriever_executed
                and self.no_new_observation is True
                and self.ig_reward_eligible
                and self.policy_credit_eligible
                and self.main_credit_eligible
            ):
                raise ValueError("N_soft facts are inconsistent")
        elif self.branch_type == ROLE_LOCALIZED_BRANCH_NORMAL:
            if not (
                self.retriever_executed
                and self.no_new_observation is False
                and self.ig_reward_eligible
                and self.policy_credit_eligible
                and self.main_credit_eligible
            ):
                raise ValueError("Normal Search facts are inconsistent")


@dataclass
class TrajectoryRecord:
    prompt_global_id: str
    trajectory_id: str
    input_ids: list[int]
    token_sources: list[TokenSource]
    turn_ids: list[int]
    turns: list[TurnRecord]
    search_prefix_end_positions: list[int]
    search_prefix_before_search_end_positions: dict[int, int] = field(
        default_factory=dict
    )
    old_action_logprobs: list[float] = field(default_factory=list)
    sampled_action_logprobs: list[float] = field(default_factory=list)
    immediate_ig: dict[int, float] = field(default_factory=dict)
    task_outcome: float = 0.0
    answer_format_indicator: int = 0
    terminal_answer_valid: bool = False
    trajectory_protocol_valid: bool = False
    trajectory_system_valid: bool = True
    parser_status: str = "valid"
    parser_error_type: str | None = None
    fallback_status: str | None = None
    environment_failure_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def action_token_mask(self) -> list[int]:
        return [int(source is TokenSource.MODEL) for source in self.token_sources]

    @property
    def action_token_count(self) -> int:
        return sum(self.action_token_mask)

    @property
    def policy_credit_mask(self) -> list[int]:
        terminal_turn = self.terminal_policy_credit_turn_index
        credit_by_turn = {}
        for turn in self.turns:
            if turn.turn_type is TurnType.SEARCH:
                credit_by_turn[turn.turn_index] = bool(
                    turn.policy_credit_eligible
                )
            else:
                # Exactly one real model-generated terminal action may receive
                # Answer/fallback credit. Earlier terminal-like records and
                # code/environment fallbacks never inherit that credit.
                credit_by_turn[turn.turn_index] = bool(
                    turn.policy_credit_eligible
                    and terminal_turn is not None
                    and turn.turn_index == terminal_turn
                )
        return [
            int(
                source is TokenSource.MODEL
                and self.trajectory_system_valid
                and credit_by_turn.get(turn_id, False)
            )
            for source, turn_id in zip(self.token_sources, self.turn_ids)
        ]

    @property
    def policy_mask(self) -> list[int]:
        """Policy-loss mask after provenance and turn eligibility are applied."""
        return self.policy_credit_mask

    @property
    def kl_mask(self) -> list[int]:
        """KL uses exactly the same eligible model-token states as policy loss."""
        return self.policy_credit_mask

    @property
    def trajectory_valid(self) -> bool:
        """Full-protocol diagnostic; never reused as a channel validity mask."""
        return bool(self.trajectory_protocol_valid and self.trajectory_system_valid)

    @property
    def outcome_reward_eligible(self) -> bool:
        return bool(self.terminal_answer_valid and self.trajectory_system_valid)

    @property
    def terminal_policy_credit_turn_index(self) -> int | None:
        model_turns = {
            turn_id
            for source, turn_id in zip(self.token_sources, self.turn_ids)
            if source is TokenSource.MODEL
        }
        terminal = [
            turn.turn_index
            for turn in self.turns
            if turn.turn_type in {TurnType.ANSWER, TurnType.FALLBACK}
            and turn.policy_credit_eligible
            and turn.turn_index in model_turns
        ]
        return terminal[-1] if terminal else None

    @property
    def optimization_ready(self) -> bool:
        return bool(self.trajectory_system_valid and sum(self.policy_credit_mask) > 0)

    @property
    def ig_reward_eligibility_by_search_index(self) -> dict[int, bool]:
        return {
            int(turn.search_index): bool(turn.ig_reward_eligible)
            for turn in self.turns
            if turn.turn_type is TurnType.SEARCH
            and turn.search_index is not None
        }

    @property
    def policy_credit_eligibility_by_search_index(self) -> dict[int, bool]:
        return {
            int(turn.search_index): bool(turn.policy_credit_eligible)
            for turn in self.turns
            if turn.turn_type is TurnType.SEARCH
            and turn.search_index is not None
        }

    @property
    def search_turn_count(self) -> int:
        return sum(turn.turn_type is TurnType.SEARCH for turn in self.turns)

    def prefix_token_ids_before_search(self, search_index: int) -> tuple[int, ...]:
        """Slice an unmodified pre-Search prefix from the original token IDs."""

        index = int(search_index)
        if index not in self.search_prefix_before_search_end_positions:
            raise ValueError(
                f"{self.trajectory_id}: missing pre-Search prefix {index}"
            )
        endpoint = int(self.search_prefix_before_search_end_positions[index])
        if endpoint <= 0 or endpoint > len(self.input_ids):
            raise ValueError(
                f"{self.trajectory_id}: invalid pre-Search endpoint {endpoint}"
            )
        return tuple(int(value) for value in self.input_ids[:endpoint])

    def prefix_token_ids_after_search_observation(
        self,
        search_index: int,
    ) -> tuple[int, ...]:
        """Slice state after Search and its full Retriever observation."""

        index = int(search_index)
        endpoint_index = index + 1
        if endpoint_index >= len(self.search_prefix_end_positions):
            raise ValueError(
                f"{self.trajectory_id}: missing post-Search prefix {index}"
            )
        endpoint = int(self.search_prefix_end_positions[endpoint_index])
        if endpoint <= 0 or endpoint > len(self.input_ids):
            raise ValueError(
                f"{self.trajectory_id}: invalid post-Search endpoint {endpoint}"
            )
        before = int(self.search_prefix_before_search_end_positions[index])
        if endpoint <= before:
            raise ValueError(
                f"{self.trajectory_id}: post-Search prefix does not extend state"
            )
        return tuple(int(value) for value in self.input_ids[:endpoint])

    def validate(self) -> None:
        token_count = len(self.input_ids)
        if len(self.token_sources) != token_count or len(self.turn_ids) != token_count:
            raise ValueError("input_ids, token_sources, and turn_ids must be aligned")
        if self.old_action_logprobs and len(self.old_action_logprobs) != token_count:
            raise ValueError("old_action_logprobs must align with input_ids")
        if self.sampled_action_logprobs and len(self.sampled_action_logprobs) != token_count:
            raise ValueError("sampled_action_logprobs must align with input_ids")

        turn_indices = [turn.turn_index for turn in self.turns]
        if len(set(turn_indices)) != len(turn_indices):
            raise ValueError("Turn indices must be unique")
        if turn_indices != sorted(turn_indices):
            raise ValueError("Turns must be ordered by turn_index")
        known_turns = set(turn_indices)
        for turn in self.turns:
            turn.validate(trajectory_system_valid=self.trajectory_system_valid)
        for source, turn_id in zip(self.token_sources, self.turn_ids):
            if source is TokenSource.MODEL:
                if turn_id < 0 or turn_id not in known_turns:
                    raise ValueError("Every sampled model token must belong to a turn")
            elif turn_id != -1:
                raise ValueError("Non-model tokens must use turn_id=-1")
        resolved_policy_mask = self.policy_credit_mask
        for turn in self.turns:
            turn.validate_role_localized_provenance(
                token_sources=self.token_sources,
                turn_ids=self.turn_ids,
                policy_mask=resolved_policy_mask,
            )

        if self.input_ids and any(source is TokenSource.MODEL for source in self.token_sources):
            if self.action_token_count == 0:
                raise ValueError("A non-empty sampled response cannot have zero action tokens")
        search_indices = [
            int(turn.search_index)
            for turn in self.turns
            if turn.turn_type is TurnType.SEARCH
            and turn.search_index is not None
        ]
        if search_indices != list(range(len(search_indices))):
            raise ValueError("Search indices must be contiguous and zero-based")
        if not self.search_prefix_end_positions:
            raise ValueError("Exact-IG prefix endpoints cannot be empty")
        if any(
            int(endpoint) <= 0 or int(endpoint) > token_count
            for endpoint in self.search_prefix_end_positions
        ):
            raise ValueError("Exact-IG prefix endpoints are out of bounds")
        if list(self.search_prefix_end_positions) != sorted(
            int(value) for value in self.search_prefix_end_positions
        ):
            raise ValueError("Exact-IG prefix endpoints must be monotonic")
        if set(self.search_prefix_before_search_end_positions) != set(search_indices):
            raise ValueError(
                "Pre-Search prefix endpoints must cover every real Search turn"
            )
        previous_pre_search = 0
        for turn in (
            item for item in self.turns if item.turn_type is TurnType.SEARCH
        ):
            search_index = int(turn.search_index)
            endpoint = int(
                self.search_prefix_before_search_end_positions[search_index]
            )
            if endpoint <= 0 or endpoint > token_count:
                raise ValueError("Pre-Search prefix endpoint is out of bounds")
            if endpoint < previous_pre_search:
                raise ValueError("Pre-Search prefix endpoints must be monotonic")
            model_positions = [
                position
                for position, (source, turn_id) in enumerate(
                    zip(self.token_sources, self.turn_ids, strict=True)
                )
                if source is TokenSource.MODEL
                and int(turn_id) == int(turn.turn_index)
            ]
            if not model_positions or endpoint != min(model_positions):
                raise ValueError(
                    "Pre-Search prefix must end immediately before its model action"
                )
            if not self.prefix_token_ids_before_search(search_index):
                raise ValueError("Pre-Search prefix cannot be empty")
            previous_pre_search = endpoint
        eligibility = self.ig_reward_eligibility_by_search_index
        for index, value in self.immediate_ig.items():
            if index not in eligibility:
                raise ValueError(f"Invalid Exact-IG search index: {index}")
            if not eligibility[index]:
                raise ValueError("Ineligible Search turns cannot carry Exact-IG")
            if not isinstance(value, (int, float)):
                raise ValueError("Exact-IG values must be numeric")

        prefix_chain_open = True
        for turn in (
            item for item in self.turns if item.turn_type is TurnType.SEARCH
        ):
            if not prefix_chain_open and turn.search_prefix_valid:
                raise ValueError(
                    "A Search prefix cannot become valid after prefix reconstruction failed"
                )
            prefix_chain_open = prefix_chain_open and turn.search_prefix_valid

        termination_reason = str(self.metadata.get("termination_reason", ""))
        if termination_reason == "maximum_search_turns_reached":
            budget_exhausted = [
                int(turn.search_index)
                for turn in self.turns
                if turn.turn_type is TurnType.SEARCH
                and turn.search_index is not None
                and is_budget_exhausted_terminal_search(self, int(turn.search_index))
            ]
            if not search_indices or budget_exhausted != [search_indices[-1]]:
                raise ValueError(
                    "maximum_search_turns_reached requires exactly one terminal "
                    "budget-exhausted Search"
                )


@dataclass(frozen=True)
class PromptTrajectoryGroup:
    prompt_global_id: str
    trajectories: tuple[TrajectoryRecord, ...]
    aliases: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, expected_group_size: int) -> None:
        if len(self.trajectories) != expected_group_size:
            raise ValueError(
                f"{self.prompt_global_id}: expected {expected_group_size} trajectories, "
                f"got {len(self.trajectories)}"
            )
        trajectory_ids = {trajectory.trajectory_id for trajectory in self.trajectories}
        if len(trajectory_ids) != len(self.trajectories):
            raise ValueError(f"{self.prompt_global_id}: duplicate trajectory IDs")
        for trajectory in self.trajectories:
            if trajectory.prompt_global_id != self.prompt_global_id:
                raise ValueError("Trajectory prompt_global_id mismatch")
            trajectory.validate()


def as_token_sources(values: Sequence[str | TokenSource]) -> list[TokenSource]:
    return [
        value if isinstance(value, TokenSource) else TokenSource(str(value))
        for value in values
    ]


def is_budget_exhausted_terminal_search(
    record: Any,
    search_index: int,
) -> bool:
    """Identify a real terminal Search emitted after retrieval budget exhaustion."""

    metadata = getattr(record, "metadata", {})
    if not isinstance(metadata, Mapping) or str(
        metadata.get("termination_reason", "")
    ) != "maximum_search_turns_reached":
        return False
    turns = list(getattr(record, "turns", ()))
    index = int(search_index)
    matching = [
        turn
        for turn in turns
        if turn.turn_type is TurnType.SEARCH
        and turn.search_index is not None
        and int(turn.search_index) == index
    ]
    if len(matching) != 1 or not turns or matching[0] is not turns[-1]:
        return False
    turn = matching[0]
    if not (
        turn.search_action_span_valid
        and not turn.search_prefix_valid
        and not turn.ig_reward_eligible
        and turn.policy_credit_eligible
        and turn.no_new_observation is True
        and turn.information_text is None
        and not turn.current_passage_keys
        and not turn.new_passage_keys
    ):
        return False
    prefix_end_positions = list(
        getattr(record, "search_prefix_end_positions", ())
    )
    if index + 1 < len(prefix_end_positions):
        return False
    pre_end_positions = getattr(
        record,
        "search_prefix_before_search_end_positions",
        {},
    )
    if not isinstance(pre_end_positions, Mapping) or index not in pre_end_positions:
        return False
    input_ids = list(getattr(record, "input_ids", ()))
    token_sources = list(getattr(record, "token_sources", ()))
    turn_ids = list(getattr(record, "turn_ids", ()))
    if not input_ids or not (
        len(input_ids) == len(token_sources) == len(turn_ids)
    ):
        return False
    model_positions = [
        position
        for position, (source, turn_id) in enumerate(
            zip(token_sources, turn_ids, strict=True)
        )
        if source is TokenSource.MODEL
        and int(turn_id) == int(turn.turn_index)
    ]
    return bool(
        model_positions
        and min(model_positions) == int(pre_end_positions[index])
        and max(model_positions) + 1 == len(input_ids)
    )
