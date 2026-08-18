from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agentic_rl.exact_ig.target_schema import ANSWER_SCAFFOLD_TEXT

from .parser import ProtocolParseResult, parse_model_trajectory
from .token_f1 import (
    IGPO_OFFICIAL_COMMIT,
    max_alias_exact_match,
    max_alias_token_f1,
)


PRODUCTION_TASK_SCORER_VERSION = (
    "production_task_scorer_max_alias_igpo_set_f1_"
    + IGPO_OFFICIAL_COMMIT
)
SUFFICIENCY_EXACT_SCORER_VERSION = (
    PRODUCTION_TASK_SCORER_VERSION + "_alias_exact_sufficiency_v1"
)


@dataclass(frozen=True)
class OutcomeResult:
    task_outcome: float
    format_indicator: int
    valid_for_selection: bool
    terminal_answer_valid: bool
    trajectory_system_valid: bool
    parse: ProtocolParseResult


def score_trajectory_outcome(
    model_actions: Sequence[str],
    aliases: Sequence[str],
    *,
    data_source: str = "",
    trajectory_system_valid: bool = True,
) -> OutcomeResult:
    parsed = parse_model_trajectory(model_actions)
    outcome_eligible = bool(
        parsed.terminal_answer_valid and trajectory_system_valid
    )
    task_outcome = (
        max_alias_token_f1(parsed.answer, aliases, data_source=data_source)
        if outcome_eligible and parsed.answer is not None
        else 0.0
    )
    return OutcomeResult(
        task_outcome=float(task_outcome),
        format_indicator=parsed.answer_format_indicator,
        valid_for_selection=outcome_eligible,
        terminal_answer_valid=parsed.terminal_answer_valid,
        trajectory_system_valid=bool(trajectory_system_valid),
        parse=parsed,
    )


def score_stop_answer_completion(
    completion_text: str,
    aliases: Sequence[str],
    *,
    data_source: str = "",
) -> OutcomeResult:
    """Score one detached Stop answer with the production trajectory scorer."""

    action = ANSWER_SCAFFOLD_TEXT + str(completion_text)
    return score_trajectory_outcome(
        [action],
        aliases,
        data_source=data_source,
        trajectory_system_valid=True,
    )


def score_sufficiency_probe_completion(
    completion_text: str,
    aliases: Sequence[str],
    *,
    data_source: str = "",
    truncated: bool = False,
) -> dict[str, object]:
    """Score one deterministic Answer-now probe without changing R_task.

    The ordinary production task score is retained as shadow telemetry.  The
    hard sufficiency bit uses only parser-valid, non-truncated alias-aware EM.
    """

    scored = score_stop_answer_completion(
        completion_text,
        aliases,
        data_source=data_source,
    )
    exact = bool(
        scored.terminal_answer_valid
        and scored.parse.answer is not None
        and max_alias_exact_match(
            scored.parse.answer,
            aliases,
            data_source=data_source,
        )
        == 1.0
    )
    parser_success = bool(scored.terminal_answer_valid)
    no_answer = bool(
        scored.parse.answer is None or not str(scored.parse.answer).strip()
    )
    sufficient = bool(
        parser_success
        and not no_answer
        and not bool(truncated)
        and exact
    )
    return {
        "sufficient_before_search": sufficient,
        "parser_success": parser_success,
        "no_answer": no_answer,
        "output_truncated": bool(truncated),
        "alias_aware_exact": exact,
        "raw_task_reward": float(scored.task_outcome),
        "alias_exact_match": exact,
        "partial_task_reward_shadow": float(scored.task_outcome),
        "format_indicator": int(scored.format_indicator),
        "terminal_answer_valid": bool(scored.terminal_answer_valid),
        "parser_status": str(scored.parse.parser_status),
        "parser_error_type": scored.parse.parser_error_type,
        "parsed_answer": scored.parse.answer,
        "truncated": bool(truncated),
        "scorer_version": SUFFICIENCY_EXACT_SCORER_VERSION,
        "task_scorer_version": PRODUCTION_TASK_SCORER_VERSION,
    }
