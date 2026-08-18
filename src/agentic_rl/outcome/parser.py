from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_INFORMATION_TAG_RE = re.compile(r"</?information>", re.IGNORECASE)
_ACTION_RE = re.compile(
    r"^\s*(?:<think>(?P<think>.*?)</think>\s*)?"
    r"(?:(?:<search>(?P<search>.*?)</search>)|"
    r"(?:<answer>(?P<answer>.*?)</answer>))\s*$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ProtocolParseResult:
    valid: bool
    answer: str | None
    search_queries: tuple[str, ...]
    answer_format_indicator: int
    parser_status: str
    parser_error_type: str | None
    model_generated_information: bool
    fallback_status: str | None
    trajectory_valid: bool
    terminal_answer_valid: bool
    action_results: tuple["ActionParseResult", ...]


@dataclass(frozen=True)
class ActionParseResult:
    action_index: int
    kind: str | None
    value: str | None
    valid: bool
    error: str | None
    model_generated_information: bool


def _balanced(text: str, tag: str) -> bool:
    return (
        len(re.findall(fr"<{tag}>", text, re.IGNORECASE))
        == len(re.findall(fr"</{tag}>", text, re.IGNORECASE))
    )


def parse_model_action(action: str, *, action_index: int = 0) -> ActionParseResult:
    generated_information = bool(_INFORMATION_TAG_RE.search(action))
    if generated_information:
        return ActionParseResult(
            action_index,
            None,
            None,
            False,
            "model_generated_information",
            True,
        )
    if any(not _balanced(action, tag) for tag in ("think", "search", "answer")):
        return ActionParseResult(
            action_index, None, None, False, "unbalanced_tags", False
        )

    think_matches = list(_THINK_RE.finditer(action))
    search_matches = list(_SEARCH_RE.finditer(action))
    answer_matches = list(_ANSWER_RE.finditer(action))
    if len(think_matches) > 1 or len(search_matches) > 1 or len(answer_matches) > 1:
        return ActionParseResult(
            action_index, None, None, False, "duplicate_action_tags", False
        )
    if bool(search_matches) == bool(answer_matches):
        return ActionParseResult(
            action_index,
            None,
            None,
            False,
            "action_must_contain_exactly_one_search_or_answer",
            False,
        )
    action_match = _ACTION_RE.fullmatch(action)
    if action_match is None:
        return ActionParseResult(
            action_index,
            None,
            None,
            False,
            "invalid_tag_order_or_nesting",
            False,
        )

    if action_match.group("search") is not None:
        query = action_match.group("search").strip()
        if not query:
            return ActionParseResult(
                action_index, None, None, False, "empty_search_query", False
            )
        return ActionParseResult(
            action_index, "search", query, True, None, False
        )
    answer = action_match.group("answer").strip()
    if not answer:
        return ActionParseResult(
            action_index, None, None, False, "empty_answer", False
        )
    return ActionParseResult(action_index, "answer", answer, True, None, False)


def parse_model_trajectory(model_actions: Sequence[str]) -> ProtocolParseResult:
    if not model_actions:
        return ProtocolParseResult(
            valid=False,
            answer=None,
            search_queries=tuple(),
            answer_format_indicator=0,
            parser_status="empty",
            parser_error_type="empty_model_trajectory",
            model_generated_information=False,
            fallback_status="empty_response",
            trajectory_valid=False,
            terminal_answer_valid=False,
            action_results=tuple(),
        )

    parsed_actions = [
        parse_model_action(str(action), action_index=index)
        for index, action in enumerate(model_actions)
    ]
    searches: list[str] = []
    first_error: str | None = None
    model_generated_information = False
    answer_count = 0
    terminal_answer: str | None = None
    terminal_answer_valid = False
    seen_answer = False

    for action_index, parsed in enumerate(parsed_actions):
        model_generated_information = (
            model_generated_information or parsed.model_generated_information
        )
        if not parsed.valid:
            if first_error is None:
                first_error = parsed.error
            continue
        if parsed.kind == "search":
            if seen_answer and first_error is None:
                first_error = "search_after_answer"
            searches.append(str(parsed.value))
            continue

        answer_count += 1
        seen_answer = True
        if action_index == len(parsed_actions) - 1 and answer_count == 1:
            terminal_answer = str(parsed.value)
            terminal_answer_valid = True
        elif first_error is None:
            first_error = "answer_not_unique_final_action"

    if answer_count == 0:
        if first_error is None:
            first_error = "missing_final_answer"
        terminal_answer_valid = False
    elif answer_count > 1:
        if first_error is None:
            first_error = "multiple_answers"
        terminal_answer_valid = False
        terminal_answer = None
    if parsed_actions[-1].kind != "answer" or not parsed_actions[-1].valid:
        terminal_answer_valid = False
        terminal_answer = None
        if first_error is None:
            first_error = "answer_not_final_action"

    trajectory_valid = first_error is None and terminal_answer_valid
    return ProtocolParseResult(
        valid=trajectory_valid,
        answer=terminal_answer,
        search_queries=tuple(searches),
        answer_format_indicator=int(terminal_answer_valid),
        parser_status="valid" if trajectory_valid else "invalid",
        parser_error_type=first_error,
        model_generated_information=model_generated_information,
        fallback_status=None if trajectory_valid else "malformed_fallback",
        trajectory_valid=trajectory_valid,
        terminal_answer_valid=terminal_answer_valid,
        action_results=tuple(parsed_actions),
    )
