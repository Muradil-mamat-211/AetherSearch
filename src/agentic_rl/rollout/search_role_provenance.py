from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence


ROLE_LOCALIZED_BRANCH_N_BUDGET = "n_budget"
ROLE_LOCALIZED_BRANCH_N_INVALID = "n_invalid"
ROLE_LOCALIZED_BRANCH_S_BEFORE = "s_before"
ROLE_LOCALIZED_BRANCH_N_SOFT = "n_soft"
ROLE_LOCALIZED_BRANCH_NORMAL = "normal"
ROLE_LOCALIZED_BRANCHES = frozenset(
    {
        ROLE_LOCALIZED_BRANCH_N_BUDGET,
        ROLE_LOCALIZED_BRANCH_N_INVALID,
        ROLE_LOCALIZED_BRANCH_S_BEFORE,
        ROLE_LOCALIZED_BRANCH_N_SOFT,
        ROLE_LOCALIZED_BRANCH_NORMAL,
    }
)

_THINK_SPAN_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_SEARCH_OPEN_RE = re.compile(r"<search>", re.IGNORECASE)
_SEARCH_CLOSE_RE = re.compile(r"</search>", re.IGNORECASE)


@dataclass(frozen=True)
class SearchRoleTokenSpans:
    """Half-open absolute token spans captured while the action is generated."""

    action: tuple[int, int]
    think: tuple[int, int] | None
    decision: tuple[int, int]
    query: tuple[int, int]

    def as_dict(self) -> dict[str, list[int] | None]:
        return {
            "action_token_span": list(self.action),
            "think_token_span": None if self.think is None else list(self.think),
            "decision_token_span": list(self.decision),
            "query_token_span": list(self.query),
        }


def action_has_search_intent(action_text: str) -> bool:
    """Recognize an exact protocol opening tag without fuzzy matching."""

    return _SEARCH_OPEN_RE.search(str(action_text)) is not None


def exact_search_payload_text(
    action_text: str,
    *,
    allow_incomplete_search: bool = False,
) -> str:
    """Return the generated payload without fuzzy post-hoc reconstruction."""

    text = str(action_text)
    openings = list(_SEARCH_OPEN_RE.finditer(text))
    if not openings:
        raise ValueError("Expected at least one Search opening tag")
    if len(openings) != 1 and not allow_incomplete_search:
        raise ValueError("Expected exactly one Search opening tag")

    opening = openings[0]
    closing = _SEARCH_CLOSE_RE.search(text, opening.end())
    if closing is None:
        if not allow_incomplete_search:
            raise ValueError("Expected one complete Search payload")
        return text[opening.end() :]
    if not allow_incomplete_search and _SEARCH_CLOSE_RE.search(text, closing.end()):
        raise ValueError("Expected exactly one Search closing tag")
    return text[opening.end() : closing.start()]


def _generated_token_byte_offsets(
    tokenizer: Any,
    token_ids: Sequence[int],
    expected_text: str,
) -> tuple[tuple[int, int], ...]:
    """Recover decoded-text byte offsets from immutable Qwen token IDs.

    ByteLevel vocab entries represent arbitrary bytes. A generated sequence may
    therefore contain an incomplete or invalid UTF-8 sequence which the Qwen
    decoder renders as U+FFFD. Preserve the original token ownership while
    projecting those raw bytes through the same lossy UTF-8 semantics.
    """

    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

    ids = tuple(int(value) for value in token_ids)
    special_ids = set(map(int, getattr(tokenizer, "all_special_ids", ())))
    pieces = tokenizer.convert_ids_to_tokens(
        list(ids),
        skip_special_tokens=False,
    )
    if not isinstance(pieces, Sequence) or len(pieces) != len(ids):
        raise ValueError("Generated Search token/piece cardinality mismatch")
    byte_decoder = {
        character: value for value, character in bytes_to_unicode().items()
    }
    chunks: list[bytes] = []
    raw_offsets: list[tuple[int, int] | None] = []
    raw_cursor = 0
    trailing_special_started = False
    for token_index, piece in enumerate(pieces):
        token_id = ids[token_index]
        if token_id in special_ids:
            decoded = tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if decoded:
                raise ValueError(
                    "Search action contains a visible special token"
                )
            trailing_special_started = True
            raw_offsets.append(None)
            continue
        if trailing_special_started:
            raise ValueError(
                "Search action contains a non-special token after its special-token suffix"
            )
        try:
            chunk = bytes(byte_decoder[character] for character in str(piece))
        except KeyError as error:
            raise ValueError(
                f"Generated Search token {token_index} is not ByteLevel reversible"
            ) from error
        chunks.append(chunk)
        raw_offsets.append((raw_cursor, raw_cursor + len(chunk)))
        raw_cursor += len(chunk)
    raw = b"".join(chunks)

    # Decode strict runs and invalid runs separately so every decoded Unicode
    # scalar retains the exact source-byte interval that produced it. This also
    # handles a valid multi-byte scalar split across two or more model tokens.
    decoded_units: list[tuple[int, int, str]] = []
    cursor = 0
    while cursor < len(raw):
        remainder = raw[cursor:]
        try:
            valid_text = remainder.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            valid_end = cursor + int(error.start)
            valid_bytes = raw[cursor:valid_end]
            valid_text = valid_bytes.decode("utf-8", errors="strict")
            valid_cursor = cursor
            for character in valid_text:
                width = len(character.encode("utf-8"))
                decoded_units.append(
                    (valid_cursor, valid_cursor + width, character)
                )
                valid_cursor += width
            invalid_start = cursor + int(error.start)
            invalid_end = cursor + int(error.end)
            if invalid_end <= invalid_start:
                raise ValueError("UTF-8 decoder reported an empty invalid span")
            decoded_units.append((invalid_start, invalid_end, "\ufffd"))
            cursor = invalid_end
        else:
            valid_cursor = cursor
            for character in valid_text:
                width = len(character.encode("utf-8"))
                decoded_units.append(
                    (valid_cursor, valid_cursor + width, character)
                )
                valid_cursor += width
            cursor = len(raw)

    decoded_text = "".join(unit[2] for unit in decoded_units)
    if decoded_text != str(expected_text):
        raise ValueError(
            "Generated Search action bytes do not round-trip to exact decoded text"
        )

    decoded_offsets: list[tuple[int, int, int, int]] = []
    decoded_cursor = 0
    for raw_start, raw_end, character in decoded_units:
        decoded_width = len(character.encode("utf-8"))
        decoded_offsets.append(
            (raw_start, raw_end, decoded_cursor, decoded_cursor + decoded_width)
        )
        decoded_cursor += decoded_width

    offsets: list[tuple[int, int]] = []
    for raw_span in raw_offsets:
        if raw_span is None:
            offsets.append((decoded_cursor, decoded_cursor))
            continue
        raw_start, raw_end = raw_span
        owned = [
            unit
            for unit in decoded_offsets
            if unit[0] < raw_end and unit[1] > raw_start
        ]
        if not owned:
            raise ValueError("Generated Search token maps to no decoded text")
        offsets.append((owned[0][2], owned[-1][3]))
    return tuple(offsets)


def _covering_token_span_for_character_span(
    text: str,
    byte_offsets: Sequence[Sequence[int]],
    character_span: tuple[int, int],
    *,
    field_name: str,
) -> tuple[int, int]:
    char_start, char_end = map(int, character_span)
    if char_start < 0 or char_end < char_start:
        raise ValueError(f"{field_name} character span is invalid")
    byte_start = len(text[:char_start].encode("utf-8"))
    byte_end = len(text[:char_end].encode("utf-8"))

    if byte_start == byte_end:
        insertion = len(byte_offsets)
        for token_index, pair in enumerate(byte_offsets):
            token_start, token_end = map(int, pair)
            if token_start >= byte_start:
                insertion = token_index
                break
            if token_start < byte_start < token_end:
                insertion = token_index + 1
                break
        return insertion, insertion

    selected: list[int] = []
    for token_index, pair in enumerate(byte_offsets):
        token_start, token_end = map(int, pair)
        if token_end <= token_start:
            continue
        if token_start < byte_end and token_end > byte_start:
            selected.append(token_index)
    if not selected:
        raise ValueError(f"{field_name} character span maps to no tokens")
    expected = list(range(selected[0], selected[-1] + 1))
    if selected != expected:
        raise ValueError(f"{field_name} token span is not contiguous")
    return selected[0], selected[-1] + 1



def _contained_token_span_for_character_span(
    text: str,
    byte_offsets: Sequence[Sequence[int]],
    character_span: tuple[int, int],
    *,
    field_name: str,
) -> tuple[int, int]:
    """Select only indivisible tokens fully owned by a semantic text span."""

    char_start, char_end = map(int, character_span)
    if char_start < 0 or char_end < char_start:
        raise ValueError(f"{field_name} character span is invalid")
    byte_start = len(text[:char_start].encode("utf-8"))
    byte_end = len(text[:char_end].encode("utf-8"))
    selected = [
        token_index
        for token_index, pair in enumerate(byte_offsets)
        if int(pair[1]) > int(pair[0])
        and int(pair[0]) >= byte_start
        and int(pair[1]) <= byte_end
    ]
    if selected:
        expected = list(range(selected[0], selected[-1] + 1))
        if selected != expected:
            raise ValueError(f"{field_name} token span is not contiguous")
        return selected[0], selected[-1] + 1

    insertion = len(byte_offsets)
    for token_index, pair in enumerate(byte_offsets):
        if int(pair[0]) >= byte_start:
            insertion = token_index
            break
    return insertion, insertion

def build_generation_time_search_role_spans(
    tokenizer: Any,
    *,
    action_token_ids: Sequence[int],
    action_text: str,
    absolute_action_start: int,
    allow_incomplete_search: bool = False,
) -> SearchRoleTokenSpans:
    """Freeze H/D/Q spans directly from the original generated token IDs.

    Decoded text is used only for the exact protocol parser. Token boundaries
    come from a reversible byte projection of the immutable vLLM token IDs.
    """

    token_ids = tuple(int(value) for value in action_token_ids)
    if not token_ids:
        raise ValueError("Search action token IDs cannot be empty")
    text = str(action_text)
    opening_matches = list(_SEARCH_OPEN_RE.finditer(text))
    if not opening_matches:
        raise ValueError(
            "Role-localized Search requires a <search> opening tag"
        )
    if len(opening_matches) != 1 and not allow_incomplete_search:
        raise ValueError(
            "Role-localized valid Search requires exactly one <search> opening tag"
        )
    opening_match = opening_matches[0]
    closing_match = _SEARCH_CLOSE_RE.search(text, opening_match.end())
    if closing_match is not None:
        if (
            not allow_incomplete_search
            and _SEARCH_CLOSE_RE.search(text, closing_match.end()) is not None
        ):
            raise ValueError(
                "Role-localized valid Search requires exactly one </search> closing tag"
            )
        query_character_span = (opening_match.end(), closing_match.start())
    elif allow_incomplete_search:
        query_character_span = (opening_match.end(), len(text))
    else:
        raise ValueError(
            "Role-localized valid Search requires one complete <search> span"
        )

    think_matches = list(_THINK_SPAN_RE.finditer(text))
    if len(think_matches) != 1 and not allow_incomplete_search:
        raise ValueError(
            "Role-localized valid Search requires exactly one complete Think span"
        )

    byte_offsets = _generated_token_byte_offsets(tokenizer, token_ids, text)
    decision_local = _covering_token_span_for_character_span(
        text,
        byte_offsets,
        opening_match.span(0),
        field_name="decision",
    )
    query_local = _contained_token_span_for_character_span(
        text,
        byte_offsets,
        query_character_span,
        field_name="query",
    )
    think_local = (
        None
        if len(think_matches) != 1
        else _contained_token_span_for_character_span(
            text,
            byte_offsets,
            think_matches[0].span(0),
            field_name="think",
        )
    )
    think_tokens = set() if think_local is None else set(range(*think_local))
    decision_tokens = set(range(*decision_local))
    query_tokens = set(range(*query_local))
    if not decision_tokens or (not think_tokens and not allow_incomplete_search):
        raise ValueError("Think/Decision native token provenance is empty")
    if query_character_span[0] < query_character_span[1] and not query_tokens:
        raise ValueError("Non-empty Search payload has no independent Query token")
    if (
        think_tokens & decision_tokens
        or think_tokens & query_tokens
        or decision_tokens & query_tokens
    ):
        raise ValueError("H/D/Q native token spans overlap")
    start = int(absolute_action_start)
    if start < 0:

        raise ValueError("absolute_action_start must be non-negative")

    def absolute(span: tuple[int, int]) -> tuple[int, int]:
        return start + span[0], start + span[1]

    return SearchRoleTokenSpans(
        action=(start, start + len(token_ids)),
        think=None if think_local is None else absolute(think_local),
        decision=absolute(decision_local),
        query=absolute(query_local),
    )


def _contiguous_local_span(token_indices: set[int]) -> tuple[int, int]:
    """Return a span only when native token ownership is contiguous."""

    if not token_indices:
        return (0, 0)
    ordered = sorted(token_indices)
    expected = list(range(ordered[0], ordered[-1] + 1))
    if ordered != expected:
        return (0, 0)
    return ordered[0], ordered[-1] + 1


def build_invalid_search_role_spans(
    tokenizer: Any,
    *,
    action_token_ids: Sequence[int],
    action_text: str,
    absolute_action_start: int,
) -> SearchRoleTokenSpans:
    """Best-effort native spans for an already-invalid Search action.

    Invalid protocol output must not abort the rollout merely because a
    tokenizer boundary makes H/D/Q ownership ambiguous. The complete model
    action is retained. We keep only independently owned D/Q tokens; an
    ambiguous Query is represented by an empty span, so the learner applies
    the Decision credit without fabricating Query provenance. This helper is
    intentionally not used for valid Search actions.
    """

    token_ids = tuple(int(value) for value in action_token_ids)
    if not token_ids:
        raise ValueError("Invalid Search action token IDs cannot be empty")
    start = int(absolute_action_start)
    if start < 0:
        raise ValueError("absolute_action_start must be non-negative")
    action = (start, start + len(token_ids))
    text = str(action_text)
    openings = list(_SEARCH_OPEN_RE.finditer(text))
    if not openings:
        # The caller only invokes this helper after search intent was found;
        # retaining the action as Decision is the fail-closed last resort.
        return SearchRoleTokenSpans(
            action=action,
            think=None,
            decision=action,
            query=(start, start),
        )

    try:
        byte_offsets = _generated_token_byte_offsets(tokenizer, token_ids, text)
    except ValueError:
        # Without a reversible native projection no finer role split is
        # defensible. Do not invent a Query span or drop model action tokens.
        return SearchRoleTokenSpans(
            action=action,
            think=None,
            decision=action,
            query=(start, start),
        )

    opening = openings[0]
    closing = _SEARCH_CLOSE_RE.search(text, opening.end())
    query_char_span = (
        (opening.end(), closing.start())
        if closing is not None
        else (opening.end(), len(text))
    )
    try:
        decision_local = _covering_token_span_for_character_span(
            text,
            byte_offsets,
            opening.span(0),
            field_name="invalid decision",
        )
    except ValueError:
        decision_local = (0, 0)
    try:
        query_local = _contained_token_span_for_character_span(
            text,
            byte_offsets,
            query_char_span,
            field_name="invalid query",
        )
    except ValueError:
        query_local = (0, 0)

    decision_tokens = set(range(*decision_local))
    # A token that belongs to the opening tag cannot also be Query credit.
    # If removing the overlap leaves a fragmented span, fail closed to an
    # empty Query rather than widening it across unrelated token ownership.
    query_tokens = set(range(*query_local)) - decision_tokens
    query_local = _contiguous_local_span(query_tokens)
    if decision_local[0] == decision_local[1]:
        decision_local = (0, len(token_ids))

    # Think is optional for N_invalid. Preserve it only when it is complete
    # and disjoint from the surviving D/Q spans; otherwise omit it explicitly.
    think_local: tuple[int, int] | None = None
    think_matches = list(_THINK_SPAN_RE.finditer(text))
    if len(think_matches) == 1:
        try:
            candidate = _contained_token_span_for_character_span(
                text,
                byte_offsets,
                think_matches[0].span(0),
                field_name="invalid think",
            )
            think_tokens = set(range(*candidate))
            if not (
                think_tokens & set(range(*decision_local))
                or think_tokens & set(range(*query_local))
            ):
                think_local = candidate
        except ValueError:
            think_local = None

    def absolute(span: tuple[int, int]) -> tuple[int, int]:
        return start + span[0], start + span[1]

    return SearchRoleTokenSpans(
        action=action,
        think=None if think_local is None else absolute(think_local),
        decision=absolute(decision_local),
        query=absolute(query_local),
    )


def classify_role_localized_search_branch(
    *,
    retrieval_budget_exhausted: bool,
    model_search_invalid: bool,
    sufficient_before_search: bool,
    retriever_executed: bool,
    no_new_observation: bool | None,
) -> str:
    """Apply the one locked, mutually exclusive branch priority."""

    if bool(retrieval_budget_exhausted):
        return ROLE_LOCALIZED_BRANCH_N_BUDGET
    if bool(model_search_invalid):
        return ROLE_LOCALIZED_BRANCH_N_INVALID
    if bool(sufficient_before_search):
        return ROLE_LOCALIZED_BRANCH_S_BEFORE
    if bool(retriever_executed) and no_new_observation is True:
        return ROLE_LOCALIZED_BRANCH_N_SOFT
    if bool(retriever_executed) and no_new_observation is False:
        return ROLE_LOCALIZED_BRANCH_NORMAL
    raise ValueError("Search facts do not define a role-localized branch")
