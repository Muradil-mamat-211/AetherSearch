"""The five-field full-trajectory format of muradil211/AetherSearch_SFT."""

from __future__ import annotations

import re
from typing import Any

from controlled_rollout import (ANSWER_ACTION, SEARCH_ACTION_SPAN, Rejected,
                                normalize, parse_action)
from token_budget import student_budget


PUBLIC_FORMAT_VERSION = "aethersearch_full_trajectory_v2_numeric_ids"
PUBLIC_FIELDS = ("id", "question", "trajectory_type", "search_count", "full_trajectory_text")
MAX_FULL_TRAJECTORY_TOKENS = 4096
QWEN_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
PUBLIC_USER_PROMPT = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>, "
    "and it will return the top searched results between <information> and </information>. "
    "You can search as many times as you want. If you find no further external knowledge needed, "
    "you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> xxx </answer>. Question: {question}"
)
EOT = "<|im_end|>"
ASSISTANT_START = "<|im_start|>assistant\n"
PUBLIC_INFORMATION_SPAN = re.compile(r"<information>(.*?)</information>", re.DOTALL)


def public_prefix(question: str) -> str:
    return ("<|im_start|>system\n" + QWEN_SYSTEM + EOT + "\n"
            "<|im_start|>user\n" + PUBLIC_USER_PROMPT.format(question=question) + EOT + "\n"
            + ASSISTANT_START)


def trajectory_type(search_count: int) -> str:
    if search_count == 0:
        return "direct_answer"
    return "single_search" if search_count == 1 else "multi_search"


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert an already validated, approved controller record for training."""
    count = record["metadata"]["search_count"]
    result = {"id": record["metadata"].get("public_id"), "question": record["question"],
              "trajectory_type": trajectory_type(count), "search_count": count,
              "full_trajectory_text": public_prefix(record["question"]) + record["response"] + EOT}
    errors = validate_public_record(result)
    if errors:
        raise ValueError("public_trajectory_invalid:" + ",".join(errors))
    return result


def validate_public_record(row: Any) -> list[str]:
    if not isinstance(row, dict) or tuple(row) != PUBLIC_FIELDS:
        return ["public_fields_mismatch"]
    uid, question, kind, count, full = (row[key] for key in PUBLIC_FIELDS)
    if (not isinstance(uid, str) or re.fullmatch(r"[0-9]{6}", uid) is None or uid == "000000"
            or not isinstance(question, str) or not question
            or type(count) is not int or not 0 <= count <= 5 or not isinstance(full, str)):
        return ["invalid_public_values"]
    if kind != trajectory_type(count):
        return ["public_trajectory_type_mismatch"]
    prefix = public_prefix(question)
    if not full.startswith(prefix) or not full.endswith(EOT) or full.count(EOT) != 3 or full.count(ASSISTANT_START) != 1:
        return ["invalid_public_chat_template"]
    if student_budget().count(full) > MAX_FULL_TRAJECTORY_TOKENS:
        return ["full_trajectory_token_budget_exceeded"]
    body, position, queries = full[len(prefix):-len(EOT)], 0, []
    if any(body.count(tag) != count for tag in ("<search>", "</search>", "<information>", "</information>")):
        return ["public_turn_count_mismatch"]
    for _ in range(count):
        action = SEARCH_ACTION_SPAN.match(body, position)
        if action is None:
            return ["missing_public_search"]
        try:
            _, query = parse_action(action.group(), "search")
        except Rejected as exc:
            return [str(exc)]
        if normalize(query) in {normalize(old) for old in queries}:
            return ["duplicate_query"]
        queries.append(query)
        position = action.end()
        info = PUBLIC_INFORMATION_SPAN.match(body, position)
        if info is None or not info.group(1).strip():
            return ["invalid_public_information"]
        position = info.end()
    final = body[position:]
    if ANSWER_ACTION.fullmatch(final) is None:
        return ["invalid_public_final"]
    try:
        parse_action(final, "answer")
    except Rejected as exc:
        return [str(exc)]
    return []
