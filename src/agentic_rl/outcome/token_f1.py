from __future__ import annotations

import json
import re
import string
from typing import Iterable, Sequence


IGPO_OFFICIAL_COMMIT = "64165e2741ed8801f977948c8128080ce87b4101"
IGPO_OFFICIAL_SOURCE = "verl/utils/reward_score/info_gain.py"
ANSWER_SPLIT = "<|answer_split|>"
SPECIAL_MULTI_LABEL_SOURCES = frozenset({"Factbench", "politifact", "liar2"})
_OFFICIAL_BALANCED_TAGS = ("code", "tool_call", "think", "answer")
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def check_tags_balance(solution_str: str) -> bool:
    """Match pinned IGPO ``check_tags_balance`` semantics."""
    for tag in _OFFICIAL_BALANCED_TAGS:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        if solution_str.count(start_tag) != solution_str.count(end_tag):
            return False
        last_pos = -1
        while True:
            start_pos = solution_str.find(start_tag, last_pos + 1)
            if start_pos == -1:
                break
            end_pos = solution_str.find(end_tag, start_pos)
            if end_pos == -1:
                return False
            last_pos = end_pos
    return True


def preprocess_text(text: str) -> str:
    """Match pinned IGPO punctuation and whitespace preprocessing."""
    value = str(text)
    for punctuation in string.punctuation:
        value = value.replace(punctuation, " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def deal_multi_labels(ground_truth: Sequence[dict[str, object]]) -> str:
    """Match the pinned IGPO special-source label reduction."""
    for item in ground_truth:
        if str(item["label"]).lower() == "false":
            return "false"
    return "true"


def compute_f1(
    solution_str: str,
    ground_truth: str,
    data_source: str,
    val_type: str = "f1",
) -> float:
    """Mechanical, dependency-free port of pinned IGPO ``compute_f1``.

    This compatibility function intentionally preserves the official first-answer
    extraction and tag-balance behavior. Project protocol validation is a stricter
    outer gate and is not implemented here.
    """
    if data_source in SPECIAL_MULTI_LABEL_SOURCES:
        ground_truth = deal_multi_labels(json.loads(ground_truth))
    solution = str(solution_str).lower()
    truth = str(ground_truth).lower()
    ground_truths = truth.split(ANSWER_SPLIT)
    if not check_tags_balance(solution):
        return 0.0 if val_type == "noformatf1" else -2.0

    answer_match = _ANSWER_RE.search(solution)
    if answer_match is None:
        return 0.0 if val_type == "noformatf1" else -2.0
    answer_content = preprocess_text(answer_match.group(1).strip())

    max_score = 0.0
    for candidate in ground_truths:
        normalized_truth = preprocess_text(candidate)
        if val_type == "em":
            if normalized_truth == answer_content:
                return 1.0
            continue
        prediction_tokens = set(answer_content.split())
        ground_truth_tokens = set(normalized_truth.split())
        if not ground_truth_tokens or not prediction_tokens:
            continue
        common_tokens = prediction_tokens & ground_truth_tokens
        precision = len(common_tokens) / len(prediction_tokens)
        recall = len(common_tokens) / len(ground_truth_tokens)
        if precision + recall > 0:
            score = 2.0 * precision * recall / (precision + recall)
            max_score = max(max_score, score)
    return float(max_score)


def serialize_aliases(aliases: Iterable[str]) -> str:
    values = [str(alias) for alias in aliases]
    if not values:
        raise ValueError("At least one ground-truth alias is required")
    return ANSWER_SPLIT.join(values)


def token_f1(prediction: str, ground_truth: str) -> float:
    """IGPO set-token F1 after the official lowercase/preprocess sequence."""
    solution = f"<answer>{str(prediction)}</answer>"
    return compute_f1(solution, str(ground_truth), data_source="", val_type="f1")


def max_alias_token_f1(
    prediction: str,
    aliases: Iterable[str],
    *,
    data_source: str = "",
) -> float:
    ground_truth = serialize_aliases(aliases)
    solution = f"<answer>{str(prediction)}</answer>"
    return compute_f1(solution, ground_truth, data_source=data_source, val_type="f1")


def max_alias_exact_match(
    prediction: str,
    aliases: Iterable[str],
    *,
    data_source: str = "",
) -> float:
    """Alias-aware exact match using the production scorer normalization."""

    ground_truth = serialize_aliases(aliases)
    solution = f"<answer>{str(prediction)}</answer>"
    return compute_f1(solution, ground_truth, data_source=data_source, val_type="em")
