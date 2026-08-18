import json
import math

import numpy as np
import pytest

from agentic_rl.outcome.format_indicator import centered_format_advantage
from agentic_rl.outcome.parser import parse_model_trajectory
from agentic_rl.outcome.token_f1 import (
    ANSWER_SPLIT,
    compute_f1,
    max_alias_token_f1,
    preprocess_text,
    token_f1,
)
from agentic_rl.outcome.workers import score_trajectory_outcome


def test_centered_binary_format_indicator() -> None:
    result = centered_format_advantage([1, 1, 0, 0])
    np.testing.assert_allclose(result, [0.5, 0.5, -0.5, -0.5])
    assert result.mean() == 0.0


@pytest.mark.parametrize(
    ("prediction", "gold", "expected"),
    [
        ("Paris", "Paris", 1.0),
        ("PARIS", "paris", 1.0),
        ("New-York", "new york", 1.0),
        ("a a b", "a b", 1.0),
        ("", "", 0.0),
        ("", "Paris", 0.0),
        ("!!!", "???", 0.0),
        ("wrong", "Paris", 0.0),
        ("The U.S.A.", "USA", 0.0),
        ("the Paris", "Paris", 2.0 / 3.0),
    ],
)
def test_pinned_igpo_set_token_f1_vectors(prediction, gold, expected) -> None:
    assert math.isclose(token_f1(prediction, gold), expected, abs_tol=1.0e-12)


def test_official_preprocess_does_not_remove_articles() -> None:
    assert preprocess_text("The Hague") == "The Hague"
    assert token_f1("The Hague", "Hague") == 2.0 / 3.0


def test_multiple_aliases_use_official_delimiter_and_max() -> None:
    aliases = ["Lutetia", "Paris", "City of Paris"]
    assert max_alias_token_f1("Paris", aliases) == 1.0
    assert compute_f1(
        "<answer>Paris</answer>",
        ANSWER_SPLIT.join(aliases),
        "",
    ) == 1.0


@pytest.mark.parametrize("source", ["Factbench", "politifact", "liar2"])
def test_pinned_igpo_special_multi_label_sources(source) -> None:
    truth = json.dumps([{"label": "true"}, {"label": "false"}])
    assert compute_f1("<answer>false</answer>", truth, source) == 1.0


def test_official_function_keeps_first_balanced_answer_semantics() -> None:
    result = compute_f1(
        "<answer>Paris</answer><answer>London</answer>",
        "Paris",
        "",
    )
    assert result == 1.0


def test_valid_protocol_and_alias_token_f1() -> None:
    result = score_trajectory_outcome(
        [
            "<think>need evidence</think><search>capital of France</search>",
            "<think>done</think><answer>Paris</answer>",
        ],
        ["Paris", "City of Paris"],
    )
    assert result.parse.valid
    assert result.task_outcome == 1.0
    assert result.format_indicator == 1
    assert result.valid_for_selection


def test_model_information_is_diagnostic_invalid_without_custom_reward() -> None:
    parsed = parse_model_trajectory(
        ["<think>x</think><information>invented</information><answer>x</answer>"]
    )
    assert not parsed.valid
    assert parsed.model_generated_information
    assert parsed.parser_error_type == "model_generated_information"
    assert parsed.answer_format_indicator == 0


@pytest.mark.parametrize(
    "action",
    [
        "<think><search>x</think></search>",
        "<think><answer>x</answer></think>",
        "<search>x</search><think>late</think>",
    ],
)
def test_project_protocol_rejects_crossed_nested_or_late_think_tags(action) -> None:
    parsed = parse_model_trajectory([action])
    assert not parsed.valid
    assert parsed.parser_error_type == "invalid_tag_order_or_nesting"
    assert parsed.answer_format_indicator == 0


def test_early_search_failure_does_not_change_terminal_answer_validity() -> None:
    result = score_trajectory_outcome(
        [
            "<think>x</think><search>missing close",
            "<think>done</think><answer>Paris</answer>",
        ],
        ["Paris"],
    )
    assert not result.parse.trajectory_valid
    assert result.parse.parser_error_type == "unbalanced_tags"
    assert result.parse.answer == "Paris"
    assert result.parse.terminal_answer_valid
    assert result.format_indicator == 1
    assert result.valid_for_selection
    assert result.task_outcome == 1.0


def test_malformed_terminal_answer_is_outcome_invalid() -> None:
    result = score_trajectory_outcome(
        [
            "<think>x</think><search>capital</search>",
            "<think>done</think><answer>Paris",
        ],
        ["Paris"],
    )
    assert not result.terminal_answer_valid
    assert not result.valid_for_selection
    assert result.task_outcome == 0.0
    assert result.format_indicator == 0


def test_retriever_system_failure_is_not_scored_as_wrong_answer() -> None:
    result = score_trajectory_outcome(
        ["<think>done</think><answer>Paris</answer>"],
        ["Paris"],
        trajectory_system_valid=False,
    )
    assert result.terminal_answer_valid
    assert not result.valid_for_selection
    assert result.task_outcome == 0.0
