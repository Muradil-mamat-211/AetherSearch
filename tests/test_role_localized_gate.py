from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest
import torch

from agentic_rl.advantage.role_localized_gate import build_role_localized_trajectory_credits
from agentic_rl.outcome.workers import (
    PRODUCTION_TASK_SCORER_VERSION,
    SUFFICIENCY_EXACT_SCORER_VERSION,
)
from agentic_rl.outcome.parser import parse_model_action
from agentic_rl.policy.gate_gradient_calibration import (
    BatchGradientProfile,
    calibrate_role_localized_gate_lambdas,
    parameter_shard_sha256,
)
from agentic_rl.policy.strict_onpolicy_loss import fixed_gate_turn_objective
from agentic_rl.policy.turn_ratio import compute_turn_ratios
from agentic_rl.rollout.search_role_provenance import (
    ROLE_LOCALIZED_BRANCH_N_BUDGET,
    ROLE_LOCALIZED_BRANCH_N_INVALID,
    ROLE_LOCALIZED_BRANCH_N_SOFT,
    ROLE_LOCALIZED_BRANCH_NORMAL,
    ROLE_LOCALIZED_BRANCH_S_BEFORE,
    build_invalid_search_role_spans,
    build_generation_time_search_role_spans,
    classify_role_localized_search_branch,
    exact_search_payload_text,
)
from agentic_rl.rollout.trajectory_schema import TokenSource, TrajectoryRecord, TurnRecord, TurnType

VERSION = 0


def probe(stage: str, sufficient: bool, reward: float) -> dict[str, object]:
    name = "sufficient_before_search" if stage == "pre" else "sufficient_after_search"
    return {
        "raw_answer_text": "ok" if sufficient else "wrong",
        "parser_success": True,
        "no_answer": False,
        "output_truncated": False,
        "alias_aware_exact": sufficient,
        "raw_task_reward": reward,
        "scorer_version": SUFFICIENCY_EXACT_SCORER_VERSION,
        "task_scorer_version": PRODUCTION_TASK_SCORER_VERSION,
        "prefix_provenance_valid": True,
        "detached": True,
        "completion_count": 1,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "n": 1,
        "max_tokens": 500,
        "stop": ["</answer>"],
        "candidate_rollout_policy_version": VERSION,
        "exact_ig_policy_version": VERSION,
        "probe_policy_version": VERSION,
        "old_logprob_policy_version": VERSION,
        name: sufficient,
    }


def spec(branch: str, **overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "branch": branch,
        "raw": 0.2,
        "norm": 0.7,
        "query_tokens": 2,
        "repeat": False,
        "pre_s": branch == ROLE_LOCALIZED_BRANCH_S_BEFORE,
        "post_s": False,
        "pre_r": 0.2,
        "post_r": 0.2,
    }
    result.update(overrides)
    return result


def credits(specs: list[dict[str, object]], z: float = 0.0):
    turns, probes, immediate, normalized = [], {}, {}, {}
    cursor = 100
    for index, row in enumerate(specs):
        branch = str(row["branch"])
        main = branch in {ROLE_LOCALIZED_BRANCH_N_SOFT, ROLE_LOCALIZED_BRANCH_NORMAL}
        no_new = True if branch in {ROLE_LOCALIZED_BRANCH_N_BUDGET, ROLE_LOCALIZED_BRANCH_N_SOFT} else False if branch == ROLE_LOCALIZED_BRANCH_NORMAL else None
        turns.append(SimpleNamespace(
            turn_type=TurnType.SEARCH,
            search_index=index,
            turn_index=index,
            policy_credit_eligible=True,
            ig_reward_eligible=main,
            main_credit_eligible=main,
            role_localized_gate_enabled=True,
            retrieval_budget_exhausted=branch == ROLE_LOCALIZED_BRANCH_N_BUDGET,
            model_search_invalid=branch == ROLE_LOCALIZED_BRANCH_N_INVALID,
            retriever_executed=main,
            no_new_observation=no_new,
            branch_type=branch,
            query_token_span=(cursor, cursor + int(row["query_tokens"])),
            exact_query_repeat=bool(row["repeat"]),
            new_passage_count=0 if no_new is True else 1,
        ))
        cursor += 8
        stages = {"pre": probe("pre", bool(row["pre_s"]), float(row["pre_r"]))}
        if main:
            stages["post"] = probe("post", bool(row["post_s"]), float(row["post_r"]))
            normalized[index] = float(row["norm"])
        probes[index] = stages
        if row["raw"] is not None:
            immediate[index] = float(row["raw"])
    record = SimpleNamespace(
        trajectory_id="trajectory",
        turns=turns,
        metadata={"routed_answer_probes": probes},
        immediate_ig=immediate,
    )
    return build_role_localized_trajectory_credits(
        record,
        normalized_ig=normalized,
        normalized_outcome=z,
        optimized_search_indices=tuple(range(len(specs))),
        expected_policy_version=VERSION,
        expected_scorer_version=PRODUCTION_TASK_SCORER_VERSION,
        probe_epsilon=1.0e-6,
    )


@pytest.mark.parametrize(
    "row, expected",
    [
        (spec(ROLE_LOCALIZED_BRANCH_NORMAL, norm=0.75), (0.75, 0.0, 0.0)),
        (spec(ROLE_LOCALIZED_BRANCH_S_BEFORE), (0.0, -0.5, 0.0)),
        (spec(ROLE_LOCALIZED_BRANCH_N_BUDGET, raw=None, pre_s=True), (0.0, -1.0, 0.0)),
        (spec(ROLE_LOCALIZED_BRANCH_N_INVALID, query_tokens=3), (0.0, -0.5, -0.5)),
        (spec(ROLE_LOCALIZED_BRANCH_N_INVALID, query_tokens=0), (0.0, -0.5, 0.0)),
        (spec(ROLE_LOCALIZED_BRANCH_N_SOFT, norm=0.4), (0.4, 0.0, 0.0)),
        (spec(ROLE_LOCALIZED_BRANCH_N_SOFT, raw=-0.2, norm=0.9, repeat=True), (0.9, 0.0, -0.25)),
        (spec(ROLE_LOCALIZED_BRANCH_N_SOFT, raw=0.0, norm=0.9, repeat=True), (0.9, 0.0, -0.25)),
        (spec(ROLE_LOCALIZED_BRANCH_N_SOFT, raw=0.1, norm=-2.0, repeat=True), (-2.0, 0.0, 0.0)),
        (spec(ROLE_LOCALIZED_BRANCH_N_SOFT, raw=-0.1, norm=2.0, repeat=True), (2.0, 0.0, -0.25)),
    ],
)
def test_locked_credit_cases(row: dict[str, object], expected: tuple[float, float, float]) -> None:
    result = credits([row])
    assert (result.main[0], result.decision[0], result.query[0]) == expected
    if row["branch"] == ROLE_LOCALIZED_BRANCH_N_BUDGET:
        assert not result.effective_cumulative_ig
        assert not result.routed_outcome
        assert not result.sufficient_after


def test_soft_n_remains_in_b_and_future_propagation() -> None:
    result = credits([
        spec(ROLE_LOCALIZED_BRANCH_NORMAL, norm=1.0),
        spec(ROLE_LOCALIZED_BRANCH_N_SOFT, norm=2.0),
        spec(ROLE_LOCALIZED_BRANCH_NORMAL, norm=3.0),
    ])
    assert result.effective_cumulative_ig_count == {0: 3, 1: 2, 2: 1}
    assert result.main[0] == pytest.approx(6 / math.sqrt(3))
    assert result.main[1] == pytest.approx(5 / math.sqrt(2))
    assert result.main[2] == 3.0


def test_branch_priority_is_total_and_mutually_exclusive() -> None:
    assert classify_role_localized_search_branch(
        retrieval_budget_exhausted=True, model_search_invalid=True,
        sufficient_before_search=True, retriever_executed=False,
        no_new_observation=True,
    ) == ROLE_LOCALIZED_BRANCH_N_BUDGET
    assert classify_role_localized_search_branch(
        retrieval_budget_exhausted=False, model_search_invalid=True,
        sufficient_before_search=True, retriever_executed=False,
        no_new_observation=None,
    ) == ROLE_LOCALIZED_BRANCH_N_INVALID
    assert classify_role_localized_search_branch(
        retrieval_budget_exhausted=False, model_search_invalid=False,
        sufficient_before_search=True, retriever_executed=True,
        no_new_observation=True,
    ) == ROLE_LOCALIZED_BRANCH_S_BEFORE


class CharacterTokenizer:
    all_special_ids: tuple[int, ...] = ()

    @staticmethod
    def convert_ids_to_tokens(token_ids, *, skip_special_tokens):
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

        assert skip_special_tokens is False
        byte_encoder = bytes_to_unicode()
        return [byte_encoder[int(token_id)] for token_id in token_ids]

def test_generation_time_spans_are_exact_disjoint_and_fail_closed() -> None:
    text = "<think>reason</think><search>bridge query</search>"
    spans = build_generation_time_search_role_spans(
        CharacterTokenizer(),
        action_token_ids=[ord(char) for char in text],
        action_text=text,
        absolute_action_start=10,
    )
    h, d, q = (set(range(*span)) for span in (spans.think, spans.decision, spans.query))
    assert not (h & d or h & q or d & q)
    assert text[spans.decision[0] - 10:spans.decision[1] - 10] == "<search>"
    assert text[spans.query[0] - 10:spans.query[1] - 10] == "bridge query"
    with pytest.raises(ValueError, match="round-trip"):
        build_generation_time_search_role_spans(
            CharacterTokenizer(), action_token_ids=[1], action_text=text,
            absolute_action_start=0,
        )
    missing = "<search>q</search>"
    with pytest.raises(ValueError, match="Think"):
        build_generation_time_search_role_spans(
            CharacterTokenizer(), action_token_ids=[ord(c) for c in missing],
            action_text=missing, absolute_action_start=0,
        )



def test_merged_think_search_boundary_preserves_native_token_provenance() -> None:
    class MergedBoundaryTokenizer:
        all_special_ids: tuple[int, ...] = ()
        pieces = (
            "<th", "ink", ">", "reason", "</", "think", "><", "search",
            ">", "bridge", "\u0120query", "</", "search", ">",
        )

        @classmethod
        def convert_ids_to_tokens(cls, token_ids, *, skip_special_tokens):
            assert skip_special_tokens is False
            return [cls.pieces[int(token_id)] for token_id in token_ids]

    text = "<think>reason</think><search>bridge query</search>"
    token_ids = list(range(len(MergedBoundaryTokenizer.pieces)))
    spans = build_generation_time_search_role_spans(
        MergedBoundaryTokenizer(),
        action_token_ids=token_ids,
        action_text=text,
        absolute_action_start=0,
    )
    think = set(range(*spans.think))
    decision = set(range(*spans.decision))
    query = set(range(*spans.query))
    assert not think & decision
    assert not decision & query
    assert spans.query == (9, 11)


def test_opening_tag_boundary_token_is_owned_by_decision_not_query() -> None:
    class CrossBoundaryTokenizer:
        all_special_ids: tuple[int, ...] = ()
        pieces = (
            "<th", "ink", ">", "reason", "</", "think", "><", "search",
            ">bridge", "\u0120query", "</", "search", ">",
        )

        @classmethod
        def convert_ids_to_tokens(cls, token_ids, *, skip_special_tokens):
            assert skip_special_tokens is False
            return [cls.pieces[int(token_id)] for token_id in token_ids]

    text = "<think>reason</think><search>bridge query</search>"
    spans = build_generation_time_search_role_spans(
        CrossBoundaryTokenizer(),
        action_token_ids=range(len(CrossBoundaryTokenizer.pieces)),
        action_text=text,
        absolute_action_start=0,
    )
    assert not set(range(*spans.decision)) & set(range(*spans.query))
    assert spans.decision == (6, 9)
    assert spans.query == (9, 10)
    assert spans.action == (0, len(CrossBoundaryTokenizer.pieces))


def test_generation_time_spans_accept_only_empty_decoded_special_suffix() -> None:
    class SpecialSuffixTokenizer(CharacterTokenizer):
        all_special_ids = (256, 257)

        @staticmethod
        def convert_ids_to_tokens(token_ids, *, skip_special_tokens):
            from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

            assert skip_special_tokens is False
            byte_encoder = bytes_to_unicode()
            return [
                f"<special:{token_id}>"
                if int(token_id) >= 256
                else byte_encoder[int(token_id)]
                for token_id in token_ids
            ]

        @staticmethod
        def decode(token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
            assert clean_up_tokenization_spaces is False
            if skip_special_tokens and all(int(value) >= 256 for value in token_ids):
                return ""
            return "visible-special"

    text = "<think>reason</think><search>bridge query</search>"
    regular_ids = [ord(char) for char in text]
    spans = build_generation_time_search_role_spans(
        SpecialSuffixTokenizer(),
        action_token_ids=[*regular_ids, 256],
        action_text=text,
        absolute_action_start=3,
    )
    assert spans.action == (3, 3 + len(regular_ids) + 1)

    with pytest.raises(ValueError, match="after its special-token suffix"):
        build_generation_time_search_role_spans(
            SpecialSuffixTokenizer(),
            action_token_ids=[*regular_ids[:5], 256, *regular_ids[5:]],
            action_text=text,
            absolute_action_start=0,
        )

    class VisibleSpecialTokenizer(SpecialSuffixTokenizer):
        @staticmethod
        def decode(token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
            return "<visible>"

    with pytest.raises(ValueError, match="visible special token"):
        build_generation_time_search_role_spans(
            VisibleSpecialTokenizer(),
            action_token_ids=[*regular_ids, 257],
            action_text=text,
            absolute_action_start=0,
        )


def test_generation_time_spans_preserve_lossy_bytelevel_token_provenance() -> None:
    class LossyByteTokenizer:
        all_special_ids: tuple[int, ...] = ()

        def __init__(self, raw_chunks: list[bytes]) -> None:
            from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

            byte_encoder = bytes_to_unicode()
            self.pieces = [
                "".join(byte_encoder[value] for value in chunk)
                for chunk in raw_chunks
            ]

        def convert_ids_to_tokens(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens is False
            return [self.pieces[int(token_id)] for token_id in token_ids]

    prefix = b"<think>reason</think><search>bridge "
    suffix = b" query</search>"
    raw_chunks = [bytes([value]) for value in prefix]
    invalid_token_index = len(raw_chunks)
    raw_chunks.append(b"\xff")
    raw_chunks.extend(bytes([value]) for value in suffix)
    text = b"".join(raw_chunks).decode("utf-8", errors="replace")
    spans = build_generation_time_search_role_spans(
        LossyByteTokenizer(raw_chunks),
        action_token_ids=range(len(raw_chunks)),
        action_text=text,
        absolute_action_start=0,
    )
    assert spans.query[0] <= invalid_token_index < spans.query[1]
    assert not set(range(*spans.decision)) & set(range(*spans.query))


def test_generation_time_spans_map_utf8_scalar_split_across_tokens() -> None:
    class SplitUtf8Tokenizer:
        all_special_ids: tuple[int, ...] = ()

        def __init__(self, raw_chunks: list[bytes]) -> None:
            from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

            byte_encoder = bytes_to_unicode()
            self.pieces = [
                "".join(byte_encoder[value] for value in chunk)
                for chunk in raw_chunks
            ]

        def convert_ids_to_tokens(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens is False
            return [self.pieces[int(token_id)] for token_id in token_ids]

    prefix = b"<think>reason</think><search>cost "
    suffix = b" query</search>"
    raw_chunks = [bytes([value]) for value in prefix]
    split_start = len(raw_chunks)
    raw_chunks.extend((b"\xe2", b"\x82", b"\xac"))
    raw_chunks.extend(bytes([value]) for value in suffix)
    text = b"".join(raw_chunks).decode("utf-8", errors="replace")
    spans = build_generation_time_search_role_spans(
        SplitUtf8Tokenizer(raw_chunks),
        action_token_ids=range(len(raw_chunks)),
        action_text=text,
        absolute_action_start=0,
    )
    assert all(
        spans.query[0] <= token_index < spans.query[1]
        for token_index in range(split_start, split_start + 3)
    )
    assert not set(range(*spans.decision)) & set(range(*spans.query))

def test_observation_is_outside_policy_ratio_loss_and_action_denominator() -> None:
    sources = [TokenSource.PROMPT] + [TokenSource.MODEL] * 8 + [TokenSource.ENVIRONMENT] * 2 + [TokenSource.MODEL] * 2
    record = TrajectoryRecord(
        prompt_global_id="p", trajectory_id="t", input_ids=list(range(13)),
        token_sources=sources, turn_ids=[-1] + [0] * 8 + [-1] * 2 + [1] * 2,
        turns=[
            TurnRecord(
                turn_index=0, turn_type=TurnType.SEARCH, search_index=0,
                model_text="<think>x</think><search>q</search>",
                search_action_span_valid=True, search_prefix_valid=True,
                ig_reward_eligible=True, policy_credit_eligible=True,
                no_new_observation=False, new_passage_keys=("passage",),
                role_localized_gate_enabled=True, retriever_executed=True,
                main_credit_eligible=True, branch_type=ROLE_LOCALIZED_BRANCH_NORMAL,
                raw_query="q", canonical_query="q", new_passage_count=1,
                stable_passage_keys_before=(), stable_passage_keys_after=("passage",),
                action_token_span=(1, 9), think_token_span=(1, 4),
                decision_token_span=(4, 6), query_token_span=(6, 7),
                observation_token_span=(9, 11),
            ),
            TurnRecord(turn_index=1, turn_type=TurnType.ANSWER,
                       model_text="<answer>a</answer>", policy_credit_eligible=True),
        ],
        search_prefix_end_positions=[11],
        search_prefix_before_search_end_positions={0: 1}, immediate_ig={0: 0.1},
        terminal_answer_valid=True, trajectory_protocol_valid=True,
        trajectory_system_valid=True,
    )
    record.validate()
    assert record.policy_mask[9:11] == [0, 0]
    assert sum(record.policy_mask) == 10
    assert record.policy_mask == record.kl_mask


def ratio(mask: list[int]) -> float:
    current = torch.tensor([0.1, 0.3, -0.2, 0.4], requires_grad=True)
    old = torch.tensor([0.0, 0.1, -0.1, 0.0])
    return float(compute_turn_ratios(
        current, old, torch.tensor(mask, dtype=torch.bool),
        torch.tensor([0, 0, 0, 0]), expected_turn_ids=[0],
    )[0].detach())


@pytest.mark.parametrize(
    "mask, expected",
    [
        ([1, 1, 1, 1], math.exp((0.1 + 0.2 - 0.1 + 0.4) / 4)),
        ([0, 1, 0, 0], math.exp(0.2)),
        ([0, 0, 1, 1], math.exp(0.15)),
    ],
)
def test_main_decision_query_ratios_match_hand_calculation(mask, expected) -> None:
    assert ratio(mask) == pytest.approx(expected)


def test_gate_surrogate_and_three_reductions_match_contract() -> None:
    r = torch.tensor(1.1, requires_grad=True)
    result = fixed_gate_turn_objective({0: r}, {0: -0.5})
    assert result.lower_bound_by_turn[0] == 0.997
    assert result.upper_bound_by_turn[0] == 1.004
    assert float(result.objective_by_turn[0].detach()) == pytest.approx(-0.55)
    main = (3 * torch.tensor(0.4) + 2 * torch.tensor(-0.2)) / 5
    explicit = torch.tensor([0.4] * 3 + [-0.2] * 2).mean()
    assert torch.allclose(main, explicit, rtol=0.0, atol=1.0e-7)
    assert torch.equal(torch.tensor(-0.25) / 3, torch.tensor(-0.25) / 3)


def test_only_soft_duplicate_allows_main_query_overlap() -> None:
    soft = credits([spec(ROLE_LOCALIZED_BRANCH_N_SOFT, raw=-0.1, repeat=True)])
    invalid = credits([spec(ROLE_LOCALIZED_BRANCH_N_INVALID)])
    assert soft.allowed_soft_duplicate_main_query_overlap_count == 1
    assert invalid.main[0] == 0.0 and invalid.query[0] == -0.5


def test_static_calibration_and_no_update_safety() -> None:
    rows = [BatchGradientProfile(
        batch_id=f"b{i}", main_gradient_norm=10.0,
        decision_gradient_norm=20.0, query_gradient_norm=5.0,
        dot_main_decision=0.0, dot_main_query=0.0, dot_decision_query=0.0,
        cos_main_decision=0.0, cos_main_query=0.0, cos_decision_query=0.0,
        decision_gate_event_count=50, query_gate_event_count=25,
        parameters_bitwise_unchanged=True, gradients_cleared=True,
        rank_metadata_consistent=True,
    ) for i in range(3)]
    calibrated = calibrate_role_localized_gate_lambdas(rows)
    assert calibrated["median_gate_to_main_gradient_ratio"] <= 0.15
    assert calibrated["optimizer_steps"] == calibrated["scheduler_steps"] == 0
    assert calibrated["checkpoint_writes"] == 0
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    before = parameter_shard_sha256([parameter])
    parameter.square().sum().backward()
    parameter.grad = None
    assert parameter_shard_sha256([parameter]) == before
    assert parameter.grad is None

@pytest.mark.parametrize(
    "pre_reward, post_reward, z, expected",
    [
        (0.0, 1.0, 0.8, 0.8),
        (0.0, 1.0, -0.8, 0.0),
        (1.0, 0.0, -0.8, -0.8),
        (1.0, 0.0, 0.8, 0.0),
        (0.5, 0.5, 0.8, 0.0),
    ],
)
def test_probe_routed_outcome_in_b_is_unchanged(
    pre_reward: float,
    post_reward: float,
    z: float,
    expected: float,
) -> None:
    result = credits([
        spec(
            ROLE_LOCALIZED_BRANCH_NORMAL,
            norm=0.0,
            pre_r=pre_reward,
            post_r=post_reward,
        )
    ], z=z)
    assert result.routed_outcome[0] == expected
    assert result.main[0] == expected


def test_s_after_still_truncates_b_after_including_current_search() -> None:
    result = credits([
        spec(ROLE_LOCALIZED_BRANCH_NORMAL, norm=1.0, post_s=True),
        spec(ROLE_LOCALIZED_BRANCH_NORMAL, norm=9.0),
    ])
    assert result.effective_cumulative_ig_count[0] == 1
    assert result.effective_cumulative_ig[0] == 1.0


def test_incomplete_invalid_search_preserves_generated_query_span() -> None:
    text = "<think>x</think><search>unfinished query"
    spans = build_generation_time_search_role_spans(
        CharacterTokenizer(),
        action_token_ids=[ord(char) for char in text],
        action_text=text,
        absolute_action_start=0,
        allow_incomplete_search=True,
    )
    assert text[slice(*spans.query)] == "unfinished query"


def test_negative_main_gate_cosine_halves_static_budget() -> None:
    rows = [BatchGradientProfile(
        batch_id=f"negative-{i}", main_gradient_norm=10.0,
        decision_gradient_norm=20.0, query_gradient_norm=5.0,
        dot_main_decision=-120.0, dot_main_query=0.0,
        dot_decision_query=0.0, cos_main_decision=-0.6,
        cos_main_query=0.0, cos_decision_query=0.0,
        decision_gate_event_count=50, query_gate_event_count=25,
        parameters_bitwise_unchanged=True, gradients_cleared=True,
        rank_metadata_consistent=True,
    ) for i in range(3)]
    result = calibrate_role_localized_gate_lambdas(rows)
    assert result["eta_decision_effective"] == 0.05
    assert result["eta_query_effective"] == 0.05


def test_empty_invalid_query_never_fabricates_query_credit() -> None:
    result = credits([spec(ROLE_LOCALIZED_BRANCH_N_INVALID, query_tokens=0)])
    assert result.empty_query_without_query_span_count == 1
    assert result.query[0] == 0.0


def test_role_localized_branch_and_credit_formula_fuzz() -> None:
    generator = random.Random(20260804)
    for _ in range(1_000):
        budget = generator.choice((False, True))
        invalid = generator.choice((False, True))
        sufficient = generator.choice((False, True))
        retriever_executed = generator.choice((False, True))
        no_new = generator.choice((False, True))
        if budget:
            expected_branch = ROLE_LOCALIZED_BRANCH_N_BUDGET
        elif invalid:
            expected_branch = ROLE_LOCALIZED_BRANCH_N_INVALID
        elif sufficient:
            expected_branch = ROLE_LOCALIZED_BRANCH_S_BEFORE
        elif retriever_executed and no_new:
            expected_branch = ROLE_LOCALIZED_BRANCH_N_SOFT
        elif retriever_executed:
            expected_branch = ROLE_LOCALIZED_BRANCH_NORMAL
        else:
            with pytest.raises(ValueError):
                classify_role_localized_search_branch(
                    retrieval_budget_exhausted=budget,
                    model_search_invalid=invalid,
                    sufficient_before_search=sufficient,
                    retriever_executed=retriever_executed,
                    no_new_observation=no_new,
                )
            continue
        assert classify_role_localized_search_branch(
            retrieval_budget_exhausted=budget,
            model_search_invalid=invalid,
            sufficient_before_search=sufficient,
            retriever_executed=retriever_executed,
            no_new_observation=no_new,
        ) == expected_branch

    for _ in range(1_000):
        branch = generator.choice(
            (
                ROLE_LOCALIZED_BRANCH_N_BUDGET,
                ROLE_LOCALIZED_BRANCH_N_INVALID,
                ROLE_LOCALIZED_BRANCH_S_BEFORE,
                ROLE_LOCALIZED_BRANCH_N_SOFT,
                ROLE_LOCALIZED_BRANCH_NORMAL,
            )
        )
        normalized = generator.uniform(-4.0, 4.0)
        raw = generator.uniform(-1.0, 1.0)
        query_tokens = generator.randrange(0, 8)
        repeat = bool(
            query_tokens > 0 and generator.choice((False, True))
        )
        result = credits(
            [
                spec(
                    branch,
                    norm=normalized,
                    raw=None if branch == ROLE_LOCALIZED_BRANCH_N_BUDGET else raw,
                    repeat=repeat,
                    query_tokens=query_tokens,
                )
            ]
        )
        expected_main = (
            normalized
            if branch in {ROLE_LOCALIZED_BRANCH_N_SOFT, ROLE_LOCALIZED_BRANCH_NORMAL}
            else 0.0
        )
        expected_decision = (
            -1.0
            if branch == ROLE_LOCALIZED_BRANCH_N_BUDGET
            else -0.5
            if branch in {ROLE_LOCALIZED_BRANCH_N_INVALID, ROLE_LOCALIZED_BRANCH_S_BEFORE}
            else 0.0
        )
        expected_query = (
            -0.5
            if branch == ROLE_LOCALIZED_BRANCH_N_INVALID and query_tokens > 0
            else -0.25
            if branch == ROLE_LOCALIZED_BRANCH_N_SOFT and repeat and raw <= 0.0
            else 0.0
        )
        assert result.main[0] == pytest.approx(expected_main)
        assert result.decision[0] == expected_decision
        assert result.query[0] == expected_query
        if result.decision[0] and result.query[0]:
            assert branch == ROLE_LOCALIZED_BRANCH_N_INVALID
            assert query_tokens > 0
        assert not (
            result.main[0]
            and (result.decision[0] or result.query[0])
            and branch != ROLE_LOCALIZED_BRANCH_N_SOFT
        )


def test_incomplete_think_is_allowed_only_for_model_invalid_search() -> None:
    text = "<think>reason<search>bridge query</search>"
    ids = [ord(char) for char in text]
    spans = build_generation_time_search_role_spans(
        CharacterTokenizer(),
        action_token_ids=ids,
        action_text=text,
        absolute_action_start=4,
        allow_incomplete_search=True,
    )
    assert spans.think is None
    assert spans.decision[0] < spans.decision[1]
    assert spans.query[0] < spans.query[1]
    with pytest.raises(ValueError, match="complete Think"):
        build_generation_time_search_role_spans(
            CharacterTokenizer(),
            action_token_ids=ids,
            action_text=text,
            absolute_action_start=0,
            allow_incomplete_search=False,
        )


def test_multiple_think_spans_route_to_invalid_only() -> None:
    text = (
        "<think>first</think><think>second</think>"
        "<search>bridge query</search>"
    )
    ids = [ord(char) for char in text]
    spans = build_generation_time_search_role_spans(
        CharacterTokenizer(),
        action_token_ids=ids,
        action_text=text,
        absolute_action_start=0,
        allow_incomplete_search=True,
    )
    assert spans.think is None
    assert spans.decision[0] < spans.decision[1]
    assert spans.query[0] < spans.query[1]
    with pytest.raises(ValueError, match="exactly one complete Think"):
        build_generation_time_search_role_spans(
            CharacterTokenizer(),
            action_token_ids=ids,
            action_text=text,
            absolute_action_start=0,
            allow_incomplete_search=False,
        )


def test_duplicate_search_tags_route_to_invalid_with_native_first_call_spans() -> None:
    text = (
        "<think>reason</think><search>first query</search>"
        "<search>second query</search>"
    )
    parsed = parse_model_action(text)
    assert not parsed.valid
    assert parsed.error == "duplicate_action_tags"
    assert exact_search_payload_text(
        text,
        allow_incomplete_search=True,
    ) == "first query"

    ids = [ord(character) for character in text]
    spans = build_generation_time_search_role_spans(
        CharacterTokenizer(),
        action_token_ids=ids,
        action_text=text,
        absolute_action_start=7,
        allow_incomplete_search=True,
    )
    decision = text.index("<search>")
    query = text.index("first query")
    assert spans.decision == (7 + decision, 7 + decision + len("<search>"))
    assert spans.query == (7 + query, 7 + query + len("first query"))
    assert not set(range(*spans.decision)) & set(range(*spans.query))

    with pytest.raises(ValueError, match="exactly one <search> opening tag"):
        build_generation_time_search_role_spans(
            CharacterTokenizer(),
            action_token_ids=ids,
            action_text=text,
            absolute_action_start=0,
            allow_incomplete_search=False,
        )


def test_unbalanced_duplicate_search_uses_first_native_payload_for_invalid() -> None:
    text = (
        "<think>reason</think><search>first query<search>second query</search>"
    )
    parsed = parse_model_action(text)
    assert not parsed.valid
    assert parsed.error == "unbalanced_tags"
    assert exact_search_payload_text(
        text,
        allow_incomplete_search=True,
    ) == "first query<search>second query"
    spans = build_generation_time_search_role_spans(
        CharacterTokenizer(),
        action_token_ids=[ord(character) for character in text],
        action_text=text,
        absolute_action_start=0,
        allow_incomplete_search=True,
    )
    assert spans.decision[0] < spans.decision[1]
    assert spans.query[0] < spans.query[1]
    assert not set(range(*spans.decision)) & set(range(*spans.query))


def test_invalid_overlapping_native_spans_fail_closed_without_dropping_action() -> None:
    # The malformed Think span contains the Search tag. The strict builder
    # must reject the overlap; the invalid-action fallback keeps the complete
    # generated action and exposes only disjoint D/Q ownership.
    text = "<think>reason<search>bridge query</search></think>"
    ids = [ord(character) for character in text]
    with pytest.raises(ValueError, match="overlap"):
        build_generation_time_search_role_spans(
            CharacterTokenizer(),
            action_token_ids=ids,
            action_text=text,
            absolute_action_start=11,
            allow_incomplete_search=True,
        )
    spans = build_invalid_search_role_spans(
        CharacterTokenizer(),
        action_token_ids=ids,
        action_text=text,
        absolute_action_start=11,
    )
    action = set(range(*spans.action))
    decision = set(range(*spans.decision))
    query = set(range(*spans.query))
    assert spans.action == (11, 11 + len(ids))
    assert decision <= action
    assert query <= action
    assert not decision & query
    assert spans.think is None
    assert spans.query[0] < spans.query[1]


def test_invalid_unrecoverable_token_projection_keeps_action_and_empty_query():
    class UnreversibleTokenizer(CharacterTokenizer):
        @staticmethod
        def convert_ids_to_tokens(token_ids, *, skip_special_tokens):
            assert skip_special_tokens is False
            return ["not-byte-level"] * len(token_ids)

    text = "<think>x</think><search>query</search>"
    ids = [ord(character) for character in text]
    spans = build_invalid_search_role_spans(
        UnreversibleTokenizer(),
        action_token_ids=ids,
        action_text=text,
        absolute_action_start=3,
    )
    assert spans.action == (3, 3 + len(ids))
    assert spans.decision == spans.action
    assert spans.query == (3, 3)
    assert spans.think is None
