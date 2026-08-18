from pathlib import Path

import pytest
import torch

from agentic_rl.policy.reduction import (
    TrajectoryTokenValues,
    distributed_local_mean_loss,
    prompt_trajectory_action_token_reduce,
)
from agentic_rl.policy.reference_kl import (
    actor_to_reference_full_vocab_kl,
    assert_reference_frozen,
    causal_action_state_mask,
    select_action_logit_rows,
)
from agentic_rl.policy.strict_onpolicy_loss import (
    ADAPTIVE_CLIP_BETA,
    ADAPTIVE_CLIP_EPSILON_HIGH,
    ADAPTIVE_CLIP_EPSILON_LOW,
    ANSWER_CLIP_SCALE,
    CLIPPING_MODE,
    a2tgpo_adaptive_turn_objective,
    adaptive_clip_scale,
    combine_task_and_kl,
)
from agentic_rl.policy.turn_ratio import compute_turn_ratios


def test_prompt_trajectory_token_reduction_equalizes_lengths() -> None:
    records = [
        TrajectoryTokenValues("p0", "t0", torch.ones(100), torch.ones(100)),
        TrajectoryTokenValues("p0", "t1", torch.ones(500) * 3, torch.ones(500)),
        TrajectoryTokenValues("p1", "t2", torch.ones(2) * 5, torch.ones(2)),
        TrajectoryTokenValues("p1", "t3", torch.ones(7) * 7, torch.ones(7)),
    ]
    reduced = prompt_trajectory_action_token_reduce(records, expected_group_size=2)
    assert torch.isclose(reduced.prompt_means["p0"], torch.tensor(2.0))
    assert torch.isclose(reduced.prompt_means["p1"], torch.tensor(6.0))
    assert torch.isclose(reduced.local_prompt_sum, torch.tensor(8.0))


def test_full_vocab_kl_is_actor_to_reference_and_keeps_actor_gradient() -> None:
    actor = torch.tensor([[2.0, 0.0], [0.5, -0.5]], requires_grad=True)
    reference = torch.tensor([[0.0, 2.0], [0.5, -0.5]])
    row_kl = actor_to_reference_full_vocab_kl(
        actor,
        reference,
        vocabulary_chunk_size=1,
    )
    assert row_kl[0] > 0
    assert torch.isclose(row_kl[1], torch.tensor(0.0), atol=1.0e-7)
    row_kl.sum().backward()
    assert actor.grad is not None
    assert reference.grad is None


def test_kl_action_tokens_use_preceding_causal_state() -> None:
    action_tokens = torch.tensor([[0, 0, 1, 1, 0]], dtype=torch.bool)
    states = causal_action_state_mask(action_tokens)
    assert states.tolist() == [[False, True, True, False, False]]
    logits = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
    selected = select_action_logit_rows(logits, action_tokens)
    torch.testing.assert_close(selected, logits[0, 1:3])


def test_action_token_at_position_zero_fails_closed() -> None:
    with pytest.raises(ValueError, match="index 0"):
        causal_action_state_mask(torch.tensor([1, 0], dtype=torch.bool))


def test_reference_model_must_be_eval_and_frozen() -> None:
    reference = torch.nn.Linear(2, 2)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    assert_reference_frozen(reference)
    reference.train()
    with pytest.raises(ValueError, match="eval mode"):
        assert_reference_frozen(reference)


def test_turn_ratio_is_near_one_but_differentiable() -> None:
    current = torch.tensor([0.1, 0.2, -0.3], requires_grad=True)
    old = current.detach().clone()
    ratios = compute_turn_ratios(
        current,
        old,
        torch.tensor([1, 1, 1]),
        torch.tensor([0, 0, 1]),
    )
    assert torch.isclose(ratios[0], torch.tensor(1.0))
    (ratios[0] + ratios[1]).backward()
    assert current.grad is not None
    assert current.grad.abs().sum() > 0


def test_turn_ratio_rejects_non_differentiable_current_policy() -> None:
    current = torch.tensor([0.1])
    with pytest.raises(ValueError, match="retain gradients"):
        compute_turn_ratios(
            current,
            current.clone(),
            torch.tensor([1]),
            torch.tensor([0]),
        )


def test_adaptive_clip_constants_are_frozen() -> None:
    assert CLIPPING_MODE == "a2tgpo_adaptive_turn_level"
    assert ADAPTIVE_CLIP_BETA == 0.3
    assert ADAPTIVE_CLIP_EPSILON_LOW == 0.003
    assert ADAPTIVE_CLIP_EPSILON_HIGH == 0.004
    assert ANSWER_CLIP_SCALE == 1.0


def test_zero_normalized_ig_uses_neutral_base_window() -> None:
    assert adaptive_clip_scale(0.0) == 1.0
    ratio = torch.tensor(1.01, requires_grad=True)
    result = a2tgpo_adaptive_turn_objective(
        {0: ratio},
        {0: 1.0},
        {0: 0.0},
        answer_turn_ids=(),
    )
    assert result.clip_scale_by_turn == {0: 1.0}
    assert result.lower_bound_by_turn == {0: 0.997}
    assert result.upper_bound_by_turn == {0: 1.004}


def test_positive_ig_widens_and_negative_ig_narrows_window() -> None:
    positive = adaptive_clip_scale(2.0)
    negative = adaptive_clip_scale(-2.0)
    assert positive > 1.0
    assert negative < 1.0
    assert 0.7 < negative < positive < 1.3
    assert positive * ADAPTIVE_CLIP_EPSILON_HIGH > ADAPTIVE_CLIP_EPSILON_HIGH
    assert negative * ADAPTIVE_CLIP_EPSILON_HIGH < ADAPTIVE_CLIP_EPSILON_HIGH


@pytest.mark.parametrize("normalized_ig", [-100.0, -2.0, 0.0, 2.0, 100.0])
def test_adaptive_scale_always_stays_in_open_bounds(normalized_ig) -> None:
    assert 0.7 < adaptive_clip_scale(normalized_ig) < 1.3


def test_answer_turn_uses_neutral_scale_without_fake_ig() -> None:
    ratio = torch.tensor(1.01, requires_grad=True)
    result = a2tgpo_adaptive_turn_objective(
        {4: ratio},
        {4: 1.0},
        {},
        answer_turn_ids=(4,),
    )
    assert result.clip_scale_by_turn == {4: 1.0}
    assert result.lower_bound_by_turn == {4: 0.997}
    assert result.upper_bound_by_turn == {4: 1.004}


def test_adaptive_surrogate_preserves_current_policy_gradient() -> None:
    ratio = torch.tensor(1.0, requires_grad=True)
    result = a2tgpo_adaptive_turn_objective(
        {0: ratio},
        {0: 2.0},
        {0: 1.0},
        answer_turn_ids=(),
    )
    result.objective_by_turn[0].backward()
    assert ratio.grad is not None
    assert ratio.grad > 0


def test_fixed_dapo_boundaries_are_absent_from_active_implementation() -> None:
    project = Path(__file__).resolve().parents[1]
    active = (
        (project / "src/agentic_rl/policy/strict_onpolicy_loss.py").read_text()
        + (project / "configs/base.yaml").read_text()
    )
    assert "fixed_" + "dapo" not in active.lower()
    assert ("0." + "8") not in active
    assert ("1." + "28") not in active


def test_world_size_compensation_and_total_loss() -> None:
    task_sum = torch.tensor(3.0, requires_grad=True)
    kl_sum = torch.tensor(0.5, requires_grad=True)
    total, task, kl = combine_task_and_kl(
        task_sum,
        kl_sum,
        global_prompt_count=4,
        world_size=4,
        kl_coefficient=0.01,
    )
    assert torch.isclose(task, torch.tensor(3.0))
    assert torch.isclose(kl, torch.tensor(0.5))
    assert torch.isclose(total, torch.tensor(-2.995))


def test_uneven_rank_prompt_counts_reconstruct_global_prompt_mean_gradient() -> None:
    parameter = torch.tensor(1.0, requires_grad=True)
    rank0_sum = parameter * 2.0
    rank1_sum = parameter * 10.0
    rank0_loss = distributed_local_mean_loss(
        rank0_sum,
        global_prompt_count=4,
        world_size=2,
    )
    rank1_loss = distributed_local_mean_loss(
        rank1_sum,
        global_prompt_count=4,
        world_size=2,
    )
    ddp_mean_loss = (rank0_loss + rank1_loss) / 2.0
    ddp_mean_loss.backward()
    assert torch.isclose(ddp_mean_loss, torch.tensor(3.0))
    assert torch.isclose(parameter.grad, torch.tensor(3.0))
