from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence


@dataclass(frozen=True)
class KLMemoryEstimate:
    action_token_count: int
    vocabulary_size: int
    actor_dtype_bytes: int
    reference_dtype_bytes: int
    actor_logits_bytes: int
    reference_logits_bytes: int
    combined_logits_bytes: int


def estimate_action_logit_memory(
    *,
    action_token_count: int,
    vocabulary_size: int,
    actor_dtype_bytes: int,
    reference_dtype_bytes: int,
) -> KLMemoryEstimate:
    actor = int(action_token_count) * int(vocabulary_size) * int(actor_dtype_bytes)
    reference = (
        int(action_token_count)
        * int(vocabulary_size)
        * int(reference_dtype_bytes)
    )
    return KLMemoryEstimate(
        action_token_count=int(action_token_count),
        vocabulary_size=int(vocabulary_size),
        actor_dtype_bytes=int(actor_dtype_bytes),
        reference_dtype_bytes=int(reference_dtype_bytes),
        actor_logits_bytes=actor,
        reference_logits_bytes=reference,
        combined_logits_bytes=actor + reference,
    )


def causal_action_state_mask(eligible_policy_token_mask: Any) -> Any:
    """Map eligible model-token positions to preceding causal prediction states."""
    import torch

    if eligible_policy_token_mask.ndim < 1:
        raise ValueError(
            "eligible_policy_token_mask must have at least one dimension"
        )
    mask = eligible_policy_token_mask.bool()
    if bool(mask[..., 0].any().detach().cpu().item()):
        raise ValueError("An action token at sequence index 0 has no causal state")
    state_mask = torch.zeros_like(mask)
    state_mask[..., :-1] = mask[..., 1:]
    return state_mask


def select_action_logit_rows(logits: Any, eligible_policy_token_mask: Any) -> Any:
    if logits.shape[:-1] != eligible_policy_token_mask.shape:
        raise ValueError(
            "Logits leading dimensions must align with eligible_policy_token_mask"
        )
    selected = logits[causal_action_state_mask(eligible_policy_token_mask)]
    if selected.shape[0] == 0:
        raise ValueError("At least one action-token state is required for KL")
    return selected


def actor_to_reference_full_vocab_kl(
    actor_action_logits: Any,
    reference_action_logits: Any,
    *,
    vocabulary_chunk_size: int,
) -> Any:
    import torch

    if actor_action_logits.shape != reference_action_logits.shape:
        raise ValueError("Actor and reference logits must have identical shapes")
    if actor_action_logits.ndim != 2:
        raise ValueError("Action logits must have shape [action_states, vocabulary]")
    if reference_action_logits.requires_grad:
        raise ValueError("Reference logits must be detached")
    if not actor_action_logits.requires_grad:
        raise ValueError("Actor logits must retain gradients")
    if vocabulary_chunk_size <= 0:
        raise ValueError("vocabulary_chunk_size must be positive")

    actor_float = actor_action_logits.float()
    reference_float = reference_action_logits.float().detach()
    actor_log_normalizer = torch.logsumexp(actor_float, dim=-1)
    reference_log_normalizer = torch.logsumexp(reference_float, dim=-1)
    row_kl = torch.zeros_like(actor_log_normalizer)
    vocabulary_size = actor_float.shape[-1]
    for start in range(0, vocabulary_size, vocabulary_chunk_size):
        end = min(start + vocabulary_chunk_size, vocabulary_size)
        actor_log_probability = (
            actor_float[:, start:end] - actor_log_normalizer.unsqueeze(1)
        )
        reference_log_probability = (
            reference_float[:, start:end] - reference_log_normalizer.unsqueeze(1)
        )
        probability = torch.exp(actor_log_probability)
        row_kl = row_kl + torch.sum(
            probability * (actor_log_probability - reference_log_probability),
            dim=-1,
        )
    return row_kl


def assert_reference_frozen(reference_model: Any) -> None:
    if bool(getattr(reference_model, "training", False)):
        raise ValueError("Reference model must be in eval mode")
    trainable = [name for name, parameter in reference_model.named_parameters() if parameter.requires_grad]
    if trainable:
        raise ValueError(
            "Reference model contains trainable parameters: "
            + ", ".join(trainable[:5])
        )


def action_state_full_vocab_kl_from_models(
    actor_model: Any,
    reference_model: Any,
    model_inputs: dict[str, Any],
    eligible_policy_token_mask: Any,
    *,
    action_state_chunk_size: int,
    vocabulary_chunk_size: int,
) -> Any:
    import torch

    assert_reference_frozen(reference_model)
    if action_state_chunk_size <= 0:
        raise ValueError("action_state_chunk_size must be positive")
    actor_backbone = getattr(actor_model, "model", None)
    actor_head = getattr(actor_model, "lm_head", None)
    reference_backbone = getattr(reference_model, "model", None)
    reference_head = getattr(reference_model, "lm_head", None)
    if any(
        component is None
        for component in (
            actor_backbone,
            actor_head,
            reference_backbone,
            reference_head,
        )
    ):
        raise TypeError("Expected causal-LM model/model-backbone/lm_head structure")

    forward_inputs = dict(model_inputs)
    forward_inputs["use_cache"] = False
    actor_hidden = actor_backbone(**forward_inputs).last_hidden_state
    if actor_hidden.shape[:-1] != eligible_policy_token_mask.shape:
        raise ValueError(
            "Actor hidden states and eligible policy-token mask do not align"
        )
    action_state_mask = causal_action_state_mask(eligible_policy_token_mask)
    actor_action_hidden = actor_hidden[action_state_mask]
    if actor_action_hidden.shape[0] == 0:
        raise ValueError("KL requires at least one action-token state")

    with torch.inference_mode():
        reference_hidden = reference_backbone(**forward_inputs).last_hidden_state
        reference_action_hidden = reference_hidden[action_state_mask].detach()
    del actor_hidden, reference_hidden

    row_blocks: list[Any] = []
    for start in range(0, actor_action_hidden.shape[0], action_state_chunk_size):
        end = min(start + action_state_chunk_size, actor_action_hidden.shape[0])
        actor_logits = actor_head(actor_action_hidden[start:end])
        with torch.inference_mode():
            reference_logits = reference_head(
                reference_action_hidden[start:end]
            ).detach()
        row_blocks.append(
            actor_to_reference_full_vocab_kl(
                actor_logits,
                reference_logits,
                vocabulary_chunk_size=vocabulary_chunk_size,
            )
        )
        del actor_logits, reference_logits
    return torch.cat(row_blocks, dim=0)
