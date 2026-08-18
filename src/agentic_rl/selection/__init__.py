"""Dual-channel RAGEN selection primitives."""

from .candidate_pool import (
    ANSWER_OUTCOME_ONLY_SELECTION_SIGNAL,
    ANSWER_OUTCOME_ONLY_RAGEN2_PAPER_VARIANCE_TOP_P_MODE,
    ANSWER_OUTCOME_ONLY_SCALED_TOP_P_MODE,
    DUAL_CHANNEL_SELECTION_SIGNAL,
    DUAL_CHANNEL_SCALED_TOP_P_MODE,
    CandidatePool,
    PromptGroup,
    SelectionDecision,
)
from .channel_scale import ChannelScaleState
from .paper_ragen2 import (
    compute_ragen2_paper_sample_variance,
    select_ragen2_raw_variance_mass_top_p,
)

__all__ = [
    "ANSWER_OUTCOME_ONLY_SELECTION_SIGNAL",
    "ANSWER_OUTCOME_ONLY_RAGEN2_PAPER_VARIANCE_TOP_P_MODE",
    "ANSWER_OUTCOME_ONLY_SCALED_TOP_P_MODE",
    "CandidatePool",
    "DUAL_CHANNEL_SELECTION_SIGNAL",
    "DUAL_CHANNEL_SCALED_TOP_P_MODE",
    "PromptGroup",
    "SelectionDecision",
    "ChannelScaleState",
    "compute_ragen2_paper_sample_variance",
    "select_ragen2_raw_variance_mass_top_p",
]
