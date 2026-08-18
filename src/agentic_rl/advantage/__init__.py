"""A2TGPO and selected-only Stop/Continue Search advantage construction."""

from .a2tgpo import (
    A2TGPOPromptResult,
    SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_MODE,
    SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_ROLE_LOCALIZED_GATE_MODE,
    SUFFICIENCY_NOVELTY_LOCAL_IG_MODE,
    compute_prompt_advantages,
    rebuild_search_advantages,
)
from .stop_continue import (
    NORMALIZED_OUTCOME_MODE,
    STOP_CONTINUE_ADVANTAGE_VERSION,
    STOP_CONTINUE_CONSENSUS_MODE,
    StopContinueAdvantage,
    StopContinueRewardTriple,
    compute_stop_continue_advantages,
)
from .mica_ig import (
    ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE,
    MicaPromptResult,
    MicaSearchCredit,
    MicaTrajectoryResult,
    PromptDepthStats,
    compute_mica_local_advantage,
    compute_mica_return_advantage,
    compute_mica_search_advantage,
    compute_normalized_terminal_outcomes,
    compute_prompt_depth_group_stats,
    compute_raw_ig_returns,
    compute_singleton_outcome_fallback,
)

__all__ = [
    "A2TGPOPromptResult",
    "ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE",
    "MicaPromptResult",
    "MicaSearchCredit",
    "MicaTrajectoryResult",
    "PromptDepthStats",
    "NORMALIZED_OUTCOME_MODE",
    "STOP_CONTINUE_ADVANTAGE_VERSION",
    "STOP_CONTINUE_CONSENSUS_MODE",
    "SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_MODE",
    "SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_ROLE_LOCALIZED_GATE_MODE",
    "SUFFICIENCY_NOVELTY_LOCAL_IG_MODE",
    "StopContinueAdvantage",
    "StopContinueRewardTriple",
    "compute_prompt_advantages",
    "compute_mica_local_advantage",
    "compute_mica_return_advantage",
    "compute_mica_search_advantage",
    "compute_normalized_terminal_outcomes",
    "compute_prompt_depth_group_stats",
    "compute_raw_ig_returns",
    "compute_singleton_outcome_fallback",
    "compute_stop_continue_advantages",
    "rebuild_search_advantages",
]
