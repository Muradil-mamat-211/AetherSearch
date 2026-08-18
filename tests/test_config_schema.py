from pathlib import Path

import pytest

from agentic_rl.config import ConfigError, DEFAULT_CONFIG, load_config, validate_config
from agentic_rl.workers.resource_plan import build_resource_plan


def test_resolved_config_locks_requested_topology_and_algorithm() -> None:
    config = load_config(DEFAULT_CONFIG)
    plan = build_resource_plan(config)
    assert plan.retriever_cuda_visible_devices == "0"
    assert plan.rl_cuda_visible_devices == "1,2,3,4"
    assert plan.rl_world_size == 4
    assert config["rollout"]["group_size"] == 16
    assert config["rollout"]["candidate_prompts_initial"] == 64
    assert config["rollout"]["candidate_prompts_max"] == 128
    assert config["rollout"]["max_num_seqs"] == 64
    assert config["rollout"]["gpu_memory_utilization"] == 0.46
    assert config["hardware"]["expected_cpu_cores"] == 125
    assert config["ray"]["outcome_worker_count"] == 24
    assert config["ray"]["exact_ig_task_builder_count"] == 20
    assert config["data"]["logical_view_mode"] == "deterministic_nq_hotpotqa_40_60"
    assert config["data"]["source_rows"] == 169615
    assert config["data"]["selection_seed"] == 20260708
    assert config["data"]["expected_rows"] == 150745
    assert config["data"]["expected_source_counts"] == {
        "nq": 60298,
        "hotpotqa": 90447,
    }
    assert config["selection"]["minimum_selected_prompts"] == 32
    assert config["selection"]["maximum_selected_prompts"] == 36
    assert config["selection"]["selection_epsilon"] == 0.0
    assert config["selection"]["alpha_ig"] == 0.5
    assert config["selection"]["alpha_outcome"] == 0.5
    assert config["selection"]["minimum_positive_prompts"] == 4
    assert config["selection"]["scale_ema_half_life"] == 10
    assert config["selection"]["health_reference_valid_updates"] == 10
    assert config["selection"]["health_threshold_ratio"] == 0.1
    assert config["policy"]["ppo_epochs"] == 1
    assert config["policy"]["optimizer_mini_steps"] == 1
    assert config["policy"]["clipping_mode"] == "a2tgpo_adaptive_turn_level"
    assert config["policy"]["adaptive_clip_beta"] == 0.3
    assert config["policy"]["adaptive_clip_epsilon_low"] == 0.003
    assert config["policy"]["adaptive_clip_epsilon_high"] == 0.004
    assert config["policy"]["answer_clip_scale"] == 1.0
    assert (
        config["advantage"]["search_task_mode"]
        == "sufficiency_novelty_cumulative_ig_probe_routed_outcome"
    )
    assert config["advantage"]["search_advantage_formula"] == (
        "-1.0 if S_before else -1.0 if N else D_ig_eff + O_route"
    )
    assert config["advantage"]["outcome_fallback_to_search"] is False
    assert config["advantage"]["search_formula_terms"] == [
        "sufficient_before_search",
        "sufficient_after_search",
        "no_new_observation",
        "effective_cumulative_normalized_local_ig",
        "probe_routed_normalized_outcome",
    ]
    assert config["advantage"]["probe_epsilon"] == 1.0e-6
    assert config["advantage"]["future_ig_accumulation"] is True
    assert config["advantage"]["sqrt_n_rescale"] is True
    assert config["advantage"]["sufficiency_probe"] == {
        "enabled": True,
        "pre_search_enabled": True,
        "post_search_enabled": True,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "n": 1,
        "max_tokens": 500,
        "stop": ["</answer>"],
        "exact_match_only": True,
    }
    assert config["advantage"]["answer_formula_terms"] == [
        "normalized_outcome",
        "centered_format_indicator",
    ]
    assert config["advantage"]["sc"] == {
        "enabled": False,
        "shadow_only": True,
        "actor_loss_enabled": False,
    }
    stages = config["update_stages"]
    assert (
        stages["update_1"]["bootstrap_activation_controls_scale_commit"] is False
    )
    assert (
        stages["update_2_to_health_ready"][
            "bootstrap_activation_controls_scale_update"
        ]
        is False
    )
    assert (
        stages["health_ready"]["inactive_channel_freezes_scale_update"] is True
    )
    assert config["exact_ig"]["include_eos_in_target"] is False
    assert (
        config["exact_ig"]["context_overflow_policy"]
        == "official_sequential_then_fail_closed"
    )
    assert (
        config["exact_ig"]["production_precision_mode"]
        == "fp32_exact_ig"
    )
    assert config["exact_ig"]["numerical_gate_status"] == "PASS"
    assert config["exact_ig"]["numerical_gate_reason"] == (
        "structural_and_semantic_audit_pass_numeric_drift_is_telemetry"
    )
    assert config["exact_ig"]["structural_audit_status"] == "PASS"
    assert config["exact_ig"]["autocast_enabled"] is False
    assert config["exact_ig"]["autocast_dtype"] is None
    assert config["exact_ig"]["parameter_dtype"] == "float32"
    assert config["exact_ig"]["logits_dtype"] == "float32"
    assert config["exact_ig"]["log_probs_dtype"] == "float32"
    assert config["exact_ig"]["allow_tf32"] is False
    assert config["exact_ig"]["temperature"] == 1.0
    assert config["exact_ig"]["scoring_logits_mode"] == "official_full_logits"
    assert config["exact_ig"]["selected_positions_enabled"] is False
    assert config["exact_ig"]["parity_rtol"] == 1.0e-5
    assert config["exact_ig"]["parity_atol"] == 2.0e-5
    assert config["exact_ig"]["oracle_canary_rate"] == 0.01
    assert config["exact_ig"]["oracle_canary_fail_closed"] is False
    assert config["exact_ig"]["maximum_phi_safety_abs_diff"] == 1.0e-3
    assert config["exact_ig"]["maximum_ig_safety_abs_diff"] == 1.0e-3
    assert config["exact_ig"]["numeric_ambiguity_epsilon"] > 0
    assert config["exact_ig"]["calibration_p99_ig_abs_diff"] > 0
    assert (
        config["exact_ig"]["runtime_smoke_oracle_canary_fail_closed"] is False
    )
    assert config["exact_ig"]["sequential_oracle_is_production_path"] is False
    schedule = config["formal_schedule"]
    assert schedule["checkpoint_every_successful_updates"] == 20
    assert config["checkpoint"]["smoke_model_checkpoints"] is False
    assert schedule["fixed_eval_every_successful_updates"] == 20
    assert schedule["optimizer_family"] is None
    assert schedule["learning_rate"] is None
    assert schedule["learner_micro_batch_size"] is None
    assert schedule["total_successful_updates"] is None
    smoke_schedule = config["runtime_smoke_schedule"]
    assert smoke_schedule["optimizer_family"] == "AdamW"
    assert smoke_schedule["learning_rate"] == 2.0e-7
    assert smoke_schedule["optimizer_betas"] == [0.9, 0.999]
    assert smoke_schedule["optimizer_epsilon"] == 1.0e-8
    assert smoke_schedule["scheduler"] == "constant"
    assert smoke_schedule["warmup"] == 0
    assert smoke_schedule["learner_micro_batch_size"] == 6


def test_unexpected_third_search_optimization_field_is_rejected() -> None:
    config = load_config(DEFAULT_CONFIG)
    config["advantage"]["third_search_term"] = 1.0
    with pytest.raises(ConfigError, match="Unexpected advantage fields"):
        validate_config(config)


def test_fixed_clipping_fields_are_rejected_as_unknown_policy_fields() -> None:
    config = load_config(DEFAULT_CONFIG)
    config["policy"]["clip_low"] = 0.2
    with pytest.raises(ConfigError, match="Unexpected policy fields"):
        validate_config(config)


def test_oracle_canary_is_observation_only() -> None:
    config = load_config(DEFAULT_CONFIG)
    config["exact_ig"]["oracle_canary_fail_closed"] = True
    with pytest.raises(
        ConfigError,
        match="exact_ig.oracle_canary_fail_closed must be False",
    ):
        validate_config(config)


def test_all_external_assets_are_absolute_references() -> None:
    config = load_config(DEFAULT_CONFIG)
    for key in (
        "actor_model",
        "reference_model",
        "train_data",
        "validation_data",
    ):
        assert Path(config["paths"][key]).is_absolute()
        assert Path(config["paths"][key]).exists()
