from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from agentic_rl.exact_ig.target_schema import (
    ANSWER_SPAN_RESOLUTION_POLICY,
    ANSWER_SCAFFOLD_TEXT,
    CANONICAL_ALIAS_POLICY,
    DEFAULT_TARGET_TEMPLATE,
    EXACT_IG_VERSION,
    FAST_PATH_STRUCTURE,
    INFO_GAIN_TYPE,
    MASK_BUILDER_VERSION,
    OFFICIAL_IGPO_COMMIT_SHA,
    POSITION_BUILDER_VERSION,
    PRODUCTION_PRECISION_MODE,
    SCAFFOLD_SHA256,
    SCORE_MASK_POLICY,
    TARGET_SCHEMA_PREFIX,
    TARGET_SCHEMA_SUFFIX,
    TARGET_TOKENIZATION_POLICY,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


class ConfigError(ValueError):
    pass


def _merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ConfigError(f"YAML root must be a mapping: {path}")
    return value


def _load_config_tree(config_path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    config_path = config_path.resolve()
    if config_path in stack:
        cycle = " -> ".join(str(path) for path in (*stack, config_path))
        raise ConfigError(f"Configuration inheritance cycle: {cycle}")
    config = _read_yaml(config_path)
    extends = config.pop("extends", None)
    if extends is not None:
        parent_path = (config_path.parent / str(extends)).resolve()
        parent = _load_config_tree(parent_path, (*stack, config_path))
        config = _merge(parent, config)
    includes = config.pop("includes", [])
    if not isinstance(includes, list):
        raise ConfigError("includes must be a list")
    for include in includes:
        include_path = (config_path.parent / str(include)).resolve()
        config = _merge(config, _read_yaml(include_path))
    return config


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = _load_config_tree(config_path)
    validate_config(config)
    return config


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ConfigError(f"{field} must be {expected!r}, got {actual!r}")


def _require_paths(config: Mapping[str, Any], keys: Iterable[str]) -> None:
    paths = config.get("paths", {})
    for key in keys:
        raw = paths.get(key)
        if not raw:
            raise ConfigError(f"Missing paths.{key}")
        if not Path(str(raw)).exists():
            raise ConfigError(f"Path does not exist: paths.{key}={raw}")


def validate_config(config: Mapping[str, Any]) -> None:
    _require_equal(
        config.get("project", {}).get("schema_version"),
        "2.1",
        "project.schema_version",
    )
    for key in ("design_report", "algorithm_overrides"):
        raw = config.get("project", {}).get(key)
        if not raw or not Path(str(raw)).exists():
            raise ConfigError(f"Path does not exist: project.{key}={raw}")
    _require_paths(
        config,
        (
            "search_r1_root",
            "environment_script",
            "rl_python",
            "retriever_python",
            "actor_model",
            "reference_model",
            "train_data",
            "validation_data",
        ),
    )
    hardware = config["hardware"]
    rollout = config["rollout"]
    selection = config["selection"]
    advantage = config["advantage"]
    policy = config["policy"]
    learner = config["learner"]
    exact_ig = config["exact_ig"]
    retriever = config["retriever"]
    data = config["data"]

    _require_equal(hardware["retriever_physical_gpu"], 0, "hardware.retriever_physical_gpu")
    migration_mode = bool(hardware.get("allow_world_size_change_on_resume", False))
    rl_world_size = int(hardware["rl_world_size"])
    expected_cpu_cores = int(hardware["expected_cpu_cores"])
    if migration_mode:
        if rl_world_size not in {3, 4}:
            raise ConfigError("Resume world-size migration supports only 3 or 4 ranks")
        if expected_cpu_cores == 48:
            _require_equal(
                hardware["rl_physical_gpus"],
                [1, 2, 3],
                "hardware.rl_physical_gpus",
            )
            _require_equal(
                hardware["rl_visible_gpus"],
                [0, 1, 2],
                "hardware.rl_visible_gpus",
            )
            _require_equal(rl_world_size, 3, "hardware.rl_world_size")
        elif expected_cpu_cores != 125:
            raise ConfigError(
                "World-size migration CPU profiles must use 125 or 48 cores"
            )
    else:
        _require_equal(hardware["rl_physical_gpus"], [1, 2, 3, 4], "hardware.rl_physical_gpus")
        _require_equal(rl_world_size, 4, "hardware.rl_world_size")
        _require_equal(expected_cpu_cores, 125, "hardware.expected_cpu_cores")
    ray_config = config["ray"]
    agent_loop_worker_count = int(ray_config.get("agent_loop_worker_count", 32))
    if agent_loop_worker_count < 1:
        raise ConfigError("ray.agent_loop_worker_count must be positive")
    if expected_cpu_cores == 48:
        _require_equal(
            hardware["cpu_reserved_for_os"],
            4,
            "hardware.cpu_reserved_for_os",
        )
        _require_equal(ray_config["retriever_pool_cpus"], 8, "ray.retriever_pool_cpus")
        _require_equal(ray_config["rl_engine_cpus_per_gpu"], 3, "ray.rl_engine_cpus_per_gpu")
        _require_equal(ray_config["controller_cpu_workers"], 5, "ray.controller_cpu_workers")
        _require_equal(ray_config["outcome_worker_count"], 4, "ray.outcome_worker_count")
        _require_equal(
            ray_config["exact_ig_task_builder_count"],
            2,
            "ray.exact_ig_task_builder_count",
        )
        _require_equal(agent_loop_worker_count, 12, "ray.agent_loop_worker_count")
    else:
        _require_equal(ray_config["retriever_pool_cpus"], 20, "ray.retriever_pool_cpus")
        _require_equal(ray_config["rl_engine_cpus_per_gpu"], 6, "ray.rl_engine_cpus_per_gpu")
        _require_equal(ray_config["controller_cpu_workers"], 71, "ray.controller_cpu_workers")
        _require_equal(ray_config["outcome_worker_count"], 24, "ray.outcome_worker_count")
        _require_equal(
            ray_config["exact_ig_task_builder_count"],
            20,
            "ray.exact_ig_task_builder_count",
        )
    cpu_budget = (
        int(hardware["cpu_reserved_for_os"])
        + int(ray_config["retriever_pool_cpus"])
        + int(hardware["rl_world_size"])
        * int(ray_config["rl_engine_cpus_per_gpu"])
        + int(ray_config["controller_cpu_workers"])
    )
    if cpu_budget > int(hardware["expected_cpu_cores"]):
        raise ConfigError("hardware/ray CPU budget exceeds expected CPU cores")
    _require_equal(rollout["data_parallel_size"], rl_world_size, "rollout.data_parallel_size")
    _require_equal(rollout["tensor_parallel_size"], 1, "rollout.tensor_parallel_size")
    _require_equal(rollout["group_size"], 16, "rollout.group_size")
    _require_equal(rollout["prompt_wave_size"], 32, "rollout.prompt_wave_size")
    _require_equal(rollout["candidate_prompts_initial"], 64, "rollout.candidate_prompts_initial")
    _require_equal(rollout["refill_prompts"], 32, "rollout.refill_prompts")
    _require_equal(rollout["candidate_prompts_max"], 128, "rollout.candidate_prompts_max")
    _require_equal(rollout["max_num_seqs"], 64, "rollout.max_num_seqs")
    expected_gpu_memory_utilization = 0.48 if expected_cpu_cores == 48 else 0.46
    _require_equal(
        float(rollout["gpu_memory_utilization"]),
        expected_gpu_memory_utilization,
        "rollout.gpu_memory_utilization",
    )
    if expected_cpu_cores == 48:
        _require_equal(
            int(config["formal_schedule"]["learner_micro_batch_size"]),
            6,
            "formal_schedule.learner_micro_batch_size",
        )
    _require_equal(
        data["logical_view_mode"],
        "deterministic_nq_hotpotqa_40_60",
        "data.logical_view_mode",
    )
    _require_equal(data["source_rows"], 169615, "data.source_rows")
    _require_equal(data["selection_seed"], 20260708, "data.selection_seed")
    _require_equal(data["nq_ratio"], 0.4, "data.nq_ratio")
    _require_equal(data["expected_rows"], 150745, "data.expected_rows")
    _require_equal(
        data["expected_source_counts"],
        {"nq": 60298, "hotpotqa": 90447},
        "data.expected_source_counts",
    )
    _require_equal(data["derive_or_copy_data"], False, "data.derive_or_copy_data")
    _require_equal(selection["minimum_selected_prompts"], 32, "selection.minimum_selected_prompts")
    _require_equal(selection["target_selected_prompts"], 36, "selection.target_selected_prompts")
    _require_equal(selection["maximum_selected_prompts"], 36, "selection.maximum_selected_prompts")
    _require_equal(selection["top_p_mass"], 0.9, "selection.top_p_mass")
    _require_equal(selection["include_zero"], False, "selection.include_zero")
    _require_equal(
        selection["selection_epsilon"], 0.0, "selection.selection_epsilon"
    )
    _require_equal(selection["alpha_ig"], 0.5, "selection.alpha_ig")
    _require_equal(selection["alpha_outcome"], 0.5, "selection.alpha_outcome")
    _require_equal(
        selection["minimum_positive_prompts"],
        4,
        "selection.minimum_positive_prompts",
    )
    _require_equal(
        selection["scale_ema_half_life"],
        10,
        "selection.scale_ema_half_life",
    )
    _require_equal(
        selection["health_reference_valid_updates"],
        10,
        "selection.health_reference_valid_updates",
    )
    _require_equal(
        selection["health_threshold_ratio"],
        0.1,
        "selection.health_threshold_ratio",
    )
    _require_equal(
        selection["recompute_after_refill_on_full_pool"],
        True,
        "selection.recompute_after_refill_on_full_pool",
    )
    _require_equal(policy["ppo_epochs"], 1, "policy.ppo_epochs")
    _require_equal(policy["optimizer_mini_steps"], 1, "policy.optimizer_mini_steps")
    _require_equal(policy["optimizer_steps_per_successful_update"], 1, "policy.optimizer_steps_per_successful_update")
    _require_equal(policy["kl_coefficient"], 0.01, "policy.kl_coefficient")
    _require_equal(policy["strict_on_policy"], True, "policy.strict_on_policy")
    _require_equal(policy["ratio_level"], "turn", "policy.ratio_level")
    _require_equal(policy["ratio_hardcoded"], False, "policy.ratio_hardcoded")
    _require_equal(
        policy["clipping_mode"],
        "a2tgpo_adaptive_turn_level",
        "policy.clipping_mode",
    )
    _require_equal(
        policy["adaptive_clip_beta"],
        0.3,
        "policy.adaptive_clip_beta",
    )
    _require_equal(
        policy["adaptive_clip_epsilon_low"],
        0.003,
        "policy.adaptive_clip_epsilon_low",
    )
    _require_equal(
        policy["adaptive_clip_epsilon_high"],
        0.004,
        "policy.adaptive_clip_epsilon_high",
    )
    _require_equal(
        policy["answer_clip_scale"],
        1.0,
        "policy.answer_clip_scale",
    )
    _require_equal(
        policy["task_reduction"],
        "prompt_trajectory_action_token_mean",
        "policy.task_reduction",
    )
    _require_equal(
        policy["kl_reduction"],
        "prompt_trajectory_action_token_mean",
        "policy.kl_reduction",
    )
    _require_equal(
        policy["full_vocab_reference_kl"],
        True,
        "policy.full_vocab_reference_kl",
    )
    _require_equal(
        policy["entropy_coefficient"],
        0.0,
        "policy.entropy_coefficient",
    )
    _require_equal(
        policy["value_coefficient"],
        0.0,
        "policy.value_coefficient",
    )
    _require_equal(policy["max_grad_norm"], 1.0, "policy.max_grad_norm")
    _require_equal(learner["strategy"], "fsdp2", "learner.strategy")
    _require_equal(learner["world_size"], rl_world_size, "learner.world_size")
    _require_equal(learner["reshard_after_forward"], False, "learner.reshard_after_forward")
    _require_equal(
        learner["reference_reshard_after_forward"],
        False,
        "learner.reference_reshard_after_forward",
    )
    _require_equal(advantage["gamma"], 1.0, "advantage.gamma")
    search_task_mode = str(advantage["search_task_mode"])
    legacy_probe_routed_mode = (
        "sufficiency_novelty_cumulative_ig_probe_routed_outcome"
    )
    role_localized_mode = (
        "sufficiency_novelty_cumulative_ig_probe_routed_outcome_"
        "role_localized_gate"
    )
    mica_mode = "answer_only_ragen2_mica_ig_v1_singleton_outcome"
    if search_task_mode not in {
        legacy_probe_routed_mode,
        role_localized_mode,
        mica_mode,
    }:
        raise ConfigError(
            "advantage.search_task_mode is not an approved production mode: "
            f"{search_task_mode}"
        )
    _require_equal(
        advantage["lambda_outcome"],
        1.0,
        "advantage.lambda_outcome",
    )
    _require_equal(
        advantage["lambda_format"],
        1.0,
        "advantage.lambda_format",
    )
    if search_task_mode == role_localized_mode:
        expected_search_formula = (
            "J_main + lambda_d*J_decision + lambda_q*J_query"
        )
    elif search_task_mode == mica_mode:
        expected_search_formula = (
            "Z_O if peer_count == 1 else 0.5*A_ret + 0.5*A_loc"
        )
    else:
        expected_search_formula = (
            "-1.0 if S_before else -1.0 if N else D_ig_eff + O_route"
        )
    _require_equal(
        advantage["search_advantage_formula"],
        expected_search_formula,
        "advantage.search_advantage_formula",
    )
    _require_equal(
        advantage["outcome_fallback_to_search"],
        search_task_mode == mica_mode,
        "advantage.outcome_fallback_to_search",
    )
    _require_equal(
        exact_ig["exact_ig_version"],
        EXACT_IG_VERSION,
        "exact_ig.exact_ig_version",
    )
    _require_equal(
        exact_ig["official_igpo_commit_sha"],
        OFFICIAL_IGPO_COMMIT_SHA,
        "exact_ig.official_igpo_commit_sha",
    )
    _require_equal(
        exact_ig["canonical_alias_policy"],
        CANONICAL_ALIAS_POLICY,
        "exact_ig.canonical_alias_policy",
    )
    _require_equal(
        exact_ig["score_mask_policy"],
        SCORE_MASK_POLICY,
        "exact_ig.score_mask_policy",
    )
    _require_equal(
        exact_ig["info_gain_type"],
        INFO_GAIN_TYPE,
        "exact_ig.info_gain_type",
    )
    _require_equal(
        exact_ig["fast_path_structure"],
        FAST_PATH_STRUCTURE,
        "exact_ig.fast_path_structure",
    )
    _require_equal(
        exact_ig["target_tokenization_policy"],
        TARGET_TOKENIZATION_POLICY,
        "exact_ig.target_tokenization_policy",
    )
    _require_equal(
        exact_ig["encode_complete_target_once_per_prompt"],
        True,
        "exact_ig.encode_complete_target_once_per_prompt",
    )
    _require_equal(
        exact_ig["answer_span_resolution"],
        ANSWER_SPAN_RESOLUTION_POLICY,
        "exact_ig.answer_span_resolution",
    )
    _require_equal(
        exact_ig["score_answer_body_tokens_only"],
        True,
        "exact_ig.score_answer_body_tokens_only",
    )
    _require_equal(exact_ig["stop_gradient"], True, "exact_ig.stop_gradient")
    _require_equal(exact_ig["use_cache"], False, "exact_ig.use_cache")
    _require_equal(
        exact_ig["production_precision_mode"],
        PRODUCTION_PRECISION_MODE,
        "exact_ig.production_precision_mode",
    )
    if exact_ig["numerical_gate_status"] not in {"PASS", "FAIL"}:
        raise ConfigError("exact_ig.numerical_gate_status must be PASS or FAIL")
    _require_equal(
        exact_ig["structural_audit_status"],
        "PASS",
        "exact_ig.structural_audit_status",
    )
    gate_path = Path(str(exact_ig["structural_audit_path"]))
    if not gate_path.is_absolute():
        raise ConfigError("exact_ig.structural_audit_path must be absolute")
    if exact_ig["numerical_gate_status"] == "PASS":
        if not gate_path.is_file():
            raise ConfigError(
                f"Exact-IG structural audit result does not exist: {gate_path}"
            )
    _require_equal(
        exact_ig["sequential_oracle_is_production_path"],
        False,
        "exact_ig.sequential_oracle_is_production_path",
    )
    _require_equal(
        exact_ig["autocast_enabled"],
        False,
        "exact_ig.autocast_enabled",
    )
    _require_equal(
        exact_ig["autocast_dtype"],
        None,
        "exact_ig.autocast_dtype",
    )
    for key in (
        "parameter_dtype",
        "activation_dtype",
        "logits_dtype",
        "log_probs_dtype",
    ):
        _require_equal(exact_ig[key], "float32", f"exact_ig.{key}")
    for key in (
        "allow_tf32",
        "allow_bf16_reduced_precision_reduction",
        "allow_fp16_reduced_precision_reduction",
    ):
        _require_equal(exact_ig[key], False, f"exact_ig.{key}")
    _require_equal(
        exact_ig["float32_matmul_precision"],
        "highest",
        "exact_ig.float32_matmul_precision",
    )
    _require_equal(
        float(exact_ig["temperature"]),
        1.0,
        "exact_ig.temperature",
    )
    if exact_ig["scoring_logits_mode"] not in {
        "official_full_logits",
        "selected_positions",
    }:
        raise ConfigError("Unsupported exact_ig.scoring_logits_mode")
    if exact_ig["selected_positions_gate_status"] not in {
        "UNVERIFIED",
        "PASS",
        "FAIL",
        "PARITY_PASS_DISABLED_BY_BASELINE_FAILURE",
    }:
        raise ConfigError(
            "exact_ig.selected_positions_gate_status must be "
            "UNVERIFIED, PASS, FAIL, or "
            "PARITY_PASS_DISABLED_BY_BASELINE_FAILURE"
        )
    if bool(exact_ig["selected_positions_enabled"]):
        _require_equal(
            exact_ig["selected_positions_gate_status"],
            "PASS",
            "exact_ig.selected_positions_gate_status",
        )
        _require_equal(
            exact_ig["scoring_logits_mode"],
            "selected_positions",
            "exact_ig.scoring_logits_mode",
        )
    else:
        _require_equal(
            exact_ig["scoring_logits_mode"],
            "official_full_logits",
            "exact_ig.scoring_logits_mode",
        )
    if exact_ig["attention_mask_mode"] not in {
        "official_additive",
        "boolean_4d",
    }:
        raise ConfigError("Unsupported exact_ig.attention_mask_mode")
    if exact_ig["boolean_4d_gate_status"] not in {
        "UNVERIFIED",
        "PASS",
        "FAIL",
    }:
        raise ConfigError(
            "exact_ig.boolean_4d_gate_status must be UNVERIFIED, PASS, or FAIL"
        )
    if bool(exact_ig["boolean_4d_enabled"]):
        _require_equal(
            exact_ig["boolean_4d_gate_status"],
            "PASS",
            "exact_ig.boolean_4d_gate_status",
        )
        _require_equal(
            exact_ig["attention_mask_mode"],
            "boolean_4d",
            "exact_ig.attention_mask_mode",
        )
    else:
        _require_equal(
            exact_ig["attention_mask_mode"],
            "official_additive",
            "exact_ig.attention_mask_mode",
        )
    _require_equal(
        float(exact_ig["parity_rtol"]),
        1.0e-5,
        "exact_ig.parity_rtol",
    )
    _require_equal(
        float(exact_ig["parity_atol"]),
        2.0e-5,
        "exact_ig.parity_atol",
    )
    for key, expected in (
        ("maximum_token_log_prob_abs_diff", 2.0e-5),
        ("maximum_phi_abs_diff", 2.0e-5),
        ("maximum_ig_abs_diff", 2.0e-5),
        ("maximum_phi_safety_abs_diff", 1.0e-3),
        ("maximum_ig_safety_abs_diff", 1.0e-3),
        ("maximum_telescoping_error", 1.0e-10),
    ):
        _require_equal(float(exact_ig[key]), expected, f"exact_ig.{key}")
    if float(exact_ig["numeric_ambiguity_epsilon"]) <= 0:
        raise ConfigError("exact_ig.numeric_ambiguity_epsilon must be positive")
    if float(exact_ig["calibration_p99_ig_abs_diff"]) <= 0:
        raise ConfigError(
            "exact_ig.calibration_p99_ig_abs_diff must be positive"
        )
    if int(exact_ig["minimum_canary_samples_for_p99"]) < 2:
        raise ConfigError(
            "exact_ig.minimum_canary_samples_for_p99 must be at least 2"
        )
    _require_equal(
        exact_ig["oracle_numeric_difference_policy"],
        "telemetry_only_unless_semantic_or_safety_drift",
        "exact_ig.oracle_numeric_difference_policy",
    )
    _require_equal(
        int(exact_ig["logits_element_size"]),
        4,
        "exact_ig.logits_element_size",
    )
    if not 0.0 <= float(exact_ig["oracle_canary_rate"]) <= 0.02:
        raise ConfigError("exact_ig.oracle_canary_rate must be in [0, 0.02]")
    _require_equal(
        bool(exact_ig["oracle_canary_fail_closed"]),
        False,
        "exact_ig.oracle_canary_fail_closed",
    )
    _require_equal(
        exact_ig["target_template"],
        DEFAULT_TARGET_TEMPLATE,
        "exact_ig.target_template",
    )
    _require_equal(
        exact_ig["scaffold_text"],
        ANSWER_SCAFFOLD_TEXT,
        "exact_ig.scaffold_text",
    )
    _require_equal(
        exact_ig["scaffold_sha256"],
        SCAFFOLD_SHA256,
        "exact_ig.scaffold_sha256",
    )
    _require_equal(
        exact_ig["target_schema_prefix"],
        TARGET_SCHEMA_PREFIX,
        "exact_ig.target_schema_prefix",
    )
    _require_equal(
        exact_ig["target_schema_suffix"],
        TARGET_SCHEMA_SUFFIX,
        "exact_ig.target_schema_suffix",
    )
    _require_equal(
        exact_ig["structural_attention_mask"],
        True,
        "exact_ig.structural_attention_mask",
    )
    _require_equal(
        exact_ig["logical_position_ids"],
        True,
        "exact_ig.logical_position_ids",
    )
    _require_equal(
        exact_ig["causal_target_shift"],
        True,
        "exact_ig.causal_target_shift",
    )
    _require_equal(
        exact_ig["mask_builder_version"],
        MASK_BUILDER_VERSION,
        "exact_ig.mask_builder_version",
    )
    _require_equal(
        exact_ig["position_builder_version"],
        POSITION_BUILDER_VERSION,
        "exact_ig.position_builder_version",
    )
    if int(exact_ig["max_records_per_forward"]) < 1:
        raise ConfigError("exact_ig.max_records_per_forward must be positive")
    for key in (
        "max_attention_cost_per_batch",
        "max_extended_tokens_per_batch",
        "max_full_logits_bytes",
        "max_selected_logits_bytes",
    ):
        value = exact_ig[key]
        if value is not None and int(value) < 1:
            raise ConfigError(f"exact_ig.{key} must be positive or null")
    _require_equal(
        exact_ig["include_eos_in_target"], False, "exact_ig.include_eos_in_target"
    )
    _require_equal(
        exact_ig["tokenizer_add_special_tokens"],
        False,
        "exact_ig.tokenizer_add_special_tokens",
    )
    _require_equal(
        exact_ig["context_overflow_policy"],
        "official_sequential_then_fail_closed",
        "exact_ig.context_overflow_policy",
    )
    expected_search_terms = (
        [
            "raw_exact_ig",
            "literal_raw_ig_suffix_return",
            "prompt_search_depth_local_advantage",
            "prompt_search_depth_return_advantage",
            "singleton_normalized_terminal_outcome",
        ]
        if search_task_mode == mica_mode
        else [
            "sufficient_before_search",
            "sufficient_after_search",
            "no_new_observation",
            "effective_cumulative_normalized_local_ig",
            "probe_routed_normalized_outcome",
        ]
    )
    if search_task_mode == role_localized_mode:
        expected_search_terms.extend(
            [
                "role_localized_main_credit",
                "decision_segment_gate",
                "query_segment_gate",
            ]
        )
    _require_equal(
        advantage["search_formula_terms"],
        expected_search_terms,
        "advantage.search_formula_terms",
    )
    _require_equal(
        advantage["answer_formula_terms"],
        ["normalized_outcome", "centered_format_indicator"],
        "advantage.answer_formula_terms",
    )

    allowed_advantage_keys = {
        "mode",
        "search_task_mode",
        "gamma",
        "lambda_outcome",
        "lambda_format",
        "probe_epsilon",
        "search_advantage_formula",
        "outcome_fallback_to_search",
        "normalization_epsilon",
        "zero_variance_tolerance",
        "search_formula_terms",
        "answer_formula_terms",
        "rescale_count_mode",
        "future_ig_accumulation",
        "sqrt_n_rescale",
        "external_ig_multiplier",
        "sufficiency_probe",
        "role_localized_gate",
        "sc",
    }
    unexpected_advantage = set(advantage) - allowed_advantage_keys
    if unexpected_advantage:
        raise ConfigError(
            "Unexpected advantage fields: "
            + ", ".join(sorted(unexpected_advantage))
        )
    _require_equal(
        advantage["future_ig_accumulation"],
        search_task_mode != mica_mode,
        "advantage.future_ig_accumulation",
    )
    _require_equal(
        advantage["sqrt_n_rescale"],
        search_task_mode != mica_mode,
        "advantage.sqrt_n_rescale",
    )
    _require_equal(
        advantage["external_ig_multiplier"],
        None,
        "advantage.external_ig_multiplier",
    )
    _require_equal(
        advantage["rescale_count_mode"],
        (
            "none_mica_raw_suffix_return"
            if search_task_mode == mica_mode
            else "effective_valid_ig_until_first_s_after"
        ),
        "advantage.rescale_count_mode",
    )
    _require_equal(
        float(advantage["probe_epsilon"]),
        1.0e-6,
        "advantage.probe_epsilon",
    )
    probe = advantage["sufficiency_probe"]
    _require_equal(
        probe["enabled"],
        search_task_mode != mica_mode,
        "advantage.sufficiency_probe.enabled",
    )
    _require_equal(
        probe["pre_search_enabled"],
        search_task_mode != mica_mode,
        "advantage.sufficiency_probe.pre_search_enabled",
    )
    _require_equal(
        probe["post_search_enabled"],
        search_task_mode != mica_mode,
        "advantage.sufficiency_probe.post_search_enabled",
    )
    _require_equal(
        probe["do_sample"], False, "advantage.sufficiency_probe.do_sample"
    )
    _require_equal(
        float(probe["temperature"]),
        0.0,
        "advantage.sufficiency_probe.temperature",
    )
    _require_equal(
        float(probe["top_p"]), 1.0, "advantage.sufficiency_probe.top_p"
    )
    _require_equal(
        int(probe["top_k"]),
        -1,
        "advantage.sufficiency_probe.top_k",
    )
    _require_equal(
        float(probe["min_p"]),
        0.0,
        "advantage.sufficiency_probe.min_p",
    )
    _require_equal(
        int(probe["n"]),
        1,
        "advantage.sufficiency_probe.n",
    )
    _require_equal(
        probe["exact_match_only"],
        True,
        "advantage.sufficiency_probe.exact_match_only",
    )
    _require_equal(
        int(probe["max_tokens"]),
        500,
        "advantage.sufficiency_probe.max_tokens",
    )
    _require_equal(
        list(probe["stop"]),
        ["</answer>"],
        "advantage.sufficiency_probe.stop",
    )
    allowed_probe_keys = {
        "enabled",
        "pre_search_enabled",
        "post_search_enabled",
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "n",
        "max_tokens",
        "stop",
        "exact_match_only",
    }
    unexpected_probe = set(probe) - allowed_probe_keys
    if unexpected_probe:
        raise ConfigError(
            "Unexpected advantage.sufficiency_probe fields: "
            + ", ".join(sorted(unexpected_probe))
        )

    sc = advantage["sc"]
    _require_equal(sc["enabled"], False, "advantage.sc.enabled")
    _require_equal(
        sc["shadow_only"], True, "advantage.sc.shadow_only"
    )
    _require_equal(
        sc["actor_loss_enabled"],
        False,
        "advantage.sc.actor_loss_enabled",
    )
    allowed_sc_keys = {
        "enabled",
        "shadow_only",
        "actor_loss_enabled",
    }
    unexpected_sc = set(sc) - allowed_sc_keys
    if unexpected_sc:
        raise ConfigError(
            "Unexpected advantage.sc fields: "
            + ", ".join(sorted(unexpected_sc))
        )

    role_gate = advantage.get("role_localized_gate")
    if search_task_mode == role_localized_mode:
        if not isinstance(role_gate, Mapping):
            raise ConfigError("advantage.role_localized_gate must be a mapping")
        allowed_role_gate_keys = {
            "enabled",
            "lambda_decision",
            "lambda_query",
            "calibration_manifest",
            "calibration_manifest_sha256",
            "branch_priority",
            "token_provenance_required",
            "decision_clip_mode",
            "query_clip_mode",
            "decision_reduction",
            "query_reduction",
            "eta_decision",
            "eta_query",
            "max_gate_to_main_grad_ratio",
            "online_lambda_updates",
            "calibration_pending",
        }
        unexpected_role_gate = set(role_gate) - allowed_role_gate_keys
        if unexpected_role_gate:
            raise ConfigError(
                "Unexpected advantage.role_localized_gate fields: "
                + ", ".join(sorted(unexpected_role_gate))
            )
        _require_equal(role_gate["enabled"], True, "role_localized_gate.enabled")
        _require_equal(
            list(role_gate["branch_priority"]),
            ["n_budget", "n_invalid", "s_before", "n_soft", "normal"],
            "role_localized_gate.branch_priority",
        )
        _require_equal(
            role_gate["token_provenance_required"],
            True,
            "role_localized_gate.token_provenance_required",
        )
        _require_equal(
            role_gate["decision_clip_mode"],
            "fixed_dapo",
            "role_localized_gate.decision_clip_mode",
        )
        _require_equal(
            role_gate["query_clip_mode"],
            "fixed_dapo",
            "role_localized_gate.query_clip_mode",
        )
        _require_equal(
            role_gate["decision_reduction"],
            "prompt_rollout_search_event_mean",
            "role_localized_gate.decision_reduction",
        )
        _require_equal(
            role_gate["query_reduction"],
            "prompt_rollout_search_event_mean",
            "role_localized_gate.query_reduction",
        )
        _require_equal(
            float(role_gate["eta_decision"]),
            0.10,
            "role_localized_gate.eta_decision",
        )
        _require_equal(
            float(role_gate["eta_query"]),
            0.05,
            "role_localized_gate.eta_query",
        )
        _require_equal(
            float(role_gate["max_gate_to_main_grad_ratio"]),
            0.15,
            "role_localized_gate.max_gate_to_main_grad_ratio",
        )
        _require_equal(
            role_gate["online_lambda_updates"],
            False,
            "role_localized_gate.online_lambda_updates",
        )
        lambda_decision = float(role_gate["lambda_decision"])
        lambda_query = float(role_gate["lambda_query"])
        if not (
            math.isfinite(lambda_decision)
            and math.isfinite(lambda_query)
            and 0.0 <= lambda_decision <= 1.0
            and 0.0 <= lambda_query <= 1.0
        ):
            raise ConfigError("Calibrated role-localized lambdas must be in [0,1]")
        manifest_path = Path(str(role_gate["calibration_manifest"])).resolve()
        calibration_pending = bool(role_gate.get("calibration_pending", False))
        if calibration_pending:
            if os.environ.get("AGENTIC_RL_RUNTIME_STAGE", "").upper() != (
                "GATE_CALIBRATION"
            ):
                raise ConfigError(
                    "Pending gate calibration config is calibration-stage only"
                )
            if lambda_decision != 0.0 or lambda_query != 0.0:
                raise ConfigError("Pending gate calibration must use zero lambdas")
            if manifest_path.exists():
                raise ConfigError(
                    "Pending gate calibration output already exists: "
                    f"{manifest_path}"
                )
            if role_gate["calibration_manifest_sha256"] not in {None, "PENDING"}:
                raise ConfigError(
                    "Pending gate calibration manifest hash must be PENDING"
                )
        else:
            if not manifest_path.is_file():
                raise ConfigError(
                    f"Gate calibration manifest does not exist: {manifest_path}"
                )
            manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            _require_equal(
                str(role_gate["calibration_manifest_sha256"]),
                manifest_sha256,
                "role_localized_gate.calibration_manifest_sha256",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            _require_equal(manifest.get("status"), "PASS", "gate calibration status")
            _require_equal(
                manifest.get("search_task_mode"),
                role_localized_mode,
                "gate calibration mode",
            )
            _require_equal(
                float(manifest.get("lambda_decision")),
                lambda_decision,
                "gate calibration lambda_decision",
            )
            _require_equal(
                float(manifest.get("lambda_query")),
                lambda_query,
                "gate calibration lambda_query",
            )
            if int(manifest.get("batch_count", 0)) < 3:
                raise ConfigError("Gate calibration requires at least three U0 batches")
            if int(manifest.get("decision_gate_event_count", 0)) < 128:
                raise ConfigError("Gate calibration has fewer than 128 Decision events")
            if int(manifest.get("query_gate_event_count", 0)) < 64:
                raise ConfigError("Gate calibration has fewer than 64 Query events")
            if (
                float(manifest.get("median_gate_to_main_gradient_ratio", 1.0))
                > 0.15
            ):
                raise ConfigError("Weighted gate gradient budget exceeds 0.15")
            for field_name in (
                "parameters_bitwise_unchanged",
                "gradients_cleared",
                "all_rank_metadata_consistent",
            ):
                _require_equal(
                    manifest.get(field_name),
                    True,
                    f"gate calibration {field_name}",
                )
            for field_name in (
                "optimizer_steps",
                "scheduler_steps",
                "checkpoint_writes",
            ):
                _require_equal(
                    int(manifest.get(field_name, -1)),
                    0,
                    f"gate calibration {field_name}",
                )
    elif role_gate is not None:
        raise ConfigError(
            "Legacy Search modes cannot carry advantage.role_localized_gate"
        )

    selection_signal = str(
        selection.get("signal", "dual_channel_ig_outcome")
    )
    if search_task_mode == mica_mode:
        _require_equal(
            config.get("algorithm_mode"),
            mica_mode,
            "algorithm_mode",
        )
        selection_mode = str(selection["mode"])
        allowed_answer_only_selection_modes = {
            "answer_outcome_only_scaled_top_p",
            "answer_outcome_only_ragen2_paper_variance_top_p",
        }
        if selection_mode not in allowed_answer_only_selection_modes:
            raise ConfigError(
                "MICA selection.mode must be one of: "
                + ", ".join(sorted(allowed_answer_only_selection_modes))
            )
        _require_equal(
            selection_signal,
            "answer_outcome_only",
            "selection.signal",
        )
        if selection_mode == "answer_outcome_only_ragen2_paper_variance_top_p":
            _require_equal(
                selection.get("health_gate_active_for_selection"),
                False,
                "selection.health_gate_active_for_selection",
            )
            _require_equal(
                selection.get("scale_active_for_selection"),
                False,
                "selection.scale_active_for_selection",
            )
        mica = config.get("mica")
        if not isinstance(mica, Mapping):
            raise ConfigError("mica must be a mapping in the MICA algorithm mode")
        allowed_mica_keys = {
            "gamma",
            "alpha",
            "normalization_scope",
            "singleton_fallback",
            "cross_prompt_normalization",
            "cross_depth_normalization",
            "raw_ig_fallback",
            "routed_outcome",
            "role_gate",
            "debug_answer_probes",
        }
        unexpected_mica = set(mica) - allowed_mica_keys
        if unexpected_mica:
            raise ConfigError(
                "Unexpected mica fields: "
                + ", ".join(sorted(unexpected_mica))
            )
        _require_equal(float(mica["gamma"]), 1.0, "mica.gamma")
        _require_equal(float(mica["alpha"]), 0.5, "mica.alpha")
        _require_equal(
            mica["normalization_scope"],
            "prompt_search_depth",
            "mica.normalization_scope",
        )
        _require_equal(
            mica["singleton_fallback"],
            "normalized_terminal_outcome",
            "mica.singleton_fallback",
        )
        for key in (
            "cross_prompt_normalization",
            "cross_depth_normalization",
            "raw_ig_fallback",
            "routed_outcome",
            "role_gate",
            "debug_answer_probes",
        ):
            _require_equal(mica[key], False, f"mica.{key}")
    else:
        _require_equal(
            selection["mode"],
            "dual_channel_scaled_top_p",
            "selection.mode",
        )
        _require_equal(
            selection_signal,
            "dual_channel_ig_outcome",
            "selection.signal",
        )
    allowed_policy_keys = {
        "strict_on_policy",
        "ppo_epochs",
        "optimizer_mini_steps",
        "zero_grad_calls_per_successful_update",
        "optimizer_steps_per_successful_update",
        "scheduler_steps_per_successful_update",
        "ratio_level",
        "ratio_hardcoded",
        "clipping_mode",
        "adaptive_clip_beta",
        "adaptive_clip_epsilon_low",
        "adaptive_clip_epsilon_high",
        "answer_clip_scale",
        "task_reduction",
        "kl_reduction",
        "full_vocab_reference_kl",
        "kl_action_state_chunk_size",
        "kl_vocabulary_chunk_size",
        "kl_coefficient",
        "entropy_coefficient",
        "value_coefficient",
        "max_grad_norm",
    }
    unexpected_policy = set(policy) - allowed_policy_keys
    if unexpected_policy:
        raise ConfigError(
            "Unexpected policy fields: "
            + ", ".join(sorted(unexpected_policy))
        )

    stages = config["update_stages"]
    _require_equal(
        stages["update_1"]["bootstrap_activation_controls_scale_commit"],
        False,
        "update_stages.update_1.bootstrap_activation_controls_scale_commit",
    )
    _require_equal(
        stages["update_2_to_health_ready"][
            "bootstrap_activation_controls_scale_update"
        ],
        False,
        "update_stages.update_2_to_health_ready."
        "bootstrap_activation_controls_scale_update",
    )
    _require_equal(
        stages["health_ready"]["inactive_channel_freezes_scale_update"],
        True,
        "update_stages.health_ready.inactive_channel_freezes_scale_update",
    )

    candidate = config.get("candidate_pool")
    if candidate is not None:
        _require_equal(
            candidate["initial_prompts"],
            rollout["candidate_prompts_initial"],
            "candidate_pool.initial_prompts",
        )
        _require_equal(
            candidate["refill_prompts"],
            rollout["refill_prompts"],
            "candidate_pool.refill_prompts",
        )
        _require_equal(
            candidate["max_prompts"],
            rollout["candidate_prompts_max"],
            "candidate_pool.max_prompts",
        )
        _require_equal(
            candidate["group_size"],
            rollout["group_size"],
            "candidate_pool.group_size",
        )
    if "sampling_top_p" in rollout:
        _require_equal(
            float(rollout["sampling_top_p"]),
            0.95,
            "rollout.sampling_top_p",
        )
        _require_equal(
            float(selection["top_p_mass"]),
            0.90,
            "selection.top_p_mass",
        )
    if "optimizer" in config:
        optimizer = config["optimizer"]
        _require_equal(optimizer["name"], "adamw", "optimizer.name")
        _require_equal(
            float(optimizer["learning_rate"]),
            2.0e-7,
            "optimizer.learning_rate",
        )
        _require_equal(
            [float(value) for value in optimizer["betas"]],
            [0.9, 0.999],
            "optimizer.betas",
        )
        _require_equal(
            float(optimizer["epsilon"]), 1.0e-8, "optimizer.epsilon"
        )
        _require_equal(
            float(optimizer["weight_decay"]),
            0.0,
            "optimizer.weight_decay",
        )
        _require_equal(
            float(optimizer["max_grad_norm"]),
            1.0,
            "optimizer.max_grad_norm",
        )
    if "pilot" in config:
        _require_equal(
            config["pilot"]["successful_updates"],
            20,
            "pilot.successful_updates",
        )
        _require_equal(
            config["pilot"]["stop_after_successful_update"],
            20,
            "pilot.stop_after_successful_update",
        )
        _require_equal(
            config["pilot"]["checkpoints"],
            [20],
            "pilot.checkpoints",
        )
        _require_equal(
            config["pilot"]["evaluations"],
            [],
            "pilot.evaluations",
        )
    for key in (
        "server_source",
        "server_config_source",
        "corpus_path",
        "bm25_index_path",
        "dense_index_path",
        "dense_encoder_path",
    ):
        raw = retriever.get(key)
        if not raw or not Path(str(raw)).exists():
            raise ConfigError(f"Retriever asset does not exist: retriever.{key}={raw}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--format", choices=("json", "yaml"), default="yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.format == "json":
        print(json.dumps(config, indent=2, sort_keys=True))
    else:
        print(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()
