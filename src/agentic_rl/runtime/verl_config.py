from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from agentic_rl.config import runtime_owned_section, runtime_section
from agentic_rl.topology import TopologyPlan

from .environment import retriever_runtime_options


class RuntimeConfigurationError(RuntimeError):
    pass


FORMAL_REQUIRED_FIELDS = (
    "optimizer_family",
    "learning_rate",
    "weight_decay",
    "optimizer_betas",
    "optimizer_epsilon",
    "scheduler",
    "warmup",
    "learner_micro_batch_size",
    "total_successful_updates",
    "maximum_prompt_length",
    "maximum_response_length",
    "maximum_model_length",
    "maximum_extended_sequence_length",
    "maximum_position_id_exclusive",
    "sampling_temperature_approval",
    "sampling_top_p_approval",
)


def unresolved_formal_fields(config: Mapping[str, Any]) -> tuple[str, ...]:
    schedule = config["formal_schedule"]
    return tuple(field for field in FORMAL_REQUIRED_FIELDS if schedule.get(field) is None)


def assert_formal_hyperparameters_approved(config: Mapping[str, Any]) -> None:
    unresolved = unresolved_formal_fields(config)
    if unresolved:
        raise RuntimeConfigurationError(
            "Formal hyperparameters require user approval: " + ", ".join(unresolved)
        )


def _generated_verl_config_path() -> Path:
    import verl

    root = Path(verl.__file__).resolve().parent
    path = root / "trainer" / "config" / "_generated_ppo_trainer.yaml"
    if not path.is_file():
        raise RuntimeConfigurationError(
            f"Installed veRL generated PPO config is missing: {path}"
        )
    return path


def build_verl_config(
    project_config: Mapping[str, Any],
    *,
    require_optimizer: bool,
) -> Any:
    """Map frozen project settings to the installed veRL 0.6.1 schema.

    The current adapter represents aggregate rollout DP as one independent
    TP=1 replica per learner rank. The replica count is derived from the
    resolved TopologyPlan rather than from a reference-machine world size.
    """

    from omegaconf import OmegaConf, open_dict

    if require_optimizer:
        assert_formal_hyperparameters_approved(project_config)
    path = _generated_verl_config_path()
    config = OmegaConf.load(path)
    model_path = str(project_config["paths"]["actor_model"])
    reference_path = str(project_config["paths"]["reference_model"])
    topology = TopologyPlan.from_config(project_config)
    rollout = runtime_section(project_config, "rollout")
    learner = project_config["learner"]
    schedule = project_config["formal_schedule"]
    exact_ig = project_config["exact_ig"]
    ray_config = runtime_owned_section(project_config, "ray")
    retriever_runtime = retriever_runtime_options(project_config)
    runtime_root = Path(str(project_config["paths"]["runtime_root"])).resolve()

    with open_dict(config):
        config.actor_rollout_ref.model.path = model_path
        config.actor_rollout_ref.model.trust_remote_code = True
        config.actor_rollout_ref.model.use_shm = False
        config.actor_rollout_ref.model.enable_gradient_checkpointing = bool(
            learner["gradient_checkpointing"]
        )
        config.actor_rollout_ref.model.use_remove_padding = False
        config.actor_rollout_ref.model.override_config = {
            "attn_implementation": "flash_attention_2",
        }
        config.actor_rollout_ref.exact_ig_precision_mode = str(
            exact_ig["production_precision_mode"]
        )
        config.actor_rollout_ref.exact_ig_attention_backend = str(
            exact_ig["attention_backend"]
        )
        config.actor_rollout_ref.exact_ig_oracle_canary_rate = float(
            exact_ig["oracle_canary_rate"]
        )
        config.actor_rollout_ref.exact_ig_oracle_canary_fail_closed = bool(
            exact_ig["oracle_canary_fail_closed"]
        )
        config.actor_rollout_ref.exact_ig_parity_rtol = float(
            exact_ig["parity_rtol"]
        )
        config.actor_rollout_ref.exact_ig_parity_atol = float(
            exact_ig["parity_atol"]
        )
        config.actor_rollout_ref.exact_ig_maximum_token_log_prob_abs_diff = (
            float(exact_ig["maximum_token_log_prob_abs_diff"])
        )
        config.actor_rollout_ref.exact_ig_maximum_phi_abs_diff = float(
            exact_ig["maximum_phi_abs_diff"]
        )
        config.actor_rollout_ref.exact_ig_maximum_ig_abs_diff = float(
            exact_ig["maximum_ig_abs_diff"]
        )
        config.actor_rollout_ref.exact_ig_maximum_phi_safety_abs_diff = float(
            exact_ig["maximum_phi_safety_abs_diff"]
        )
        config.actor_rollout_ref.exact_ig_maximum_ig_safety_abs_diff = float(
            exact_ig["maximum_ig_safety_abs_diff"]
        )
        config.actor_rollout_ref.exact_ig_numeric_ambiguity_epsilon = float(
            exact_ig["numeric_ambiguity_epsilon"]
        )
        config.actor_rollout_ref.exact_ig_calibration_p99_ig_abs_diff = float(
            exact_ig["calibration_p99_ig_abs_diff"]
        )
        config.actor_rollout_ref.exact_ig_minimum_canary_samples_for_p99 = int(
            exact_ig["minimum_canary_samples_for_p99"]
        )
        config.actor_rollout_ref.exact_ig_maximum_telescoping_error = float(
            exact_ig["maximum_telescoping_error"]
        )
        config.actor_rollout_ref.exact_ig_scoring_logits_mode = str(
            exact_ig["scoring_logits_mode"]
        )
        config.actor_rollout_ref.exact_ig_selected_positions_enabled = bool(
            exact_ig["selected_positions_enabled"]
        )
        config.actor_rollout_ref.exact_ig_attention_mask_mode = str(
            exact_ig["attention_mask_mode"]
        )
        config.actor_rollout_ref.exact_ig_structural_audit_path = str(
            exact_ig["structural_audit_path"]
        )
        config.actor_rollout_ref.exact_ig_version = str(
            exact_ig["exact_ig_version"]
        )
        config.actor_rollout_ref.exact_ig_info_gain_type = str(
            exact_ig["info_gain_type"]
        )
        config.actor_rollout_ref.exact_ig_max_records_per_forward = int(
            exact_ig["max_records_per_forward"]
        )
        config.actor_rollout_ref.exact_ig_max_attention_cost_per_batch = (
            None
            if exact_ig["max_attention_cost_per_batch"] is None
            else int(exact_ig["max_attention_cost_per_batch"])
        )
        config.actor_rollout_ref.exact_ig_max_extended_tokens_per_batch = (
            None
            if exact_ig["max_extended_tokens_per_batch"] is None
            else int(exact_ig["max_extended_tokens_per_batch"])
        )
        config.actor_rollout_ref.exact_ig_max_full_logits_bytes = (
            None
            if exact_ig["max_full_logits_bytes"] is None
            else int(exact_ig["max_full_logits_bytes"])
        )
        config.actor_rollout_ref.exact_ig_max_selected_logits_bytes = (
            None
            if exact_ig["max_selected_logits_bytes"] is None
            else int(exact_ig["max_selected_logits_bytes"])
        )

        actor = config.actor_rollout_ref.actor
        actor.strategy = "fsdp2"
        actor.ppo_epochs = 1
        actor.shuffle = False
        actor.loss_agg_mode = "token-mean"
        actor.entropy_coeff = 0.0
        actor.use_kl_loss = False
        actor.grad_clip = float(project_config["policy"]["max_grad_norm"])
        actor.fsdp_config.strategy = "fsdp2"
        actor.fsdp_config.reshard_after_forward = False
        actor.fsdp_config.fsdp_size = int(topology.learner_world_size)
        actor.fsdp_config.model_dtype = "fp32"
        actor.fsdp_config.dtype = "bfloat16"
        actor.fsdp_config.param_offload = False
        actor.fsdp_config.optimizer_offload = False
        actor.fsdp_config.offload_policy = False
        actor.fsdp_config.use_torch_compile = False

        ref = config.actor_rollout_ref.ref
        ref.strategy = "fsdp2"
        ref.model = {"path": reference_path}
        ref.fsdp_config.strategy = "fsdp2"
        ref.fsdp_config.reshard_after_forward = False
        ref.fsdp_config.fsdp_size = int(topology.learner_world_size)
        ref.fsdp_config.model_dtype = "bfloat16"
        ref.fsdp_config.dtype = "bfloat16"
        ref.fsdp_config.use_torch_compile = False
        ref.fsdp_config.param_offload = False
        ref.fsdp_config.offload_policy = False

        verl_rollout = config.actor_rollout_ref.rollout
        verl_rollout.name = "vllm"
        verl_rollout.mode = "async"
        verl_rollout.dtype = "bfloat16"
        # Aggregate DP is implemented by independent one-GPU replicas.
        verl_rollout.data_parallel_size = 1
        verl_rollout.tensor_model_parallel_size = 1
        verl_rollout.pipeline_model_parallel_size = 1
        verl_rollout.n = int(rollout["group_size"])
        verl_rollout.do_sample = bool(rollout.get("do_sample", True))
        verl_rollout.temperature = float(rollout["temperature"])
        verl_rollout.top_p = float(
            rollout.get("sampling_top_p", rollout["top_p"])
        )
        configured_top_k = int(rollout["top_k"])
        # vLLM uses -1 for disabled top-k; the project-facing formal config
        # follows the user-facing convention top_k=0.
        verl_rollout.top_k = -1 if configured_top_k == 0 else configured_top_k
        verl_rollout.val_kwargs.temperature = 0.0
        verl_rollout.val_kwargs.top_p = 1.0
        verl_rollout.val_kwargs.top_k = -1
        verl_rollout.val_kwargs.do_sample = False
        verl_rollout.val_kwargs.n = 1
        verl_rollout.gpu_memory_utilization = float(
            rollout["gpu_memory_utilization"]
        )
        verl_rollout.max_num_seqs = int(rollout["max_num_seqs"])
        verl_rollout.enable_chunked_prefill = bool(
            rollout["enable_chunked_prefill"]
        )
        verl_rollout.enable_prefix_caching = bool(
            rollout["enable_prefix_caching"]
        )
        verl_rollout.free_cache_engine = True
        verl_rollout.load_format = "dummy"
        verl_rollout.calculate_log_probs = True
        verl_rollout.agent.num_workers = int(ray_config["agent_loop_worker_count"])
        verl_rollout.project_http_server_num_cpus = float(
            ray_config["vllm_http_server_cpus"]
        )
        verl_rollout.project_retriever_service_url = str(
            project_config["retriever"]["service_url"]
        )
        verl_rollout.project_retriever_top_k = int(
            project_config["retriever"]["top_k"]
        )
        verl_rollout.project_retriever_client_batch_wait_ms = float(
            retriever_runtime["client_batch_wait_ms"]
        )
        verl_rollout.project_retriever_client_max_concurrency = int(
            retriever_runtime["client_max_concurrency"]
        )
        verl_rollout.project_retriever_client_max_batch_queries = int(
            retriever_runtime["client_max_batch_queries"]
        )
        verl_rollout.project_retriever_client_request_timeout_seconds = float(
            retriever_runtime["client_request_timeout_seconds"]
        )
        verl_rollout.project_retriever_client_network_retries = int(
            retriever_runtime["client_network_retries"]
        )
        verl_rollout.agent.default_agent_loop = "search_exact_ig"
        search_task_mode = str(project_config["advantage"]["search_task_mode"])
        agent_loop_filename = (
            "verl_agent_loop_role_localized_gate.yaml"
            if search_task_mode
            == "sufficiency_novelty_cumulative_ig_probe_routed_outcome_role_localized_gate"
            else "verl_agent_loop.yaml"
        )
        verl_rollout.agent.agent_loop_config_path = str(
            Path(__file__).resolve().parents[3]
            / "configs"
            / agent_loop_filename
        )

        maximum_prompt = schedule.get("maximum_prompt_length")
        maximum_response = schedule.get("maximum_response_length")
        maximum_model = schedule.get("maximum_model_length")
        if maximum_prompt is not None:
            verl_rollout.prompt_length = int(maximum_prompt)
            config.data.max_prompt_length = int(maximum_prompt)
        if maximum_response is not None:
            verl_rollout.response_length = int(maximum_response)
            config.data.max_response_length = int(maximum_response)
        if maximum_model is not None:
            verl_rollout.max_model_len = int(maximum_model)

        config.actor_rollout_ref.hybrid_engine = True
        config.actor_rollout_ref.nccl_timeout = 1200
        config.reward_model.enable = False
        config.reward_model.enable_resource_pool = False

        config.data.train_files = str(project_config["paths"]["train_data"])
        config.data.val_files = str(project_config["paths"]["validation_data"])
        config.data.prompt_key = str(project_config["data"]["prompt_key"])
        config.data.return_raw_chat = True
        config.data.shuffle = False
        config.data.truncation = "error"

        config.trainer.n_gpus_per_node = int(topology.rl_gpus_per_node)
        config.trainer.nnodes = int(topology.nnodes)
        config.trainer.project_name = str(project_config["project"]["name"])
        config.trainer.experiment_name = "strict_runtime"
        config.trainer.default_local_dir = str(runtime_root / "checkpoints")
        config.trainer.device = "cuda"

        if require_optimizer:
            optimizer = actor.optim
            optimizer.optimizer = str(schedule["optimizer_family"])
            optimizer.lr = float(schedule["learning_rate"])
            optimizer.weight_decay = float(schedule["weight_decay"])
            optimizer.betas = [float(value) for value in schedule["optimizer_betas"]]
            optimizer.override_optimizer_config = {
                "eps": float(schedule["optimizer_epsilon"])
            }
            optimizer.lr_scheduler_type = str(schedule["scheduler"])
            optimizer.total_training_steps = int(
                schedule["total_successful_updates"]
            )
            warmup = schedule["warmup"]
            if isinstance(warmup, int):
                optimizer.lr_warmup_steps = int(warmup)
            else:
                optimizer.lr_warmup_steps_ratio = float(warmup)
            micro_batch = int(schedule["learner_micro_batch_size"])
            actor.ppo_micro_batch_size_per_gpu = micro_batch
            actor.ppo_mini_batch_size = (
                int(project_config["selection"]["maximum_selected_prompts"])
                * int(rollout["group_size"])
            )
            verl_rollout.log_prob_micro_batch_size_per_gpu = micro_batch
            ref.log_prob_micro_batch_size_per_gpu = micro_batch
            config.actor_rollout_ref.project_scheduler_mode = (
                "successful_update_constant_with_warmup"
            )
            config.actor_rollout_ref.project_warmup_successful_updates = int(
                warmup
            )
        else:
            # Topology/config inspection may not initialize a model or optimizer.
            actor.ppo_micro_batch_size_per_gpu = 1
            actor.ppo_mini_batch_size = 4

    return config


def effective_rollout_topology(config: Any) -> dict[str, int]:
    world_size = int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes)
    per_replica_world = (
        int(config.actor_rollout_ref.rollout.tensor_model_parallel_size)
        * int(config.actor_rollout_ref.rollout.data_parallel_size)
        * int(config.actor_rollout_ref.rollout.pipeline_model_parallel_size)
    )
    return {
        "worker_world_size": world_size,
        "per_replica_world_size": per_replica_world,
        "replica_count": world_size // per_replica_world,
        "aggregate_data_parallel_size": world_size,
        "tensor_parallel_size": int(
            config.actor_rollout_ref.rollout.tensor_model_parallel_size
        ),
    }
