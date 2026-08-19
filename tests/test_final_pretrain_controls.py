from __future__ import annotations

from pathlib import Path

from agentic_rl.config import load_config
from agentic_rl.runtime.fsdp_worker import (
    successful_update_warmup_factor,
)
from agentic_rl.runtime.pretrain_controls import (
    exercise_forced_skip_transaction,
)
from agentic_rl.runtime.verl_config import build_verl_config

from config_support import FORMAL_CONFIG, PILOT_CONFIG


ROOT = Path(__file__).resolve().parents[1]


def test_pilot_20_resolved_contract() -> None:
    config = load_config(PILOT_CONFIG)
    assert config["data"]["expected_rows"] == 150_745
    assert config["data"]["shuffle_seed"] == 20_260_724
    assert config["rollout"]["group_size"] == 16
    assert config["rollout"]["candidate_prompts_max"] == 128
    assert config["rollout"]["max_num_seqs"] == 64
    assert config["rollout"]["gpu_memory_utilization"] == 0.46
    assert config["rollout"]["sampling_top_p"] == 0.95
    assert config["selection"]["top_p_mass"] == 0.90
    assert config["pilot"]["checkpoints"] == [20]
    assert config["pilot"]["evaluations"] == []
    assert config["evaluation"]["asynchronous"] is True
    assert config["evaluation"]["role"] == "eval"
    assert config["advantage"]["external_ig_multiplier"] is None
    assert config["advantage"]["future_ig_accumulation"] is True
    assert config["advantage"]["sqrt_n_rescale"] is True
    assert config["formal_schedule"]["total_successful_updates"] == 20
    assert config["formal_schedule"]["checkpoint_every_successful_updates"] == 20
    assert config["formal_schedule"]["fixed_eval_every_successful_updates"] == 20
    assert config["formal_schedule"]["learner_micro_batch_size"] == 6


def test_formal_total_remains_user_supplied() -> None:
    config = load_config(FORMAL_CONFIG)
    assert config["formal"]["total_successful_updates"] is None
    assert config["formal_schedule"]["total_successful_updates"] is None
    assert config["formal_schedule"]["checkpoint_every_successful_updates"] == 20
    assert config["formal_schedule"]["fixed_eval_every_successful_updates"] == 20
    assert config["formal_schedule"]["learner_micro_batch_size"] == 8
    assert config["rollout"]["candidate_prompts_max"] == 128
    assert config["evaluation"]["asynchronous"] is True
    assert config["evaluation"]["role"] == "eval"
    assert config["checkpoint"]["formal_limit"] == 2


def test_pilot_warmup_uses_half_lr_then_base_lr() -> None:
    assert successful_update_warmup_factor(0, 2) == 0.5
    assert successful_update_warmup_factor(1, 2) == 1.0
    assert successful_update_warmup_factor(2, 2) == 1.0


def test_forced_skip_has_no_state_commit_or_step() -> None:
    result = exercise_forced_skip_transaction()
    assert result["status"] == "PASS"
    assert result["optimizer_steps"] == 0
    assert result["scheduler_steps"] == 0
    assert result["successful_update_after"] == 0
    assert result["scale_unchanged"] is True
    assert result["health_unchanged"] is True


def test_pilot_rollout_config_instantiates_against_installed_verl() -> None:
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.config.rollout import RolloutConfig

    project = load_config(PILOT_CONFIG)
    resolved = build_verl_config(project, require_optimizer=True)
    rollout = omega_conf_to_dataclass(resolved.actor_rollout_ref.rollout)
    assert isinstance(rollout, RolloutConfig)
    assert rollout.do_sample is True
    assert rollout.temperature == 1.0
    assert rollout.top_p == 0.95
    assert rollout.top_k == -1
    assert rollout.max_num_seqs == 64
    assert rollout.gpu_memory_utilization == 0.46
    assert rollout.val_kwargs.do_sample is False
