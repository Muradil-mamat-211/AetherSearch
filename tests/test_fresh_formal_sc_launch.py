from __future__ import annotations

from pathlib import Path

from agentic_rl.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_fresh_formal_config_is_u0_probe_routed_cumulative_ig() -> None:
    config = load_config(ROOT / "configs" / "formal_train.yaml")
    assert config["formal"]["fresh_start_required"] is True
    assert config["formal"]["resume_from_successful_update"] == 0
    assert config["formal"]["total_successful_updates"] is None
    assert config["advantage"]["search_task_mode"] == (
        "sufficiency_novelty_cumulative_ig_probe_routed_outcome"
    )
    assert config["advantage"]["external_ig_multiplier"] is None
    assert config["rollout"]["candidate_prompts_max"] == 128
    assert config["rollout"]["max_num_seqs"] == 64
    assert config["rollout"]["gpu_memory_utilization"] == 0.46
    assert config["formal_schedule"]["learner_micro_batch_size"] == 8
    assert config["evaluation"]["expected_manifest_sha256"] == (
        "a37096d3cab04dfee994318a7059e1151eef1a0df4eb444d6f8544f57ea65baa"
    )
    assert config["evaluation"]["manifest_mode"] == "full_validation"
    assert config["evaluation"]["expected_row_count"] == 51713


def test_isolated_formal_entry_is_fresh_and_uses_mica_preflight() -> None:
    launcher = (ROOT / "scripts" / "train_formal_manual.sh").read_text()
    supervisor = (ROOT / "scripts" / "_run_runtime_job.sh").read_text()
    assert "final_pretrain_gate.json" not in launcher
    assert "preflight_mica_formal.py" in launcher
    assert "preflight_fresh_formal_sc.py" not in launcher
    assert "test_mica_ig_v1.py" in launcher
    assert "test_answer_only_ragen2_mica_integration.py" in launcher
    assert "formal_fresh_u000_to_u500_answer_ragen2_mica_ig_v1" in launcher
    assert "AGENTIC_RL_RESUME_CHECKPOINT" in launcher
    assert "unset AGENTIC_RL_RESUME_CHECKPOINT" in supervisor
    assert '"${STAGE}" == "PILOT20" || "${STAGE}" == "FORMAL"' in supervisor
    assert (
        'FORMAL "${RUN_DIR}/configs/resolved_config.yaml" "${RUN_DIR}"'
        in launcher
    )


def test_formal_preflight_locks_conditional_second_refill_capacity() -> None:
    preflight = (ROOT / "scripts" / "preflight_fresh_formal_sc.py").read_text(
        encoding="utf-8"
    )
    assert 'config["candidate_pool"]["max_prompts"]) == 128' in preflight
    assert 'config["candidate_pool"]["max_prompts"]) == 96' not in preflight
