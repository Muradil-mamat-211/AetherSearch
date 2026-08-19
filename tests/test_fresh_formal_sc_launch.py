from __future__ import annotations

from pathlib import Path

import yaml

from agentic_rl.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_public_recipe_is_u0_mica_and_full_eval() -> None:
    recipe_path = ROOT / "recipes" / "rl" / "train_4x48gb.yaml"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    config = load_config(
        ROOT / "configs" / "formal_train_answer_only_ragen2_mica_ig_v1.yaml"
    )
    assert recipe["extends"] == (
        "../../configs/formal_train_answer_only_ragen2_mica_ig_v1.yaml"
    )
    assert config["formal"]["fresh_start_required"] is True
    assert config["formal"]["resume_from_successful_update"] == 0
    assert config["formal"]["total_successful_updates"] is None
    assert config["advantage"]["search_task_mode"] == (
        "answer_only_ragen2_mica_ig_v1_singleton_outcome"
    )
    assert config["rollout"]["max_num_seqs"] == 64
    assert config["rollout"]["gpu_memory_utilization"] == 0.48
    assert config["formal_schedule"]["learner_micro_batch_size"] == 6
    assert recipe["evaluation"]["expected_manifest_sha256"] == (
        "a37096d3cab04dfee994318a7059e1151eef1a0df4eb444d6f8544f57ea65baa"
    )
    assert recipe["evaluation"]["manifest_mode"] == "full_validation"
    assert recipe["evaluation"]["expected_row_count"] == 51713


def test_isolated_formal_entry_is_fresh_and_uses_mica_preflight() -> None:
    launcher = (ROOT / "scripts" / "train_rl.sh").read_text()
    supervisor = (ROOT / "scripts" / "_run_runtime_job.sh").read_text()
    assert "recipes/rl/train_4x48gb.yaml" in launcher
    assert "preflight_mica_formal.py" in launcher
    assert "fresh_formal_sc" not in launcher
    assert "unset AGENTIC_RL_RESUME_CHECKPOINT" in supervisor
    assert '"${STAGE}" == "PILOT20" || "${STAGE}" == "FORMAL"' in supervisor
