from __future__ import annotations

from pathlib import Path

import yaml

from agentic_rl.config import load_config

from config_support import PAPER_MICA_CONFIG


ROOT = Path(__file__).resolve().parents[1]


def test_public_recipe_is_u0_mica_and_full_eval() -> None:
    recipe_path = ROOT / "recipes" / "rl" / "train_4x48gb.yaml"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    config = load_config(PAPER_MICA_CONFIG)
    assert recipe["extends"] == (
        "../../configs/formal_train_answer_only_ragen2_paper_mica_ig_v1.yaml"
    )
    assert config["formal"]["fresh_start_required"] is True
    assert config["formal"]["resume_from_successful_update"] == 0
    assert config["formal"]["total_successful_updates"] is None
    assert config["advantage"]["search_task_mode"] == (
        "answer_only_ragen2_mica_ig_v1_singleton_outcome"
    )
    assert config["selection"]["mode"] == (
        "answer_outcome_only_ragen2_paper_variance_top_p"
    )
    assert config["selection"]["health_gate_active_for_selection"] is False
    assert config["selection"]["scale_active_for_selection"] is False
    assert config["rollout"]["max_num_seqs"] == 64
    assert config["rollout"]["gpu_memory_utilization"] == 0.48
    assert config["formal_schedule"]["learner_micro_batch_size"] == 6
    assert recipe["evaluation"]["expected_manifest_sha256"] == (
        "7e0ce6c65b056788c049811b2c2a7a525e205be2359184a5be5183ce6db86d74"
    )
    assert recipe["evaluation"]["manifest_mode"] == "full_validation"
    assert recipe["evaluation"]["expected_row_count"] == 1400
    recipe_evaluation = {
        key: value
        for key, value in recipe["evaluation"].items()
        if key != "manifest_path"
    }
    assert recipe_evaluation == {
        key: config["evaluation"][key] for key in recipe_evaluation
    }
    assets = yaml.safe_load(
        (ROOT / "configs" / "assets" / "aethersearch_release_v1.yaml").read_text()
    )["assets"]["validation"]
    assert assets["source"]["repo_id"] == "muradil211/AetherSearch_Eval_1400"
    assert assets["source"]["file"] == "eval_1400.jsonl"
    assert assets["sha256"] == config["evaluation"]["expected_validation_sha256"]
    assert assets["manifest_sha256"] == (
        config["evaluation"]["expected_manifest_sha256"]
    )
    assert assets["expected_row_count"] == (
        config["evaluation"]["expected_row_count"]
    )
    assert assets["expected_source_counts"] == (
        config["evaluation"]["expected_source_counts"]
    )
    assert config["formal_schedule"]["fixed_eval_every_successful_updates"] == 20
    assert config["formal_schedule"]["checkpoint_every_successful_updates"] == 20
    assert config["evaluation"]["do_sample"] is False
    assert config["evaluation"]["temperature"] == 0.0
    assert config["evaluation"]["sampling_top_p"] == 1.0


def test_isolated_formal_entry_is_fresh_and_uses_mica_preflight() -> None:
    launcher = (ROOT / "scripts" / "train_rl.sh").read_text()
    supervisor = (ROOT / "scripts" / "_run_runtime_job.sh").read_text()
    assert "recipes/rl/train_4x48gb.yaml" in launcher
    assert "preflight_mica_formal.py" in launcher
    assert "fresh_formal_sc" not in launcher
    assert "unset AGENTIC_RL_RESUME_CHECKPOINT" in supervisor
    assert '"${STAGE}" == "PILOT20" || "${STAGE}" == "FORMAL"' in supervisor
