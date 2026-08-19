from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from agentic_rl.config import load_config
from agentic_rl.controller.dataset_view import DeterministicNQHotpotLogicalView
from agentic_rl.runtime.fixed_eval import create_or_validate_eval_manifest_from_config
from agentic_rl.runtime.verl_config import assert_formal_hyperparameters_approved
from agentic_rl.runtime.verl_runtime_adapter import _sha256_file, _sha256_tree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _tokenizer_hash(model_root: Path) -> str:
    digest = hashlib.sha256()
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        path = model_root / name
        if path.is_file():
            digest.update(path.name.encode("utf-8"))
            digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _validate_formula_audit(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Missing formula audit: {path}")
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text[text.find("{") :])
    _require(payload.get("result") == "PASS", "Formula audit did not PASS")
    _require(not payload.get("failed_checks"), "Formula audit has failed checks")
    return payload


def run_preflight(
    config_path: Path,
    *,
    formula_audit_path: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    assert_formal_hyperparameters_approved(config)
    formal = config["formal"]
    _require(formal.get("fresh_start_required") is True, "Fresh start is not locked")
    _require(
        int(formal.get("resume_from_successful_update", -1)) == 0,
        "Formal run is not locked to successful_update=0",
    )
    _require(
        "AGENTIC_RL_RESUME_CHECKPOINT" not in os.environ,
        "A resume checkpoint is present in the launch environment",
    )
    _require(int(config["formal_schedule"]["total_successful_updates"]) == 500, "Target is not U500")
    _require(float(config["formal_schedule"]["learning_rate"]) == 2.0e-7, "LR changed")
    _require(float(config["policy"]["kl_coefficient"]) == 1.0e-2, "KL beta changed")
    _require(int(config["rollout"]["group_size"]) == 16, "G changed")
    _require(float(config["rollout"]["temperature"]) == 1.0, "Temperature changed")
    _require(float(config["rollout"]["sampling_top_p"]) == 0.95, "Rollout top-p changed")
    _require(int(config["rollout"]["max_num_seqs"]) == 64, "vLLM max_num_seqs changed")
    _require(
        float(config["rollout"]["gpu_memory_utilization"]) == 0.46,
        "vLLM GPU-memory utilization changed",
    )
    _require(
        int(config["formal_schedule"]["learner_micro_batch_size"]) == 8,
        "Learner micro-batch size changed",
    )
    _require(
        int(config["hardware"]["expected_cpu_cores"]) == 125,
        "Ray CPU capacity changed",
    )
    _require(int(config["candidate_pool"]["initial_prompts"]) == 64, "Initial pool changed")
    _require(int(config["candidate_pool"]["max_prompts"]) == 128, "Maximum pool changed")
    _require(int(config["selection"]["min_selected"]) == 32, "Selected minimum changed")
    _require(int(config["selection"]["max_selected"]) == 36, "Selected maximum changed")
    search_mode = str(config["advantage"]["search_task_mode"])
    legacy_mode = "sufficiency_novelty_cumulative_ig_probe_routed_outcome"
    role_mode = (
        "sufficiency_novelty_cumulative_ig_probe_routed_outcome_"
        "role_localized_gate"
    )
    _require(
        search_mode in {legacy_mode, role_mode},
        "An approved cumulative-IG/Probe-routed mode is inactive",
    )
    expected_formula = (
        "J_main + lambda_d*J_decision + lambda_q*J_query"
        if search_mode == role_mode
        else "-1.0 if S_before else -1.0 if N else D_ig_eff + O_route"
    )
    _require(
        config["advantage"]["search_advantage_formula"] == expected_formula,
        "Search advantage formula changed",
    )
    if search_mode == role_mode:
        role_gate = config["advantage"]["role_localized_gate"]
        _require(
            role_gate.get("calibration_pending", False) is False,
            "Formal launch cannot use a pending gate calibration",
        )
        _require(
            Path(str(role_gate["calibration_manifest"])).is_file(),
            "Immutable gate calibration manifest is missing",
        )
    _require(
        config["advantage"]["outcome_fallback_to_search"] is False,
        "Outcome fallback still enters Search advantage",
    )
    _require(
        config["advantage"]["future_ig_accumulation"] is True
        and config["advantage"]["sqrt_n_rescale"] is True
        and config["advantage"]["external_ig_multiplier"] is None,
        "Effective cumulative IG/rescale contract is inactive",
    )
    _require(
        float(config["advantage"]["probe_epsilon"]) == 1.0e-6,
        "Probe routing epsilon changed",
    )

    paths = config["paths"]
    actor = Path(str(paths["actor_model"])).resolve()
    reference = Path(str(paths["reference_model"])).resolve()
    _require(actor == reference, "Actor init and frozen Reference are not the locked DPO-V2 path")
    actor_hash = _sha256_tree(actor)
    reference_hash = _sha256_tree(reference)
    tokenizer_hash = _tokenizer_hash(actor)
    _require(actor_hash == formal["actor_init_tree_sha256"], "Actor init hash changed")
    _require(reference_hash == formal["reference_tree_sha256"], "Reference hash changed")
    _require(tokenizer_hash == formal["tokenizer_sha256"], "Tokenizer hash changed")

    train_path = Path(str(paths["train_data"])).resolve()
    train_hash = _sha256_file(train_path)
    _require(train_hash == config["data"]["source_sha256"], "Training parquet hash changed")
    logical = DeterministicNQHotpotLogicalView(
        train_path,
        selection_seed=int(config["data"]["selection_seed"]),
        expected_source_rows=int(config["data"]["source_rows"]),
        expected_logical_rows=int(config["data"]["expected_rows"]),
        expected_nq_rows=int(config["data"]["expected_source_counts"]["nq"]),
        expected_hotpotqa_rows=int(config["data"]["expected_source_counts"]["hotpotqa"]),
        expected_identity_sha256=str(config["data"]["ordered_view_identity_sha256"]),
    )

    evaluation = config["evaluation"]
    eval_manifest = create_or_validate_eval_manifest_from_config(
        validation_path=paths["validation_data"],
        evaluation=evaluation,
    )
    _require(
        eval_manifest["manifest_sha256"] == evaluation["expected_manifest_sha256"],
        "Fixed-eval manifest hash changed",
    )

    retriever = config["retriever"]
    retriever_index = Path(str(retriever["dense_index_path"])).resolve()
    retriever_config = Path(str(retriever["server_config_source"])).resolve()
    retriever_index_hash = _sha256_file(retriever_index)
    retriever_config_hash = _sha256_file(retriever_config)
    _require(
        retriever_index_hash == formal["retriever_index_sha256"],
        "Retriever index hash changed",
    )
    _require(
        retriever_config_hash == formal["retriever_config_sha256"],
        "Retriever config hash changed",
    )

    formula_audit = _validate_formula_audit(formula_audit_path)
    manifest_path = PROJECT_ROOT / "MANIFEST.sha256"
    return {
        "status": "PASS",
        "fresh_start": {
            "successful_update": 0,
            "optimizer_state": "fresh",
            "scheduler_state": "fresh",
            "ragen_state": "fresh",
            "selected_history": "empty",
            "resume_checkpoint": None,
        },
        "target_successful_updates": 500,
        "actor_init_path": str(actor),
        "actor_init_model_hash": actor_hash,
        "reference_model_path": str(reference),
        "reference_model_hash": reference_hash,
        "tokenizer_hash": tokenizer_hash,
        "train_data_path": str(train_path),
        "train_data_sha256": train_hash,
        "logical_view": logical.identity.__dict__,
        "fixed_eval_manifest_path": str(Path(evaluation["manifest_path"]).resolve()),
        "fixed_eval_manifest_sha256": eval_manifest["manifest_sha256"],
        "fixed_eval_count": len(eval_manifest["rows"]),
        "retriever_index_path": str(retriever_index),
        "retriever_index_sha256": retriever_index_hash,
        "retriever_config_path": str(retriever_config),
        "retriever_config_sha256": retriever_config_hash,
        "resolved_config_path": str(config_path.resolve()),
        "resolved_config_sha256": _sha256_file(config_path),
        "source_identity_type": "manifest_sha256",
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "git_commit": None,
        "git_note": "Project root is not a Git worktree; MANIFEST.sha256 is authoritative.",
        "exact_ig": {
            "version": config["exact_ig"]["exact_ig_version"],
            "precision": config["exact_ig"]["production_precision_mode"],
            "structural_audit": config["exact_ig"]["structural_audit_status"],
            "numeric_difference_policy": config["exact_ig"][
                "oracle_numeric_difference_policy"
            ],
        },
        "search_advantage": {
            "mode": config["advantage"]["search_task_mode"],
            "lambda_outcome": float(config["advantage"]["lambda_outcome"]),
            "lambda_format": float(config["advantage"]["lambda_format"]),
            "search_advantage_formula": config["advantage"][
                "search_advantage_formula"
            ],
            "outcome_fallback_to_search": config["advantage"][
                "outcome_fallback_to_search"
            ],
            "formula_audit_path": str(formula_audit_path.resolve()),
            "formula_audit_sha256": _sha256_file(formula_audit_path),
            "formula_audit": formula_audit,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Immutable preflight for the fresh S/N/local-IG formal run"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--formula-audit", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_preflight(
        args.config,
        formula_audit_path=args.formula_audit,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
