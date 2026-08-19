from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from agentic_rl.advantage import (
    ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE,
)
from agentic_rl.config import load_config
from agentic_rl.controller.dataset_view import DeterministicNQHotpotLogicalView
from agentic_rl.runtime.fixed_eval import create_or_validate_eval_manifest_from_config
from agentic_rl.runtime.resource_guard import validate_runtime_resource_budget
from agentic_rl.runtime.verl_config import assert_formal_hyperparameters_approved
from agentic_rl.runtime.verl_runtime_adapter import _sha256_file, _sha256_tree
from agentic_rl.selection import ANSWER_OUTCOME_ONLY_SELECTION_SIGNAL


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


def run_preflight(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    resource_budget = validate_runtime_resource_budget(config)
    assert_formal_hyperparameters_approved(config)
    mode = str(config["advantage"]["search_task_mode"])
    _require(
        mode == ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE,
        "Formal preflight is not on the frozen MICA V1 mode",
    )
    _require(
        str(config["selection"].get("signal"))
        == ANSWER_OUTCOME_ONLY_SELECTION_SIGNAL,
        "Formal preflight is not using Answer-only RAGEN",
    )
    mica = config["mica"]
    _require(float(mica["gamma"]) == 1.0, "MICA gamma drifted")
    _require(float(mica["alpha"]) == 0.5, "MICA alpha drifted")
    _require(
        str(mica["normalization_scope"]) == "prompt_search_depth",
        "MICA normalization scope drifted",
    )
    _require(
        str(mica["singleton_fallback"])
        == "normalized_terminal_outcome",
        "MICA singleton fallback drifted",
    )
    for key in (
        "cross_prompt_normalization",
        "cross_depth_normalization",
        "raw_ig_fallback",
        "routed_outcome",
        "role_gate",
        "debug_answer_probes",
    ):
        _require(mica[key] is False, f"MICA forbidden switch enabled: {key}")
    probe = config["advantage"]["sufficiency_probe"]
    _require(
        probe["enabled"] is False
        and probe["pre_search_enabled"] is False
        and probe["post_search_enabled"] is False,
        "Diagnostic Answer probes are enabled",
    )
    _require(
        config["advantage"]["sc"]["actor_loss_enabled"] is False,
        "Legacy Stop/Continue actor credit is enabled",
    )
    _require(
        int(config["formal"]["resume_from_successful_update"]) == 0
        and config["formal"]["fresh_start_required"] is True,
        "Formal MICA launch is not a fresh U0 run",
    )
    _require(
        not os.environ.get("AGENTIC_RL_RESUME_CHECKPOINT"),
        "Formal MICA preflight rejects AGENTIC_RL_RESUME_CHECKPOINT",
    )

    paths = config["paths"]
    formal = config["formal"]
    actor = Path(str(paths["actor_model"])).resolve()
    reference = Path(str(paths["reference_model"])).resolve()
    _require(actor == reference, "Actor and frozen Reference paths differ")
    actor_hash = _sha256_tree(actor)
    reference_hash = _sha256_tree(reference)
    tokenizer_hash = _tokenizer_hash(actor)
    _require(actor_hash == formal["actor_init_tree_sha256"], "Actor hash changed")
    _require(reference_hash == formal["reference_tree_sha256"], "Reference hash changed")
    _require(tokenizer_hash == formal["tokenizer_sha256"], "Tokenizer hash changed")

    train_path = Path(str(paths["train_data"])).resolve()
    train_hash = _sha256_file(train_path)
    _require(train_hash == config["data"]["source_sha256"], "Training data changed")
    logical = DeterministicNQHotpotLogicalView(
        train_path,
        selection_seed=int(config["data"]["selection_seed"]),
        expected_source_rows=int(config["data"]["source_rows"]),
        expected_logical_rows=int(config["data"]["expected_rows"]),
        expected_nq_rows=int(config["data"]["expected_source_counts"]["nq"]),
        expected_hotpotqa_rows=int(
            config["data"]["expected_source_counts"]["hotpotqa"]
        ),
        expected_identity_sha256=str(
            config["data"]["ordered_view_identity_sha256"]
        ),
    )
    evaluation = config["evaluation"]
    manifest = create_or_validate_eval_manifest_from_config(
        validation_path=paths["validation_data"],
        evaluation=evaluation,
    )
    _require(
        manifest["manifest_sha256"] == evaluation["expected_manifest_sha256"],
        "Fixed-eval manifest changed",
    )

    retriever = config["retriever"]
    index_path = Path(str(retriever["dense_index_path"])).resolve()
    server_config = Path(str(retriever["server_config_source"])).resolve()
    index_hash = _sha256_file(index_path)
    server_hash = _sha256_file(server_config)
    _require(index_hash == formal["retriever_index_sha256"], "Retriever index changed")
    _require(server_hash == formal["retriever_config_sha256"], "Retriever config changed")

    return {
        "status": "PASS",
        "algorithm_mode": mode,
        "selection_signal": config["selection"]["signal"],
        "mica": dict(mica),
        "fresh_start": {
            "resume_checkpoint": None,
            "successful_update": 0,
            "optimizer_step": 0,
            "scheduler_step": 0,
        },
        "runtime_resource_budget": resource_budget,
        "actor_model": str(actor),
        "actor_checksum": actor_hash,
        "reference_model": str(reference),
        "reference_checksum": reference_hash,
        "tokenizer_checksum": tokenizer_hash,
        "train_data": str(train_path),
        "train_data_checksum": train_hash,
        "logical_view": logical.identity.__dict__,
        "eval_manifest": str(Path(evaluation["manifest_path"]).resolve()),
        "eval_manifest_checksum": manifest["manifest_sha256"],
        "retriever_index_checksum": index_hash,
        "retriever_config_checksum": server_hash,
        "config": str(config_path.resolve()),
        "config_checksum": _sha256_file(config_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_preflight(args.config)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
