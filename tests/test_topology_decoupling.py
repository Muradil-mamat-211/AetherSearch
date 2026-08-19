from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
import yaml

from agentic_rl.assets import load_asset_manifest
from agentic_rl.config import ConfigError, DEFAULT_CONFIG, load_config, validate_config
from agentic_rl.qualification import QualificationError, validate_reference_qualification
from agentic_rl.runtime.resource_guard import BYTES_PER_GIB, validate_runtime_resource_budget
from agentic_rl.runtime.verl_config import build_verl_config, effective_rollout_topology
from agentic_rl.topology import TopologyPlan


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_FIXTURE = ROOT / "tests" / "fixtures" / "reference_4x48gb_resolved.yaml"


def _asset_backed_environment_available() -> bool:
    return all(
        os.environ.get(name)
        for name in (
            "AETHERSEARCH_SEARCH_R1_ROOT",
            "AETHERSEARCH_ENV_SCRIPT",
            "AETHERSEARCH_RL_PYTHON",
            "AETHERSEARCH_RETRIEVER_PYTHON",
            "AETHERSEARCH_ACTOR_MODEL",
            "AETHERSEARCH_REFERENCE_MODEL",
            "AETHERSEARCH_TRAIN_DATA",
            "AETHERSEARCH_VALIDATION_DATA",
            "AETHERSEARCH_RUNTIME_ROOT",
            "AETHERSEARCH_CORPUS_PATH",
            "AETHERSEARCH_BM25_INDEX_PATH",
            "AETHERSEARCH_DENSE_INDEX_PATH",
            "AETHERSEARCH_DENSE_ENCODER_PATH",
            "AETHERSEARCH_ASSET_MANIFEST",
            "AETHERSEARCH_QUALIFICATION_MODE",
        )
    )


def _normalise_paths(value: object) -> object:
    if isinstance(value, dict):
        return {key: _normalise_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalise_paths(item) for item in value]
    if not isinstance(value, str):
        return value
    result = value
    replacements = {
        name: os.environ.get(name)
        for name in (
            "AETHERSEARCH_PROJECT_ROOT",
            "AETHERSEARCH_EXACT_IG_AUDIT_ROOT",
            "AETHERSEARCH_SEARCH_R1_ROOT",
            "AETHERSEARCH_ENV_SCRIPT",
            "AETHERSEARCH_RL_PYTHON",
            "AETHERSEARCH_RETRIEVER_PYTHON",
            "AETHERSEARCH_ACTOR_MODEL",
            "AETHERSEARCH_REFERENCE_MODEL",
            "AETHERSEARCH_TRAIN_DATA",
            "AETHERSEARCH_VALIDATION_DATA",
            "AETHERSEARCH_RUNTIME_ROOT",
            "AETHERSEARCH_CORPUS_PATH",
            "AETHERSEARCH_BM25_INDEX_PATH",
            "AETHERSEARCH_DENSE_INDEX_PATH",
            "AETHERSEARCH_DENSE_ENCODER_PATH",
        )
    }
    for name, raw in sorted(
        replacements.items(), key=lambda item: len(item[1] or ""), reverse=True
    ):
        if raw:
            result = result.replace(str(raw), "${" + name + "}")
    return result


@pytest.mark.skipif(
    not _asset_backed_environment_available(),
    reason="reference snapshot requires the local asset environment",
)
def test_reference_recipe_preserves_semantic_snapshot() -> None:
    resolved = load_config(ROOT / "recipes" / "rl" / "train_4x48gb.yaml")
    # train_rl.sh materializes the recipe's experiment-level update budget
    # into the legacy formal/scheduler fields before runtime consumption.
    total_updates = int(resolved["training"]["total_successful_updates"])
    resolved["formal"]["total_successful_updates"] = total_updates
    resolved["formal_schedule"]["total_successful_updates"] = total_updates
    resolved["scheduler"]["total_successful_updates"] = total_updates
    expected = yaml.safe_load(REFERENCE_FIXTURE.read_text(encoding="utf-8"))
    expected["evaluation"].pop("physical_gpu", None)
    expected["evaluation"]["role"] = "eval"
    sections = (
        "project",
        "data",
        "rollout",
        "selection",
        "advantage",
        "policy",
        "learner",
        "formal",
        "formal_schedule",
        "runtime",
        "runtime_smoke_schedule",
        "exact_ig",
        "retriever",
        "evaluation",
        "checkpoint",
        "candidate_pool",
        "update_stages",
        "optimizer",
        "scheduler",
    )
    for section in sections:
        assert _normalise_paths(resolved[section]) == expected[section], section

    plan = TopologyPlan.from_config(resolved)
    assert plan.as_dict() == {
        **plan.as_dict(),
        "learner_world_size": 3,
        "rl_physical_gpus": [1, 2, 3],
        "rl_visible_gpus": [0, 1, 2],
        "rollout_data_parallel_size": 3,
        "rollout_tensor_parallel_size": 1,
    }


def _eight_gpu_config() -> dict:
    config = copy.deepcopy(load_config(DEFAULT_CONFIG))
    config["hardware"].update(
        {
            "total_physical_gpus": 8,
            "gpu_memory_gb": 80,
            "expected_cpu_cores": 96,
            "expected_host_ram_gb": 700,
            "ray_object_store_gb": 96,
            "memory_safety_reserve_gb": 64,
        }
    )
    config["ray"].update(
        {
            "retriever_pool_cpus": 8,
            "rl_engine_cpus_per_gpu": 4,
            "controller_cpu_workers": 8,
            "outcome_worker_count": 4,
            "exact_ig_task_builder_count": 2,
            "agent_loop_worker_count": 16,
            "placement_strategy": "SPREAD",
            "cluster_mode": "local",
        }
    )
    config["topology"] = {
        "cluster_mode": "local",
        "nnodes": 1,
        "roles": {
            "retriever": {"physical_gpu": 0},
            "eval": {"colocate_with": "retriever"},
            "rl": {"physical_gpus_by_node": [[1, 2, 3, 4, 5, 6, 7]]},
        },
        "ray": {"placement_strategy": "SPREAD"},
        "rollout": {"data_parallel_size": 7, "tensor_parallel_size": 1},
    }
    config["rollout"].update({"data_parallel_size": 7, "tensor_parallel_size": 1})
    config["learner"]["world_size"] = 7
    return config


def test_eight_gpu_topology_is_planned_without_python_case_logic() -> None:
    config = _eight_gpu_config()
    validate_config(config)
    plan = TopologyPlan.from_config(config)
    assert plan.learner_world_size == 7
    assert plan.rl_gpus_per_node == 7
    assert plan.rl_visible_gpus == tuple(range(7))
    assert plan.placement_strategy == "SPREAD"
    assert len(plan.ray_bundles(4)) == 7
    assert sum(bundle["GPU"] for bundle in plan.ray_bundles(4)) == 7.0

    resolved = build_verl_config(config, require_optimizer=False)
    assert resolved.trainer.nnodes == 1
    assert resolved.trainer.n_gpus_per_node == 7
    assert effective_rollout_topology(resolved) == {
        "worker_world_size": 7,
        "per_replica_world_size": 1,
        "replica_count": 7,
        "aggregate_data_parallel_size": 7,
        "tensor_parallel_size": 1,
    }


def test_portable_topology_rejects_invalid_invariants() -> None:
    overlap = _eight_gpu_config()
    overlap["topology"]["roles"]["retriever"]["physical_gpu"] = 1
    with pytest.raises(ConfigError, match="overlap"):
        validate_config(overlap)

    out_of_range = _eight_gpu_config()
    out_of_range["topology"]["roles"]["rl"]["physical_gpus_by_node"] = [
        [1, 2, 3, 4, 5, 6, 8]
    ]
    with pytest.raises(ConfigError, match="outside"):
        validate_config(out_of_range)

    world_mismatch = _eight_gpu_config()
    world_mismatch["topology"]["learner"] = {"world_size": 6}
    with pytest.raises(ConfigError, match="does not match derived"):
        validate_config(world_mismatch)

    checked_legacy = _eight_gpu_config()
    checked_legacy["topology"]["compatibility"] = {"validate_legacy_fields": True}
    checked_legacy["hardware"]["rl_world_size"] = 3
    with pytest.raises(ConfigError, match="Deprecated"):
        validate_config(checked_legacy)


def test_portable_resource_guard_uses_minimums_not_exact_host_shape() -> None:
    config = _eight_gpu_config()
    snapshot = {
        "memory_limit_bytes": 720 * BYTES_PER_GIB,
        "memory_current_bytes": 200 * BYTES_PER_GIB,
        "cpu_quota_cores": 100.0,
        "gpu_count": 8,
        "gpu_memory_gib": 80.0,
        "memory_events": {},
    }
    result = validate_runtime_resource_budget(config, snapshot=snapshot)
    assert result["status"] == "PASS"

    config["hardware"]["expected_cpu_cores"] = 101
    with pytest.raises(RuntimeError, match="CPU count exceeds"):
        validate_runtime_resource_budget(config, snapshot=snapshot)

    config = _eight_gpu_config()
    with pytest.raises(RuntimeError, match="Physical GPU count"):
        validate_runtime_resource_budget(
            config,
            snapshot={**snapshot, "gpu_count": 7},
        )
    with pytest.raises(RuntimeError, match="GPU memory"):
        validate_runtime_resource_budget(
            config,
            snapshot={**snapshot, "gpu_memory_gib": 40.0},
        )


def test_backend_and_resource_negative_paths_are_explicit() -> None:
    bad_tp = _eight_gpu_config()
    bad_tp["topology"]["rollout"].update(
        {"data_parallel_size": 1, "tensor_parallel_size": 7}
    )
    with pytest.raises(ConfigError, match="backend compatibility"):
        validate_config(bad_tp)

    invalid_parallel_product = _eight_gpu_config()
    invalid_parallel_product["topology"]["rollout"].update(
        {"data_parallel_size": 7, "tensor_parallel_size": 2}
    )
    with pytest.raises(ConfigError, match="must divide"):
        validate_config(invalid_parallel_product)

    bad_bundles = _eight_gpu_config()
    bad_bundles["ray"]["rl_engine_cpus_per_gpu"] = 100
    with pytest.raises(ConfigError, match="CPU budget"):
        validate_config(bad_bundles)


def test_reference_qualification_is_separate_from_portable_validation() -> None:
    config = _eight_gpu_config()
    with pytest.raises(QualificationError, match="topology"):
        validate_reference_qualification(
            config,
            ROOT / "configs" / "qualification" / "official_4x48gb_v1.yaml",
        )


@pytest.mark.skipif(
    not os.environ.get("AETHERSEARCH_ASSET_MANIFEST"),
    reason="asset manifest path is supplied by the local environment",
)
def test_asset_manifest_is_a_separate_checksum_contract() -> None:
    manifest = load_asset_manifest(os.environ["AETHERSEARCH_ASSET_MANIFEST"])
    assert manifest["version"] == 1
    assert set(manifest["assets"]) >= {
        "actor",
        "reference",
        "tokenizer",
        "train",
        "validation",
        "retriever_index",
    }
    assert all(
        len(str(asset["sha256"])) == 64
        for asset in manifest["assets"].values()
    )
    validation = manifest["assets"]["validation"]
    assert validation["manifest_sha256"] == (
        "a37096d3cab04dfee994318a7059e1151eef1a0df4eb444d6f8544f57ea65baa"
    )
