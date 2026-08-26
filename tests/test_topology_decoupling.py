from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from agentic_rl.assets import load_asset_manifest
from agentic_rl.config import (
    ConfigError,
    _load_config_tree,
    load_config,
    validate_backend_compatibility,
    validate_resources,
    validate_topology,
)
from agentic_rl.qualification import QualificationError, validate_reference_qualification
from agentic_rl.runtime.resource_guard import BYTES_PER_GIB, validate_runtime_resource_budget
from agentic_rl.runtime.verl_config import build_verl_config, effective_rollout_topology
from agentic_rl.topology import TopologyPlan, materialize_topology


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
    # Runtime ownership metadata is a source-layer contract.  The legacy
    # snapshot intentionally compares only effective algorithm/runtime flags,
    # not the nested owner declarations used to materialize them.
    resolved["runtime"] = {
        key: value
        for key, value in resolved["runtime"].items()
        if key
        not in {
            "environment",
            "ray",
            "rollout",
            "learner",
            "evaluation",
            "resume",
            "safety",
            "retriever",
        }
    }
    sections = (
        "project",
        "data",
        "topology",
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


def test_algorithm_layers_do_not_select_server_capacity() -> None:
    base_path = ROOT / "configs" / "base.yaml"
    base_source = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    assert "hardware_5x48gb.yaml" not in base_source.get("includes", [])

    for path in (
        base_path,
        ROOT / "configs" / "formal_train_answer_only_ragen2_mica_ig_v1.yaml",
    ):
        config = _load_config_tree(path)
        assert "hardware" not in config
        assert "topology" not in config
        assert "world_size" not in config.get("learner", {})
        assert not {
            "data_parallel_size",
            "replicas",
            "gpu_memory_utilization",
            "max_num_seqs",
        } & set(config.get("rollout", {}))


def test_official_recipe_resolves_through_explicit_reference_profile() -> None:
    config = _load_config_tree(ROOT / "recipes" / "rl" / "train_4x48gb.yaml")
    plan = TopologyPlan.from_config(config)
    assert plan.retriever_physical_gpu == 0
    assert plan.eval_physical_gpu == 0
    assert plan.rl_physical_gpus == (1, 2, 3)
    assert plan.rl_visible_gpus == (0, 1, 2)
    assert plan.learner_world_size == 3
    assert plan.rollout_data_parallel_size == 3
    assert plan.rollout_tensor_parallel_size == 1


def _non_reference_multi_gpu_config() -> dict:
    # Use the checked resolved snapshot as a data-only algorithm fixture. It
    # avoids requiring local assets while keeping this test about topology
    # planning rather than launch preflight.
    config = yaml.safe_load(REFERENCE_FIXTURE.read_text(encoding="utf-8"))
    config["hardware"].pop("gpu_memory_gb", None)
    config["hardware"].update(
        {
            "total_physical_gpus": 8,
            "expected_cpu_cores": 96,
            "expected_host_ram_gb": 700,
        }
    )
    config["runtime"]["ray"] = {
        "object_store_gb": 96,
        "memory_safety_reserve_gb": 64,
        "retriever_pool_cpus": 8,
        "rl_engine_cpus_per_gpu": 4,
        "vllm_http_server_cpus": 1,
        "control_actor_cpus": {
            "prompt_sampler": 2,
            "candidate_pool": 2,
            "metrics": 1,
            "outcome_worker": 1,
            "checkpoint_commit": 1,
            "exact_ig_task_builder": 2,
        },
        "outcome_worker_count": 4,
        "exact_ig_task_builder_count": 2,
        "agent_loop_worker_count": 16,
        "memory_monitor_refresh_ms": 1000,
        "memory_usage_threshold": 0.8,
        "object_spilling_directory": "runtime/ray_spill",
        "placement_strategy": "SPREAD",
        "cluster_mode": "local",
    }
    config["runtime"]["retriever"] = {
        "query_max_length": 256,
        "dense_query_batch_size": 64,
        "bm25_workers": 16,
        "request_batch_wait_ms": 5.0,
        "request_batch_max_queries": 256,
        "request_wait_timeout_seconds": 180.0,
        "client_batch_wait_ms": 5.0,
        "client_max_concurrency": 128,
        "client_max_batch_queries": 256,
        "client_request_timeout_seconds": 30.0,
        "client_network_retries": 2,
        "health_timeout_seconds": 30.0,
        "retrieval_use_fp16": True,
        "faiss_gpu": True,
        "require_faiss_gpu": True,
        "faiss_gpu_stream_flat": True,
        "faiss_gpu_device": 0,
        "faiss_gpu_use_fp16": True,
        "faiss_temp_memory_mb": 256,
        "faiss_add_batch_size": 0,
        "dense_device": "cuda",
    }
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


def test_topology_plan_supports_non_reference_multi_gpu_layout() -> None:
    """CPU-only planner coverage, not multi-GPU runtime qualification."""

    config = _non_reference_multi_gpu_config()
    validate_topology(config)
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
    overlap = _non_reference_multi_gpu_config()
    overlap["topology"]["roles"]["retriever"]["physical_gpu"] = 1
    with pytest.raises(ConfigError, match="overlap"):
        validate_topology(overlap)

    out_of_range = _non_reference_multi_gpu_config()
    out_of_range["topology"]["roles"]["rl"]["physical_gpus_by_node"] = [
        [1, 2, 3, 4, 5, 6, 8]
    ]
    with pytest.raises(ConfigError, match="outside"):
        validate_topology(out_of_range)

    world_mismatch = _non_reference_multi_gpu_config()
    world_mismatch["topology"]["learner"] = {"world_size": 6}
    with pytest.raises(ConfigError, match="does not match derived"):
        validate_topology(world_mismatch)

    checked_legacy = _non_reference_multi_gpu_config()
    checked_legacy["topology"]["compatibility"] = {"validate_legacy_fields": True}
    checked_legacy["hardware"]["rl_world_size"] = 3
    with pytest.raises(ConfigError, match="Deprecated"):
        validate_topology(checked_legacy)


def _non_reference_layout_config() -> dict:
    config = yaml.safe_load(REFERENCE_FIXTURE.read_text(encoding="utf-8"))
    config["hardware"].pop("gpu_memory_gb", None)
    for field in (
        "retriever_physical_gpu",
        "rl_physical_gpus",
        "rl_visible_gpus",
        "rl_world_size",
        "vllm_data_parallel_size",
        "vllm_tensor_parallel_size",
    ):
        config["hardware"].pop(field, None)
    config["rollout"].pop("data_parallel_size", None)
    config["rollout"].pop("tensor_parallel_size", None)
    config["learner"].pop("world_size", None)
    config["hardware"].update(
        {
            "total_physical_gpus": 2,
            "expected_cpu_cores": 16,
            "expected_host_ram_gb": 128,
            "cpu_reserved_for_os": 2,
            "ray_object_store_gb": 16,
            "memory_safety_reserve_gb": 16,
        }
    )
    config["ray"].update(
        {
            "retriever_pool_cpus": 2,
            "rl_engine_cpus_per_gpu": 2,
            "controller_cpu_workers": 2,
            "outcome_worker_count": 1,
            "exact_ig_task_builder_count": 1,
            "agent_loop_worker_count": 1,
            "placement_strategy": "STRICT_PACK",
        }
    )
    config["topology"] = {
        "cluster_mode": "local",
        "nnodes": 1,
        "roles": {
            "retriever": {"physical_gpu": 0},
            "eval": {"colocate_with": "retriever"},
            "rl": {"physical_gpus_by_node": [[1]]},
        },
        "ray": {"placement_strategy": "STRICT_PACK"},
        "rollout": {"data_parallel_size": 1, "tensor_parallel_size": 1},
    }
    return config


def test_topology_plan_supports_non_reference_layout() -> None:
    """Verify role and logical-device derivation without runtime claims."""

    config = _non_reference_layout_config()
    plan = TopologyPlan.from_config(config)
    assert plan.total_physical_gpus == 2
    assert plan.retriever_physical_gpu == 0
    assert plan.eval_physical_gpu == 0
    assert plan.rl_physical_gpus == (1,)
    assert plan.rl_visible_gpus == (0,)
    assert plan.learner_world_size == 1
    assert plan.rollout_data_parallel_size == 1
    assert plan.rollout_tensor_parallel_size == 1
    assert plan.rl_cuda_visible_devices == "1"
    assert plan.retriever_cuda_visible_devices == "0"

    materialized = materialize_topology(config)
    assert materialized == plan
    assert config["hardware"]["retriever_physical_gpu"] == 0
    assert config["hardware"]["rl_physical_gpus"] == [1]
    assert config["hardware"]["rl_visible_gpus"] == [0]
    assert config["hardware"]["rl_world_size"] == 1
    assert config["rollout"]["data_parallel_size"] == 1
    assert config["rollout"]["tensor_parallel_size"] == 1
    assert config["learner"]["world_size"] == 1


def test_portable_resource_guard_uses_minimums_not_exact_host_shape() -> None:
    config = _non_reference_multi_gpu_config()
    # This value belongs only to the resource-guard arithmetic test, not to
    # the topology-layout coverage above.
    config["hardware"]["gpu_memory_gb"] = 1
    snapshot = {
        "memory_limit_bytes": 720 * BYTES_PER_GIB,
        "memory_current_bytes": 200 * BYTES_PER_GIB,
        "cpu_quota_cores": 100.0,
        "gpu_count": 8,
        "gpu_memory_gib": 1.0,
        "memory_events": {},
    }
    result = validate_runtime_resource_budget(config, snapshot=snapshot)
    assert result["status"] == "PASS"

    config["hardware"]["expected_cpu_cores"] = 101
    with pytest.raises(RuntimeError, match="CPU count exceeds"):
        validate_runtime_resource_budget(config, snapshot=snapshot)

    config = _non_reference_multi_gpu_config()
    config["hardware"]["gpu_memory_gb"] = 1
    with pytest.raises(RuntimeError, match="Physical GPU count"):
        validate_runtime_resource_budget(
            config,
            snapshot={**snapshot, "gpu_count": 7},
        )
    with pytest.raises(RuntimeError, match="GPU memory"):
        validate_runtime_resource_budget(
            config,
            snapshot={**snapshot, "gpu_memory_gib": 0.5},
        )


def test_backend_and_resource_negative_paths_are_explicit() -> None:
    bad_tp = _non_reference_multi_gpu_config()
    bad_tp["topology"]["rollout"].update(
        {"data_parallel_size": 1, "tensor_parallel_size": 7}
    )
    bad_tp_plan = TopologyPlan.from_config(bad_tp)
    with pytest.raises(ConfigError, match="backend compatibility"):
        validate_backend_compatibility(bad_tp, bad_tp_plan)

    invalid_parallel_product = _non_reference_multi_gpu_config()
    invalid_parallel_product["topology"]["rollout"].update(
        {"data_parallel_size": 7, "tensor_parallel_size": 2}
    )
    with pytest.raises(ConfigError, match="must divide"):
        validate_topology(invalid_parallel_product)

    bad_bundles = _non_reference_multi_gpu_config()
    bad_bundles["hardware"]["gpu_memory_gb"] = 1
    bad_bundles["runtime"]["ray"]["rl_engine_cpus_per_gpu"] = 100
    bad_bundle_plan = TopologyPlan.from_config(bad_bundles)
    with pytest.raises(ConfigError, match="CPU budget"):
        validate_resources(bad_bundles, bad_bundle_plan)


def test_reference_qualification_is_separate_from_portable_validation() -> None:
    config = _non_reference_multi_gpu_config()
    with pytest.raises(QualificationError, match="topology"):
        validate_reference_qualification(
            config,
            ROOT / "configs" / "qualification" / "official_4x48gb_v1.yaml",
        )


def test_release_manifest_pins_complete_retriever_asset_set() -> None:
    manifest_path = ROOT / "configs" / "assets" / "aethersearch_release_v1.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assets = manifest["assets"]
    assert set(assets) >= {
        "retriever_corpus",
        "retriever_bm25_index",
        "retriever_index",
        "retriever_encoder_weights",
        "retriever_encoder_config",
        "retriever_encoder_tokenizer",
        "retriever_config",
    }
    expected_sources = {
        "retriever_corpus": (
            "PeterJinGo/wiki-18-corpus",
            "69c1c00ffe7c5554c68d8548355cb22e46aabc51",
        ),
        "retriever_bm25_index": (
            "PeterJinGo/wiki-18-bm25-index",
            "2c7554f25f425038c4bcb155735a0f831851fd78",
        ),
        "retriever_index": (
            "PeterJinGo/wiki-18-e5-index",
            "a4d31160a035f30764604f4827cd8f1d0315eb86",
        ),
        "retriever_encoder_weights": (
            "intfloat/e5-base-v2",
            "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        ),
        "retriever_encoder_config": (
            "intfloat/e5-base-v2",
            "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        ),
        "retriever_encoder_tokenizer": (
            "intfloat/e5-base-v2",
            "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        ),
    }
    for name, (repo_id, revision) in expected_sources.items():
        assert assets[name]["source"]["repo_id"] == repo_id
        assert assets[name]["source"]["revision"] == revision
        assert len(assets[name]["sha256"]) == 64
    assert assets["retriever_corpus"]["expected_passage_count"] == 21_015_324
    assert assets["retriever_index"]["expected_passage_count"] == 21_015_324
    assert assets["retriever_index"]["embedding_dimension"] == 768


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
