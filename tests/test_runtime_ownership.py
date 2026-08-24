from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml

from agentic_rl.config import (
    ConfigError,
    _materialize_runtime_compatibility,
    validate_backend_compatibility,
)
from agentic_rl.runtime.environment import runtime_environment, retriever_runtime_options
from agentic_rl.runtime.ray_topology import RuntimeRayTopology
from agentic_rl.topology import TopologyPlan


ROOT = Path(__file__).resolve().parents[1]
HARDWARE = ROOT / "configs" / "hardware" / "4x48gb_3rl.yaml"
RUNTIME = ROOT / "configs" / "runtime" / "verl_fsdp2_vllm_4x48_reference.yaml"


def _runtime_config(tmp_path: Path) -> dict:
    config = {
        "paths": {"runtime_root": str(tmp_path)},
        "hardware": {
            "total_physical_gpus": 2,
            "gpu_memory_gb": 1,
            "expected_cpu_cores": 4,
            "expected_host_ram_gb": 8,
            "cpu_reserved_for_os": 1,
        },
        "topology": {
            "cluster_mode": "local",
            "nnodes": 1,
            "roles": {
                "retriever": {"physical_gpu": 0},
                "eval": {"colocate_with": "retriever"},
                "rl": {"physical_gpus_by_node": [[1]]},
            },
            "ray": {"placement_strategy": "STRICT_PACK"},
            "rollout": {"data_parallel_size": 1, "tensor_parallel_size": 1},
        },
        "ray": {
            "memory_monitor_refresh_ms": 17,
            "memory_usage_threshold": 0.5,
            "object_store_gb": 1,
            "memory_safety_reserve_gb": 1,
            "object_spilling_directory": "runtime/ray_spill",
            "rl_engine_cpus_per_gpu": 1,
        },
        "runtime": {
            "environment": {
                "driver": {"OMP_NUM_THREADS": "7", "RAYON_NUM_THREADS": "3"},
                "worker": {"OMP_NUM_THREADS": "9"},
                "process": {"VLLM_WORKER_MULTIPROC_METHOD": "fork"},
                "retriever": {"OMP_NUM_THREADS": "11"},
            }
        },
        "learner": {"strategy": "fsdp2", "world_size": 1},
        "rollout": {"data_parallel_size": 1, "tensor_parallel_size": 1},
    }
    return config


def test_active_hardware_file_has_no_runtime_tuning() -> None:
    hardware = yaml.safe_load(HARDWARE.read_text(encoding="utf-8"))
    runtime = yaml.safe_load(RUNTIME.read_text(encoding="utf-8"))
    assert set(hardware) == {"hardware", "topology"}
    assert set(runtime["runtime"]) >= {
        "ray",
        "rollout",
        "learner",
        "evaluation",
        "environment",
        "retriever",
    }
    assert "ray_object_store_gb" not in hardware["hardware"]
    assert "gpu_memory_utilization" not in hardware.get("rollout", {})


def test_runtime_profile_materializes_legacy_fields_without_changing_values() -> None:
    config = {
        "hardware": {"expected_cpu_cores": 48},
        "ray": {},
        "rollout": {},
        "formal_schedule": {},
        "runtime_smoke_schedule": {},
        "evaluation": {},
        "retriever": {},
        "runtime": yaml.safe_load(RUNTIME.read_text(encoding="utf-8"))["runtime"],
    }
    _materialize_runtime_compatibility(config)
    assert config["ray"]["rl_engine_cpus_per_gpu"] == 3
    assert config["ray"]["object_store_gb"] == 48
    assert config["rollout"]["gpu_memory_utilization"] == 0.48
    assert config["formal_schedule"]["learner_micro_batch_size"] == 6
    assert config["runtime_smoke_schedule"]["learner_micro_batch_size"] == 6
    assert config["evaluation"]["max_memory_fraction"] == 0.24
    assert config["retriever"]["faiss_device_inside_retriever_namespace"] == 0


def test_ray_driver_environment_follows_runtime_profile(monkeypatch, tmp_path: Path) -> None:
    config = _runtime_config(tmp_path)
    captured: dict[str, object] = {}
    fake_ray = types.SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **kwargs: captured.update(kwargs),
        cluster_resources=lambda: {"GPU": 1, "CPU": 4},
        available_resources=lambda: {"GPU": 1, "CPU": 4},
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(
        "agentic_rl.runtime.ray_topology.validate_runtime_resource_budget",
        lambda _config: {"status": "PASS"},
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("AGENTIC_RL_EXPECTED_RL_CUDA_VISIBLE_DEVICES", "1")

    RuntimeRayTopology(config).initialize_ray()
    assert captured["runtime_env"] == {
        "env_vars": {"OMP_NUM_THREADS": "7", "RAYON_NUM_THREADS": "3"}
    }
    assert runtime_environment(config, "worker")["OMP_NUM_THREADS"] == "9"

    config["runtime"]["environment"]["driver"]["OMP_NUM_THREADS"] = "13"
    assert runtime_environment(config, "driver")["OMP_NUM_THREADS"] == "13"


def test_existing_ray_capability_is_rejected_before_runtime_start() -> None:
    config = _runtime_config(Path("/tmp/aethersearch-existing-ray"))
    config["topology"]["cluster_mode"] = "existing"
    plan = TopologyPlan.from_config(config)
    with pytest.raises(ConfigError, match="supports local Ray only"):
        validate_backend_compatibility(config, plan)


def test_retriever_backend_options_follow_runtime_profile() -> None:
    config = {
        "runtime": {
            "retriever": {
                "query_max_length": 256,
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
        }
    }
    options = retriever_runtime_options(config)
    assert options["query_max_length"] == 256
    assert options["faiss_temp_memory_mb"] == 256
    config["runtime"]["retriever"]["query_max_length"] = 512
    config["runtime"]["retriever"]["dense_device"] = "cpu"
    changed = retriever_runtime_options(config)
    assert changed["query_max_length"] == 512
    assert changed["dense_device"] == "cpu"


def test_active_launchers_do_not_own_runtime_constants() -> None:
    supervisor = (ROOT / "scripts" / "_run_runtime_job.sh").read_text(encoding="utf-8")
    retriever = (ROOT / "scripts" / "launch_retriever.sh").read_text(encoding="utf-8")
    for source in (supervisor, retriever):
        assert "OMP_NUM_THREADS=1" not in source
        assert "MKL_NUM_THREADS=1" not in source
        assert "OPENBLAS_NUM_THREADS=1" not in source
    assert "VLLM_WORKER_MULTIPROC_METHOD=spawn" not in supervisor
    assert "--query-max-length 256" not in retriever
    assert "--faiss-temp-memory-mb 256" not in retriever
    assert "runtime_process_environment" in supervisor
    assert "runtime_retriever_environment" in retriever


def test_algorithm_sources_have_no_server_or_machine_ownership() -> None:
    roots = (
        ROOT / "src" / "agentic_rl" / "advantage",
        ROOT / "src" / "agentic_rl" / "exact_ig",
        ROOT / "src" / "agentic_rl" / "outcome",
        ROOT / "src" / "agentic_rl" / "policy",
        ROOT / "src" / "agentic_rl" / "selection",
    )
    forbidden = (
        "CUDA_VISIBLE_DEVICES=",
        "/root/autodl",
        "ray.init(",
        "placement_group(",
        "import verl",
        "from verl",
    )
    for root in roots:
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                assert token not in source, f"{token} leaked into {path}"
