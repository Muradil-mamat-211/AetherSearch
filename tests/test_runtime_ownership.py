from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml

from agentic_rl.config import (
    ConfigError,
    _materialize_runtime_compatibility,
    load_config,
    runtime_owned_section,
    validate_backend_compatibility,
    validate_resources,
)
from agentic_rl.runtime.environment import runtime_environment, retriever_runtime_options
from agentic_rl.runtime.ray_topology import RuntimeRayTopology
from agentic_rl.runtime.resource_plan import build_runtime_cpu_resource_plan
from agentic_rl.runtime.retriever_command import build_retriever_command
from agentic_rl.runtime.verl_config import build_verl_config
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
            "expected_cpu_cores": 16,
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
        "runtime": {
            "ray": {
                "memory_monitor_refresh_ms": 17,
                "memory_usage_threshold": 0.5,
                "object_store_gb": 1,
                "memory_safety_reserve_gb": 1,
                "object_spilling_directory": "runtime/ray_spill",
                "retriever_pool_cpus": 1,
                "rl_engine_cpus_per_gpu": 1,
                "vllm_http_server_cpus": 1,
                "control_actor_cpus": {
                    "prompt_sampler": 1,
                    "candidate_pool": 1,
                    "metrics": 1,
                    "outcome_worker": 1,
                    "checkpoint_commit": 1,
                    "exact_ig_task_builder": 1,
                },
                "outcome_worker_count": 1,
                "exact_ig_task_builder_count": 1,
                "agent_loop_worker_count": 1,
            },
            "environment": {
                "driver": {"OMP_NUM_THREADS": "7", "RAYON_NUM_THREADS": "3"},
                "worker": {"OMP_NUM_THREADS": "9"},
                "process": {"VLLM_WORKER_MULTIPROC_METHOD": "fork"},
                "retriever": {"OMP_NUM_THREADS": "11"},
            }
        },
        "learner": {"strategy": "fsdp2", "world_size": 1},
        "rollout": {"data_parallel_size": 1, "tensor_parallel_size": 1},
        "formal_schedule": {
            "maximum_extended_sequence_length": 8,
            "maximum_position_id_exclusive": 16,
        },
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
    assert "control_actor_cpus" in runtime["runtime"]["ray"]
    assert "controller_cpu_workers" not in runtime["runtime"]["ray"]
    assert "resume" not in runtime["runtime"]
    assert "safety" not in runtime["runtime"]


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
    assert config["ray"]["controller_cpu_workers"] == 5
    assert config["rollout"]["gpu_memory_utilization"] == 0.48
    assert config["formal_schedule"]["learner_micro_batch_size"] == 6
    assert config["runtime_smoke_schedule"]["learner_micro_batch_size"] == 6
    assert config["evaluation"]["max_memory_fraction"] == 0.24
    assert config["retriever"]["faiss_device_inside_retriever_namespace"] == 0
    assert config["runtime"]["retriever"]["client_batch_wait_ms"] == 5.0
    assert config["runtime"]["retriever"]["client_max_concurrency"] == 128
    assert config["runtime"]["retriever"]["client_max_batch_queries"] == 256
    assert (
        config["runtime"]["retriever"]["client_request_timeout_seconds"]
        == 30.0
    )
    assert config["runtime"]["retriever"]["client_network_retries"] == 2
    assert config["runtime"]["retriever"]["health_timeout_seconds"] == 30.0


def test_ray_driver_environment_follows_runtime_profile(monkeypatch, tmp_path: Path) -> None:
    config = _runtime_config(tmp_path)
    captured: dict[str, object] = {}
    fake_ray = types.SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **kwargs: captured.update(kwargs),
        cluster_resources=lambda: {"GPU": 1, "CPU": 16},
        available_resources=lambda: {"GPU": 1, "CPU": 16},
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

    multi_node = _runtime_config(Path("/tmp/aethersearch-multi-node"))
    multi_node["topology"]["nnodes"] = 2
    multi_node["topology"]["roles"]["rl"]["physical_gpus_by_node"] = [[1], [2]]
    multi_node["topology"]["rollout"]["data_parallel_size"] = 2
    multi_node["hardware"]["total_physical_gpus"] = 3
    multi_node_plan = TopologyPlan.from_config(multi_node)
    with pytest.raises(ConfigError, match="nnodes=1 only"):
        validate_backend_compatibility(multi_node, multi_node_plan)


def test_retriever_backend_options_follow_runtime_profile() -> None:
    config = {
        "runtime": {
            "retriever": {
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
        }
    }
    options = retriever_runtime_options(config)
    assert options["query_max_length"] == 256
    assert options["faiss_temp_memory_mb"] == 256
    config["runtime"]["retriever"]["query_max_length"] = 512
    config["runtime"]["retriever"]["bm25_workers"] = 3
    config["runtime"]["retriever"]["client_max_concurrency"] = 17
    config["runtime"]["retriever"]["dense_device"] = "cpu"
    changed = retriever_runtime_options(config)
    assert changed["query_max_length"] == 512
    assert changed["bm25_workers"] == 3
    assert changed["client_max_concurrency"] == 17
    assert changed["dense_device"] == "cpu"


def test_new_profiles_do_not_fall_back_to_hardware_runtime_fields() -> None:
    config = {
        "hardware": {
            "ray_object_store_gb": 48,
            "memory_safety_reserve_gb": 64,
        },
        "ray": {"rl_engine_cpus_per_gpu": 3},
    }
    with pytest.raises(ConfigError, match="legacy runtime fallback is disabled"):
        runtime_owned_section(config, "ray")

    partial = {
        **config,
        "runtime": {"ray": {"rl_engine_cpus_per_gpu": 3}},
    }
    with pytest.raises(ConfigError, match="missing runtime-owned fields"):
        runtime_owned_section(partial, "ray")

    config["compatibility"] = {"allow_legacy_runtime_fields": True}
    legacy = runtime_owned_section(config, "ray")
    assert legacy["object_store_gb"] == 48
    assert legacy["memory_safety_reserve_gb"] == 64


def _retriever_command_config(tmp_path: Path) -> dict:
    config = _runtime_config(tmp_path)
    config["paths"].update(
        {
            "retriever_python": sys.executable,
            "runtime_root": str(tmp_path),
        }
    )
    config["retriever"] = {
        "service_url": "http://127.0.0.1:8123",
        "server_source": str(ROOT / "runtime_assets/retriever/hybrid_retrieval_server.py"),
        "bm25_index_path": str(tmp_path / "bm25"),
        "dense_index_path": str(tmp_path / "dense.index"),
        "corpus_path": str(tmp_path / "corpus.jsonl"),
        "dense_encoder_name": "e5",
        "dense_encoder_path": str(tmp_path / "encoder"),
        "top_k": 3,
        "bm25_top_n": 20,
        "dense_top_n": 20,
        "fusion_alpha": 0.5,
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
    return config


def _argument(command: tuple[str, ...], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_retriever_command_follows_complete_runtime_options(tmp_path: Path) -> None:
    config = _retriever_command_config(tmp_path)
    command = build_retriever_command(config)
    assert _argument(command, "--query-max-length") == "256"
    assert _argument(command, "--dense-query-batch-size") == "64"
    assert _argument(command, "--bm25-workers") == "16"
    assert _argument(command, "--faiss-temp-memory-mb") == "256"
    assert _argument(command, "--dense-device") == "cuda"
    assert "--retrieval-use-fp16" in command
    assert "--faiss-gpu" in command
    assert "--require-faiss-gpu" in command
    assert "--faiss-gpu-stream-flat" in command
    assert "--faiss-gpu-use-fp16" in command

    runtime = config["runtime"]["retriever"]
    runtime.update(
        {
            "query_max_length": 384,
            "dense_query_batch_size": 7,
            "bm25_workers": 3,
            "request_batch_wait_ms": 9.5,
            "request_batch_max_queries": 17,
            "request_wait_timeout_seconds": 42.0,
            "faiss_temp_memory_mb": 99,
            "dense_device": "cpu",
            "retrieval_use_fp16": False,
            "faiss_gpu": False,
            "require_faiss_gpu": False,
            "faiss_gpu_stream_flat": False,
            "faiss_gpu_use_fp16": False,
        }
    )
    changed = build_retriever_command(config)
    assert _argument(changed, "--query-max-length") == "384"
    assert _argument(changed, "--dense-query-batch-size") == "7"
    assert _argument(changed, "--bm25-workers") == "3"
    assert _argument(changed, "--request-batch-wait-ms") == "9.5"
    assert _argument(changed, "--request-batch-max-queries") == "17"
    assert _argument(changed, "--request-wait-timeout-seconds") == "42.0"
    assert _argument(changed, "--faiss-temp-memory-mb") == "99"
    assert _argument(changed, "--dense-device") == "cpu"
    assert "--retrieval-use-fp16" not in changed
    assert "--faiss-gpu" not in changed
    assert "--require-faiss-gpu" not in changed
    assert "--faiss-gpu-stream-flat" not in changed
    assert "--faiss-gpu-use-fp16" not in changed


def test_ray_actor_construction_uses_the_shared_resource_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _runtime_config(tmp_path)
    train_data = tmp_path / "train.jsonl"
    train_data.write_text("{}\n", encoding="utf-8")
    actor_model = tmp_path / "actor"
    actor_model.mkdir()
    config["paths"].update(
        {"train_data": str(train_data), "actor_model": str(actor_model)}
    )
    config["data"] = {
        "expected_rows": 1,
        "shuffle_seed": 1,
        "selection_seed": 2,
        "source_rows": 1,
        "expected_source_counts": {"nq": 1, "hotpotqa": 0},
        "ordered_view_identity_sha256": "0" * 64,
    }
    config["rollout"].update({"group_size": 16, "candidate_prompts_max": 128})
    captured: dict[str, float] = {}

    class FakeRemote:
        @staticmethod
        def remote(*_args: object, **_kwargs: object) -> object:
            return object()

    def fake_remote_class(
        implementation: type,
        *,
        num_cpus: float,
        num_gpus: float,
        resources: object = None,
    ) -> type[FakeRemote]:
        del num_gpus, resources
        captured[implementation.__name__] = float(num_cpus)
        return FakeRemote

    monkeypatch.setitem(sys.modules, "ray", types.SimpleNamespace())
    monkeypatch.setattr(
        "agentic_rl.runtime.ray_topology.ray_remote_class",
        fake_remote_class,
    )
    RuntimeRayTopology(config).instantiate_control_actors()
    assert captured == {
        "PromptSamplerActor": 1.0,
        "CandidatePoolActor": 1.0,
        "MetricsActor": 1.0,
        "OutcomeWorkerActor": 1.0,
        "CheckpointCommitActor": 1.0,
        "ExactIGTaskBuilderActor": 1.0,
    }

    config["runtime"]["ray"]["control_actor_cpus"]["outcome_worker"] = 2
    captured.clear()
    topology = RuntimeRayTopology(config)
    topology.instantiate_control_actors()
    assert captured["OutcomeWorkerActor"] == 2.0
    cpu_plan = build_runtime_cpu_resource_plan(
        config["hardware"],
        topology.ray_config,
        learner_world_size=1,
        formal_schedule=config["formal_schedule"],
    )
    assert cpu_plan.control_actors.outcome_worker_cpus == 2.0


def _write_pure_runtime_composition(tmp_path: Path) -> Path:
    parent = yaml.safe_load(
        (ROOT / "tests/fixtures/reference_4x48gb_resolved.yaml").read_text(
            encoding="utf-8"
        )
    )
    parent.pop("hardware")
    parent.pop("topology")
    parent.pop("ray")
    for key in (
        "data_parallel_size",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "max_num_seqs",
    ):
        parent["rollout"].pop(key, None)
    parent["learner"].pop("world_size", None)
    parent["formal_schedule"].pop("learner_micro_batch_size", None)
    parent["runtime_smoke_schedule"].pop("learner_micro_batch_size", None)
    for key in ("minimum_free_memory_gib", "max_memory_fraction"):
        parent["evaluation"].pop(key, None)
    for key in (
        "dense_index_type",
        "require_faiss_gpu",
        "faiss_device_inside_retriever_namespace",
        "dense_query_batch_size",
        "bm25_workers",
        "request_batch_wait_ms",
        "request_batch_max_queries",
        "timeout_seconds",
    ):
        parent["retriever"].pop(key, None)

    asset_root = tmp_path / "assets"
    actor_model = asset_root / "actor"
    reference_model = asset_root / "reference"
    dense_encoder = asset_root / "encoder"
    bm25_index = asset_root / "bm25"
    for directory in (
        actor_model,
        reference_model,
        dense_encoder,
        bm25_index,
    ):
        directory.mkdir(parents=True)
    train_data = asset_root / "train.parquet"
    validation_data = asset_root / "validation.parquet"
    corpus = asset_root / "corpus.jsonl"
    dense_index = asset_root / "dense.index"
    environment_script = asset_root / "env.sh"
    audit = asset_root / "audit_status.json"
    for path in (train_data, validation_data, corpus, dense_index):
        path.write_bytes(b"fixture")
    environment_script.write_text("# fixture\n", encoding="utf-8")
    audit.write_text("{}\n", encoding="utf-8")
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()

    parent["paths"] = {
        "search_r1_root": str(tmp_path),
        "environment_script": str(environment_script),
        "rl_python": sys.executable,
        "retriever_python": sys.executable,
        "actor_model": str(actor_model),
        "reference_model": str(reference_model),
        "train_data": str(train_data),
        "validation_data": str(validation_data),
        "runtime_root": str(runtime_root),
    }
    parent["exact_ig"]["structural_audit_path"] = str(audit)
    parent["exact_ig"]["numerical_gate_path"] = str(audit)
    parent["retriever"].update(
        {
            "server_source": str(
                ROOT / "runtime_assets/retriever/hybrid_retrieval_server.py"
            ),
            "server_config_source": str(
                ROOT / "runtime_assets/retriever/retriever.yaml"
            ),
            "corpus_path": str(corpus),
            "bm25_index_path": str(bm25_index),
            "dense_index_path": str(dense_index),
            "dense_encoder_path": str(dense_encoder),
        }
    )
    parent["evaluation"]["manifest_path"] = str(
        runtime_root / "validation_manifest.json"
    )

    hardware = {
        "hardware": {
            "total_physical_gpus": 2,
            "gpu_memory_gb": 1,
            "expected_host_ram_gb": 32,
            "expected_cpu_cores": 16,
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
    }
    runtime = {
        "runtime": {
            "ray": {
                "retriever_pool_cpus": 1,
                "rl_engine_cpus_per_gpu": 2,
                "vllm_http_server_cpus": 1,
                "control_actor_cpus": {
                    "prompt_sampler": 1,
                    "candidate_pool": 1,
                    "metrics": 1,
                    "outcome_worker": 1,
                    "checkpoint_commit": 1,
                    "exact_ig_task_builder": 1,
                },
                "outcome_worker_count": 1,
                "exact_ig_task_builder_count": 1,
                "agent_loop_worker_count": 2,
                "memory_monitor_refresh_ms": 17,
                "memory_usage_threshold": 0.5,
                "object_store_gb": 2,
                "memory_safety_reserve_gb": 2,
                "object_spilling_directory": "runtime/ray_spill",
            },
            "rollout": {"gpu_memory_utilization": 0.48, "max_num_seqs": 64},
            "learner": {"micro_batch_size": 6},
            "evaluation": {
                "minimum_free_memory_gib": 1,
                "max_memory_fraction": 0.24,
            },
            "environment": {
                "driver": {"OMP_NUM_THREADS": "2"},
                "worker": {"OMP_NUM_THREADS": "3"},
                "process": {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
                "retriever": {"OMP_NUM_THREADS": "4"},
            },
            "retriever": {
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
                "dense_index_type": "GpuIndexFlatIP",
            },
        }
    }
    parent_path = tmp_path / "experiment.yaml"
    hardware_path = tmp_path / "hardware.yaml"
    runtime_path = tmp_path / "runtime.yaml"
    profile_path = tmp_path / "profile.yaml"
    parent_path.write_text(yaml.safe_dump(parent, sort_keys=False), encoding="utf-8")
    hardware_path.write_text(yaml.safe_dump(hardware, sort_keys=False), encoding="utf-8")
    runtime_path.write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")
    profile_path.write_text(
        yaml.safe_dump(
            {
                "extends": parent_path.name,
                "includes": [hardware_path.name, runtime_path.name],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return profile_path


def test_pure_hardware_runtime_portable_composition_runs_full_config_chain(
    tmp_path: Path,
) -> None:
    profile = _write_pure_runtime_composition(tmp_path)
    source = yaml.safe_load((tmp_path / "experiment.yaml").read_text(encoding="utf-8"))
    hardware = yaml.safe_load((tmp_path / "hardware.yaml").read_text(encoding="utf-8"))
    assert "hardware" not in source
    assert "topology" not in source
    assert "ray" not in source
    assert "ray_object_store_gb" not in hardware["hardware"]
    assert "memory_safety_reserve_gb" not in hardware["hardware"]

    config = load_config(profile)
    topology = TopologyPlan.from_config(config)
    validate_resources(config, topology)
    ray = runtime_owned_section(config, "ray")
    cpu_plan = build_runtime_cpu_resource_plan(
        config["hardware"],
        ray,
        learner_world_size=topology.learner_world_size,
        formal_schedule=config["formal_schedule"],
    )
    runtime_topology = RuntimeRayTopology(config)
    verl = build_verl_config(config, require_optimizer=False)
    assert topology.learner_world_size == 1
    assert cpu_plan.total_cpus == 11
    assert runtime_topology.control_actor_resource_plan.total_cpus == 6
    assert verl.trainer.nnodes == 1
    assert verl.trainer.n_gpus_per_node == 1
    assert verl.actor_rollout_ref.rollout.agent.num_workers == 2
    assert verl.actor_rollout_ref.rollout.project_retriever_top_k == 3
    assert (
        verl.actor_rollout_ref.rollout.project_retriever_client_max_concurrency
        == 128
    )
    assert (
        verl.actor_rollout_ref.rollout.project_retriever_client_request_timeout_seconds
        == 30.0
    )


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
    assert "build_retriever_command" in retriever


def test_active_sources_and_layers_have_single_runtime_owners() -> None:
    ray_topology = (
        ROOT / "src/agentic_rl/runtime/ray_topology.py"
    ).read_text(encoding="utf-8")
    assert "num_cpus=1" not in ray_topology
    assert "num_cpus=2" not in ray_topology
    for field in (
        "prompt_sampler_cpus",
        "candidate_pool_cpus",
        "metrics_cpus",
        "outcome_worker_cpus",
        "checkpoint_commit_cpus",
        "exact_ig_task_builder_cpus",
    ):
        assert f"num_cpus=resources.{field}" in ray_topology
    capped_vllm = (
        ROOT / "src/agentic_rl/runtime/capped_vllm.py"
    ).read_text(encoding="utf-8")
    verl_config = (
        ROOT / "src/agentic_rl/runtime/verl_config.py"
    ).read_text(encoding="utf-8")
    assert "num_cpus=float(self.config.project_http_server_num_cpus)" in capped_vllm
    assert 'ray_config["vllm_http_server_cpus"]' in verl_config
    retriever_client = (
        ROOT / "src/agentic_rl/retriever/client.py"
    ).read_text(encoding="utf-8")
    retriever_health = (
        ROOT / "src/agentic_rl/retriever/health.py"
    ).read_text(encoding="utf-8")
    assert "timeout_seconds: float = 180.0" not in retriever_client
    assert "timeout_seconds: float = 30.0" not in retriever_health

    retriever_behavior = yaml.safe_load(
        (ROOT / "configs/retriever_external.yaml").read_text(encoding="utf-8")
    )["retriever"]
    scheduling = {
        "dense_query_batch_size",
        "bm25_workers",
        "request_batch_wait_ms",
        "request_batch_max_queries",
        "timeout_seconds",
    }
    assert not scheduling & set(retriever_behavior)
    runtime = yaml.safe_load(RUNTIME.read_text(encoding="utf-8"))["runtime"]
    assert {
        "dense_query_batch_size",
        "bm25_workers",
        "request_batch_wait_ms",
        "request_batch_max_queries",
        "request_wait_timeout_seconds",
        "client_batch_wait_ms",
        "client_max_concurrency",
        "client_max_batch_queries",
        "client_request_timeout_seconds",
        "client_network_retries",
        "health_timeout_seconds",
    } <= set(runtime["retriever"])
    for loop_config in (
        ROOT / "configs" / "verl_agent_loop.yaml",
        ROOT / "configs" / "verl_agent_loop_role_localized_gate.yaml",
    ):
        source = loop_config.read_text(encoding="utf-8")
        assert "retriever_url" not in source
        assert "retriever_batch_window_ms" not in source
        assert "retriever_max_concurrency" not in source
        assert "retriever_request_timeout_seconds" not in source
    assert "allow_world_size_change_on_resume" not in RUNTIME.read_text(
        encoding="utf-8"
    )
    assert "forbid_retriever_in_rl_process_group" not in RUNTIME.read_text(
        encoding="utf-8"
    )


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
