from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agentic_rl.controller.dataset_view import DeterministicNQHotpotLogicalView
from agentic_rl.controller.prompt_sampler import (
    ImmutableDatasetPromptSampler,
    PromptCursorState,
)
from agentic_rl.exact_ig.task_builder import (
    ExactIGTaskBuilder,
    assert_same_prompt_target_consistency,
)
from agentic_rl.metrics.schema import MetricScope
from agentic_rl.metrics.sinks import JsonlMetricSink
from agentic_rl.outcome.workers import (
    PRODUCTION_TASK_SCORER_VERSION,
    score_sufficiency_probe_completion,
    score_stop_answer_completion,
    score_trajectory_outcome,
)
from agentic_rl.selection.candidate_pool import CandidatePool, PromptGroup


def _run_probe(python: str, source: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [python, "-c", source],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": -1,
            "stderr": f"Compatibility probe timed out after {exc.timeout} seconds",
            "timeout": True,
        }
    stdout = completed.stdout.strip()
    parsed: dict[str, Any]
    stdout_lines = [line for line in stdout.splitlines() if line.strip()]
    try:
        parsed = json.loads(stdout_lines[-1]) if stdout_lines else {}
    except json.JSONDecodeError:
        parsed = {"stdout": stdout}
    else:
        if len(stdout_lines) > 1:
            parsed["stdout_prefix"] = "\n".join(stdout_lines[:-1])
    parsed["returncode"] = completed.returncode
    parsed["stderr"] = completed.stderr.strip()
    return parsed


def probe_runtime_compatibility(config: Mapping[str, Any]) -> dict[str, Any]:
    python = str(config["paths"]["rl_python"])
    source = r'''
import importlib
import inspect
import json

result = {}
try:
    import torch
    result["torch_version"] = torch.__version__
    try:
        from torch.distributed.fsdp import fully_shard, FSDPModule
        result["fsdp2_public_api"] = True
        result["fsdp2_fully_shard_signature"] = str(inspect.signature(fully_shard))
        result["fsdp2_has_reshard_setter"] = hasattr(FSDPModule, "set_reshard_after_forward")
    except Exception as exc:
        result["fsdp2_public_api"] = False
        result["fsdp2_error"] = repr(exc)
except Exception as exc:
    result["torch_error"] = repr(exc)

try:
    import ray
    result["ray_version"] = ray.__version__
except Exception as exc:
    result["ray_error"] = repr(exc)

try:
    import verl
    result["verl_file"] = inspect.getfile(verl)
    result["verl_version"] = getattr(verl, "__version__", None)
    agent_modules = (
        "verl.experimental.agent_loop",
        "verl.workers.rollout.schemas",
        "verl.workers.rollout.vllm_rollout.vllm_async_server",
    )
    result["verl_agent_loop_modules"] = {}
    for name in agent_modules:
        try:
            importlib.import_module(name)
            result["verl_agent_loop_modules"][name] = True
        except Exception:
            result["verl_agent_loop_modules"][name] = False
except Exception as exc:
    result["verl_error"] = repr(exc)

try:
    import vllm
    result["vllm_version"] = vllm.__version__
    classes = []
    for module_name, class_name in (
        ("vllm", "LLM"),
        ("vllm.worker.worker", "Worker"),
    ):
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
            classes.append(cls)
        except Exception:
            pass
    result["vllm_sleep_api"] = any(hasattr(cls, "sleep") for cls in classes)
    result["vllm_wake_api"] = any(
        hasattr(cls, "wake_up") or hasattr(cls, "wake")
        for cls in classes
    )
except Exception as exc:
    result["vllm_error"] = repr(exc)

print(json.dumps(result, sort_keys=True))
'''
    probe = _run_probe(python, source)
    blockers: list[str] = []
    if probe.get("returncode") != 0:
        blockers.append(
            f"RL Python compatibility probe failed: {probe.get('stderr', '')}"
        )
    if not probe.get("fsdp2_public_api"):
        blockers.append("Public PyTorch FSDP2 fully_shard API is unavailable")
    if not probe.get("fsdp2_has_reshard_setter"):
        blockers.append("FSDP2 set_reshard_after_forward API is unavailable")
    if "vllm_error" in probe:
        blockers.append(f"vLLM import failed: {probe['vllm_error']}")
    if not probe.get("vllm_sleep_api") or not probe.get("vllm_wake_api"):
        blockers.append("Installed vLLM lacks the required sleep/wake API")
    agent_modules = probe.get("verl_agent_loop_modules", {})
    if not any(agent_modules.values()):
        blockers.append("Installed veRL lacks the required current AgentLoop extension API")
    unresolved_schedule = [
        key
        for key, value in config.get("formal_schedule", {}).items()
        if key != "source" and value is None
    ]
    if unresolved_schedule:
        blockers.append(
            "Formal schedule is not approved: " + ", ".join(unresolved_schedule)
        )
    if not config.get("runtime", {}).get("adapter_factory"):
        blockers.append(
            "No audited veRL/FSDP2/vLLM runtime adapter factory is configured"
        )
    return {
        "rl_python": python,
        "probe": probe,
        "compatible": not blockers,
        "blockers": blockers,
    }


def ray_remote_class(
    implementation: type,
    *,
    num_cpus: float,
    num_gpus: float,
    resources: Mapping[str, float] | None = None,
) -> Any:
    import ray

    return ray.remote(
        num_cpus=float(num_cpus),
        num_gpus=float(num_gpus),
        resources=dict(resources or {}),
    )(implementation)


class PromptSamplerActor:
    def __init__(
        self,
        dataset_size: int,
        shuffle_seed: int,
        logical_view_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.sampler = ImmutableDatasetPromptSampler(
            dataset_size=dataset_size,
            shuffle_seed=shuffle_seed,
        )
        self.view = (
            DeterministicNQHotpotLogicalView(**dict(logical_view_kwargs))
            if logical_view_kwargs is not None
            else None
        )

    def allocate(self, count: int) -> tuple[int, ...]:
        return self.sampler.allocate(count)

    def state(self) -> dict[str, Any]:
        return self.sampler.state().__dict__.copy()

    def restore_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        restored = PromptCursorState(
            epoch=int(state["epoch"]),
            cursor=int(state["cursor"]),
            dataset_size=int(state["dataset_size"]),
            shuffle_seed=int(state["shuffle_seed"]),
            permutation_hash=str(state["permutation_hash"]),
        )
        self.sampler = ImmutableDatasetPromptSampler.restore(restored)
        return self.state()

    def allocate_rows(self, count: int) -> tuple[dict[str, Any], ...]:
        if self.view is None:
            raise RuntimeError("PromptSamplerActor has no logical dataset view")
        logical_indices = self.sampler.allocate(count)
        return self.view.rows(logical_indices)

    def dataset_identity(self) -> dict[str, Any]:
        if self.view is None:
            raise RuntimeError("PromptSamplerActor has no logical dataset view")
        return self.view.identity.__dict__.copy()


class CandidatePoolActor:
    def __init__(self, group_size: int = 16, maximum_prompts: int = 128) -> None:
        self.pool = CandidatePool(
            group_size=group_size,
            maximum_prompts=maximum_prompts,
        )

    def add_global_groups(self, groups: list[PromptGroup]) -> int:
        self.pool.add(groups)
        return len(self.pool)

    def all_prompt_ids(self) -> tuple[str, ...]:
        return tuple(group.prompt_global_id for group in self.pool.groups())


class MetricsActor:
    def __init__(self, paths_by_scope: Mapping[str, str]) -> None:
        self.sink = JsonlMetricSink(
            {
                MetricScope(scope): path
                for scope, path in paths_by_scope.items()
            }
        )

    def write(self, scope: str, record: Mapping[str, Any]) -> None:
        self.sink.write(MetricScope(scope), record)

    def write_many(
        self,
        scope: str,
        records: Sequence[Mapping[str, Any]],
    ) -> int:
        self.sink.write_many(MetricScope(scope), records)
        return len(records)


class OutcomeWorkerActor:
    def score_batch(
        self,
        trajectories: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for item in trajectories:
            scored = score_trajectory_outcome(
                [str(value) for value in item["model_actions"]],
                [str(value) for value in item["gold_aliases"]],
                data_source=str(item.get("data_source", "")),
                trajectory_system_valid=bool(
                    item.get("trajectory_system_valid", True)
                ),
            )
            results.append(
                {
                    "trajectory_id": str(item["trajectory_id"]),
                    "task_outcome": scored.task_outcome,
                    "format_indicator": scored.format_indicator,
                    "valid_for_selection": scored.valid_for_selection,
                    "terminal_answer_valid": scored.terminal_answer_valid,
                    "trajectory_system_valid": scored.trajectory_system_valid,
                    "parser_status": scored.parse.parser_status,
                    "parser_error_type": scored.parse.parser_error_type,
                    "scorer_version": PRODUCTION_TASK_SCORER_VERSION,
                }
            )
        return results

    def score_stop_branch_batch(
        self,
        completions: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for item in completions:
            scored = score_stop_answer_completion(
                str(item["completion_text"]),
                [str(value) for value in item["gold_aliases"]],
                data_source=str(item.get("data_source", "")),
            )
            results.append(
                {
                    "prompt_global_id": str(item["prompt_global_id"]),
                    "trajectory_id": str(item["trajectory_id"]),
                    "search_index": int(item["search_index"]),
                    "sample_index": int(item["sample_index"]),
                    "task_outcome": float(scored.task_outcome),
                    "format_indicator": int(scored.format_indicator),
                    "terminal_answer_valid": bool(
                        scored.terminal_answer_valid
                    ),
                    "parser_status": str(scored.parse.parser_status),
                    "parser_error_type": scored.parse.parser_error_type,
                    "scorer_version": PRODUCTION_TASK_SCORER_VERSION,
                }
            )
        return results

    def score_sufficiency_probe_batch(
        self,
        completions: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for item in completions:
            scored = score_sufficiency_probe_completion(
                str(item["completion_text"]),
                [str(value) for value in item["gold_aliases"]],
                data_source=str(item.get("data_source", "")),
                truncated=bool(item.get("truncated", False)),
            )
            results.append(
                {
                    "prompt_global_id": str(item["prompt_global_id"]),
                    "trajectory_id": str(item["trajectory_id"]),
                    "search_index": int(item["search_index"]),
                    **(
                        {"probe_stage": str(item["probe_stage"])}
                        if "probe_stage" in item
                        else {}
                    ),
                    **scored,
                }
            )
        return results

    def score_rollout_chunk(self, rollout_chunk: Any) -> dict[str, Any]:
        extras = [
            {
                key: values[index]
                for key, values in rollout_chunk.non_tensor_batch.items()
            }
            for index in range(len(rollout_chunk))
        ]
        return {
            "extras": extras,
            "outcomes": self.score_batch(extras),
            "timing": dict(rollout_chunk.meta_info.get("timing", {})),
        }


class ExactIGTaskBuilderActor:
    def __init__(
        self,
        model_path: str,
        *,
        maximum_extended_sequence_length: int,
        maximum_position_id_exclusive: int,
    ) -> None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        self.builder = ExactIGTaskBuilder(
            tokenizer,
            maximum_extended_sequence_length=maximum_extended_sequence_length,
            maximum_position_id_exclusive=maximum_position_id_exclusive,
        )

    def build_batch(
        self,
        trajectories: Sequence[Mapping[str, Any]],
    ) -> list[Any]:
        tasks: list[Any] = []
        for item in trajectories:
            tasks.append(
                self.builder.build(
                    prompt_global_id=str(item["prompt_global_id"]),
                    trajectory_id=str(item["trajectory_id"]),
                    full_trajectory_input_ids=item["full_input_ids_unpadded"],
                    original_attention_mask=[
                        1
                    ]
                    * len(item["full_input_ids_unpadded"]),
                    prefix_end_positions=item["prefix_end_positions"],
                    canonical_answer=item["canonical_answer"],
                )
            )
        prompt_ids = sorted({task.prompt_global_id for task in tasks})
        for prompt_id in prompt_ids:
            assert_same_prompt_target_consistency(
                [task for task in tasks if task.prompt_global_id == prompt_id]
            )
        return tasks

    def build_rollout_chunk(self, rollout_chunk: Any) -> list[Any]:
        trajectories = [
            {
                key: values[index]
                for key, values in rollout_chunk.non_tensor_batch.items()
            }
            for index in range(len(rollout_chunk))
            if bool(
                rollout_chunk.non_tensor_batch["trajectory_system_valid"][index]
            )
        ]
        return self.build_batch(trajectories)


class CheckpointCommitActor:
    """Rank-zero checkpoint event coordinator; tensor state stays on FSDP ranks."""

    def __init__(self, event_log_path: str) -> None:
        self.event_log_path = Path(event_log_path)
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: Mapping[str, Any]) -> None:
        with self.event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(event), sort_keys=True) + "\n")
