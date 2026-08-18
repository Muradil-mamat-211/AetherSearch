#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentic_rl.exact_ig.fsdp_scoring_window import (
    FSDPReshardStateRegistry,
    exact_ig_scoring_window,
    fsdp2_modules,
)
from agentic_rl.exact_ig.precision_policy import production_precision_policy
from agentic_rl.exact_ig.sequential_oracle import (
    sequential_teacher_forced_oracle,
)
from agentic_rl.exact_ig.task_builder import (
    ExactIGTaskBuilder,
    VectorizedExactIGTask,
)
from agentic_rl.exact_ig.vectorized_scorer import (
    OFFICIAL_ADDITIVE_MASK,
    OFFICIAL_FULL_LOGITS,
    VectorizedExactIGScorer,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "/root/autodl-tmp/search-r1-workspace/models/"
            "dpo_v2_final_model"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def _local_parameter_checksum(model: torch.nn.Module) -> str:
    """Hash distributed local-shard samples without materializing full weights."""

    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            local = (
                parameter.to_local()
                if callable(getattr(parameter, "to_local", None))
                else parameter
            )
            flat = local.detach().reshape(-1)
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(parameter.shape)).encode("ascii"))
            digest.update(str(parameter.dtype).encode("ascii"))
            if flat.numel():
                edge = min(16, int(flat.numel()))
                sample_tensor = torch.cat(
                    (flat[:edge], flat[-edge:]),
                    dim=0,
                )
                sample = sample_tensor.contiguous().cpu().numpy().tobytes()
                digest.update(sample)
    return digest.hexdigest()


def _normalize_sharded_residency(model: torch.nn.Module) -> None:
    for module in reversed(fsdp2_modules(model)):
        module.reshard()
    dist.barrier()


def _build_real_task(
    tokenizer: Any,
    maximum_context: int,
) -> VectorizedExactIGTask:
    prompt = (
        "<|im_start|>system\nYou answer with search evidence."
        "<|im_end|>\n<|im_start|>user\n"
        "Which city is the capital of France?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    search = (
        "<search>capital of France</search>"
        "<information>Paris is the capital and largest city of France."
        "</information>"
    )
    original = tokenizer(
        prompt + search,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    prompt_end = len(
        tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
    )
    task = ExactIGTaskBuilder(
        tokenizer,
        maximum_extended_sequence_length=maximum_context,
        maximum_position_id_exclusive=maximum_context,
    ).build(
        prompt_global_id="fsdp4-real-prompt",
        trajectory_id="fsdp4-real-trajectory",
        full_trajectory_input_ids=original,
        original_attention_mask=[1] * len(original),
        prefix_end_positions=[prompt_end, len(original)],
        canonical_answer="Paris",
    )
    if not isinstance(task, VectorizedExactIGTask):
        raise RuntimeError("The FSDP4 validation task unexpectedly required fallback")
    return task


def _allclose_rows(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
) -> tuple[bool, float]:
    passed = True
    maximum = 0.0
    for left_row, right_row in zip(left, right, strict=True):
        left_tensor = torch.tensor(left_row, dtype=torch.float32)
        right_tensor = torch.tensor(right_row, dtype=torch.float32)
        maximum = max(
            maximum,
            float(torch.max(torch.abs(left_tensor - right_tensor)).item()),
        )
        passed = passed and bool(
            torch.allclose(left_tensor, right_tensor, rtol=1e-4, atol=1e-6)
        )
    return passed, maximum


def main() -> None:
    args = _arguments()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise RuntimeError(f"Expected FSDP2 world size 4, got {world_size}")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)

    local_result: dict[str, Any]
    phase = "initialization"
    try:
        phase = "load_tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            use_fast=True,
            trust_remote_code=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        phase = "load_model"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.float32,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        ).to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        phase = "fully_shard"
        mesh = init_device_mesh("cuda", (world_size,))
        initial_states: dict[Any, bool] = {}
        layers = tuple(model.model.layers)
        for layer_index, layer in enumerate(layers):
            initial = bool(layer_index % 2 == 0)
            fully_shard(
                layer,
                mesh=mesh,
                reshard_after_forward=initial,
            )
            initial_states[layer] = initial
        fully_shard(
            model,
            mesh=mesh,
            reshard_after_forward=False,
        )
        initial_states[model] = False

        phase = "register_reshard_states"
        registry = FSDPReshardStateRegistry()
        fsdp_modules = fsdp2_modules(model)
        if set(fsdp_modules) != set(initial_states):
            raise RuntimeError(
                "FSDP2 module discovery differs from the explicit state registry"
            )
        registry.register_model(model, initial_states)
        _normalize_sharded_residency(model)
        phase = "checksum_before"
        checksum_before = _local_parameter_checksum(model)
        phase = "build_task"
        task = _build_real_task(
            tokenizer,
            int(getattr(model.config, "max_position_embeddings", 32768)),
        )
        policy = production_precision_policy("official_bf16_autocast")
        scorer = VectorizedExactIGScorer(
            precision_policy=policy,
            padding_token_id=int(tokenizer.pad_token_id),
            tokenizer=tokenizer,
            scoring_logits_mode=OFFICIAL_FULL_LOGITS,
            attention_mask_mode=OFFICIAL_ADDITIVE_MASK,
        )
        phase = "fast_scoring_window"
        with exact_ig_scoring_window(
            model,
            registry=registry,
            reshard_after_forward=False,
            synchronize=dist.barrier,
            memory_snapshot=lambda: (
                torch.cuda.memory_allocated(device),
                torch.cuda.memory_reserved(device),
            ),
        ) as report:
            window_observed = tuple(
                registry.state_for(module) for module in fsdp_modules
            )
            phase = "fast_score"
            fast = scorer.score(model, task, device)
            phase = "sequential_score"
            sequential = sequential_teacher_forced_oracle(
                model=model,
                tokenizer=tokenizer,
                full_trajectory_input_ids=task.input_ids[
                    : task.original_token_count
                ],
                original_attention_mask=task.original_attention_mask,
                original_position_ids=task.original_position_ids,
                prefix_end_positions=task.prefix_end_positions,
                canonical_answer=task.canonical_answer,
                encoded_target=task.canonical_target,
                device=device,
                precision_policy=policy,
            )
        phase = "checksum_after"
        _normalize_sharded_residency(model)
        checksum_after = _local_parameter_checksum(model)
        token_allclose, max_abs_diff = _allclose_rows(
            fast.answer_token_log_probs_by_prefix,
            sequential.answer_token_log_probs_by_prefix,
        )
        state_after = tuple(
            registry.state_for(module) for module in fsdp_modules
        )

        exception_restored = False
        phase = "exception_restore"
        try:
            with exact_ig_scoring_window(
                model,
                registry=registry,
                reshard_after_forward=False,
                synchronize=dist.barrier,
            ):
                raise ValueError("intentional FSDP scoring-window body failure")
        except ValueError as error:
            exception_restored = (
                str(error) == "intentional FSDP scoring-window body failure"
                and tuple(
                    registry.state_for(module) for module in fsdp_modules
                )
                == tuple(initial_states[module] for module in fsdp_modules)
            )

        phase = "complete"
        local_result = {
            "rank": rank,
            "local_rank": local_rank,
            "device": str(device),
            "module_count": len(fsdp_modules),
            "before_states": list(report.before_states),
            "window_states_declared": list(report.window_states),
            "window_states_observed": list(window_observed),
            "after_states": list(state_after),
            "restore_succeeded": bool(report.restore_succeeded),
            "exception_restore_succeeded": bool(exception_restored),
            "exit_allocated_bytes": report.exit_allocated_bytes,
            "exit_reserved_bytes": report.exit_reserved_bytes,
            "model_checksum_before": checksum_before,
            "model_checksum_after": checksum_after,
            "model_checksum_unchanged": checksum_before == checksum_after,
            "residency_normalized_for_checksum": True,
            "fast_phi": list(fast.score_by_prefix),
            "sequential_phi": list(sequential.score_by_prefix),
            "fast_ig": list(fast.immediate_ig),
            "sequential_ig": list(sequential.immediate_ig),
            "official_allclose": token_allclose,
            "max_abs_token_log_prob_diff": max_abs_diff,
            "metadata_hash": hashlib.sha256(
                json.dumps(
                    {
                        "fast_phi": fast.score_by_prefix,
                        "sequential_phi": sequential.score_by_prefix,
                        "target_hash": task.target_token_ids_hash,
                        "answer_range": [
                            task.canonical_target.answer_token_start,
                            task.canonical_target.answer_token_end,
                        ],
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
            "error": None,
        }
    except BaseException as error:
        local_result = {
            "rank": rank,
            "local_rank": local_rank,
            "phase": phase,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        rank_error = args.output.parent / f"fsdp_rank_{rank}_error.json"
        rank_error.write_text(
            json.dumps(local_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise

    phase = "all_gather_metadata"
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_result)
    if rank == 0:
        rows = [row for row in gathered if row is not None]
        metadata_hashes = {
            row.get("metadata_hash") for row in rows if row.get("error") is None
        }
        payload = {
            "schema": "exact_ig_fsdp2_window_restore_v3",
            "world_size": world_size,
            "rows": rows,
            "all_ranks_completed": (
                len(rows) == world_size
                and all(row.get("error") is None for row in rows)
            ),
            "all_ranks_restore_succeeded": all(
                row.get("restore_succeeded") is True for row in rows
            ),
            "all_ranks_exception_restore_succeeded": all(
                row.get("exception_restore_succeeded") is True for row in rows
            ),
            "all_rank_checksums_unchanged": all(
                row.get("model_checksum_unchanged") is True for row in rows
            ),
            "rank_metadata_consistent": len(metadata_hashes) == 1,
            "official_fast_sequential_allclose": all(
                row.get("official_allclose") is True for row in rows
            ),
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
        }
        payload["fsdp_window_restore_pass"] = bool(
            payload["all_ranks_completed"]
            and payload["all_ranks_restore_succeeded"]
            and payload["all_ranks_exception_restore_succeeded"]
            and payload["all_rank_checksums_unchanged"]
            and payload["rank_metadata_consistent"]
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output)
        print(
            json.dumps(
                {
                    "fsdp_window_restore_pass": payload[
                        "fsdp_window_restore_pass"
                    ],
                    "official_fast_sequential_allclose": payload[
                        "official_fast_sequential_allclose"
                    ],
                    "rank_metadata_consistent": payload[
                        "rank_metadata_consistent"
                    ],
                },
                sort_keys=True,
            )
        )
    dist.barrier()
    dist.destroy_process_group()
    if local_result.get("error") is not None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
