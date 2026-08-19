from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from agentic_rl.config import load_config
from agentic_rl.outcome.parser import parse_model_action, parse_model_trajectory
from agentic_rl.outcome.workers import score_trajectory_outcome
from agentic_rl.retriever.client import HybridRetrieverClient
from agentic_rl.retriever.health import query_health
from agentic_rl.topology import TopologyPlan

from .fixed_eval import create_or_validate_eval_manifest_from_config, load_eval_rows
from .formal_state import (
    append_jsonl,
    atomic_write_json,
    claim_next_eval,
    complete_eval,
    defer_eval,
    eval_queue_snapshot,
    read_json,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gpu_snapshot(physical_gpu: int) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    for line in completed.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        if values and int(values[0]) == int(physical_gpu):
            return {
                "physical_gpu": int(physical_gpu),
                "memory_used_mib": int(values[1]),
                "memory_free_mib": int(values[2]),
                "memory_total_mib": int(values[3]),
                "temperature_c": int(values[4]),
            }
    raise RuntimeError(
        f"Configured eval GPU {int(physical_gpu)} was not reported by nvidia-smi"
    )


def _information_token_ids(
    tokenizer: Any,
    documents: Sequence[Any],
    maximum_tokens: int,
) -> tuple[list[int], str]:
    body = "\n".join(str(document.contents) for document in documents)
    prefix = tokenizer(
        "<information>",
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    suffix = tokenizer(
        "</information>",
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    budget = int(maximum_tokens) - len(prefix) - len(suffix)
    if budget < 0:
        raise RuntimeError("Information-token cap is smaller than protocol tags")
    body_ids = tokenizer(
        body,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"][:budget]
    token_ids = [int(value) for value in (*prefix, *body_ids, *suffix)]
    return token_ids, tokenizer.decode(token_ids, skip_special_tokens=False)


def _evaluate_row(
    *,
    model: Any,
    tokenizer: Any,
    retriever: HybridRetrieverClient,
    row: Mapping[str, Any],
    update: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    prompt_ids = tokenizer.apply_chat_template(
        list(row["prompt_messages"]),
        add_generation_prompt=True,
        tokenize=True,
    )
    prompt_ids = [int(value) for value in prompt_ids]
    maximum_prompt = int(config["formal_schedule"]["maximum_prompt_length"])
    maximum_response = int(config["formal_schedule"]["maximum_response_length"])
    maximum_model = int(config["formal_schedule"]["maximum_model_length"])
    if not prompt_ids or len(prompt_ids) > maximum_prompt:
        raise RuntimeError("Eval prompt is empty or exceeds the formal prompt limit")

    response_ids: list[int] = []
    model_actions: list[str] = []
    queries: list[str] = []
    system_valid = True
    environment_failure_code: str | None = None
    max_search_turns = int(config["rollout"]["max_search_turns"])
    max_model_tokens = int(config["rollout"]["max_model_tokens_per_turn"])
    max_information_tokens = int(
        config["rollout"]["max_information_tokens_per_turn"]
    )

    for turn_index in range(max_search_turns + 1):
        remaining_response = maximum_response - len(response_ids)
        remaining_context = maximum_model - len(prompt_ids) - len(response_ids)
        maximum_new = min(max_model_tokens, remaining_response, remaining_context)
        if maximum_new <= 0:
            break
        input_ids = torch.tensor(
            [prompt_ids + response_ids],
            dtype=torch.long,
            device="cuda",
        )
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=int(maximum_new),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        action_ids = [
            int(value)
            for value in generated[0, input_ids.shape[1] :].detach().cpu().tolist()
        ]
        del input_ids, attention_mask, generated
        if not action_ids:
            break
        action_text = tokenizer.decode(action_ids, skip_special_tokens=True)
        response_ids.extend(action_ids)
        model_actions.append(action_text)
        parsed = parse_model_action(action_text, action_index=turn_index)
        if not parsed.valid or parsed.kind == "answer":
            break
        if len(queries) >= max_search_turns:
            break
        query = str(parsed.value)
        queries.append(query)
        try:
            retrieval = retriever.retrieve([query])
            documents = retrieval.documents_by_query[0]
            information_ids, _ = _information_token_ids(
                tokenizer,
                documents,
                max_information_tokens,
            )
            if len(response_ids) + len(information_ids) > maximum_response:
                raise RuntimeError("retriever_information_exceeds_response_budget")
            if len(prompt_ids) + len(response_ids) + len(information_ids) > maximum_model:
                raise RuntimeError("retriever_information_exceeds_model_context")
            response_ids.extend(information_ids)
        except BaseException as exc:
            system_valid = False
            environment_failure_code = type(exc).__name__
            break

    aliases = tuple(str(value) for value in row["gold_aliases"])
    outcome = score_trajectory_outcome(
        model_actions,
        aliases,
        data_source=str(row["data_source"]),
        trajectory_system_valid=system_valid,
    )
    parsed_trajectory = parse_model_trajectory(model_actions)
    return {
        "successful_update_step": int(update),
        "prompt_global_id": str(row["prompt_global_id"]),
        "dataset_row_id": str(row.get("id", "")),
        "dataset_source_index": int(row["source_index"]),
        "domain": str(row["data_source"]),
        "trajectory_id": f"{row['prompt_global_id']}:eval-update-{update:03d}",
        "R_task": float(outcome.task_outcome),
        "exact": math.isclose(
            float(outcome.task_outcome), 1.0, rel_tol=0.0, abs_tol=1.0e-12
        ),
        "F_ans": int(outcome.format_indicator),
        "terminal_answer_valid": bool(outcome.terminal_answer_valid),
        "system_valid": bool(system_valid),
        "environment_failure_code": environment_failure_code,
        "search_count": len(queries),
        "queries": queries,
        "model_actions": model_actions,
        "answer": parsed_trajectory.answer,
    }


def _aggregate(
    predictions: Sequence[Mapping[str, Any]],
    *,
    update: int,
    manifest_sha256: str,
    actor_checksum: str,
    wall_seconds: float,
) -> list[dict[str, Any]]:
    domains = sorted({str(item["domain"]) for item in predictions})
    records: list[dict[str, Any]] = []
    for domain in (*domains, "overall"):
        subset = (
            list(predictions)
            if domain == "overall"
            else [item for item in predictions if str(item["domain"]) == domain]
        )
        count = len(subset)
        outcomes = [float(item["R_task"]) for item in subset]
        searches = [int(item["search_count"]) for item in subset]
        all_queries = [
            " ".join(str(query).lower().split())
            for item in subset
            for query in item["queries"]
        ]
        repeats = sum(
            len(item["queries"])
            != len({" ".join(str(query).lower().split()) for query in item["queries"]})
            for item in subset
        )
        records.append(
            {
                "successful_update_step": int(update),
                "domain": domain,
                "count": count,
                "f1": float(np.mean(outcomes, dtype=np.float64)) if outcomes else 0.0,
                "exact": sum(bool(item["exact"]) for item in subset) / count if count else 0.0,
                "format_rate": sum(int(item["F_ans"]) for item in subset) / count if count else 0.0,
                "answer_rate": sum(bool(item["terminal_answer_valid"]) for item in subset) / count if count else 0.0,
                "no_answer_rate": sum(not bool(item["terminal_answer_valid"]) for item in subset) / count if count else 0.0,
                "avg_search": float(np.mean(searches, dtype=np.float64)) if searches else 0.0,
                "multi_search_rate": sum(value >= 2 for value in searches) / count if count else 0.0,
                "repeat_query_rate": repeats / count if count else 0.0,
                "max_turn_rate": sum(value >= 5 for value in searches) / count if count else 0.0,
                "query_diversity": len(set(all_queries)) / len(all_queries) if all_queries else 0.0,
                "template_similarity": 1.0 - len(set(all_queries)) / len(all_queries) if all_queries else 0.0,
                "manifest_sha256": str(manifest_sha256),
                "actor_checksum": str(actor_checksum),
                "wall_seconds": float(wall_seconds),
                "evaluation_device": f"physical_gpu_{int(config['_eval_physical_gpu'])}",
            }
        )
    return records


def _verify_model_checkpoint(model_path: Path) -> dict[str, Any]:
    completed = model_path / "COMPLETED"
    metadata_path = model_path / "training_metadata.json"
    weights = model_path / "model.safetensors"
    if not completed.is_file() or not metadata_path.is_file() or not weights.is_file():
        raise RuntimeError("Eval model checkpoint is not atomically committed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = metadata["manifest"].get("model.safetensors")
    if not expected or _sha256_file(weights) != expected:
        raise RuntimeError("Eval model checkpoint checksum failed")
    return metadata


def _run_task(
    *,
    task: Mapping[str, Any],
    config: Mapping[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(str(task["model_path"])).resolve()
    metadata = _verify_model_checkpoint(model_path)
    if metadata["actor_checksum"] != str(task["actor_checksum"]):
        raise RuntimeError("Eval queue/model Actor checksums differ")
    evaluation = config["evaluation"]
    torch.cuda.set_per_process_memory_fraction(
        float(evaluation["max_memory_fraction"]),
        device=0,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=str(evaluation["attention_backend"]),
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    manifest = create_or_validate_eval_manifest_from_config(
        validation_path=config["paths"]["validation_data"],
        evaluation=evaluation,
    )
    if manifest["manifest_sha256"] != str(evaluation["expected_manifest_sha256"]):
        raise RuntimeError("Fixed Eval manifest SHA-256 differs from Pilot")
    rows = load_eval_rows(manifest=manifest)
    retriever = HybridRetrieverClient(
        str(config["retriever"]["service_url"]),
        timeout_seconds=float(config["retriever"]["timeout_seconds"]),
        default_top_k=int(config["retriever"].get("top_k", 3)),
    )

    update = int(task["update"])
    eval_root = run_dir / "eval"
    temporary = eval_root / f".tmp_update_{update:03d}"
    destination = eval_root / f"update_{update:03d}"
    if destination.exists():
        raise FileExistsError(f"Eval output already exists: {destination}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    predictions_path = temporary / "predictions.jsonl"
    log_path = temporary / "eval.log"
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    peak_memory = 0
    with predictions_path.open("w", encoding="utf-8", buffering=1) as output, log_path.open(
        "w", encoding="utf-8", buffering=1
    ) as log:
        for index, row in enumerate(rows, start=1):
            prediction = _evaluate_row(
                model=model,
                tokenizer=tokenizer,
                retriever=retriever,
                row=row,
                update=update,
                config=config,
            )
            predictions.append(prediction)
            output.write(json.dumps(prediction, sort_keys=True) + "\n")
            if index % 25 == 0 or index == len(rows):
                peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated()))
                log.write(
                    f"{time.time():.6f} update={update} completed={index}/{len(rows)} "
                    f"peak_memory_bytes={peak_memory}\n"
                )
        output.flush()
        os.fsync(output.fileno())
        log.flush()
        os.fsync(log.fileno())
    wall_seconds = time.perf_counter() - started
    aggregate = _aggregate(
        predictions,
        update=update,
        manifest_sha256=manifest["manifest_sha256"],
        actor_checksum=str(task["actor_checksum"]),
        wall_seconds=wall_seconds,
    )
    summary = {
        "status": "PASS",
        "successful_update_step": update,
        "model_path": str(model_path),
        "actor_checksum": str(task["actor_checksum"]),
        "manifest_sha256": manifest["manifest_sha256"],
        "metrics": aggregate,
        "predictions": str(destination / "predictions.jsonl"),
        "wall_seconds": wall_seconds,
        "peak_memory_bytes": peak_memory,
    }
    (temporary / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (temporary / "COMPLETED").write_text("status=PASS\n", encoding="utf-8")
    os.replace(temporary, destination)
    for record in aggregate:
        append_jsonl(run_dir / "metrics" / "eval_metrics.jsonl", record)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _target_successful_update(config: Mapping[str, Any]) -> int:
    target_successful_update = int(
        config["formal_schedule"]["total_successful_updates"]
    )
    if target_successful_update <= 0:
        raise RuntimeError("Async Eval requires a positive target update")
    return target_successful_update


def run_worker(config_path: Path, run_dir: Path) -> int:
    config = load_config(config_path)
    topology = TopologyPlan.from_config(config)
    if topology.eval_physical_gpu is None:
        raise RuntimeError("Async evaluation requires a configured eval role")
    config = dict(config)
    config["_eval_physical_gpu"] = int(topology.eval_physical_gpu)
    target_successful_update = _target_successful_update(config)
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    retry_seconds = int(config["evaluation"]["retry_seconds"])
    minimum_free_mib = int(float(config["evaluation"]["minimum_free_memory_gib"]) * 1024)
    while not stop_requested:
        task = claim_next_eval(run_dir, worker_pid=os.getpid())
        if task is None:
            trainer = read_json(run_dir / "state" / "trainer_result.json", {})
            queue = eval_queue_snapshot(run_dir)
            complete = {
                int(item["update"])
                for item in queue["tasks"]
                if item["status"] == "completed"
            }
            if (
                trainer.get("status") == "PASS"
                and int(trainer.get("successful_update_step", -1))
                == target_successful_update
                and target_successful_update in complete
                and all(item["status"] == "completed" for item in queue["tasks"])
            ):
                atomic_write_json(
                    run_dir / "state" / "eval_worker_result.json",
                    {
                        "status": "PASS",
                        "target_successful_update": target_successful_update,
                        "completed_updates": sorted(complete),
                        "timestamp": time.time(),
                    },
                )
                return 0
            time.sleep(30)
            continue

        update = int(task["update"])
        try:
            query_health(str(config["retriever"]["service_url"]))
            gpu = _gpu_snapshot(int(config["_eval_physical_gpu"]))
            if int(gpu["memory_free_mib"]) < minimum_free_mib:
                reason = (
                    f"gpu{gpu['physical_gpu']}_free_memory_{gpu['memory_free_mib']}MiB_below_"
                    f"{minimum_free_mib}MiB"
                )
                defer_eval(run_dir, update=update, reason=reason)
                time.sleep(retry_seconds)
                continue
            summary = _run_task(task=task, config=config, run_dir=run_dir)
            complete_eval(run_dir, update=update, error=None)
            append_jsonl(
                run_dir / "logs" / "eval_worker_events.jsonl",
                {"event": "completed", "update": update, "summary": summary, "timestamp": time.time()},
            )
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            complete_eval(run_dir, update=update, error=error)
            append_jsonl(
                run_dir / "logs" / "eval_worker_events.jsonl",
                {"event": "failed_retry", "update": update, "error": error, "timestamp": time.time()},
            )
            time.sleep(retry_seconds)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Serialized topology-routed eval worker")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    arguments = parser.parse_args()
    raise SystemExit(
        run_worker(Path(arguments.config).resolve(), Path(arguments.run_dir).resolve())
    )


if __name__ == "__main__":
    main()
