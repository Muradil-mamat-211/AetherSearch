from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROLE_LOCALIZED_GATE_MODE = (
    "sufficiency_novelty_cumulative_ig_probe_routed_outcome_"
    "role_localized_gate"
)


@dataclass(frozen=True)
class BatchGradientProfile:
    batch_id: str
    main_gradient_norm: float
    decision_gradient_norm: float
    query_gradient_norm: float
    dot_main_decision: float
    dot_main_query: float
    dot_decision_query: float
    cos_main_decision: float
    cos_main_query: float
    cos_decision_query: float
    decision_gate_event_count: int
    query_gate_event_count: int
    parameters_bitwise_unchanged: bool
    gradients_cleared: bool
    rank_metadata_consistent: bool

    def validate(self) -> None:
        scalar_fields = (
            self.main_gradient_norm,
            self.decision_gradient_norm,
            self.query_gradient_norm,
            self.dot_main_decision,
            self.dot_main_query,
            self.dot_decision_query,
            self.cos_main_decision,
            self.cos_main_query,
            self.cos_decision_query,
        )
        if not all(math.isfinite(float(value)) for value in scalar_fields):
            raise ValueError(f"{self.batch_id}: gradient profile is non-finite")
        if min(
            self.main_gradient_norm,
            self.decision_gradient_norm,
            self.query_gradient_norm,
        ) < 0.0:
            raise ValueError(f"{self.batch_id}: gradient norm is negative")
        if self.decision_gate_event_count < 0 or self.query_gate_event_count < 0:
            raise ValueError(f"{self.batch_id}: event count is negative")
        if not (
            self.parameters_bitwise_unchanged
            and self.gradients_cleared
            and self.rank_metadata_consistent
        ):
            raise ValueError(f"{self.batch_id}: no-update safety contract failed")


def _median_positive(values: Sequence[float], *, field_name: str) -> float:
    nonzero = [float(value) for value in values if float(value) > 0.0]
    if not nonzero:
        raise ValueError(f"No nonzero {field_name} values were observed")
    return float(statistics.median(nonzero))


def _cosine(dot: float, left_norm: float, right_norm: float, epsilon: float) -> float:
    denominator = float(left_norm) * float(right_norm)
    return float(dot / (denominator + epsilon)) if denominator else 0.0


def calibrate_role_localized_gate_lambdas(
    profiles: Sequence[BatchGradientProfile | Mapping[str, Any]],
    *,
    eta_decision: float = 0.10,
    eta_query: float = 0.05,
    maximum_gate_to_main_ratio: float = 0.15,
    epsilon: float = 1.0e-12,
) -> dict[str, Any]:
    """Derive immutable gate coefficients from detached fresh-U0 profiles."""

    rows = tuple(
        profile
        if isinstance(profile, BatchGradientProfile)
        else BatchGradientProfile(**dict(profile))
        for profile in profiles
    )
    if len(rows) < 3:
        raise ValueError("Calibration requires at least three representative batches")
    for row in rows:
        row.validate()
    decision_events = sum(row.decision_gate_event_count for row in rows)
    query_events = sum(row.query_gate_event_count for row in rows)
    if decision_events < 128:
        raise ValueError(
            f"Decision calibration requires >=128 events, observed {decision_events}"
        )
    if query_events < 64:
        raise ValueError(
            f"Query calibration requires >=64 events, observed {query_events}"
        )
    if not (
        float(eta_decision) == 0.10
        and float(eta_query) == 0.05
        and float(maximum_gate_to_main_ratio) == 0.15
        and float(epsilon) > 0.0
    ):
        raise ValueError("Gate calibration budgets are locked")

    median_main = _median_positive(
        [row.main_gradient_norm for row in rows],
        field_name="Main gradient norm",
    )
    median_decision = _median_positive(
        [row.decision_gradient_norm for row in rows],
        field_name="Decision gradient norm",
    )
    median_query = _median_positive(
        [row.query_gradient_norm for row in rows],
        field_name="Query gradient norm",
    )
    median_cos_md = float(statistics.median(row.cos_main_decision for row in rows))
    median_cos_mq = float(statistics.median(row.cos_main_query for row in rows))
    median_cos_dq = float(statistics.median(row.cos_decision_query for row in rows))
    effective_eta_decision = float(eta_decision) * (
        0.5 if median_cos_md < -0.5 else 1.0
    )
    effective_eta_query = float(eta_query) * (
        0.5 if median_cos_mq < -0.5 else 1.0
    )
    lambda_decision = min(
        1.0,
        max(0.0, effective_eta_decision * median_main / (median_decision + epsilon)),
    )
    lambda_query = min(
        1.0,
        max(0.0, effective_eta_query * median_main / (median_query + epsilon)),
    )

    def weighted_gate_norm(row: BatchGradientProfile) -> float:
        squared = (
            lambda_decision * lambda_decision * row.decision_gradient_norm**2
            + lambda_query * lambda_query * row.query_gradient_norm**2
            + 2.0
            * lambda_decision
            * lambda_query
            * row.dot_decision_query
        )
        return math.sqrt(max(float(squared), 0.0))

    gate_norms = [weighted_gate_norm(row) for row in rows]
    gate_to_main = [
        norm / (row.main_gradient_norm + epsilon)
        for norm, row in zip(gate_norms, rows, strict=True)
    ]
    median_gate_ratio = float(statistics.median(gate_to_main))
    common_scale = 1.0
    if median_gate_ratio > maximum_gate_to_main_ratio:
        common_scale = float(maximum_gate_to_main_ratio / median_gate_ratio)
        lambda_decision *= common_scale
        lambda_query *= common_scale
        gate_norms = [weighted_gate_norm(row) for row in rows]
        gate_to_main = [
            norm / (row.main_gradient_norm + epsilon)
            for norm, row in zip(gate_norms, rows, strict=True)
        ]
        median_gate_ratio = float(statistics.median(gate_to_main))
    if median_gate_ratio > maximum_gate_to_main_ratio + 1.0e-12:
        raise RuntimeError("Weighted gate gradient budget was not enforced")

    return {
        "schema_version": 1,
        "status": "PASS",
        "search_task_mode": ROLE_LOCALIZED_GATE_MODE,
        "batch_count": len(rows),
        "decision_gate_event_count": decision_events,
        "query_gate_event_count": query_events,
        "eta_decision_requested": float(eta_decision),
        "eta_query_requested": float(eta_query),
        "eta_decision_effective": effective_eta_decision,
        "eta_query_effective": effective_eta_query,
        "lambda_decision": float(lambda_decision),
        "lambda_query": float(lambda_query),
        "common_budget_scale": common_scale,
        "median_main_gradient_norm": median_main,
        "median_decision_gradient_norm": median_decision,
        "median_query_gradient_norm": median_query,
        "median_weighted_gate_gradient_norm": float(statistics.median(gate_norms)),
        "median_gate_to_main_gradient_ratio": median_gate_ratio,
        "median_cos_main_decision": median_cos_md,
        "median_cos_main_query": median_cos_mq,
        "median_cos_decision_query": median_cos_dq,
        "maximum_gate_to_main_gradient_ratio": float(maximum_gate_to_main_ratio),
        "parameters_bitwise_unchanged": all(
            row.parameters_bitwise_unchanged for row in rows
        ),
        "gradients_cleared": all(row.gradients_cleared for row in rows),
        "all_rank_metadata_consistent": all(
            row.rank_metadata_consistent for row in rows
        ),
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "checkpoint_writes": 0,
        "profiles": [asdict(row) for row in rows],
    }


def write_immutable_calibration_manifest(
    path: str | Path,
    payload: Mapping[str, Any],
) -> str:
    """Atomically create one read-only manifest and return its SHA-256."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Calibration manifest already exists: {destination}")
    serialized = (
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        destination.chmod(0o444)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(serialized).hexdigest()


def parameter_shard_sha256(parameters: Sequence[Any]) -> str:
    """Hash local parameter shards without gathering or changing model state."""

    digest = hashlib.sha256()
    for parameter in parameters:
        value = parameter.detach()
        if hasattr(value, "to_local"):
            value = value.to_local()
        contiguous = value.contiguous().cpu()
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        # NumPy cannot represent every Torch dtype (notably bfloat16). Hash
        # the exact underlying bytes without numerically casting the shard.
        byte_view = contiguous.view(__import__("torch").uint8).numpy()
        digest.update(byte_view.tobytes(order="C"))
    return digest.hexdigest()


def global_gradient_profile_from_shards(
    main_gradients: Sequence[Any],
    decision_gradients: Sequence[Any],
    query_gradients: Sequence[Any],
    *,
    epsilon: float = 1.0e-12,
) -> dict[str, float]:
    """All-reduce exact squared norms/dots for matching FSDP parameter shards."""

    import torch
    import torch.distributed as dist

    if not (
        len(main_gradients) == len(decision_gradients) == len(query_gradients)
    ):
        raise ValueError("Channel gradient shard cardinalities differ")
    local = torch.zeros(6, dtype=torch.float64)
    for main, decision, query in zip(
        main_gradients,
        decision_gradients,
        query_gradients,
        strict=True,
    ):
        main_flat = main.detach().float().cpu().reshape(-1)
        decision_flat = decision.detach().float().cpu().reshape(-1)
        query_flat = query.detach().float().cpu().reshape(-1)
        if not (
            main_flat.shape == decision_flat.shape == query_flat.shape
        ):
            raise ValueError("Channel gradient shard shapes differ")
        for start in range(0, main_flat.numel(), 1_048_576):
            end = min(start + 1_048_576, main_flat.numel())
            main_chunk = main_flat[start:end].double()
            decision_chunk = decision_flat[start:end].double()
            query_chunk = query_flat[start:end].double()
            local[0] += torch.dot(main_chunk, main_chunk)
            local[1] += torch.dot(decision_chunk, decision_chunk)
            local[2] += torch.dot(query_chunk, query_chunk)
            local[3] += torch.dot(main_chunk, decision_chunk)
            local[4] += torch.dot(main_chunk, query_chunk)
            local[5] += torch.dot(decision_chunk, query_chunk)
    if dist.is_available() and dist.is_initialized():
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL FSDP calibration requires CUDA")
        reduced = local.to(torch.device("cuda", torch.cuda.current_device()))
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        local = reduced.cpu()
    main_norm = math.sqrt(max(float(local[0]), 0.0))
    decision_norm = math.sqrt(max(float(local[1]), 0.0))
    query_norm = math.sqrt(max(float(local[2]), 0.0))
    return {
        "main_gradient_norm": main_norm,
        "decision_gradient_norm": decision_norm,
        "query_gradient_norm": query_norm,
        "dot_main_decision": float(local[3]),
        "dot_main_query": float(local[4]),
        "dot_decision_query": float(local[5]),
        "cos_main_decision": _cosine(
            float(local[3]), main_norm, decision_norm, epsilon
        ),
        "cos_main_query": _cosine(
            float(local[4]), main_norm, query_norm, epsilon
        ),
        "cos_decision_query": _cosine(
            float(local[5]), decision_norm, query_norm, epsilon
        ),
    }
