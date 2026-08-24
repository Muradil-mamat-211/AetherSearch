"""Portable runtime resource checks plus explicit reference qualification.

These checks are deliberately independent of algorithm configuration. They
prevent Ray from being started with a resource contract larger than the
actual cgroup and make the memory budget visible in the run metadata.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

from agentic_rl.config import runtime_section

BYTES_PER_GIB = 1024**3
CHECKPOINT_MEMORY_HEADROOM_GIB = 24.0
CHECKPOINT_MEMORY_IO_SAFETY_GIB = 4.0
CHECKPOINT_MIN_FREE_DISK_GIB = 48.0
CHECKPOINT_DISK_SAFETY_GIB = 8.0
CHECKPOINT_SIZE_MARGIN = 1.10


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _read_limit(path: Path) -> int | None:
    value = _read_text(path)
    if value is None or value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid cgroup limit in {path}: {value!r}") from exc
    return parsed if parsed > 0 else None


def _read_cpu_quota(path: Path) -> float | None:
    value = _read_text(path)
    if value is None:
        return None
    fields = value.split()
    if len(fields) != 2 or fields[0] == "max":
        return None
    try:
        quota, period = int(fields[0]), int(fields[1])
    except ValueError as exc:
        raise RuntimeError(f"Invalid cgroup CPU quota in {path}: {value!r}") from exc
    if quota <= 0 or period <= 0:
        return None
    return quota / period


def _read_gpu_inventory() -> dict[str, Any]:
    """Read physical GPU count and the smallest installed GPU memory size."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return {"gpu_count": None, "gpu_memory_gib": None}
    memories_mib: list[float] = []
    for line in completed.stdout.splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            memories_mib.append(float(raw))
        except ValueError:
            return {"gpu_count": None, "gpu_memory_gib": None}
    if not memories_mib:
        return {"gpu_count": None, "gpu_memory_gib": None}
    return {
        "gpu_count": len(memories_mib),
        "gpu_memory_gib": min(memories_mib) / 1024.0,
    }


def read_runtime_resource_snapshot() -> dict[str, Any]:
    """Read the effective cgroup limits, not the unconstrained host values."""

    memory_limit = _read_limit(Path("/sys/fs/cgroup/memory.max"))
    memory_current = _read_limit(Path("/sys/fs/cgroup/memory.current"))
    cpu_quota = _read_cpu_quota(Path("/sys/fs/cgroup/cpu.max"))
    events: dict[str, int] = {}
    raw_events = _read_text(Path("/sys/fs/cgroup/memory.events"))
    if raw_events:
        for line in raw_events.splitlines():
            key, _, value = line.partition(" ")
            if key and value.isdigit():
                events[key] = int(value)
    gpu_inventory = _read_gpu_inventory()
    return {
        "memory_limit_bytes": memory_limit,
        "memory_current_bytes": memory_current,
        "memory_limit_gib": (
            memory_limit / BYTES_PER_GIB if memory_limit is not None else None
        ),
        "memory_current_gib": (
            memory_current / BYTES_PER_GIB if memory_current is not None else None
        ),
        "cpu_quota_cores": cpu_quota,
        "os_cpu_count": os.cpu_count(),
        **gpu_inventory,
        "memory_events": events,
    }


def validate_runtime_resource_budget(
    config: Mapping[str, Any],
    *,
    snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail before Ray startup when the configured contract is unsafe."""

    snapshot = dict(snapshot or read_runtime_resource_snapshot())
    hardware = config["hardware"]
    ray_config = runtime_section(config, "ray")
    expected_cpu = int(hardware["expected_cpu_cores"])
    expected_ram_gib = float(hardware["expected_host_ram_gb"])
    required_gpu_count = int(hardware.get("total_physical_gpus", 0))
    minimum_gpu_memory_gib = float(hardware.get("gpu_memory_gb", 0))
    object_store_value = ray_config.get(
        "object_store_gb", hardware.get("ray_object_store_gb")
    )
    if object_store_value is None:
        raise RuntimeError("runtime.ray.object_store_gb must be configured")
    object_store_gib = float(object_store_value)
    reserve_value = ray_config.get(
        "memory_safety_reserve_gb", hardware.get("memory_safety_reserve_gb")
    )
    if reserve_value is None:
        raise RuntimeError(
            "runtime.ray.memory_safety_reserve_gb must be configured"
        )
    reserve_gib = float(reserve_value)
    actual_memory = snapshot.get("memory_limit_bytes")
    actual_cpu = snapshot.get("cpu_quota_cores")

    if actual_memory is None:
        raise RuntimeError("Cannot verify the cgroup memory.max resource contract")
    actual_ram_gib = float(actual_memory) / BYTES_PER_GIB
    if actual_ram_gib + 1.0e-9 < expected_ram_gib:
        raise RuntimeError(
            "Configured host RAM requirement does not match the cgroup limit "
            "and exceeds the available resource: "
            f"required={expected_ram_gib:g} GiB actual={actual_ram_gib:.3f} GiB"
        )
    if actual_cpu is None:
        raise RuntimeError("Cannot verify the cgroup CPU quota contract")
    if actual_cpu + 1.0e-9 < expected_cpu:
        raise RuntimeError(
            "Configured CPU count exceeds the cgroup quota: "
            f"configured={expected_cpu} quota={actual_cpu:.3f}"
        )
    if required_gpu_count > 0:
        actual_gpu_count = snapshot.get("gpu_count")
        if actual_gpu_count is None:
            raise RuntimeError(
                "Cannot verify the physical GPU count for the configured topology"
            )
        if int(actual_gpu_count) < required_gpu_count:
            raise RuntimeError(
                "Physical GPU count is below the configured topology: "
                f"required={required_gpu_count} actual={int(actual_gpu_count)}"
            )
        actual_gpu_memory_gib = snapshot.get("gpu_memory_gib")
        if minimum_gpu_memory_gib > 0 and actual_gpu_memory_gib is None:
            raise RuntimeError(
                "Cannot verify the minimum GPU memory for the configured topology"
            )
        if (
            minimum_gpu_memory_gib > 0
            and float(actual_gpu_memory_gib) + 1.0e-9 < minimum_gpu_memory_gib
        ):
            raise RuntimeError(
                "GPU memory is below the configured topology minimum: "
                f"required={minimum_gpu_memory_gib:g} GiB "
                f"actual={float(actual_gpu_memory_gib):.3f} GiB"
            )
    if object_store_gib <= 0 or reserve_gib <= 0:
        raise RuntimeError("Object-store and memory reserve must be positive")
    if object_store_gib + reserve_gib >= actual_ram_gib:
        raise RuntimeError(
            "Ray object store plus safety reserve consumes the memory limit: "
            f"object_store={object_store_gib:g} GiB reserve={reserve_gib:g} GiB "
            f"limit={actual_ram_gib:.3f} GiB"
        )
    current = snapshot.get("memory_current_bytes")
    current_gib = float(current) / BYTES_PER_GIB if current is not None else None
    return {
        "status": "PASS",
        "configured_expected_host_ram_gib": expected_ram_gib,
        "configured_expected_cpu_cores": expected_cpu,
        "configured_ray_object_store_gib": object_store_gib,
        "configured_memory_safety_reserve_gib": reserve_gib,
        "cgroup_memory_limit_gib": actual_ram_gib,
        "cgroup_memory_current_gib": current_gib,
        "cgroup_cpu_quota_cores": float(actual_cpu),
        "physical_gpu_count": (
            int(snapshot["gpu_count"])
            if snapshot.get("gpu_count") is not None
            else None
        ),
        "minimum_gpu_memory_gib": minimum_gpu_memory_gib,
        "smallest_gpu_memory_gib": (
            float(snapshot["gpu_memory_gib"])
            if snapshot.get("gpu_memory_gib") is not None
            else None
        ),
        "memory_events": dict(snapshot.get("memory_events", {})),
        "headroom_after_object_store_and_reserve_gib": (
            actual_ram_gib - object_store_gib - reserve_gib
        ),
    }


def validate_reference_resource_budget(
    config: Mapping[str, Any],
    *,
    snapshot: Mapping[str, Any] | None = None,
    ram_tolerance_gib: float = 1.0,
) -> dict[str, Any]:
    """Apply the exact official 4x48GB host qualification.

    This check is intentionally opt-in.  Portable runs use
    :func:`validate_runtime_resource_budget`, which only requires that the
    machine meet the declared minimum resources.
    """

    result = validate_runtime_resource_budget(config, snapshot=snapshot)
    expected_cpu = int(config["hardware"]["expected_cpu_cores"])
    expected_ram = float(config["hardware"]["expected_host_ram_gb"])
    actual_cpu = float(result["cgroup_cpu_quota_cores"])
    actual_ram = float(result["cgroup_memory_limit_gib"])
    if expected_cpu != 48:
        raise RuntimeError(
            "Official reference qualification requires expected_cpu_cores=48, "
            f"got {expected_cpu}"
        )
    if not math.isclose(expected_ram, 360.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError(
            "Official reference qualification requires expected_host_ram_gb=360, "
            f"got {expected_ram:g}"
        )
    if not math.isclose(actual_cpu, 48.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError(
            "Official reference qualification requires a 48-core cgroup quota, "
            f"got {actual_cpu:.3f}"
        )
    if not math.isclose(actual_ram, 360.0, rel_tol=0.0, abs_tol=ram_tolerance_gib):
        raise RuntimeError(
            "Official reference qualification requires a 360 GiB cgroup limit, "
            f"got {actual_ram:.3f} GiB"
        )
    result["reference_qualification"] = "official_4x48gb_v1"
    return result


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        int(item.stat().st_size)
        for item in path.rglob("*")
        if item.is_file()
    )


def _max_file_size(path: Path, *, suffix: str | None = None) -> int:
    if not path.exists():
        return 0
    sizes = [
        int(item.stat().st_size)
        for item in path.rglob("*")
        if item.is_file() and (suffix is None or item.name.endswith(suffix))
    ]
    return max(sizes, default=0)


def validate_checkpoint_runtime_budget(
    runtime_root: str | Path,
    *,
    snapshot: Mapping[str, Any] | None = None,
    source_checkpoint: str | Path | None = None,
    include_checkpoint_write: bool = True,
    memory_headroom_gib: float = CHECKPOINT_MEMORY_HEADROOM_GIB,
    min_free_disk_gib: float = CHECKPOINT_MIN_FREE_DISK_GIB,
    disk_safety_gib: float = CHECKPOINT_DISK_SAFETY_GIB,
) -> dict[str, Any]:
    """Fail closed before a cadence update when checkpoint I/O is unsafe.

    The guard is intentionally independent of optimizer and algorithm state.
    It accounts for the observed checkpoint/model artifact sizes plus a
    margin, and requires cgroup memory headroom before the update is allowed
    to begin.  It does not raise the cgroup limit or reserve extra workers.
    """

    root = Path(runtime_root).resolve()
    snapshot = dict(snapshot or read_runtime_resource_snapshot())
    limit = snapshot.get("memory_limit_bytes")
    current = snapshot.get("memory_current_bytes")
    if limit is None or current is None:
        raise RuntimeError("Cannot verify cgroup memory headroom before checkpoint")
    limit = int(limit)
    current = int(current)
    memory_limit_gib = limit / BYTES_PER_GIB
    memory_current_gib = current / BYTES_PER_GIB
    memory_budget_gib = memory_limit_gib - float(memory_headroom_gib)
    if current > limit - int(float(memory_headroom_gib) * BYTES_PER_GIB):
        raise RuntimeError(
            "Checkpoint preflight blocked before optimizer: insufficient cgroup "
            f"headroom current={memory_current_gib:.2f}GiB limit="
            f"{memory_limit_gib:.2f}GiB required_headroom={memory_headroom_gib:.2f}GiB"
        )

    resume_root = root / "checkpoints" / "resume"
    model_root = root / "checkpoints" / "models"
    observed_resume_bytes = max(
        (
            _directory_size(child)
            for child in resume_root.iterdir()
            if child.is_dir() and not child.name.endswith(".tmp")
        ),
        default=0,
    ) if resume_root.exists() else 0
    if source_checkpoint is not None:
        observed_resume_bytes = max(
            observed_resume_bytes,
            _directory_size(Path(source_checkpoint).resolve()),
        )
    observed_model_bytes = _max_file_size(model_root, suffix=".safetensors")
    # The fallback covers a fresh recovery run before its first local export.
    estimated_resume_bytes = (
        max(observed_resume_bytes, 40 * BYTES_PER_GIB)
        if include_checkpoint_write
        else 0
    )
    estimated_model_bytes = max(observed_model_bytes, 8 * BYTES_PER_GIB)
    estimated_write_bytes = int(
        CHECKPOINT_SIZE_MARGIN * (estimated_resume_bytes + estimated_model_bytes)
        + float(disk_safety_gib) * BYTES_PER_GIB
    )
    estimated_resume_gib = estimated_resume_bytes / BYTES_PER_GIB
    estimated_model_gib = estimated_model_bytes / BYTES_PER_GIB
    projected_checkpoint_peak_gib = (
        memory_current_gib
        + estimated_resume_gib
        + estimated_model_gib
        + CHECKPOINT_MEMORY_IO_SAFETY_GIB
    )
    if projected_checkpoint_peak_gib > memory_limit_gib:
        raise RuntimeError(
            "Checkpoint preflight blocked before optimizer: projected checkpoint "
            f"peak={projected_checkpoint_peak_gib:.2f}GiB exceeds "
            f"cgroup limit={memory_limit_gib:.2f}GiB"
        )
    disk = shutil.disk_usage(root)
    free_gib = disk.free / BYTES_PER_GIB
    required_free_gib = max(
        (float(min_free_disk_gib) if include_checkpoint_write else 16.0),
        estimated_write_bytes / BYTES_PER_GIB,
    )
    if disk.free < estimated_write_bytes or disk.free < int(
        float(min_free_disk_gib) * BYTES_PER_GIB
    ):
        raise RuntimeError(
            "Checkpoint preflight blocked before optimizer: insufficient disk "
            f"free={free_gib:.2f}GiB required={required_free_gib:.2f}GiB"
        )
    return {
        "status": "PASS",
        "runtime_root": str(root),
        "memory_limit_gib": memory_limit_gib,
        "memory_current_gib": memory_current_gib,
        "memory_headroom_required_gib": float(memory_headroom_gib),
        "memory_budget_before_checkpoint_gib": memory_budget_gib,
        "estimated_resume_write_gib": estimated_resume_gib,
        "estimated_model_export_gib": estimated_model_gib,
        "projected_checkpoint_peak_gib": projected_checkpoint_peak_gib,
        "checkpoint_memory_io_safety_gib": CHECKPOINT_MEMORY_IO_SAFETY_GIB,
        "disk_free_gib": free_gib,
        "disk_required_free_gib": required_free_gib,
        "observed_resume_max_bytes": int(observed_resume_bytes),
        "observed_model_max_bytes": int(observed_model_bytes),
        "estimated_checkpoint_write_bytes": int(estimated_write_bytes),
        "memory_events": dict(snapshot.get("memory_events", {})),
    }
