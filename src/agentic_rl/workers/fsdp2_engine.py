from __future__ import annotations

from typing import Any, Iterable


def apply_qwen_fsdp2(
    model: Any,
    *,
    mesh: Any,
    reshard_after_forward: bool,
    mixed_precision_dtype: Any,
) -> Any:
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("Expected a Qwen-compatible model.model.layers structure")
    policy = MixedPrecisionPolicy(
        param_dtype=mixed_precision_dtype,
        reduce_dtype=mixed_precision_dtype,
        output_dtype=mixed_precision_dtype,
    )
    for layer in layers:
        fully_shard(
            layer,
            mesh=mesh,
            reshard_after_forward=bool(reshard_after_forward),
            mp_policy=policy,
        )
    fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=bool(reshard_after_forward),
        mp_policy=policy,
    )
    return model


def assert_fsdp2_world(*, world_size: int, expected_world_size: int = 4) -> None:
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError("FSDP2 requires an initialized process group")
    if int(world_size) != expected_world_size:
        raise RuntimeError(
            f"Configured world size {world_size} != required {expected_world_size}"
        )
    if dist.get_world_size() != expected_world_size:
        raise RuntimeError(
            f"Process-group world size {dist.get_world_size()} != {expected_world_size}"
        )
