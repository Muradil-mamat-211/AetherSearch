from __future__ import annotations

from pathlib import Path
from typing import Any


def save_fsdp2_training_state(
    destination: str | Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler_state: dict,
) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
    )

    destination_path = Path(destination)
    actor_path = destination_path / "actor"
    optimizer_path = destination_path / "optimizer"
    scheduler_path = destination_path / "scheduler"
    actor_path.mkdir(exist_ok=True)
    optimizer_path.mkdir(exist_ok=True)
    scheduler_path.mkdir(exist_ok=True)
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    dcp.save(
        {"model": get_model_state_dict(model, options=options)},
        checkpoint_id=str(actor_path),
    )
    dcp.save(
        {
            "optimizer": get_optimizer_state_dict(
                model,
                optimizer,
                options=options,
            )
        },
        checkpoint_id=str(optimizer_path),
    )
    dcp.save(
        {"scheduler": scheduler_state},
        checkpoint_id=str(scheduler_path),
    )


def load_fsdp2_training_state(
    source: str | Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
    )

    source_path = Path(source)
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_state = {"model": get_model_state_dict(model, options=options)}
    dcp.load(model_state, checkpoint_id=str(source_path / "actor"))
    set_model_state_dict(model, model_state["model"], options=options)
    optimizer_state = {
        "optimizer": get_optimizer_state_dict(model, optimizer, options=options)
    }
    dcp.load(optimizer_state, checkpoint_id=str(source_path / "optimizer"))
    set_optimizer_state_dict(
        model,
        optimizer,
        optimizer_state["optimizer"],
        options=options,
    )
    scheduler_state = {"scheduler": scheduler.state_dict()}
    dcp.load(scheduler_state, checkpoint_id=str(source_path / "scheduler"))
    scheduler.load_state_dict(scheduler_state["scheduler"])
