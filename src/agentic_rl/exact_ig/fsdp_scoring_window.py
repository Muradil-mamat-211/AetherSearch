from __future__ import annotations

import weakref
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping


class FSDPReshardRestoreError(RuntimeError):
    pass


@dataclass
class FSDPScoringWindowReport:
    module_count: int
    before_states: tuple[bool, ...]
    window_states: tuple[bool, ...]
    after_states: tuple[bool, ...] | None = None
    restore_succeeded: bool = False
    exit_allocated_bytes: int | None = None
    exit_reserved_bytes: int | None = None


def fsdp2_modules(model: Any) -> tuple[Any, ...]:
    modules = tuple(
        module
        for module in model.modules()
        if callable(getattr(module, "set_reshard_after_forward", None))
    )
    if not modules:
        raise RuntimeError(
            "No FSDP2 modules expose set_reshard_after_forward; "
            "refusing to pretend this is an FSDP2 scoring window"
        )
    return modules


class FSDPReshardStateRegistry:
    """Tracks setter-only FSDP2 state without guessing a module's prior value."""

    def __init__(self) -> None:
        self._states: weakref.WeakKeyDictionary[Any, bool] = (
            weakref.WeakKeyDictionary()
        )

    def register_module(self, module: Any, value: bool) -> None:
        if not callable(getattr(module, "set_reshard_after_forward", None)):
            raise TypeError("Registered object is not an FSDP2 module")
        self._states[module] = bool(value)

    def register_model(
        self,
        model: Any,
        states: bool | Mapping[Any, bool],
    ) -> int:
        modules = fsdp2_modules(model)
        if isinstance(states, Mapping):
            missing = [module for module in modules if module not in states]
            if missing:
                raise RuntimeError(
                    "Explicit FSDP2 state mapping does not cover every module"
                )
            for module in modules:
                self.register_module(module, bool(states[module]))
        else:
            for module in modules:
                self.register_module(module, bool(states))
        return len(modules)

    def state_for(self, module: Any) -> bool:
        if module not in self._states:
            raise RuntimeError(
                "FSDP2 reshard state is unregistered; prior state cannot be guessed"
            )
        return bool(self._states[module])

    def snapshot(self, model: Any) -> tuple[tuple[Any, bool], ...]:
        return tuple(
            (module, self.state_for(module))
            for module in fsdp2_modules(model)
        )

    def set_module(self, module: Any, value: bool) -> None:
        setter = getattr(module, "set_reshard_after_forward", None)
        if not callable(setter):
            raise TypeError("Object no longer exposes the FSDP2 reshard setter")
        parameters = inspect.signature(setter).parameters
        if "recurse" in parameters:
            setter(bool(value), recurse=False)
        else:
            setter(bool(value))
        self._states[module] = bool(value)


def configure_reshard_after_forward(
    model: Any,
    value: bool,
    *,
    registry: FSDPReshardStateRegistry,
) -> int:
    modules = fsdp2_modules(model)
    for module in modules:
        registry.state_for(module)
    changed: list[tuple[Any, bool]] = []
    try:
        for module in modules:
            previous = registry.state_for(module)
            registry.set_module(module, bool(value))
            changed.append((module, previous))
    except BaseException:
        failures: list[str] = []
        for module, previous in reversed(changed):
            try:
                registry.set_module(module, previous)
            except BaseException as restore_error:
                failures.append(repr(restore_error))
        if failures:
            raise FSDPReshardRestoreError(
                "FSDP2 configuration failed and partial-state restoration also "
                f"failed: {failures}"
            )
        raise
    return len(modules)


def _restore_snapshot(
    snapshot: tuple[tuple[Any, bool], ...],
    registry: FSDPReshardStateRegistry,
) -> None:
    failures: list[str] = []
    for module, previous in reversed(snapshot):
        try:
            registry.set_module(module, previous)
        except BaseException as error:
            failures.append(f"{type(error).__name__}: {error}")
    if failures:
        raise FSDPReshardRestoreError(
            "Failed to restore one or more FSDP2 reshard states: "
            + "; ".join(failures)
        )


@contextmanager
def exact_ig_scoring_window(
    model: Any,
    *,
    registry: FSDPReshardStateRegistry,
    reshard_after_forward: bool = False,
    synchronize: Callable[[], None] | None = None,
    memory_snapshot: Callable[[], tuple[int, int]] | None = None,
) -> Iterator[FSDPScoringWindowReport]:
    snapshot = registry.snapshot(model)
    report = FSDPScoringWindowReport(
        module_count=len(snapshot),
        before_states=tuple(state for _, state in snapshot),
        window_states=tuple(bool(reshard_after_forward) for _ in snapshot),
    )
    try:
        configure_reshard_after_forward(
            model,
            bool(reshard_after_forward),
            registry=registry,
        )
    except BaseException:
        report.after_states = tuple(
            registry.state_for(module) for module, _ in snapshot
        )
        raise

    body_error: BaseException | None = None
    try:
        yield report
    except BaseException as error:
        body_error = error
        raise
    finally:
        try:
            _restore_snapshot(snapshot, registry)
            if synchronize is not None:
                synchronize()
            report.after_states = tuple(
                registry.state_for(module) for module, _ in snapshot
            )
            report.restore_succeeded = (
                report.after_states == report.before_states
            )
            if not report.restore_succeeded:
                raise FSDPReshardRestoreError(
                    "FSDP2 state registry differs after scoring-window restore"
                )
            if memory_snapshot is not None:
                allocated, reserved = memory_snapshot()
                report.exit_allocated_bytes = int(allocated)
                report.exit_reserved_bytes = int(reserved)
        except BaseException as restore_error:
            report.restore_succeeded = False
            if body_error is not None:
                raise FSDPReshardRestoreError(
                    "Exact-IG body failed and FSDP2 state restoration also failed"
                ) from restore_error
            raise
