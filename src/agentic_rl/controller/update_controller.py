from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from agentic_rl.config import DEFAULT_CONFIG, load_config
from agentic_rl.selection.candidate_pool import (
    CandidatePool,
    PromptGroup,
    SelectionDecision,
)
from agentic_rl.selection.candidate_pool import DUAL_CHANNEL_SELECTION_SIGNAL
from agentic_rl.workers.resource_plan import build_resource_plan
from agentic_rl.workers.ray_actors import probe_runtime_compatibility

from .attempt_state import SnapshotVersions, TrainingState
from .transaction import StrictUpdateTransaction


class AttemptRuntimeAdapter(Protocol):
    def freeze_rollout_boundary(self, successful_update_step: int) -> SnapshotVersions:
        ...

    def collect_scored_prompt_groups(
        self,
        prompt_count: int,
        *,
        snapshot_step: int,
    ) -> Sequence[PromptGroup]:
        ...

    def selected_microbatches(
        self,
        groups: Sequence[PromptGroup],
    ) -> Sequence[Any]:
        ...

    def prepare_selected_stop_branches(
        self,
        groups: Sequence[PromptGroup],
    ) -> None:
        ...

    def finalize_selected_exact_ig(
        self,
        groups: Sequence[PromptGroup],
    ) -> Sequence[PromptGroup]:
        ...

    def zero_grad(self) -> None:
        ...

    def actor_parameter_checksum(self) -> str:
        ...

    def backward_microbatch(self, microbatch: Any) -> None:
        ...

    def clip_gradients(self, max_grad_norm: float) -> float:
        ...

    def optimizer_step(self) -> None:
        ...

    def scheduler_step(self) -> None:
        ...

    def commit_successful_update(self, state: TrainingState) -> None:
        """Commit the update event; persist a checkpoint only when policy allows."""
        ...

    def rollback_pre_step_attempt(self) -> None:
        ...

    def data_cursor(self) -> int:
        ...

    def rng_state(self) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class AttemptResult:
    state: TrainingState
    selection: SelectionDecision
    optimizer_committed: bool


class StrictAttemptController:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def _select(
        self,
        pool: CandidatePool,
        state: TrainingState,
    ) -> SelectionDecision:
        selection = self.config["selection"]
        return pool.select(
            ig_state=state.ig_channel,
            outcome_state=state.outcome_channel,
            top_p_mass=float(selection["top_p_mass"]),
            alpha_ig=float(selection["alpha_ig"]),
            alpha_outcome=float(selection["alpha_outcome"]),
            noise_floor_ig=float(selection["noise_floor_ig"]),
            noise_floor_outcome=float(selection["noise_floor_outcome"]),
            minimum_positive_prompts=int(selection["minimum_positive_prompts"]),
            health_threshold_ratio=float(selection["health_threshold_ratio"]),
            minimum_selected_prompts=int(selection["minimum_selected_prompts"]),
            maximum_selected_prompts=int(selection["maximum_selected_prompts"]),
            allow_provisional_scale=state.successful_update_step == 0,
            signal_mode=str(
                selection.get("signal", DUAL_CHANNEL_SELECTION_SIGNAL)
            ),
            selection_mode=str(selection["mode"]),
        )

    def run_attempt(
        self,
        state: TrainingState,
        runtime: AttemptRuntimeAdapter,
    ) -> AttemptResult:
        rollout = self.config["rollout"]
        selection_config = self.config["selection"]
        initial_count = int(rollout["candidate_prompts_initial"])
        refill_prompts = int(rollout["refill_prompts"])
        maximum_count = int(rollout["candidate_prompts_max"])
        if (
            refill_prompts <= 0
            or maximum_count < initial_count
            or (maximum_count - initial_count) % refill_prompts != 0
        ):
            raise RuntimeError(
                "candidate_prompts_max must be reachable from "
                "candidate_prompts_initial by whole refill_prompts increments"
            )
        allowed_candidate_counts = tuple(
            range(initial_count, maximum_count + 1, refill_prompts)
        )
        transaction = StrictUpdateTransaction(
            state,
            allowed_candidate_counts=allowed_candidate_counts,
            minimum_selected_prompts=int(
                selection_config["minimum_selected_prompts"]
            ),
            maximum_selected_prompts=int(
                selection_config["maximum_selected_prompts"]
            ),
        )
        start_metrics = getattr(runtime, "start_attempt_metrics", None)
        if callable(start_metrics):
            start_metrics(state)
        versions = runtime.freeze_rollout_boundary(state.successful_update_step)
        snapshot_metrics = getattr(runtime, "record_snapshot_metrics", None)
        if callable(snapshot_metrics):
            snapshot_metrics(versions)
        transaction.freeze_snapshot(versions)
        pool = CandidatePool(
            group_size=int(rollout["group_size"]),
            maximum_prompts=int(rollout["candidate_prompts_max"]),
        )
        try:
            checkpoint_preflight = getattr(
                runtime,
                "checkpoint_resource_preflight",
                None,
            )
            if callable(checkpoint_preflight):
                checkpoint_preflight(
                    next_successful_update_step=(
                        int(state.successful_update_step) + 1
                    ),
                    phase="before_attempt",
                )
            wave_size = int(rollout["prompt_wave_size"])
            if initial_count % wave_size != 0:
                raise RuntimeError("Initial candidate count must be whole prompt waves")
            collect_initial = getattr(
                runtime,
                "collect_initial_scored_prompt_groups",
                None,
            )
            if callable(collect_initial):
                pool.add(
                    collect_initial(
                        initial_count,
                        wave_size=wave_size,
                        snapshot_step=versions.actor,
                    )
                )
            else:
                for _ in range(initial_count // wave_size):
                    pool.add(
                        runtime.collect_scored_prompt_groups(
                            wave_size,
                            snapshot_step=versions.actor,
                        )
                    )
            selection_started = time.perf_counter()
            decision = self._select(pool, state)
            selection_rounds = [
                {
                    "pool_size": len(pool),
                    "selected_count": int(decision.selected_count),
                    "requires_refill": bool(decision.requires_refill),
                    "skip_update": bool(decision.skip_update),
                }
            ]
            refill_count = 0
            while decision.requires_refill:
                if len(pool) + refill_prompts > maximum_count:
                    raise RuntimeError(
                        "Selection requested a refill beyond candidate_prompts_max"
                    )
                pool.add(
                    runtime.collect_scored_prompt_groups(
                        refill_prompts,
                        snapshot_step=versions.actor,
                    )
                )
                refill_count += 1
                decision = self._select(pool, state)
                selection_rounds.append(
                    {
                        "pool_size": len(pool),
                        "selected_count": int(decision.selected_count),
                        "requires_refill": bool(decision.requires_refill),
                        "skip_update": bool(decision.skip_update),
                    }
                )
            selection_seconds = time.perf_counter() - selection_started
            selection_metrics = getattr(runtime, "record_selection_metrics", None)
            if callable(selection_metrics):
                selection_metrics(
                    state,
                    pool.groups(),
                    decision,
                    refill_count=refill_count,
                    selection_seconds=selection_seconds,
                    selection_rounds=selection_rounds,
                )

            transaction.complete_rollout(
                candidate_prompt_count=len(pool),
                refill_count=refill_count,
            )
            transaction.complete_scoring()
            if decision.skip_update:
                next_state = transaction.skip(
                    "selected_below_minimum_after_refill",
                    data_cursor=runtime.data_cursor(),
                )
                skipped_metrics = getattr(runtime, "record_skipped_attempt", None)
                if callable(skipped_metrics):
                    skipped_metrics(
                        next_state,
                        reason="selected_below_minimum_after_refill",
                    )
                return AttemptResult(next_state, decision, False)
            transaction.select(decision.selected_count)
            selected_groups = pool.selected_groups(decision)
            checksum_before_selected_scoring = runtime.actor_parameter_checksum()
            finalize_selected_exact_ig = getattr(
                runtime,
                "finalize_selected_exact_ig",
                None,
            )
            if callable(finalize_selected_exact_ig):
                selected_groups = tuple(
                    finalize_selected_exact_ig(selected_groups)
                )
            if runtime.actor_parameter_checksum() != checksum_before_selected_scoring:
                raise RuntimeError(
                    "Actor parameters changed during selected-only Exact-IG"
                )
            checksum_before_stop = runtime.actor_parameter_checksum()
            prepare_stop_branches = getattr(
                runtime,
                "prepare_selected_stop_branches",
                None,
            )
            search_task_mode = str(
                self.config["advantage"].get(
                    "search_task_mode",
                    "normalized_outcome",
                )
            )
            if search_task_mode in {
                "stop_continue_consensus",
                "sufficiency_novelty_local_ig",
                "sufficiency_novelty_cumulative_ig_probe_routed_outcome",
                "sufficiency_novelty_cumulative_ig_probe_routed_outcome_role_localized_gate",
            }:
                if not callable(prepare_stop_branches):
                    raise RuntimeError(
                        "Search probe mode requires a bound detached probe runtime"
                    )
                prepare_stop_branches(selected_groups)
            elif callable(prepare_stop_branches):
                prepare_stop_branches(selected_groups)
            if runtime.actor_parameter_checksum() != checksum_before_stop:
                raise RuntimeError(
                    "Actor parameters changed during detached Stop branching"
                )
            selected_microbatches = tuple(
                runtime.selected_microbatches(selected_groups)
            )
            if runtime.actor_parameter_checksum() != checksum_before_stop:
                raise RuntimeError(
                    "Actor parameters changed while preparing advantages/"
                    "old log-probabilities"
                )
            runtime.zero_grad()
            transaction.record_zero_grad()
            for microbatch in selected_microbatches:
                transaction.record_backward_microbatch(
                    runtime.actor_parameter_checksum()
                )
                runtime.backward_microbatch(microbatch)
            transaction.complete_backward(runtime.actor_parameter_checksum())
            grad_norm = runtime.clip_gradients(
                float(self.config["policy"]["max_grad_norm"])
            )
            gradient_metrics = getattr(runtime, "record_gradient_norm", None)
            if callable(gradient_metrics):
                gradient_metrics(grad_norm)
            # Re-check at the last reversible boundary.  If checkpoint I/O
            # would be unsafe after the learner has built gradients, the
            # attempt can still zero gradients and abort without stepping.
            if callable(checkpoint_preflight):
                checkpoint_preflight(
                    next_successful_update_step=(
                        int(state.successful_update_step) + 1
                    ),
                    phase="before_optimizer",
                )
            transaction.begin_optimizer_step()
            runtime.optimizer_step()
            transaction.record_optimizer_step()
            transaction.begin_scheduler_step()
            runtime.scheduler_step()
            transaction.record_scheduler_step()
            next_state = transaction.propose_commit(
                ig_stats=decision.ig_stats,
                outcome_stats=decision.outcome_stats,
                ema_half_life=float(selection_config["scale_ema_half_life"]),
                health_reference_valid_updates=int(
                    selection_config["health_reference_valid_updates"]
                ),
                data_cursor=runtime.data_cursor(),
                rng_state=runtime.rng_state(),
            )
            runtime.commit_successful_update(next_state)
            transaction.record_commit_success()
            return AttemptResult(next_state, decision, True)
        except BaseException as exc:
            if transaction.can_rollback_pre_step:
                runtime.rollback_pre_step_attempt()
            transaction.abort(
                str(exc),
                data_cursor=runtime.data_cursor(),
            )
            failed_metrics = getattr(runtime, "record_failed_attempt", None)
            if callable(failed_metrics):
                failed_metrics(exc)
            raise


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "project": config["project"]["name"],
        "resource_plan": build_resource_plan(config).as_dict(),
        "runtime_compatibility": probe_runtime_compatibility(config),
    }


def _load_adapter_factory(path: str) -> Any:
    module_name, separator, attribute = str(path).partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("adapter_factory must have form package.module:function")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    if not callable(factory):
        raise TypeError(f"Configured adapter factory is not callable: {path}")
    return factory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict Agentic RL controller entry point"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Enter the runtime adapter after all fail-closed preflight checks.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Resolve configuration and inspect backend compatibility only.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    result = preflight(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    blockers = result["runtime_compatibility"]["blockers"]
    if blockers:
        raise SystemExit(
            "Runtime preflight blocked:\n- " + "\n- ".join(blockers)
        )
    if args.preflight_only or not args.execute:
        return
    factory = _load_adapter_factory(config["runtime"]["adapter_factory"])
    runtime = factory(config)
    run = getattr(runtime, "run", None)
    if not callable(run):
        raise TypeError("Runtime adapter must expose run()")
    run()


if __name__ == "__main__":
    main()
