from __future__ import annotations

import time
from typing import Any, Optional

import ray
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    _qwen2_5_vl_dedup_image_tokens,
    vLLMHttpServerBase,
    vLLMReplica,
)


def _build_stop_pair_sampling_params(
    params: dict[str, Any],
    *,
    max_tokens: int,
) -> SamplingParams:
    """Build one parallel-sampling request whose final output contains both samples."""

    stop_params = dict(params)
    stop_params["n"] = 2
    stop_params["stop"] = ["</answer>"]
    stop_params["include_stop_str_in_output"] = True
    stop_params["logprobs"] = None
    stop_params["prompt_logprobs"] = None
    stop_params["output_kind"] = RequestOutputKind.FINAL_ONLY
    return SamplingParams(max_tokens=max_tokens, **stop_params)


def _build_sufficiency_probe_sampling_params(
    params: dict[str, Any],
    *,
    max_tokens: int,
) -> SamplingParams:
    probe_params = dict(params)
    if probe_params.pop("do_sample", None) is not False:
        raise ValueError("Sufficiency probe requires do_sample=false")
    if int(probe_params.pop("n", 0)) != 1:
        raise ValueError("Sufficiency probe requires n=1")
    if float(probe_params.get("temperature", -1.0)) != 0.0:
        raise ValueError("Sufficiency probe requires temperature=0")
    if float(probe_params.get("top_p", -1.0)) != 1.0:
        raise ValueError("Sufficiency probe requires top_p=1")
    if int(probe_params.get("top_k", 0)) != -1:
        raise ValueError("Sufficiency probe requires top_k=-1")
    if float(probe_params.get("min_p", -1.0)) != 0.0:
        raise ValueError("Sufficiency probe requires min_p=0")
    probe_params["n"] = 1
    probe_params["stop"] = ["</answer>"]
    probe_params["include_stop_str_in_output"] = True
    probe_params["logprobs"] = None
    probe_params["prompt_logprobs"] = None
    probe_params["output_kind"] = RequestOutputKind.FINAL_ONLY
    return SamplingParams(max_tokens=max_tokens, **probe_params)


class CappedVLLMHttpServerBase(vLLMHttpServerBase):
    """veRL vLLM server with a request-local per-turn generation cap.

    veRL 0.6.1's stock server always expands ``max_tokens`` to all remaining
    context. Agentic Search requires a hard 500-token cap for each assistant
    action, so this override consumes the project-only
    ``project_max_tokens`` request field. No mutable server-wide length is
    changed, which keeps concurrent requests race-free.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._project_weight_version = -1
        self._project_weight_checksum = ""
        self._project_inflight_requests = 0
        self._project_max_inflight_requests = 0
        self._project_request_count = 0
        self._project_generated_tokens = 0
        self._project_generation_seconds = 0.0
        self._project_generation_errors = 0

    def set_project_weight_version(
        self,
        snapshot_step: int,
        source_checksum: str,
    ) -> dict[str, Any]:
        self._project_weight_version = int(snapshot_step)
        self._project_weight_checksum = str(source_checksum)
        return self.get_project_weight_version()

    def get_project_weight_version(self) -> dict[str, Any]:
        return {
            "replica_rank": int(self.replica_rank),
            "snapshot_step": int(self._project_weight_version),
            "source_checksum": self._project_weight_checksum,
        }

    def get_project_runtime_metrics(self) -> dict[str, Any]:
        return {
            "replica_rank": int(self.replica_rank),
            "inflight_requests": int(self._project_inflight_requests),
            "max_inflight_requests": int(self._project_max_inflight_requests),
            "request_count": int(self._project_request_count),
            "generated_tokens": int(self._project_generated_tokens),
            "generation_seconds": float(self._project_generation_seconds),
            "generation_errors": int(self._project_generation_errors),
            "snapshot_step": int(self._project_weight_version),
            "source_checksum": self._project_weight_checksum,
        }

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        params = dict(sampling_params)
        requested_cap = int(params.pop("project_max_tokens"))
        if requested_cap <= 0:
            raise ValueError("project_max_tokens must be positive")
        remaining = int(self.config.max_model_len) - len(prompt_ids)
        if remaining <= 0:
            raise RuntimeError("vLLM request has no remaining context")
        max_tokens = min(requested_cap, remaining)
        params["logprobs"] = 0 if params.pop("logprobs", False) else None
        params.setdefault(
            "repetition_penalty",
            self.config.get("repetition_penalty", 1.0),
        )
        vllm_params = SamplingParams(max_tokens=max_tokens, **params)
        prompt_ids = _qwen2_5_vl_dedup_image_tokens(
            prompt_ids,
            self.model_config.processor,
        )
        prompt = TokensPrompt(
            prompt_token_ids=prompt_ids,
            multi_modal_data={"image": image_data} if image_data else None,
        )

        lora_request = None
        if self.model_config.lora_rank > 0:
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME,
                    lora_int_id=VLLM_LORA_INT_ID,
                    lora_path=VLLM_LORA_PATH,
                )
        started = time.perf_counter()
        self._project_request_count += 1
        self._project_inflight_requests += 1
        self._project_max_inflight_requests = max(
            self._project_max_inflight_requests,
            self._project_inflight_requests,
        )
        try:
            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=vllm_params,
                request_id=request_id,
                lora_request=lora_request,
            )
            final_result: Optional[RequestOutput] = None
            async for output in generator:
                final_result = output
            if final_result is None:
                raise RuntimeError("vLLM returned no result")
        except BaseException:
            self._project_generation_errors += 1
            raise
        finally:
            self._project_inflight_requests -= 1
            self._project_generation_seconds += time.perf_counter() - started
        token_ids = final_result.outputs[0].token_ids
        self._project_generated_tokens += len(token_ids)
        log_probs = None
        if vllm_params.logprobs is not None:
            log_probs = [
                values[token_ids[index]].logprob
                for index, values in enumerate(final_result.outputs[0].logprobs)
            ]
        return TokenOutput(token_ids=token_ids, log_probs=log_probs)

    async def generate_stop_pair(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        """Generate exactly two detached Stop answers in one vLLM request."""

        params = dict(sampling_params)
        requested_cap = int(params.pop("project_max_tokens"))
        if requested_cap <= 0:
            raise ValueError("Stop answer max tokens must be positive")
        remaining = int(self.config.max_model_len) - len(prompt_ids)
        if requested_cap > remaining:
            raise RuntimeError(
                "Stop branch would require context truncation; request rejected"
            )
        if params.pop("logprobs", None) not in {None, False}:
            raise ValueError("Stop branches must not request output logprobs")
        if params.pop("prompt_logprobs", None) not in {None, False}:
            raise ValueError("Stop branches must not request prompt logprobs")
        params.setdefault(
            "repetition_penalty",
            self.config.get("repetition_penalty", 1.0),
        )
        vllm_params = _build_stop_pair_sampling_params(
            params,
            max_tokens=requested_cap,
        )
        prompt = TokensPrompt(prompt_token_ids=[int(value) for value in prompt_ids])
        lora_request = None
        if self.model_config.lora_rank > 0:
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if not lora_loaded:
                raise RuntimeError("Stop branch requires the rollout LoRA snapshot")
            lora_request = LoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
            )

        started = time.perf_counter()
        self._project_request_count += 1
        self._project_inflight_requests += 1
        self._project_max_inflight_requests = max(
            self._project_max_inflight_requests,
            self._project_inflight_requests,
        )
        try:
            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=vllm_params,
                request_id=str(request_id),
                lora_request=lora_request,
            )
            final_result: Optional[RequestOutput] = None
            async for output in generator:
                final_result = output
            if final_result is None:
                raise RuntimeError("vLLM returned no Stop result")
        except BaseException:
            self._project_generation_errors += 1
            raise
        finally:
            self._project_inflight_requests -= 1
            self._project_generation_seconds += time.perf_counter() - started

        if len(final_result.outputs) != 2:
            raise RuntimeError(
                f"Stop request returned {len(final_result.outputs)} completions"
            )
        completions = []
        generated_token_count = 0
        for sample_index, output in enumerate(final_result.outputs):
            token_ids = [int(value) for value in output.token_ids]
            generated_token_count += len(token_ids)
            completions.append(
                {
                    "sample_index": int(sample_index),
                    "text": str(output.text),
                    "token_ids": token_ids,
                    "finish_reason": (
                        None
                        if output.finish_reason is None
                        else str(output.finish_reason)
                    ),
                    "stop_reason": (
                        None
                        if output.stop_reason is None
                        else str(output.stop_reason)
                    ),
                }
            )
        self._project_generated_tokens += generated_token_count
        metrics = getattr(final_result, "metrics", None)
        cached_prompt_tokens = int(
            getattr(metrics, "num_cached_tokens", 0) or 0
        )
        return {
            "request_id": str(request_id),
            "replica_rank": int(self.replica_rank),
            "snapshot_step": int(self._project_weight_version),
            "source_checksum": self._project_weight_checksum,
            "prompt_tokens": len(prompt_ids),
            "cached_prompt_tokens": cached_prompt_tokens,
            "decode_tokens": generated_token_count,
            "generation_seconds": time.perf_counter() - started,
            "automatic_prefix_caching": bool(
                self.config.get("enable_prefix_caching", False)
            ),
            "completion_count": len(completions),
            "completions": completions,
        }

    async def generate_sufficiency_probe(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        """Generate one detached deterministic Answer-now completion."""

        params = dict(sampling_params)
        requested_cap = int(params.pop("project_max_tokens"))
        if requested_cap <= 0:
            raise ValueError("Sufficiency answer max tokens must be positive")
        remaining = int(self.config.max_model_len) - len(prompt_ids)
        if requested_cap > remaining:
            raise RuntimeError(
                "Sufficiency probe would require context truncation"
            )
        if params.pop("logprobs", None) not in {None, False}:
            raise ValueError("Sufficiency probe must not request output logprobs")
        if params.pop("prompt_logprobs", None) not in {None, False}:
            raise ValueError("Sufficiency probe must not request prompt logprobs")
        params.setdefault(
            "repetition_penalty",
            self.config.get("repetition_penalty", 1.0),
        )
        vllm_params = _build_sufficiency_probe_sampling_params(
            params,
            max_tokens=requested_cap,
        )
        prompt = TokensPrompt(prompt_token_ids=[int(value) for value in prompt_ids])
        lora_request = None
        if self.model_config.lora_rank > 0:
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if not lora_loaded:
                raise RuntimeError(
                    "Sufficiency probe requires the rollout LoRA snapshot"
                )
            lora_request = LoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
            )

        started = time.perf_counter()
        self._project_request_count += 1
        self._project_inflight_requests += 1
        self._project_max_inflight_requests = max(
            self._project_max_inflight_requests,
            self._project_inflight_requests,
        )
        try:
            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=vllm_params,
                request_id=str(request_id),
                lora_request=lora_request,
            )
            final_result: Optional[RequestOutput] = None
            async for output in generator:
                final_result = output
            if final_result is None:
                raise RuntimeError("vLLM returned no sufficiency result")
        except BaseException:
            self._project_generation_errors += 1
            raise
        finally:
            self._project_inflight_requests -= 1
            self._project_generation_seconds += time.perf_counter() - started

        if len(final_result.outputs) != 1:
            raise RuntimeError(
                "Sufficiency request did not return exactly one completion"
            )
        output = final_result.outputs[0]
        token_ids = [int(value) for value in output.token_ids]
        self._project_generated_tokens += len(token_ids)
        metrics = getattr(final_result, "metrics", None)
        cached_prompt_tokens = int(
            getattr(metrics, "num_cached_tokens", 0) or 0
        )
        return {
            "request_id": str(request_id),
            "replica_rank": int(self.replica_rank),
            "snapshot_step": int(self._project_weight_version),
            "source_checksum": self._project_weight_checksum,
            "prompt_tokens": len(prompt_ids),
            "cached_prompt_tokens": cached_prompt_tokens,
            "decode_tokens": len(token_ids),
            "generation_seconds": time.perf_counter() - started,
            "automatic_prefix_caching": bool(
                self.config.get("enable_prefix_caching", False)
            ),
            "completion_count": 1,
            "completions": [
                {
                    "sample_index": 0,
                    "text": str(output.text),
                    "token_ids": token_ids,
                    "finish_reason": (
                        None
                        if output.finish_reason is None
                        else str(output.finish_reason)
                    ),
                    "stop_reason": (
                        None
                        if output.stop_reason is None
                        else str(output.stop_reason)
                    ),
                }
            ],
        }


CappedVLLMHttpServer = ray.remote(num_cpus=1)(CappedVLLMHttpServerBase)


class CappedVLLMReplica(vLLMReplica):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.server_class = CappedVLLMHttpServer


class StrictAgentLoopManager(AgentLoopManager):
    """AgentLoopManager bound to four independent capped TP=1 replicas."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.rollout_replica_class = CappedVLLMReplica
        super().__init__(*args, **kwargs)
        self._project_sleeping = bool(
            self.config.actor_rollout_ref.rollout.free_cache_engine
        )

    def topology(self) -> dict[str, Any]:
        return {
            "replica_count": len(self.rollout_replicas),
            "server_addresses": list(self.server_addresses),
            "aggregate_data_parallel_size": len(self.rollout_replicas),
            "tensor_parallel_size": 1,
        }

    def stamp_weight_version(
        self,
        snapshot_step: int,
        source_checksum: str,
    ) -> list[dict[str, Any]]:
        versions = self._run_all_with_results(
            [
                replica.servers[0].set_project_weight_version.remote(
                    int(snapshot_step),
                    str(source_checksum),
                )
                for replica in self.rollout_replicas
            ]
        )
        if len(versions) != len(self.rollout_replicas):
            raise RuntimeError("vLLM replica version count disagrees with topology")
        if {
            (value["snapshot_step"], value["source_checksum"])
            for value in versions
        } != {(int(snapshot_step), str(source_checksum))}:
            raise RuntimeError("vLLM replica version stamps disagree")
        return versions

    def read_weight_versions(self) -> list[dict[str, Any]]:
        return self._run_all_with_results(
            [
                replica.servers[0].get_project_weight_version.remote()
                for replica in self.rollout_replicas
            ]
        )

    def synchronize_from_actor(
        self,
        snapshot_step: int,
        source_checksum: str,
    ) -> dict[str, Any]:
        """Run veRL's real hybrid weight-sync path, then stamp its version.

        In hybrid mode ``wake_up`` calls every
        ``AsyncActorRolloutRefWorker.wake_up``. That enters
        ``ActorRolloutRefWorker.rollout_mode`` and invokes vLLM
        ``update_weights`` before the coroutine completes. The project stamp is
        deliberately written only after that barrier.
        """

        if not self._project_sleeping:
            self.sleep_for_scoring()
        started = time.perf_counter()
        self.wake_for_rollout()
        versions = self.stamp_weight_version(snapshot_step, source_checksum)
        observed = self.read_weight_versions()
        if observed != versions:
            raise RuntimeError("vLLM version readback differs after weight sync")
        return {
            "snapshot_step": int(snapshot_step),
            "source_checksum": str(source_checksum),
            "replica_count": len(versions),
            "versions": versions,
            "seconds": time.perf_counter() - started,
            "sync_path": (
                "vLLMHttpServer.wake_up->"
                "AsyncActorRolloutRefWorker.wake_up->"
                "ActorRolloutRefWorker.rollout_mode->vLLM.update_weights"
            ),
        }

    def runtime_metrics(self) -> list[dict[str, Any]]:
        return self._run_all_with_results(
            [
                replica.servers[0].get_project_runtime_metrics.remote()
                for replica in self.rollout_replicas
            ]
        )

    def generate_stop_branches(
        self,
        jobs_by_replica: list[list[dict[str, Any]]],
        *,
        expected_snapshot_step: int,
        expected_source_checksum: str,
    ) -> dict[str, Any]:
        """Execute replica-local Search-depth waves with Prompt affinity."""

        if len(jobs_by_replica) != len(self.rollout_replicas):
            raise RuntimeError("Stop branching jobs do not match replica topology")
        self.wake_for_rollout()
        expected_version = {
            (
                int(expected_snapshot_step),
                str(expected_source_checksum),
            )
        }
        before = self.read_weight_versions()
        if {
            (int(row["snapshot_step"]), str(row["source_checksum"]))
            for row in before
        } != expected_version:
            raise RuntimeError("Stop branching started on the wrong policy version")

        async def run_replica(
            replica_index: int,
            jobs: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            import asyncio

            server = self.rollout_replicas[replica_index].servers[0]
            rows: list[dict[str, Any]] = []
            depths = sorted({int(job["search_index"]) for job in jobs})
            for depth in depths:
                wave = [
                    job for job in jobs if int(job["search_index"]) == depth
                ]
                outputs = await asyncio.gather(
                    *[
                        server.generate_stop_pair.remote(
                            [int(value) for value in job["stop_input_ids"]],
                            dict(job["sampling_params"]),
                            str(job["request_id"]),
                        )
                        for job in wave
                    ]
                )
                for job, output in zip(wave, outputs, strict=True):
                    rows.append(
                        {
                            **dict(job["metadata"]),
                            **dict(output),
                            "assigned_replica": int(replica_index),
                            "local_depth": int(depth),
                        }
                    )
            return rows

        async def run_all_replicas() -> list[list[dict[str, Any]]]:
            import asyncio

            return list(
                await asyncio.gather(
                    *[
                        run_replica(replica_index, jobs)
                        for replica_index, jobs in enumerate(jobs_by_replica)
                    ]
                )
            )

        import asyncio

        started = time.perf_counter()
        rows_by_replica = asyncio.run(run_all_replicas())
        after = self.read_weight_versions()
        if after != before:
            raise RuntimeError("vLLM policy version changed during Stop branching")
        return {
            "rows": [
                row for replica_rows in rows_by_replica for row in replica_rows
            ],
            "per_replica_jobs": [
                len(replica_rows) for replica_rows in rows_by_replica
            ],
            "per_replica_tokens": [
                sum(int(row["decode_tokens"]) for row in replica_rows)
                for replica_rows in rows_by_replica
            ],
            "versions_before": before,
            "versions_after": after,
            "generation_seconds": time.perf_counter() - started,
            "prompt_affinity": True,
            "local_depth_waves": True,
            "cross_replica_depth_barrier": False,
        }

    def generate_sufficiency_probes(
        self,
        jobs_by_replica: list[list[dict[str, Any]]],
        *,
        expected_snapshot_step: int,
        expected_source_checksum: str,
    ) -> dict[str, Any]:
        """Execute deterministic Prompt-affine probes in replica-local waves."""

        if len(jobs_by_replica) != len(self.rollout_replicas):
            raise RuntimeError("Sufficiency probing jobs do not match replica topology")
        self.wake_for_rollout()
        expected_version = {
            (int(expected_snapshot_step), str(expected_source_checksum))
        }
        before = self.read_weight_versions()
        if {
            (int(row["snapshot_step"]), str(row["source_checksum"]))
            for row in before
        } != expected_version:
            raise RuntimeError(
                "Sufficiency probing started on the wrong policy version"
            )

        async def run_replica(
            replica_index: int,
            jobs: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            import asyncio

            server = self.rollout_replicas[replica_index].servers[0]
            rows: list[dict[str, Any]] = []
            depths = sorted({int(job["search_index"]) for job in jobs})
            for depth in depths:
                wave = [
                    job for job in jobs if int(job["search_index"]) == depth
                ]
                outputs = await asyncio.gather(
                    *[
                        server.generate_sufficiency_probe.remote(
                            [int(value) for value in job["probe_input_ids"]],
                            dict(job["sampling_params"]),
                            str(job["request_id"]),
                        )
                        for job in wave
                    ]
                )
                for job, output in zip(wave, outputs, strict=True):
                    rows.append(
                        {
                            **dict(job["metadata"]),
                            **dict(output),
                            "assigned_replica": int(replica_index),
                            "local_depth": int(depth),
                        }
                    )
            return rows

        async def run_all_replicas() -> list[list[dict[str, Any]]]:
            import asyncio

            return list(
                await asyncio.gather(
                    *[
                        run_replica(replica_index, jobs)
                        for replica_index, jobs in enumerate(jobs_by_replica)
                    ]
                )
            )

        import asyncio

        started = time.perf_counter()
        rows_by_replica = asyncio.run(run_all_replicas())
        after = self.read_weight_versions()
        if after != before:
            raise RuntimeError(
                "vLLM policy version changed during sufficiency probing"
            )
        return {
            "rows": [
                row for replica_rows in rows_by_replica for row in replica_rows
            ],
            "per_replica_jobs": [
                len(replica_rows) for replica_rows in rows_by_replica
            ],
            "per_replica_tokens": [
                sum(int(row["decode_tokens"]) for row in replica_rows)
                for replica_rows in rows_by_replica
            ],
            "versions_before": before,
            "versions_after": after,
            "generation_seconds": time.perf_counter() - started,
            "prompt_affinity": True,
            "local_depth_waves": True,
            "cross_replica_depth_barrier": False,
        }

    def wake_for_rollout(self) -> None:
        if self._project_sleeping:
            self.wake_up()
            self._project_sleeping = False

    def sleep_for_scoring(self) -> None:
        if not self._project_sleeping:
            self.sleep()
            self._project_sleeping = True

    def generate_sequences_keep_awake(self, prompts: Any) -> Any:
        """Generate one wave without sleeping before the next wave."""
        import ray
        from verl.protocol import DataProto

        outputs = ray.get(self.dispatch_sequences_keep_awake(prompts))
        output = DataProto.concat(outputs)
        metrics = [item.meta_info.pop("metrics") for item in outputs]
        output.meta_info = {
            "timing": self._performance_metrics(metrics, output),
            **outputs[0].meta_info,
        }
        return output

    def dispatch_sequences_keep_awake(self, prompts: Any) -> list[Any]:
        """Return ObjectRefs so CPU postprocessing can overlap the next wave."""
        self.wake_for_rollout()
        worker_count = min(len(self.agent_loop_workers), len(prompts))
        while worker_count > 1 and len(prompts) % worker_count:
            worker_count -= 1
        workers = self.agent_loop_workers[:worker_count]
        chunks = prompts.chunk(worker_count)
        return [
            worker.generate_sequences.remote(chunk)
            for worker, chunk in zip(
                workers,
                chunks,
                strict=True,
            )
        ]

    @staticmethod
    def _run_all_with_results(tasks: list[Any]) -> list[Any]:
        import asyncio

        async def run_all() -> list[Any]:
            return list(await asyncio.gather(*tasks))

        return asyncio.run(run_all())
