from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

from .target_schema import PRODUCTION_PRECISION_MODE


@dataclass(frozen=True)
class ExactIGPrecisionPolicy:
    """Auditable numerical policy for Exact-IG teacher forcing.

    Dtypes observed from a real forward are metadata, not configuration claims.
    CPU tests deliberately run in their native dtype.
    """

    mode: str
    autocast_enabled: bool
    autocast_dtype: str | None
    temperature: float
    attention_implementation: str
    sdpa_backend: str | None
    allow_tf32: bool = False
    allow_bf16_reduced_precision_reduction: bool = False
    allow_fp16_reduced_precision_reduction: bool = False


_FP32_POLICY = ExactIGPrecisionPolicy(
    mode=PRODUCTION_PRECISION_MODE,
    autocast_enabled=False,
    autocast_dtype=None,
    temperature=1.0,
    attention_implementation="sdpa",
    sdpa_backend="math",
    allow_tf32=False,
    allow_bf16_reduced_precision_reduction=False,
    allow_fp16_reduced_precision_reduction=False,
)


def production_precision_policy(mode: str) -> ExactIGPrecisionPolicy:
    if str(mode) != _FP32_POLICY.mode:
        raise ValueError(
            f"Exact-IG production precision must be {PRODUCTION_PRECISION_MODE}"
        )
    return _FP32_POLICY


def _set_attention_implementation(
    model: Any,
    implementation: str,
) -> list[tuple[Any, str]]:
    changed: list[tuple[Any, str]] = []
    seen: set[int] = set()
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is None or id(config) in seen:
            continue
        seen.add(id(config))
        previous = str(getattr(config, "_attn_implementation", "eager"))
        if previous != implementation:
            changed.append((config, previous))
            config._attn_implementation = implementation
    return changed


def _model_device_type(model: Any) -> str:
    try:
        return str(next(model.parameters()).device.type)
    except StopIteration:
        return "cpu"


def _floating_parameter_dtypes(model: Any) -> set[str]:
    return {
        str(parameter.dtype).removeprefix("torch.")
        for parameter in model.parameters()
        if parameter.is_floating_point()
    }


def assert_fp32_exact_ig_runtime(
    *,
    model: Any,
    policy: ExactIGPrecisionPolicy,
    logits: Any | None = None,
    log_probs: Any | None = None,
) -> None:
    """Fail closed if a production Exact-IG forward is not genuinely FP32."""

    import torch

    if policy.mode != PRODUCTION_PRECISION_MODE:
        return
    parameter_dtypes = _floating_parameter_dtypes(model)
    if parameter_dtypes != {"float32"}:
        raise RuntimeError(
            "FP32 Exact-IG requires every floating Reward Snapshot parameter "
            f"to be float32, got {sorted(parameter_dtypes)}"
        )
    if policy.autocast_enabled or policy.autocast_dtype is not None:
        raise RuntimeError("FP32 Exact-IG forbids autocast")
    device_type = _model_device_type(model)
    if device_type == "cuda" and torch.is_autocast_enabled("cuda"):
        raise RuntimeError("CUDA autocast remained enabled inside FP32 Exact-IG")
    for name, value in (("logits", logits), ("log_probs", log_probs)):
        if value is not None and value.dtype is not torch.float32:
            raise RuntimeError(
                f"FP32 Exact-IG requires {name} dtype float32, got {value.dtype}"
            )


@contextmanager
def exact_ig_precision_context(
    model: Any,
    policy: ExactIGPrecisionPolicy,
) -> Iterator[None]:
    """Apply and exactly restore the project-locked FP32 scoring environment."""

    import torch

    if float(policy.temperature) != 1.0:
        raise ValueError("Exact-IG temperature is fixed to 1.0")
    if policy.mode == PRODUCTION_PRECISION_MODE:
        if policy.autocast_enabled or policy.autocast_dtype is not None:
            raise ValueError("fp32_exact_ig must disable autocast completely")
        if (
            policy.allow_tf32
            or policy.allow_bf16_reduced_precision_reduction
            or policy.allow_fp16_reduced_precision_reduction
        ):
            raise ValueError("fp32_exact_ig forbids reduced-precision CUDA math")
        assert_fp32_exact_ig_runtime(model=model, policy=policy)
    changed_configs = _set_attention_implementation(
        model,
        policy.attention_implementation,
    )
    device_type = _model_device_type(model)
    previous_cuda_state: dict[str, Any] = {}
    previous_matmul_precision = torch.get_float32_matmul_precision()
    try:
        if device_type == "cuda":
            previous_cuda_state = {
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "allow_bf16_reduced_precision_reduction": (
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                ),
                "allow_fp16_reduced_precision_reduction": (
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
                ),
            }
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.set_float32_matmul_precision("highest")
        with ExitStack() as stack:
            if (
                device_type == "cuda"
                and policy.attention_implementation == "sdpa"
                and policy.sdpa_backend
            ):
                backend = {
                    "math": torch.nn.attention.SDPBackend.MATH,
                    "flash": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                    "efficient": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                    "cudnn": torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
                }[policy.sdpa_backend]
                stack.enter_context(torch.nn.attention.sdpa_kernel(backend))
            if device_type == "cuda":
                stack.enter_context(torch.autocast(device_type="cuda", enabled=False))
            assert_fp32_exact_ig_runtime(model=model, policy=policy)
            yield
    finally:
        if previous_cuda_state:
            torch.backends.cuda.matmul.allow_tf32 = previous_cuda_state[
                "matmul_allow_tf32"
            ]
            torch.backends.cudnn.allow_tf32 = previous_cuda_state[
                "cudnn_allow_tf32"
            ]
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                previous_cuda_state["allow_bf16_reduced_precision_reduction"]
            )
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
                previous_cuda_state["allow_fp16_reduced_precision_reduction"]
            )
            torch.set_float32_matmul_precision(previous_matmul_precision)
        for config, implementation in changed_configs:
            config._attn_implementation = implementation


def precision_runtime_metadata(
    model: Any,
    policy: ExactIGPrecisionPolicy,
    observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    parameter = next(model.parameters())
    observed = dict(observation or {})
    return {
        "exact_ig_production_precision_mode": policy.mode,
        "actual_model_parameter_dtype": str(parameter.dtype).removeprefix("torch."),
        "actual_logits_dtype": observed.get("actual_logits_dtype"),
        "actual_log_probs_dtype": observed.get("actual_log_probs_dtype"),
        "autocast_enabled": bool(observed.get("autocast_enabled", False)),
        "autocast_dtype": policy.autocast_dtype,
        "attention_backend": (
            f"{policy.attention_implementation}:{policy.sdpa_backend or 'native'}"
        ),
        "temperature": float(policy.temperature),
        "allow_tf32": bool(policy.allow_tf32),
        "allow_bf16_reduced_precision_reduction": bool(
            policy.allow_bf16_reduced_precision_reduction
        ),
        "allow_fp16_reduced_precision_reduction": bool(
            policy.allow_fp16_reduced_precision_reduction
        ),
        "float32_matmul_precision": "highest",
        "cuda_available": bool(torch.cuda.is_available()),
    }
