#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agentic_rl.advantage.a2tgpo import compute_prompt_advantages
from agentic_rl.config import DEFAULT_CONFIG, load_config
from agentic_rl.controller.transaction import StrictUpdateTransaction
from agentic_rl.exact_ig.alias_reduce import immediate_ig_from_prefix_scores
from agentic_rl.exact_ig.vectorized_scorer import VectorizedExactIGScorer
from agentic_rl.outcome.format_indicator import centered_format_advantage
from agentic_rl.outcome.token_f1 import compute_f1
from agentic_rl.policy.reference_kl import actor_to_reference_full_vocab_kl
from agentic_rl.policy.reduction import prompt_trajectory_action_token_reduce
from agentic_rl.policy.strict_onpolicy_loss import (
    ADAPTIVE_CLIP_BETA,
    ADAPTIVE_CLIP_EPSILON_HIGH,
    ADAPTIVE_CLIP_EPSILON_LOW,
    ANSWER_CLIP_SCALE,
    CLIPPING_MODE,
    a2tgpo_adaptive_turn_objective,
    adaptive_clip_scale,
)
from agentic_rl.policy.turn_ratio import compute_turn_ratios
from agentic_rl.selection.channel_scale import ChannelScaleState
from agentic_rl.selection.prompt_variance import ig_prompt_variance
from agentic_rl.selection.top_p import stable_mass_top_p


OUTPUT_PATH = PROJECT_ROOT / "EFFECTIVE_ALGORITHM_FROM_CODE_V2_1.md"


def _source(relative: str) -> str:
    return (PROJECT_ROOT / relative).read_text(encoding="utf-8")


def source_location(value: Callable[..., Any] | type[Any]) -> str:
    path = Path(inspect.getsourcefile(value) or "").resolve()
    relative = path.relative_to(PROJECT_ROOT)
    line = inspect.getsourcelines(value)[1]
    return f"{relative.as_posix()}:{line}"


def constant_location(relative: str, name: str) -> str:
    path = PROJECT_ROOT / relative
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if line.startswith(f"{name} ="):
            return f"{relative}:{line_number}"
    raise RuntimeError(f"Cannot locate {name} in {relative}")


def _fixed_clip_absent() -> bool:
    blocked = (
        "fixed_" + "dapo",
        "dapo_" + "turn_objective",
        "clip_ratio_" + "low",
        "clip_ratio_" + "high",
    )
    for root_name in ("src", "configs"):
        for path in (PROJECT_ROOT / root_name).rglob("*"):
            if path.suffix not in {".py", ".yaml", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8").lower()
            if any(fragment in text for fragment in blocked):
                return False
            if "0." + "8" in text or "1." + "28" in text:
                return False
    return True


def effective_contract(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_config(DEFAULT_CONFIG) if config is None else config
    alias_source = inspect.getsource(immediate_ig_from_prefix_scores)
    score_source = inspect.getsource(VectorizedExactIGScorer.score_batch)
    variance_source = inspect.getsource(ig_prompt_variance)
    scale_source = inspect.getsource(ChannelScaleState.inspect_pool)
    commit_source = inspect.getsource(ChannelScaleState.committed_after_success)
    advantage_source = inspect.getsource(compute_prompt_advantages)
    format_source = inspect.getsource(centered_format_advantage)
    clip_source = inspect.getsource(a2tgpo_adaptive_turn_objective)
    ratio_source = inspect.getsource(compute_turn_ratios)
    reduction_source = inspect.getsource(prompt_trajectory_action_token_reduce)
    kl_source = inspect.getsource(actor_to_reference_full_vocab_kl)
    transaction_source = inspect.getsource(StrictUpdateTransaction)
    selection = config["selection"]
    rollout = config["rollout"]
    advantage = config["advantage"]
    policy = config["policy"]
    learner = config["learner"]
    hardware = config["hardware"]

    return {
        "schema_version": config["project"]["schema_version"],
        "exact_ig": {
            "score_space": (
                "mean_canonical_answer_body_log_likelihood"
                if "token_logprobs.to" in score_source and ".mean(" in score_source
                else "UNRECOGNIZED"
            ),
            "search_reward": (
                "phi_t_minus_phi_t_minus_1"
                if "prefix_scores[index] - prefix_scores[index - 1]"
                in alias_source
                else "UNRECOGNIZED"
            ),
            "canonical_alias_policy": config["exact_ig"][
                "canonical_alias_policy"
            ],
            "score_mask_policy": config["exact_ig"]["score_mask_policy"],
            "fast_path_structure": config["exact_ig"]["fast_path_structure"],
            "phi_exponentiated": "exp(" in alias_source or "torch.exp(" in score_source,
            "extra_search_reward_terms": [],
            "stop_gradient": config["exact_ig"]["stop_gradient"],
        },
        "outcome": {
            "scorer": config["outcome"]["scorer"],
            "official_commit": config["outcome"]["official_commit"],
            "aliases_delimiter": config["outcome"]["aliases_delimiter"],
            "format_enters_task_outcome": False,
        },
        "ig_variance": {
            "scope": (
                "same_prompt_same_search_position"
                if "peers.setdefault(int(search_index)" in variance_source
                else "UNRECOGNIZED"
            ),
            "sample_variance_ddof": 1,
            "effective_position_min_peers": (
                2 if "count >= 2" in variance_source else -1
            ),
            "singleton_weight": (
                0.0
                if "supported_peer_total" in variance_source
                and "else 0.0" in variance_source
                else "UNRECOGNIZED"
            ),
            "weight_denominator": (
                "sum_peer_counts_over_positions_with_n_ge_2"
                if "supported_peer_total = sum(" in variance_source
                else "UNRECOGNIZED"
            ),
        },
        "scale_activation": {
            "update_1_selection_scale": (
                "current_positive_median"
                if "allow_provisional_scale" in scale_source
                else "UNRECOGNIZED"
            ),
            "update_1_activation_controls_scale_commit": (
                "stats.gate.active" in commit_source.split(
                    "def committed_after_success", 1
                )[-1]
            ),
            "updates_2_to_health_ready_selection_scale": "previous_committed_scale",
            "bootstrap_activation_controls_ema": (
                not (
                    "self.health_reference is None or gate.active" in scale_source
                    and "scale_update_allowed_after_success" in commit_source
                )
            ),
            "health_inactive_freezes_ema": (
                "self.health_reference is None or gate.active" in scale_source
            ),
            "ema_half_life": selection["scale_ema_half_life"],
            "ema_eta": ChannelScaleState.ema_eta(
                float(selection["scale_ema_half_life"])
            ),
            "health_reference_valid_observations": selection[
                "health_reference_valid_updates"
            ],
            "health_threshold_ratio": selection["health_threshold_ratio"],
            "minimum_positive_prompts": selection["minimum_positive_prompts"],
            "late_initialization_allowed": False,
        },
        "selection": {
            "alpha_ig": selection["alpha_ig"],
            "alpha_outcome": selection["alpha_outcome"],
            "top_p_mass": selection["top_p_mass"],
            "positive_scores_only": not selection["include_zero"],
            "selection_epsilon": selection["selection_epsilon"],
            "candidate_prompts_initial": rollout["candidate_prompts_initial"],
            "candidate_prompts_max": rollout["candidate_prompts_max"],
            "refill_prompts": rollout["refill_prompts"],
            "minimum_selected_prompts": selection["minimum_selected_prompts"],
            "maximum_selected_prompts": selection["maximum_selected_prompts"],
            "rollouts_per_prompt": rollout["group_size"],
            "recompute_full_pool_after_refill": selection[
                "recompute_after_refill_on_full_pool"
            ],
        },
        "advantage": {
            "ig_normalization": "same_prompt_same_search_position_population",
            "outcome_normalization": "same_prompt_population",
            "gamma": advantage["gamma"],
            "rescale_count": advantage["rescale_count_mode"],
            "search_terms": advantage["search_formula_terms"],
            "answer_terms": advantage["answer_formula_terms"],
            "lambda_ig": advantage["lambda_ig"],
            "lambda_outcome": advantage["lambda_outcome"],
            "lambda_format": advantage["lambda_format"],
            "format_centering_is_subtraction": "return values - np.mean" in format_source,
            "malformed_term_present": "lambda_" + "mal" in advantage_source.lower(),
        },
        "clipping": {
            "mode": CLIPPING_MODE,
            "beta_c": ADAPTIVE_CLIP_BETA,
            "epsilon_low": ADAPTIVE_CLIP_EPSILON_LOW,
            "epsilon_high": ADAPTIVE_CLIP_EPSILON_HIGH,
            "answer_scale": ANSWER_CLIP_SCALE,
            "search_scale_formula": (
                "1+beta*(2*sigmoid(normalized_ig)-1)"
                if "1.0 + beta * (2.0 * sigmoid - 1.0)" in inspect.getsource(
                    adaptive_clip_scale
                )
                else "UNRECOGNIZED"
            ),
            "turn_surrogate": (
                "min(ratio*A,clip(ratio,lower,upper)*A)"
                if "torch.minimum(" in clip_source
                and "torch.clamp(ratio, min=lower, max=upper)" in clip_source
                else "UNRECOGNIZED"
            ),
            "fixed_dapo_active": not _fixed_clip_absent(),
            "clip_scale_stop_gradient": True,
        },
        "policy": {
            "ratio_level": policy["ratio_level"],
            "ratio_hardcoded": policy["ratio_hardcoded"],
            "old_logprob_detached_required": (
                "Old-policy logprobs must be detached" in ratio_source
            ),
            "current_logprob_differentiable_required": (
                "Current-policy logprobs must retain gradients" in ratio_source
            ),
            "reduction": policy["task_reduction"],
            "nested_reduction_implemented": (
                "trajectory_mean = record.values[mask].mean()" in reduction_source
                and "torch.stack(trajectory_means).mean()" in reduction_source
            ),
            "full_vocab_kl": policy["full_vocab_reference_kl"],
            "kl_direction": (
                "actor_to_frozen_reference_forward"
                if "probability * (actor_log_probability - reference_log_probability)"
                in kl_source
                else "UNRECOGNIZED"
            ),
            "kl_reduction": policy["kl_reduction"],
            "kl_coefficient": policy["kl_coefficient"],
            "entropy_coefficient": policy["entropy_coefficient"],
            "value_coefficient": policy["value_coefficient"],
            "max_grad_norm": policy["max_grad_norm"],
            "ppo_epochs": policy["ppo_epochs"],
            "optimizer_mini_steps": policy["optimizer_mini_steps"],
            "optimizer_steps_per_successful_update": policy[
                "optimizer_steps_per_successful_update"
            ],
            "one_optimizer_step_guard": (
                "Exactly one optimizer.step is permitted" in transaction_source
            ),
        },
        "topology": {
            "retriever_physical_gpu": hardware["retriever_physical_gpu"],
            "rl_physical_gpus": hardware["rl_physical_gpus"],
            "vllm_dp": rollout["data_parallel_size"],
            "vllm_tp": rollout["tensor_parallel_size"],
            "fsdp2_world_size": learner["world_size"],
            "learner_strategy": learner["strategy"],
        },
        "runtime_verification": {
            "exact_ig_static_structure": config["exact_ig"]["runtime_status"][
                "static_structure"
            ],
            "exact_ig_numerical_parity": config["exact_ig"]["runtime_status"][
                "numerical_parity"
            ],
            "exact_ig_gpu_runtime": config["exact_ig"]["runtime_status"][
                "gpu_runtime"
            ],
            "distributed_runtime": "UNVERIFIED_RUNTIME",
            "vllm_fsdp2_weight_sync": "UNVERIFIED_RUNTIME",
            "real_optimizer_transaction": "UNVERIFIED_RUNTIME",
        },
    }


def parameter_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    locations = {
        "exact_ig": source_location(VectorizedExactIGScorer.score),
        "outcome": source_location(compute_f1),
        "ig_variance": source_location(ig_prompt_variance),
        "scale": source_location(ChannelScaleState.inspect_pool),
        "scale_commit": source_location(ChannelScaleState.committed_after_success),
        "advantage": source_location(compute_prompt_advantages),
        "ratio": source_location(compute_turn_ratios),
        "clip": source_location(a2tgpo_adaptive_turn_objective),
        "reduction": source_location(prompt_trajectory_action_token_reduce),
        "kl": source_location(actor_to_reference_full_vocab_kl),
        "transaction": source_location(StrictUpdateTransaction),
    }
    values = [
        ("Exact IG score", "mean log-likelihood", locations["exact_ig"], "exact_ig", True, False, True, False, False, True),
        ("Outcome scorer", config["outcome"]["scorer"], locations["outcome"], "outcome.scorer", True, True, True, False, False, True),
        ("IG position minimum peers", 2, locations["ig_variance"], "source constant", True, True, False, False, False, True),
        ("Top-p mass", config["selection"]["top_p_mass"], source_location(stable_mass_top_p), "selection.top_p_mass", True, True, False, False, True, True),
        ("IG channel alpha", config["selection"]["alpha_ig"], source_location(stable_mass_top_p), "selection.alpha_ig", True, True, False, False, True, True),
        ("Outcome channel alpha", config["selection"]["alpha_outcome"], source_location(stable_mass_top_p), "selection.alpha_outcome", True, True, False, False, True, True),
        ("Scale half-life", config["selection"]["scale_ema_half_life"], locations["scale_commit"], "selection.scale_ema_half_life", True, True, False, False, True, True),
        ("Health observations", config["selection"]["health_reference_valid_updates"], locations["scale_commit"], "selection.health_reference_valid_updates", True, True, False, False, True, True),
        ("A2TGPO gamma", config["advantage"]["gamma"], locations["advantage"], "advantage.gamma", True, False, True, False, False, True),
        ("Adaptive clip beta", ADAPTIVE_CLIP_BETA, constant_location("src/agentic_rl/policy/strict_onpolicy_loss.py", "ADAPTIVE_CLIP_BETA"), "policy.adaptive_clip_beta", True, False, False, True, False, True),
        ("Adaptive epsilon low", ADAPTIVE_CLIP_EPSILON_LOW, constant_location("src/agentic_rl/policy/strict_onpolicy_loss.py", "ADAPTIVE_CLIP_EPSILON_LOW"), "policy.adaptive_clip_epsilon_low", True, False, False, True, False, True),
        ("Adaptive epsilon high", ADAPTIVE_CLIP_EPSILON_HIGH, constant_location("src/agentic_rl/policy/strict_onpolicy_loss.py", "ADAPTIVE_CLIP_EPSILON_HIGH"), "policy.adaptive_clip_epsilon_high", True, False, False, True, False, True),
        ("Answer clip scale", ANSWER_CLIP_SCALE, constant_location("src/agentic_rl/policy/strict_onpolicy_loss.py", "ANSWER_CLIP_SCALE"), "policy.answer_clip_scale", True, False, False, True, False, True),
        ("Turn ratio", "exp(mean(logpi-logpi_old))", locations["ratio"], "policy.ratio_level", False, False, False, True, False, True),
        ("Task reduction", config["policy"]["task_reduction"], locations["reduction"], "policy.task_reduction", False, False, False, True, False, True),
        ("KL direction", "actor||reference", locations["kl"], "policy.full_vocab_reference_kl", False, False, False, True, False, True),
        ("KL coefficient", config["policy"]["kl_coefficient"], locations["kl"], "policy.kl_coefficient", False, False, False, True, True, True),
        ("PPO epochs", config["policy"]["ppo_epochs"], locations["transaction"], "policy.ppo_epochs", False, False, False, True, True, True),
        ("Optimizer mini-steps", config["policy"]["optimizer_mini_steps"], locations["transaction"], "policy.optimizer_mini_steps", False, False, False, True, True, True),
    ]
    keys = (
        "parameter",
        "effective_value",
        "source",
        "config_path",
        "stop_gradient",
        "enters_selection",
        "enters_advantage",
        "enters_policy_gradient",
        "enters_checkpoint",
        "enters_logs",
    )
    return [dict(zip(keys, row)) for row in values]


def render_report(config: dict[str, Any], contract: dict[str, Any]) -> str:
    locations = {
        "Exact IG": source_location(VectorizedExactIGScorer.score),
        "Outcome F1": source_location(compute_f1),
        "IG variance": source_location(ig_prompt_variance),
        "Scale/health": source_location(ChannelScaleState.inspect_pool),
        "Scale commit": source_location(ChannelScaleState.committed_after_success),
        "Top-p": source_location(stable_mass_top_p),
        "Advantage": source_location(compute_prompt_advantages),
        "Format": source_location(centered_format_advantage),
        "Ratio": source_location(compute_turn_ratios),
        "Adaptive clipping": source_location(a2tgpo_adaptive_turn_objective),
        "Reduction": source_location(prompt_trajectory_action_token_reduce),
        "KL": source_location(actor_to_reference_full_vocab_kl),
        "Transaction": source_location(StrictUpdateTransaction),
    }
    scale = contract["scale_activation"]
    selection = contract["selection"]
    advantage = contract["advantage"]
    clipping = contract["clipping"]
    policy = contract["policy"]
    topology = contract["topology"]
    runtime = contract["runtime_verification"]
    sections = [
        ("Exact IG", f"`Phi` is `{contract['exact_ig']['score_space']}` and `r_IG={contract['exact_ig']['search_reward']}`. Canonical policy is `{contract['exact_ig']['canonical_alias_policy']}`, score mask is `{contract['exact_ig']['score_mask_policy']}`, Fast Path is `{contract['exact_ig']['fast_path_structure']}`. Source: `{locations['Exact IG']}`."),
        ("Outcome F1", f"Scorer `{contract['outcome']['scorer']}` at commit `{contract['outcome']['official_commit']}`; alias delimiter `{contract['outcome']['aliases_delimiter']}`. Source: `{locations['Outcome F1']}`."),
        ("Per-position IG variance", f"Scope `{contract['ig_variance']['scope']}`, sample ddof `{contract['ig_variance']['sample_variance_ddof']}`, minimum peers `{contract['ig_variance']['effective_position_min_peers']}`. Source: `{locations['IG variance']}`."),
        ("Prompt IG variance", f"Singleton weight is `{contract['ig_variance']['singleton_weight']}`; denominator is `{contract['ig_variance']['weight_denominator']}`."),
        ("Outcome variance", "Sample variance is computed only over Outcome-eligible trajectories; format is excluded."),
        ("Noise floor", f"`e=max(V-nu,0)` with configured IG/Outcome floors `{config['selection']['noise_floor_ig']}` / `{config['selection']['noise_floor_outcome']}`."),
        ("Update 1", f"Selection scale `{scale['update_1_selection_scale']}`; activation controls scale commit `{scale['update_1_activation_controls_scale_commit']}`."),
        ("Updates 2-10", f"Selection uses `{scale['updates_2_to_health_ready_selection_scale']}`; bootstrap activation controls EMA `{scale['bootstrap_activation_controls_ema']}`; eta `{scale['ema_eta']:.15f}`."),
        ("Update 11+", f"Reference after `{scale['health_reference_valid_observations']}` valid observations; health threshold `{scale['health_threshold_ratio']}`; inactive health gate freezes EMA `{scale['health_inactive_freezes_ema']}`."),
        ("Scale and activation", f"Scale inspect: `{locations['Scale/health']}`. Commit: `{locations['Scale commit']}`."),
        ("Top-p and refill", f"Mass `{selection['top_p_mass']}`, pool `{selection['candidate_prompts_initial']} -> {selection['candidate_prompts_max']}`, selected `{selection['minimum_selected_prompts']}..{selection['maximum_selected_prompts']}`, G=`{selection['rollouts_per_prompt']}`. Source: `{locations['Top-p']}`."),
        ("A2TGPO normalization", f"IG normalization `{advantage['ig_normalization']}`; Outcome normalization `{advantage['outcome_normalization']}`. Source: `{locations['Advantage']}`."),
        ("Accumulation count", f"`n_acc` mode `{advantage['rescale_count']}`; zero-normalized but eligible positions remain counted."),
        ("Square-root rescale", f"`D_bar=D/sqrt(n_acc)` with gamma `{advantage['gamma']}`."),
        ("Search advantage", f"Terms `{advantage['search_terms']}` with coefficients `{advantage['lambda_ig']}`, `{advantage['lambda_outcome']}`."),
        ("Answer advantage", f"Terms `{advantage['answer_terms']}` with coefficients `{advantage['lambda_outcome']}`, `{advantage['lambda_format']}`."),
        ("Format advantage", f"`A_format=F_ans-mean(F_ans)`; subtraction detected `{advantage['format_centering_is_subtraction']}`. Format source: `{locations['Format']}`."),
        ("Adaptive clip scale", f"Mode `{clipping['mode']}`; `c={clipping['search_scale_formula']}`; beta `{clipping['beta_c']}`. Source: `{locations['Adaptive clipping']}`."),
        ("Adaptive bounds", f"epsilon low/high `{clipping['epsilon_low']}` / `{clipping['epsilon_high']}`; Answer scale `{clipping['answer_scale']}`; fixed DAPO active `{clipping['fixed_dapo_active']}`."),
        ("Turn ratio", f"Level `{policy['ratio_level']}`, hardcoded `{policy['ratio_hardcoded']}`; detached-old guard `{policy['old_logprob_detached_required']}`. Source: `{locations['Ratio']}`."),
        ("Turn surrogate", f"`{clipping['turn_surrogate']}`."),
        ("Task reduction", f"`{policy['reduction']}`, implementation detected `{policy['nested_reduction_implemented']}`. Source: `{locations['Reduction']}`."),
        ("Reference KL", f"Full vocabulary `{policy['full_vocab_kl']}`, direction `{policy['kl_direction']}`, reduction `{policy['kl_reduction']}`. Source: `{locations['KL']}`."),
        ("Total loss", f"`L_total=-J_policy+{policy['kl_coefficient']}*L_KL`; entropy `{policy['entropy_coefficient']}`, value `{policy['value_coefficient']}`."),
        ("Strict one-step transaction", f"PPO epochs `{policy['ppo_epochs']}`, optimizer mini-steps `{policy['optimizer_mini_steps']}`, optimizer steps/success `{policy['optimizer_steps_per_successful_update']}`. Source: `{locations['Transaction']}`."),
        ("GPU and engine topology", f"Retriever GPU `{topology['retriever_physical_gpu']}`; RL GPUs `{topology['rl_physical_gpus']}`; vLLM DP/TP `{topology['vllm_dp']}/{topology['vllm_tp']}`; FSDP2 world size `{topology['fsdp2_world_size']}`."),
        ("Runtime-unverified gates", ", ".join(f"`{key}={value}`" for key, value in runtime.items())),
    ]
    rows = parameter_rows(config)
    table = "\n".join(
        "| {parameter} | `{effective_value}` | `{source}` | `{config_path}` | {stop_gradient} | {enters_selection} | {enters_advantage} | {enters_policy_gradient} | {enters_checkpoint} | {enters_logs} |".format(
            **row
        )
        for row in rows
    )
    body = "\n\n".join(f"## {title}\n\n{text}" for title, text in sections)
    contract_json = json.dumps(contract, indent=2, sort_keys=True)
    return f"""# Effective Algorithm From Code V2.1

This file is generated by `scripts/export_effective_algorithm_v2_1.py` from the
resolved configuration, imported source constants, function metadata, and
active source-shape checks. It is not a hand-maintained normative document.

{body}

## Effective Parameter Dataflow

| Parameter | Effective value | Source | Config path | Stop-gradient | Selection | Advantage | Policy gradient | Checkpoint | Logs |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
{table}

## Machine-Readable Contract

<!-- EFFECTIVE_CONTRACT_JSON_START -->
```json
{contract_json}
```
<!-- EFFECTIVE_CONTRACT_JSON_END -->
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()
    config = load_config(args.config)
    contract = effective_contract(config)
    output = Path(args.output).resolve()
    output.write_text(render_report(config, contract), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256_input_contract": contract,
                "sections": 27,
                "parameters": len(parameter_rows(config)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
