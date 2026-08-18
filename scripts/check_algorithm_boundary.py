#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from agentic_rl.config import DEFAULT_CONFIG, load_config
from agentic_rl.policy.strict_onpolicy_loss import (
    ADAPTIVE_CLIP_BETA,
    ADAPTIVE_CLIP_EPSILON_HIGH,
    ADAPTIVE_CLIP_EPSILON_LOW,
    ANSWER_CLIP_SCALE,
    CLIPPING_MODE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _scan(paths: list[Path], fragments: tuple[str, ...]) -> list[str]:
    findings: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        for fragment in fragments:
            if fragment.lower() in text:
                findings.append(f"{path.relative_to(PROJECT_ROOT)}:{fragment}")
    return findings


def main() -> None:
    config = load_config(DEFAULT_CONFIG)
    advantage = config["advantage"]
    policy = config["policy"]
    if advantage["search_formula_terms"] != [
        "sufficient_before_search",
        "sufficient_after_search",
        "no_new_observation",
        "effective_cumulative_normalized_local_ig",
        "probe_routed_normalized_outcome",
    ]:
        raise SystemExit("Search advantage formula does not have exactly two terms")
    if advantage["answer_formula_terms"] != [
        "normalized_outcome",
        "centered_format_indicator",
    ]:
        raise SystemExit("Answer advantage formula is not locked")
    expected_keys = {
        "mode",
        "search_task_mode",
        "gamma",
        "lambda_outcome",
        "lambda_format",
        "probe_epsilon",
        "search_advantage_formula",
        "outcome_fallback_to_search",
        "normalization_epsilon",
        "zero_variance_tolerance",
        "search_formula_terms",
        "answer_formula_terms",
        "rescale_count_mode",
        "future_ig_accumulation",
        "sqrt_n_rescale",
        "external_ig_multiplier",
        "sufficiency_probe",
        "sc",
    }
    if set(advantage) != expected_keys:
        raise SystemExit("Advantage config contains an unexpected field")
    if advantage["search_task_mode"] != (
        "sufficiency_novelty_cumulative_ig_probe_routed_outcome"
    ):
        raise SystemExit("S/N/cumulative-IG/Probe-routed Search mode is not active")
    if advantage["external_ig_multiplier"] is not None:
        raise SystemExit("Production Search has an external IG multiplier")
    if not advantage["future_ig_accumulation"] or not advantage["sqrt_n_rescale"]:
        raise SystemExit("Effective cumulative IG or sqrt(valid_count) is inactive")
    if float(advantage["lambda_outcome"]) != 1.0:
        raise SystemExit("Answer lambda_outcome is not 1.0")
    if float(advantage["lambda_format"]) != 1.0:
        raise SystemExit("Answer lambda_format is not 1.0")
    expected_sc = {
        "enabled": False,
        "shadow_only": True,
        "actor_loss_enabled": False,
    }
    if advantage["sc"] != expected_sc:
        raise SystemExit("Stop/Continue config lock differs from production V1")

    active_paths = [
        path
        for root_name in ("src", "configs")
        for path in (PROJECT_ROOT / root_name).rglob("*")
        if path.suffix in {".py", ".yaml", ".yml"}
    ]
    malformed_paths = [
        PROJECT_ROOT / "src/agentic_rl/advantage/a2tgpo.py",
        PROJECT_ROOT / "configs/base.yaml",
        PROJECT_ROOT / "configs/update_stages.yaml",
    ]
    malformed_findings = _scan(
        malformed_paths,
        (
            "A_" + "mal",
            "lambda_" + "mal",
            "malformed_" + "advantage",
            "malformed_search_" + "reward",
            "malformed_search_" + "penalty",
        ),
    )
    fixed_clip_findings = _scan(
        active_paths,
        (
            "fixed_" + "dapo",
            "dapo_" + "turn_objective",
            "clip_ratio_" + "low",
            "clip_ratio_" + "high",
            "0." + "8",
            "1." + "28",
        ),
    )
    if malformed_findings:
        raise SystemExit(
            "Forbidden Search optimization fields: " + ", ".join(malformed_findings)
        )
    if fixed_clip_findings:
        raise SystemExit(
            "Forbidden fixed clipping fields: " + ", ".join(fixed_clip_findings)
        )

    clipping_values = {
        "mode": policy["clipping_mode"],
        "beta": policy["adaptive_clip_beta"],
        "epsilon_low": policy["adaptive_clip_epsilon_low"],
        "epsilon_high": policy["adaptive_clip_epsilon_high"],
        "answer_scale": policy["answer_clip_scale"],
    }
    expected_clipping = {
        "mode": CLIPPING_MODE,
        "beta": ADAPTIVE_CLIP_BETA,
        "epsilon_low": ADAPTIVE_CLIP_EPSILON_LOW,
        "epsilon_high": ADAPTIVE_CLIP_EPSILON_HIGH,
        "answer_scale": ANSWER_CLIP_SCALE,
    }
    if clipping_values != expected_clipping:
        raise SystemExit(
            f"Config/source adaptive clipping mismatch: "
            f"{clipping_values} != {expected_clipping}"
        )

    print(
        json.dumps(
            {
                "adaptive_clipping": clipping_values,
                "answer_advantage_terms": advantage["answer_formula_terms"],
                "fixed_clipping_scan": "PASS",
                "malformed_optimization_scan": "PASS",
                "search_advantage_terms": advantage["search_formula_terms"],
                "search_advantage_coefficients": {
                    "external_ig_multiplier": advantage[
                        "external_ig_multiplier"
                    ],
                },
                "answer_advantage_coefficients": {
                    "lambda_outcome": advantage["lambda_outcome"],
                    "lambda_format": advantage["lambda_format"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
