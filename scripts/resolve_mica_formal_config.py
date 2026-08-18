from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from agentic_rl.config import load_config, validate_config
from agentic_rl.runtime.verl_config import assert_formal_hyperparameters_approved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-successful-updates", type=int, required=True)
    args = parser.parse_args()

    config = load_config(args.input)
    total = int(args.total_successful_updates)
    config["formal"]["total_successful_updates"] = total
    config["formal_schedule"]["total_successful_updates"] = total
    config["scheduler"]["total_successful_updates"] = total
    if total != 500:
        raise SystemExit("The locked fresh MICA experiment must target U500")
    if config["formal"].get("fresh_start_required") is not True:
        raise SystemExit("Formal config does not require a fresh start")
    if int(config["formal"].get("resume_from_successful_update", -1)) != 0:
        raise SystemExit("Formal config is not locked to successful_update=0")
    validate_config(config)
    assert_formal_hyperparameters_approved(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


if __name__ == "__main__":
    main()
