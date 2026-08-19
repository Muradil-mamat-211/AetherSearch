from __future__ import annotations

import argparse

from agentic_rl.config import DEFAULT_CONFIG, load_config

from .verl_runtime_adapter import create_runtime_adapter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed staged runtime entry point"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    create_runtime_adapter(load_config(args.config)).run()


if __name__ == "__main__":
    main()
