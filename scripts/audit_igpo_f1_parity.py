#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import string
import urllib.request
from pathlib import Path

from agentic_rl.outcome.token_f1 import (
    IGPO_OFFICIAL_COMMIT,
    IGPO_OFFICIAL_SOURCE,
    compute_f1 as project_compute_f1,
)


RAW_URL = (
    "https://raw.githubusercontent.com/GuoqingWang1/IGPO/"
    f"{IGPO_OFFICIAL_COMMIT}/{IGPO_OFFICIAL_SOURCE}"
)
FUNCTIONS = {
    "check_tags_balance",
    "preprocess_text",
    "deal_multi_labels",
    "compute_f1",
}


def load_official_compute_f1():
    with urllib.request.urlopen(RAW_URL, timeout=30) as response:
        source = response.read().decode("utf-8")
    module = ast.parse(source, filename=RAW_URL)
    selected = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in FUNCTIONS
    ]
    namespace = {"re": re, "string": string, "json": json}
    exec(compile(ast.Module(body=selected, type_ignores=[]), RAW_URL, "exec"), namespace)
    return namespace["compute_f1"]


def fuzz_cases(count: int, seed: int):
    generator = random.Random(seed)
    words = ["Paris", "the", "New", "York", "a", "b", "U.S.A", ""]
    punctuation = ["", "-", ",", ".", "!", "  ", "\n"]
    for _ in range(count):
        prediction = generator.choice(words) + generator.choice(punctuation)
        prediction += generator.choice(words)
        aliases = [
            generator.choice(words) + generator.choice(punctuation)
            + generator.choice(words)
            for _ in range(generator.randint(1, 3))
        ]
        yield (
            f"<think>x</think><answer>{prediction}</answer>",
            "<|answer_split|>".join(aliases),
            "",
            generator.choice(["f1", "em", "noformatf1"]),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output")
    args = parser.parse_args()
    official = load_official_compute_f1()
    mismatches = []
    for index, case in enumerate(fuzz_cases(args.cases, args.seed)):
        project = project_compute_f1(*case)
        expected = official(*case)
        if abs(float(project) - float(expected)) > 1.0e-12:
            mismatches.append(
                {
                    "index": index,
                    "case": case,
                    "project": project,
                    "official": expected,
                }
            )
    result = {
        "repository": "https://github.com/GuoqingWang1/IGPO",
        "commit": IGPO_OFFICIAL_COMMIT,
        "source": IGPO_OFFICIAL_SOURCE,
        "cases": args.cases,
        "seed": args.seed,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
