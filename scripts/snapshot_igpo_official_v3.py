from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.request
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "exact_ig_official_alignment_v3_20260730"
)
REPOSITORY = "https://github.com/GuoqingWang1/IGPO"
PINNED_COMMIT = "64165e2741ed8801f977948c8128080ce87b4101"
FILES = (
    "scrl/llm_agent/vectorized_gt_logprob.py",
    "scrl/llm_agent/generation.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["git", "ls-remote", REPOSITORY + ".git", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    head = completed.stdout.split()[0]
    files = {}
    for relative in FILES:
        url = (
            "https://raw.githubusercontent.com/GuoqingWang1/IGPO/"
            + head
            + "/"
            + relative
        )
        content = urllib.request.urlopen(url, timeout=60).read()
        destination = output / "official_source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        files[relative] = {
            "url": url,
            "sha256": hashlib.sha256(content).hexdigest(),
            "snapshot_path": str(destination),
            "byte_count": len(content),
        }
    payload = {
        "official_repository": REPOSITORY,
        "current_head_commit_sha": head,
        "previous_pinned_commit_sha": PINNED_COMMIT,
        "head_matches_previous_pin": head == PINNED_COMMIT,
        "audit_date": date.today().isoformat(),
        "files": files,
        "functions_reviewed": {
            "scrl/llm_agent/vectorized_gt_logprob.py": [
                "tokenize_ground_truth",
                "get_gt_answer_token_range",
                "build_extended_sequence",
                "build_extended_attention_mask",
                "build_extended_position_ids",
                "compute_all_turns_vectorized",
                "compute_all_turns_sequential",
                "validate_vectorized_vs_sequential",
                "compute_info_gain_rewards",
            ],
            "scrl/llm_agent/generation.py": ["run_llm_loop"],
        },
    }
    (output / "EXACT_IG_OFFICIAL_SOURCE_SNAPSHOT.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
