#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEARCH_R1_ROOT = Path("/root/autodl-tmp/search-r1-workspace/code/Search-R1")
MODEL = Path(
    "/root/autodl-tmp/search-r1-workspace/models/dpo_v2_final_model"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(SEARCH_R1_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    import torch
    from torch.distributed.fsdp import FSDPModule, fully_shard

    paths = {
        "actor_model": MODEL,
        "train_data": Path(
            "/root/autodl-tmp/search-r1-workspace/data/nq_search/train.parquet"
        ),
        "validation_data": Path(
            "/root/autodl-tmp/search-r1-workspace/data/nq_search/test.parquet"
        ),
        "retriever_server": Path(
            "/root/autodl-tmp/search-r1-workspace/projects/"
            "igpo_ragen2_a2tgpo_strict_onpolicy_v1/runtime_assets/retriever/"
            "hybrid_retrieval_server.py"
        ),
        "retriever_config": Path(
            "/root/autodl-tmp/search-r1-workspace/projects/"
            "igpo_ragen2_a2tgpo_strict_onpolicy_v1/runtime_assets/retriever/"
            "retriever.yaml"
        ),
        "dense_index": Path(
            "/root/autodl-tmp/search-r1-workspace/data/nq_search/e5_Flat.index"
        ),
        "corpus": Path(
            "/root/autodl-tmp/search-r1-workspace/data/wiki18_corpus/wiki-18.jsonl"
        ),
        "bm25_index": Path(
            "/root/autodl-tmp/search-r1-workspace/data/wiki18_bm25/bm25"
        ),
        "dense_encoder": Path(
            "/root/autodl-tmp/search-r1-workspace/models/e5-base-v2"
        ),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise SystemExit("Missing required paths: " + ", ".join(missing))

    small_file_hashes = {
        name: sha256(paths[name])
        for name in (
            "train_data",
            "validation_data",
            "retriever_server",
            "retriever_config",
        )
    }
    official_root = (
        PROJECT_ROOT
        / "third_party"
        / "igpo_official_64165e2741ed8801f977948c8128080ce87b4101"
    )
    official_sources = {
        "vectorized_gt_logprob.py": official_root
        / "scrl/llm_agent/vectorized_gt_logprob.py",
        "prealigned_vectorized.py": official_root
        / "scrl/llm_agent/prealigned_vectorized.py",
        "generation.py": official_root / "scrl/llm_agent/generation.py",
        "LICENSE": official_root / "LICENSE",
    }
    output = {
        "project_root": str(PROJECT_ROOT),
        "required_paths": {
            name: {"path": str(path), "exists": path.exists()}
            for name, path in paths.items()
        },
        "small_file_sha256": small_file_hashes,
        "search_r1_commit": git_output("rev-parse", "HEAD"),
        "search_r1_status_before_and_after_project_only_work": git_output(
            "status", "--short"
        ).splitlines(),
        "framework": {
            "python": __import__("platform").python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nccl": list(torch.cuda.nccl.version()),
            "transformers": package_version("transformers"),
            "verl": package_version("verl"),
            "ray": package_version("ray"),
            "vllm": package_version("vllm"),
            "flash_attention": package_version("flash-attn"),
            "xformers": package_version("xformers"),
            "triton": package_version("triton"),
            "fsdp2_fully_shard_signature": str(inspect.signature(fully_shard)),
            "fsdp2_reshard_api": hasattr(
                FSDPModule,
                "set_reshard_after_forward",
            ),
        },
        "official_igpo": {
            "repository": "https://github.com/GuoqingWang1/IGPO",
            "commit": "64165e2741ed8801f977948c8128080ce87b4101",
            "source_sha256": {
                name: sha256(path) for name, path in official_sources.items()
            },
        },
        "project_symlinks": [
            str(path.relative_to(PROJECT_ROOT))
            for path in PROJECT_ROOT.rglob("*")
            if path.is_symlink()
        ],
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
