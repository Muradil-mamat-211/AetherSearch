"""External model/data/retriever asset manifest support."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml


_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class AssetManifestError(ValueError):
    """Raised when a declared external asset is missing or has drifted."""


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if not isinstance(value, str):
        return value
    names = _ENV_REFERENCE.findall(value)
    missing = sorted({name for name in names if name not in os.environ})
    if missing:
        raise AssetManifestError(
            "Missing asset manifest environment variables: " + ", ".join(missing)
        )
    return os.path.expanduser(os.path.expandvars(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    root = Path(path)
    if root.is_file():
        return sha256_file(root)
    digest = hashlib.sha256()
    for item in sorted(value for value in root.rglob("*") if value.is_file()):
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def tokenizer_sha256(model_root: str | Path) -> str:
    digest = hashlib.sha256()
    root = Path(model_root)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        path = root / name
        if path.is_file():
            digest.update(path.name.encode("utf-8"))
            digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def load_asset_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise AssetManifestError(f"Asset manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise AssetManifestError("Asset manifest root must be a mapping")
    assets = value.get("assets")
    if not isinstance(assets, Mapping) or not assets:
        raise AssetManifestError("Asset manifest must contain a non-empty assets mapping")
    result = dict(_expand(dict(value)))
    result["_path"] = str(manifest_path)
    return result


def validate_asset_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_asset_manifest(path)
    checked: dict[str, Any] = {}
    for name, raw in manifest["assets"].items():
        if not isinstance(raw, Mapping):
            raise AssetManifestError(f"assets.{name} must be a mapping")
        kind = str(raw.get("kind", "file"))
        asset_path = Path(str(raw.get("path", ""))).expanduser().resolve()
        expected = str(raw.get("sha256", ""))
        if len(expected) != 64:
            raise AssetManifestError(f"assets.{name}.sha256 must be SHA-256")
        if not asset_path.exists():
            raise AssetManifestError(f"Asset does not exist: assets.{name}={asset_path}")
        if kind == "file":
            actual = sha256_file(asset_path)
        elif kind == "tree":
            actual = sha256_tree(asset_path)
        elif kind == "tokenizer":
            actual = tokenizer_sha256(asset_path)
        else:
            raise AssetManifestError(f"Unsupported asset kind for {name}: {kind}")
        if actual != expected:
            raise AssetManifestError(
                f"Asset checksum changed for {name}: expected={expected} actual={actual}"
            )
        checked_entry: dict[str, Any] = {
            "kind": kind,
            "path": str(asset_path),
            "sha256": actual,
        }
        for metadata_key in (
            "manifest_sha256",
            "expected_row_count",
            "expected_source_counts",
        ):
            if metadata_key in raw:
                checked_entry[metadata_key] = raw[metadata_key]
        checked[str(name)] = checked_entry
    return {
        "status": "PASS",
        "manifest": str(manifest["_path"]),
        "assets": checked,
    }
