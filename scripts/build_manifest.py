from __future__ import annotations

import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_FILES = {"FILE_MANIFEST.md", "MANIFEST.sha256"}
EXCLUDED_PARTS = {
    ".pytest_cache",
    "__pycache__",
    "outputs",
    "runtime",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_files() -> list[Path]:
    files: list[Path] = []
    for path in PROJECT_ROOT.rglob("*"):
        relative = path.relative_to(PROJECT_ROOT)
        if not path.is_file():
            continue
        if relative.as_posix() in EXCLUDED_FILES:
            continue
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        files.append(relative)
    return sorted(files, key=lambda item: item.as_posix())


def main() -> None:
    rows = []
    for relative in project_files():
        absolute = PROJECT_ROOT / relative
        rows.append((relative.as_posix(), absolute.stat().st_size, sha256_file(absolute)))

    checksum_text = "".join(
        f"{digest}  {relative}\n" for relative, _, digest in rows
    )
    (PROJECT_ROOT / "MANIFEST.sha256").write_text(
        checksum_text,
        encoding="utf-8",
    )

    tree = "\n".join(relative for relative, _, _ in rows)
    table = "\n".join(
        f"| `{relative}` | {size} | `{digest}` |"
        for relative, size, digest in rows
    )
    manifest = f"""# File Manifest

Generated deterministically from the isolated project tree.

`FILE_MANIFEST.md`, `MANIFEST.sha256`, runtime outputs, and Python/test caches
are excluded from the checksummed set to avoid self-reference and generated
state.

## Complete Regular-File Tree

```text
{tree}
```

## SHA-256 Inventory

| File | Bytes | SHA-256 |
|---|---:|---|
{table}
"""
    (PROJECT_ROOT / "FILE_MANIFEST.md").write_text(manifest, encoding="utf-8")


if __name__ == "__main__":
    main()
