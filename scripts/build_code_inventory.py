from __future__ import annotations

import hashlib
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "CODE_INVENTORY.md"
ROOT_DOCUMENTS = {
    "EXTERNAL_ASSETS.md",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "TRAINING_REPRODUCTION.md",
}


def release_files() -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    paths = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = Path(raw_path.decode("utf-8"))
        if relative != OUTPUT.relative_to(PROJECT_ROOT):
            paths.append(relative)
    return sorted(paths, key=lambda path: path.as_posix())


def category(path: Path) -> str:
    if len(path.parts) > 1:
        return path.parts[0]
    if path.name in ROOT_DOCUMENTS:
        return "documentation"
    return "repo_root"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    groups: dict[str, list[tuple[Path, int, str]]] = defaultdict(list)
    for relative in release_files():
        absolute = PROJECT_ROOT / relative
        groups[category(relative)].append(
            (relative, absolute.stat().st_size, sha256(absolute))
        )

    lines = [
        "# Code Inventory",
        "",
        "This inventory covers every non-ignored file in the AetherSearch GitHub",
        "release. Model weights, optimizer checkpoints, eval result bundles, report",
        "archives, and runtime snapshots are not included.",
        "",
        f"Generated UTC: `{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}`",
        "",
        "## Summary",
        "",
    ]
    for name in sorted(groups):
        rows = groups[name]
        lines.append(
            f"- `{name}`: `{len(rows)}` files, "
            f"`{sum(size for _, size, _ in rows)}` bytes"
        )

    lines.extend(["", "## Files", ""])
    for name in sorted(groups):
        lines.extend(
            [
                f"### {name}",
                "",
                "| path | bytes | sha256 |",
                "|---|---:|---|",
            ]
        )
        for relative, size, digest in groups[name]:
            lines.append(f"| `{relative.as_posix()}` | {size} | `{digest}` |")
        lines.append("")

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
