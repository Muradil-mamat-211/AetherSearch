#!/usr/bin/env python3
"""Validate README text integrity and repository-local references."""

from __future__ import annotations

import argparse
import html
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path


FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(?:[^`]*)?$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
LINK_RE = re.compile(r"!?\[[^\]]*\]\((<[^>]+>|[^)\s]+)(?:\s+[^)]*)?\)")
HTML_LINK_RE = re.compile(r"\b(?:src|href)=[\"']([^\"']+)[\"']", re.IGNORECASE)
DIV_RE = re.compile(r"</?div\b[^>]*>", re.IGNORECASE)
CONFLICT_RE = re.compile(r"^(?:<<<<<<< |=======\s*$|>>>>>>> )", re.MULTILINE)
TABLE_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")


def github_slug(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = html.unescape(text).strip().lower()
    text = "".join(char for char in text if char.isalnum() or char in " -_")
    return re.sub(r"\s+", "-", text)


def split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith(r"\|"):
        stripped = stripped[:-1]

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    code_ticks = 0
    for char in stripped:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if char == "`":
            code_ticks ^= 1
            current.append(char)
            continue
        if char == "|" and not code_ticks:
            cells.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    cells.append("".join(current).strip())
    return cells


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        text = path.read_bytes().decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        return [f"strict UTF-8 decode failed: {exc}"]

    if "\ufffd" in text:
        errors.append("contains U+FFFD replacement character")
    controls = sorted({ord(char) for char in text if ord(char) < 32 and char not in "\n\r\t"})
    if controls:
        errors.append(f"contains forbidden C0 control characters: {controls}")
    if CONFLICT_RE.search(text):
        errors.append("contains a Git merge conflict marker")

    lines = text.splitlines()
    in_fence = False
    fence_char = ""
    fence_len = 0
    outside_fence: list[tuple[int, str]] = []
    for line_number, line in enumerate(lines, start=1):
        match = FENCE_RE.match(line)
        if match:
            marker = match.group(1)
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
                fence_len = len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_len:
                in_fence = False
                fence_char = ""
                fence_len = 0
            continue
        if not in_fence:
            outside_fence.append((line_number, line))
    if in_fence:
        errors.append("contains an unclosed fenced code block")

    div_depth = 0
    for match in DIV_RE.finditer("\n".join(line for _, line in outside_fence)):
        if match.group(0).lower().startswith("</"):
            div_depth -= 1
            if div_depth < 0:
                errors.append("contains a closing </div> without a matching <div>")
                div_depth = 0
        else:
            div_depth += 1
    if div_depth:
        errors.append(f"contains {div_depth} unclosed <div> element(s)")

    for line_number, line in outside_fence:
        runs = re.findall(r"(?<!\\)(`+)", line)
        pending: str | None = None
        for run in runs:
            if pending is None:
                pending = run
            elif run == pending:
                pending = None
            else:
                errors.append(
                    f"line {line_number} has mismatched inline backtick delimiters"
                )
                pending = None
                break
        if pending is not None:
            errors.append(f"line {line_number} has an unclosed inline code span")

    slugs: set[str] = set()
    slug_counts: dict[str, int] = {}
    for _, line in outside_fence:
        heading = HEADING_RE.match(line)
        if not heading:
            continue
        base = github_slug(heading.group(2))
        count = slug_counts.get(base, 0)
        slug_counts[base] = count + 1
        slugs.add(base if count == 0 else f"{base}-{count}")

    references: list[str] = []
    outside_text = "\n".join(line for _, line in outside_fence)
    references.extend(match.group(1) for match in LINK_RE.finditer(outside_text))
    references.extend(match.group(1) for match in HTML_LINK_RE.finditer(outside_text))
    for raw_reference in references:
        reference = html.unescape(raw_reference.strip("<>"))
        if reference.startswith("#"):
            anchor = urllib.parse.unquote(reference[1:]).lower()
            if anchor not in slugs:
                errors.append(f"missing README anchor: {reference}")
            continue
        parsed = urllib.parse.urlsplit(reference)
        if parsed.scheme or reference.startswith("//"):
            continue
        relative_path = urllib.parse.unquote(parsed.path)
        if not relative_path:
            continue
        target = (path.parent / relative_path).resolve()
        if not target.exists():
            errors.append(f"missing repository-local link target: {reference}")
            continue
        if target.suffix.lower() == ".svg":
            try:
                ET.parse(target)
            except ET.ParseError as exc:
                errors.append(f"invalid SVG/XML target {reference}: {exc}")

    table_index = 0
    while table_index + 1 < len(outside_fence):
        _, header = outside_fence[table_index]
        _, separator = outside_fence[table_index + 1]
        if "|" not in header or "|" not in separator:
            table_index += 1
            continue
        separator_cells = split_table_row(separator)
        if not separator_cells or not all(
            TABLE_SEPARATOR_RE.fullmatch(cell) for cell in separator_cells
        ):
            table_index += 1
            continue
        expected = len(split_table_row(header))
        if len(separator_cells) != expected:
            errors.append(
                f"table near line {outside_fence[table_index][0]} has a malformed separator"
            )
        row_index = table_index + 2
        while row_index < len(outside_fence) and "|" in outside_fence[row_index][1]:
            line_number, row = outside_fence[row_index]
            if len(split_table_row(row)) != expected:
                errors.append(f"table row at line {line_number} has the wrong column count")
            row_index += 1
        table_index = row_index

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "README.md",
    )
    args = parser.parse_args()
    errors = validate(args.path.resolve())
    if errors:
        for error in errors:
            print(f"README validation failed: {error}", file=sys.stderr)
        return 1
    print(f"README validation passed: {args.path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
