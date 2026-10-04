from __future__ import annotations

import importlib.util
from pathlib import Path


VALIDATOR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_readme.py"
SPEC = importlib.util.spec_from_file_location("aethersearch_readme_validator", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def test_explicit_anchor_supports_navigation_to_emoji_heading(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        '[Workflow](#dpo-data-construction-workflow)\n\n'
        '<a id="dpo-data-construction-workflow"></a>\n\n'
        '## 🛠️ DPO Data Construction Workflow\n',
        encoding="utf-8",
    )
    assert VALIDATOR.validate(readme) == []


def test_explicit_anchor_does_not_hide_a_missing_link_target(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        '[Missing](#missing-workflow)\n\n<a id="existing-workflow"></a>\n',
        encoding="utf-8",
    )
    assert VALIDATOR.validate(readme) == ["missing Markdown anchor: #missing-workflow"]


def test_anchor_in_fenced_example_is_not_a_navigation_target(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        '[Example](#example-workflow)\n\n```html\n'
        '<a id="example-workflow"></a>\n```\n',
        encoding="utf-8",
    )
    assert VALIDATOR.validate(readme) == ["missing Markdown anchor: #example-workflow"]
