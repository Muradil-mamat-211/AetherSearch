"""Student-token budgets and the actual observation shown to the teacher."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

WORKSPACE = Path(os.environ.get("AETHERSEARCH_SFT_WORKSPACE", str(Path(__file__).resolve().parents[3]))).expanduser().resolve()
STUDENT_TOKENIZER = Path(os.environ.get("AETHERSEARCH_STUDENT_TOKENIZER_PATH", str(WORKSPACE / "models/qwen2p5_3b_instruct_search_dpo_v1_from_dpo_v0_4493_lr5e7"))).expanduser().resolve()
TOKENIZER_DEPS = Path(os.environ.get("AETHERSEARCH_TOKENIZER_DEPS_PATH", str(WORKSPACE / "envs/teacher_tokenizer_deps"))).expanduser().resolve()
MAX_SEARCH_TURNS = 5
MAX_ACTION_TOKENS = 500
MAX_INFORMATION_TOKENS = 500
BUDGET_VERSION = "qwen_student_500_v1"
INFORMATION_POLICY = "rl_prefix_truncation_top3_required_v1"
INFO_PATTERN = re.compile(
    r'<information>\n?Doc 1\(Title: "([^"\n]+)"\) ([^\n]+)\n'
    r'Doc 2\(Title: "([^"\n]+)"\) ([^\n]+)\n'
    r'Doc 3\(Title: "([^"\n]+)"\) ([^\n]+)\n?</information>'
)


class BudgetError(ValueError):
    """An action or observation cannot be used within the student budget."""


class StudentBudget:
    def __init__(self, path: Path = STUDENT_TOKENIZER):
        if TOKENIZER_DEPS.is_dir():
            sys.path.insert(0, str(TOKENIZER_DEPS))
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise RuntimeError("student_tokenizer_dependency_missing") from None
        self.path = path.resolve()
        tokenizer_file = self.path / "tokenizer.json"
        config_file = self.path / "tokenizer_config.json"
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        config = json.loads(config_file.read_text(encoding="utf-8"))
        eos = config.get("eos_token")
        eos = eos.get("content") if isinstance(eos, dict) else eos
        if not isinstance(eos, str) or self.tokenizer.token_to_id(eos) is None:
            raise RuntimeError("student_eos_token_missing")
        self.eos_reserve = len(self.encode(eos))
        self.identity = {
            "path": str(self.path),
            "tokenizer_sha256": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
            "eos_token": eos,
        }

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def action_tokens(self, text: str) -> int:
        # Reserve the student's end-of-turn token, which is absent from API text.
        return self.count(text) + self.eos_reserve

    def check_action(self, text: str, kind: str) -> int:
        count = self.action_tokens(text)
        if count > MAX_ACTION_TOKENS:
            raise BudgetError(f"{kind}_action_token_budget_exceeded")
        return count

    def specification(self) -> dict[str, Any]:
        return {"version": BUDGET_VERSION, "student_tokenizer": dict(self.identity),
                "max_search_turns": MAX_SEARCH_TURNS,
                "max_model_tokens_per_turn": MAX_ACTION_TOKENS,
                "max_new_tokens_per_turn": MAX_ACTION_TOKENS,
                "max_information_tokens_per_turn": MAX_INFORMATION_TOKENS,
                "action_eos_reserve": self.eos_reserve,
                "information_policy": INFORMATION_POLICY}

    def information(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        if len(documents) != 3:
            raise BudgetError("information_requires_top3")
        body = "\n".join(
            f'Doc {n}(Title: "{doc["title"].replace(chr(34), chr(39))}") {doc["text"]}'
            for n, doc in enumerate(documents, 1)
        )
        if any("<" in d["title"] + d["text"] or ">" in d["title"] + d["text"] for d in documents):
            raise BudgetError("unsafe_evidence_markup")
        prefix, suffix = self.encode("<information>"), self.encode("</information>")
        body_ids = self.encode(body)
        available = MAX_INFORMATION_TOKENS - len(prefix) - len(suffix)
        ids = prefix + body_ids[:available] + suffix
        rendered = self.decode(ids)
        match = INFO_PATTERN.fullmatch(rendered)
        # Match RL's contiguous prefix clipping. Reject rather than manufacture a
        # third document if that clipping removes its title or all of its text.
        if not match:
            raise BudgetError("information_missing_top3_after_truncation")
        visible = [{"doc_id": doc["doc_id"], "title": title, "text": text}
                   for doc, (title, text) in zip(documents, zip(match.groups()[::2], match.groups()[1::2]))]
        if self.count(rendered) > MAX_INFORMATION_TOKENS:
            raise BudgetError("information_retokenization_exceeds_budget")
        return {"information": rendered, "information_token_ids": ids,
                "information_tokens": self.count(rendered), "information_injected_tokens": len(ids),
                "visible_documents": visible,
                "information_truncated": len(body_ids) > available}


@lru_cache(maxsize=1)
def student_budget() -> StudentBudget:
    return StudentBudget()
