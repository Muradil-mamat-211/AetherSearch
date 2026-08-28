from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = PROJECT_ROOT / "sft" / "scripts" / "train_sft_2000.py"
SPEC = importlib.util.spec_from_file_location("aethersearch_sft_2000_trainer", TRAINER_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAINER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAINER
SPEC.loader.exec_module(TRAINER)


class CharacterTokenizer:
    """Small reversible tokenizer used to test policy boundaries without a download."""

    eos_token = TRAINER.IM_END
    eos_token_id = 0x110001
    pad_token_id = 0
    is_fast = True
    backend_tokenizer = SimpleNamespace(normalizer=None)

    @staticmethod
    def _encode(text: str) -> list[int]:
        result: list[int] = []
        cursor = 0
        while cursor < len(text):
            if text.startswith(TRAINER.IM_END, cursor):
                result.append(CharacterTokenizer.eos_token_id)
                cursor += len(TRAINER.IM_END)
            else:
                result.append(ord(text[cursor]) + 1)
                cursor += 1
        return result

    def __call__(self, text, **_kwargs):
        if isinstance(text, list):
            return SimpleNamespace(input_ids=[self._encode(item) for item in text])
        return SimpleNamespace(input_ids=self._encode(text))

    def decode(self, ids, **_kwargs) -> str:
        pieces: list[str] = []
        for token_id in ids:
            if token_id == self.eos_token_id:
                pieces.append(TRAINER.IM_END)
            else:
                pieces.append(chr(token_id - 1))
        return "".join(pieces)


def canonical_record() -> dict:
    question = "Who wrote the example?"
    text = (
        f"{TRAINER.SYSTEM_MARKER}System instructions.{TRAINER.IM_END}\n"
        f"{TRAINER.USER_MARKER}Answer with search. Question: {question}"
        f"{TRAINER.IM_END}\n{TRAINER.ASSISTANT_MARKER}"
        "<think>I need evidence.</think>"
        "<search>example author</search>"
        "<information>Retrieved evidence.</information>"
        "<think>The evidence is sufficient.</think>"
        f"<answer>Ada</answer>{TRAINER.IM_END}"
    )
    return {
        "id": "000001",
        "question": question,
        "trajectory_type": "single_search",
        "search_count": 1,
        "full_trajectory_text": text,
    }


def test_record_layout_and_exact_mask_segments() -> None:
    record = canonical_record()
    text = record["full_trajectory_text"]
    layout = TRAINER.validate_public_record(record, line_number=1, strict_public_schema=True)
    segments = TRAINER.build_mask_segments(text, layout, line_number=1)

    assert [kind for _, kind in segments] == [
        "prompt",
        "supervised",
        "information",
        "supervised",
    ]
    assert "".join(segment for segment, _ in segments) == text
    assert segments[2][0] == "<information>Retrieved evidence.</information>"
    assert segments[-1][0].endswith(f"</answer>{TRAINER.IM_END}")


def test_dataset_masks_prompt_information_padding_and_supervises_eot(
    tmp_path: Path,
) -> None:
    record = canonical_record()
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    tokenizer = CharacterTokenizer()
    dataset = TRAINER.SearchSFT2000Dataset(
        train_file=str(train_file),
        tokenizer=tokenizer,
        max_seq_len=4096,
        expected_num_samples=1,
        tokenization_batch_size=1,
    )

    sample = dataset[0]
    layout = TRAINER.validate_public_record(record, line_number=1, strict_public_schema=True)
    expected_labels: list[int] = []
    for segment, kind in TRAINER.build_mask_segments(
        record["full_trajectory_text"], layout, line_number=1
    ):
        segment_ids = tokenizer._encode(segment)
        expected_labels.extend(
            segment_ids
            if kind == "supervised"
            else [TRAINER.IGNORE_INDEX] * len(segment_ids)
        )

    assert sample["labels"] == expected_labels
    assert sample["labels"][-1] == tokenizer.eos_token_id
    assert dataset.audit.prompt_masked_tokens["sum"] > 0
    assert dataset.audit.information_masked_tokens["sum"] > 0
    assert dataset.audit.supervised_tokens["sum"] > 0

    batch = TRAINER.SearchSFT2000Collator(tokenizer)(
        [sample, {"input_ids": [11, 12], "labels": [11, 12]}]
    )
    assert batch["input_ids"].shape[0] == 2
    assert torch.all(batch["attention_mask"][1, 2:] == 0)
    assert torch.all(batch["labels"][1, 2:] == TRAINER.IGNORE_INDEX)


def test_structural_corruption_is_rejected() -> None:
    record = canonical_record()
    record["full_trajectory_text"] = record["full_trajectory_text"].replace(
        "</search><information>", "</search> gap <information>"
    )
    with pytest.raises(ValueError, match="not directly adjacent"):
        TRAINER.validate_public_record(record, line_number=9, strict_public_schema=True)


def test_cli_validation_rejects_unsafe_values(tmp_path: Path) -> None:
    train_file = tmp_path / "train.jsonl"
    train_file.write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(
        train_file=str(train_file),
        deepspeed=None,
        max_seq_len=0,
        expected_num_samples=2000,
        expected_sha256=None,
        tokenization_batch_size=64,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1.0,
        max_steps=-1,
        learning_rate=2e-6,
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=5,
        save_steps=100,
        save_total_limit=1,
        dataloader_num_workers=2,
    )
    with pytest.raises(ValueError, match="max_seq_len"):
        TRAINER.validate_cli_args(args)
