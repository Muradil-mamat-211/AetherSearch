#!/usr/bin/env python3
"""Strict preference trainer for the canonical AetherSearch DPO dataset."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import unicodedata
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


IGNORE_INDEX = -100
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
SYSTEM_MARKER = f"{IM_START}system\n"
USER_MARKER = f"{IM_START}user\n"
ASSISTANT_MARKER = f"{IM_START}assistant\n"
PUBLIC_FIELD_ORDER = (
    "id",
    "question",
    "source_dataset",
    "answers",
    "pair_type",
    "prompt_text",
    "chosen",
    "rejected",
)
PUBLIC_FIELDS = set(PUBLIC_FIELD_ORDER)
CANONICAL_DATA_SHA256 = (
    "c42adcb0f194cff3126134b37afd85e4b89aa9917e5c98dda4b09904509f61e9"
)
CANONICAL_PAIR_COUNT = 2126

STRUCTURAL_TAG_RE = re.compile(r"</?(?:think|search|information|answer)>")
COMPLETE_SPAN_RE = re.compile(
    r"<(think|search|information|answer)>(.*?)</\1>", flags=re.S
)


def rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def rank0_print(*args, **kwargs) -> None:
    if rank0():
        print(*args, **kwargs, flush=True)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_question(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = " ".join(normalized.split()).strip()
    return re.sub(r"[?？]+$", "", normalized).strip()


def percentile(values: Sequence[int], probability: float) -> int:
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(len(ordered) * probability) - 1),
    )
    return ordered[index]


def distribution(values: Sequence[int]) -> Dict[str, int]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "sum": sum(values),
    }


def validate_cli_args(args: argparse.Namespace) -> None:
    """Reject ambiguous or unsafe settings before model allocation."""
    if not os.path.isfile(args.train_file):
        raise FileNotFoundError(f"training JSONL does not exist: {args.train_file}")
    if args.deepspeed and not os.path.isfile(args.deepspeed):
        raise FileNotFoundError(f"DeepSpeed config does not exist: {args.deepspeed}")
    if args.max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if args.expected_num_samples < 0:
        raise ValueError("expected_num_samples must be non-negative")
    if args.expected_sha256 and not re.fullmatch(
        r"[0-9a-fA-F]{64}", args.expected_sha256
    ):
        raise ValueError("expected_sha256 must contain 64 hexadecimal characters")
    if args.tokenization_batch_size <= 0:
        raise ValueError("tokenization_batch_size must be positive")
    if args.beta <= 0:
        raise ValueError("beta must be positive")
    if args.per_device_train_batch_size <= 0:
        raise ValueError("per_device_train_batch_size must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if args.num_train_epochs <= 0:
        raise ValueError("num_train_epochs must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("max_steps must be -1 or a positive integer")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be between zero and one")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if args.logging_steps <= 0:
        raise ValueError("logging_steps must be positive")
    if args.save_steps <= 0:
        raise ValueError("save_steps must be positive")
    if args.save_total_limit <= 0:
        raise ValueError("save_total_limit must be positive")
    if args.dataloader_num_workers < 0:
        raise ValueError("dataloader_num_workers must be non-negative")


@dataclass(frozen=True)
class ChoiceLayout:
    terminal: str
    information_spans: List[Tuple[int, int]]


@dataclass
class DatasetAudit:
    source_file: str
    source_sha256: str
    input_pairs: int
    kept_pairs: int
    filtered_overlength_pairs: int
    unique_ids: int
    unique_normalized_questions: int
    decoded_roundtrip_choices: int
    tokenizer_normalized_choices: int
    masking_strategy: str
    source_distribution: Dict[str, int]
    pair_type_distribution: Dict[str, int]
    terminal_distribution: Dict[str, Dict[str, int]]
    information_blocks: Dict[str, int]
    total_tokens: Dict[str, Dict[str, int]]
    prompt_masked_tokens: Dict[str, Dict[str, int]]
    information_masked_tokens: Dict[str, Dict[str, int]]
    supervised_tokens: Dict[str, Dict[str, int]]


def validate_prompt(prompt: str, question: str, line_number: int) -> None:
    if not prompt.startswith(SYSTEM_MARKER):
        raise ValueError(f"line {line_number}: prompt must begin with the system marker")
    if prompt.count(IM_START) != 3 or prompt.count(IM_END) != 2:
        raise ValueError(
            f"line {line_number}: prompt must contain three ChatML starts and two ends"
        )
    if prompt.count(SYSTEM_MARKER) != 1:
        raise ValueError(f"line {line_number}: prompt must contain one system marker")
    if prompt.count(USER_MARKER) != 1:
        raise ValueError(f"line {line_number}: prompt must contain one user marker")
    if prompt.count(ASSISTANT_MARKER) != 1:
        raise ValueError(f"line {line_number}: prompt must contain one assistant marker")
    if f"Question: {question}{IM_END}" not in prompt:
        raise ValueError(
            f"line {line_number}: prompt does not contain the exact public question"
        )
    if not prompt.endswith("\n"):
        raise ValueError(f"line {line_number}: prompt must end with a newline")
    stripped = prompt.rstrip()
    if not (
        stripped.endswith(ASSISTANT_MARKER.rstrip())
        or stripped.endswith("</information>")
    ):
        raise ValueError(
            f"line {line_number}: prompt must stop at assistant generation or retrieved information"
        )


def validate_continuation(
    text: str,
    side: str,
    line_number: int,
) -> ChoiceLayout:
    if not isinstance(text, str) or not text:
        raise ValueError(f"line {line_number}: {side} must be a non-empty string")
    if text != text.strip():
        raise ValueError(f"line {line_number}: {side} has leading or trailing whitespace")
    if IM_START in text or IM_END in text:
        raise ValueError(f"line {line_number}: {side} contains an unexpected ChatML marker")
    if not text.startswith("<think>"):
        raise ValueError(f"line {line_number}: {side} must begin with <think>")

    for tag in ("think", "search", "information", "answer"):
        if text.count(f"<{tag}>") != text.count(f"</{tag}>"):
            raise ValueError(f"line {line_number}: unbalanced {tag} tags in {side}")

    matches = list(COMPLETE_SPAN_RE.finditer(text))
    if not matches:
        raise ValueError(f"line {line_number}: {side} contains no complete structural span")
    if matches[0].start() != 0 or matches[-1].end() != len(text):
        raise ValueError(f"line {line_number}: content exists outside {side} structural spans")
    for previous, following in zip(matches, matches[1:]):
        if text[previous.end() : following.start()].strip():
            raise ValueError(
                f"line {line_number}: non-whitespace content exists between {side} spans"
            )
    for index, match in enumerate(matches, start=1):
        if not match.group(2).strip():
            raise ValueError(
                f"line {line_number}: empty <{match.group(1)}> body in {side} span {index}"
            )

    names = [match.group(1) for match in matches]
    structural_tags = [match.group(0) for match in STRUCTURAL_TAG_RE.finditer(text)]
    if len(structural_tags) != 2 * len(matches):
        raise ValueError(f"line {line_number}: nested or unmatched structural tag in {side}")

    if text.endswith("</search>"):
        terminal = "search"
        if names != ["think", "search"]:
            raise ValueError(
                f"line {line_number}: search-terminal {side} must be think then search"
            )
    elif text.endswith("</answer>"):
        terminal = "answer"
        cursor = 0
        while names[cursor : cursor + 3] == ["think", "search", "information"]:
            cursor += 3
        if names[cursor:] != ["think", "answer"]:
            raise ValueError(
                f"line {line_number}: answer-terminal {side} has an invalid trajectory sequence"
            )
    else:
        raise ValueError(
            f"line {line_number}: {side} must end with </search> or </answer>"
        )

    information_spans = [
        (match.start(), match.end())
        for match in matches
        if match.group(1) == "information"
    ]
    return ChoiceLayout(terminal=terminal, information_spans=information_spans)


def validate_public_record(
    record: Dict[str, Any],
    line_number: int,
    strict_public_schema: bool,
) -> Tuple[ChoiceLayout, ChoiceLayout]:
    missing = PUBLIC_FIELDS - set(record)
    if missing:
        raise ValueError(f"line {line_number}: missing fields: {sorted(missing)}")
    if strict_public_schema and tuple(record) != PUBLIC_FIELD_ORDER:
        raise ValueError(
            f"line {line_number}: fields must be exactly {list(PUBLIC_FIELD_ORDER)} in order"
        )

    sample_id = record["id"]
    question = record["question"]
    source_dataset = record["source_dataset"]
    answers = record["answers"]
    pair_type = record["pair_type"]
    prompt = record["prompt_text"]
    chosen = record["chosen"]
    rejected = record["rejected"]

    if not isinstance(sample_id, str) or not re.fullmatch(r"\d{6}", sample_id):
        raise ValueError(f"line {line_number}: id must be a six-digit string")
    for field_name, value in (
        ("question", question),
        ("source_dataset", source_dataset),
        ("pair_type", pair_type),
        ("prompt_text", prompt),
        ("chosen", chosen),
        ("rejected", rejected),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"line {line_number}: {field_name} must be a non-empty string")
    if not isinstance(answers, list) or not answers:
        raise ValueError(f"line {line_number}: answers must be a non-empty list")
    if not all(isinstance(answer, str) and answer.strip() for answer in answers):
        raise ValueError(f"line {line_number}: every answer alias must be non-empty")
    if chosen == rejected:
        raise ValueError(f"line {line_number}: chosen and rejected are identical")

    validate_prompt(prompt, question, line_number)
    chosen_layout = validate_continuation(chosen, "chosen", line_number)
    rejected_layout = validate_continuation(rejected, "rejected", line_number)
    return chosen_layout, rejected_layout


def build_continuation_segments(
    continuation: str,
    layout: ChoiceLayout,
    line_number: int,
    side: str,
) -> Tuple[List[Tuple[str, str]], str]:
    """Split continuation text at every preference-loss mask boundary."""
    segments: List[Tuple[str, str]] = []
    cursor = 0
    for information_start, information_end in layout.information_spans:
        if not (cursor <= information_start < information_end <= len(continuation)):
            raise ValueError(f"line {line_number}: invalid information span in {side}")
        if information_start > cursor:
            segments.append((continuation[cursor:information_start], "supervised"))
        segments.append(
            (continuation[information_start:information_end], "information")
        )
        cursor = information_end
    tail = continuation[cursor:]
    if layout.terminal == "answer":
        tail += IM_END
    if tail:
        segments.append((tail, "supervised"))

    if not segments or any(not segment_text for segment_text, _ in segments):
        raise ValueError(f"line {line_number}: empty tokenization segment in {side}")
    final_text = continuation + (IM_END if layout.terminal == "answer" else "")
    if "".join(segment_text for segment_text, _ in segments) != final_text:
        raise ValueError(f"line {line_number}: {side} segments do not reconstruct text")
    return segments, final_text


class AetherSearchDPODataset(Dataset):
    def __init__(
        self,
        train_file: str,
        tokenizer,
        max_seq_len: int = 4096,
        long_sample_policy: str = "error",
        expected_num_samples: int = CANONICAL_PAIR_COUNT,
        expected_sha256: Optional[str] = CANONICAL_DATA_SHA256,
        strict_public_schema: bool = True,
        tokenization_batch_size: int = 64,
    ):
        if long_sample_policy not in {"error", "filter"}:
            raise ValueError("long_sample_policy must be error or filter")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if tokenization_batch_size <= 0:
            raise ValueError("tokenization_batch_size must be positive")
        final_eot_ids = tokenizer(IM_END, add_special_tokens=False).input_ids
        if final_eot_ids != [tokenizer.eos_token_id]:
            raise ValueError(
                f"Qwen final EOT must tokenize to the EOS id, got {final_eot_ids}"
            )

        source_sha256 = sha256_file(train_file)
        if expected_sha256 and source_sha256.lower() != expected_sha256.lower():
            raise ValueError(
                f"dataset SHA256 mismatch: expected={expected_sha256}, actual={source_sha256}"
            )

        records: List[Dict[str, Any]] = []
        layouts: List[Tuple[ChoiceLayout, ChoiceLayout]] = []
        line_numbers: List[int] = []
        seen_ids: Dict[str, int] = {}
        seen_questions: Dict[str, int] = {}
        source_distribution = collections.Counter()
        pair_type_distribution = collections.Counter()
        terminal_distribution = {
            "chosen": collections.Counter(),
            "rejected": collections.Counter(),
        }
        information_blocks = collections.Counter()

        with open(train_file, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"line {line_number}: invalid JSON: {error}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"line {line_number}: each JSONL row must be an object")

                chosen_layout, rejected_layout = validate_public_record(
                    record,
                    line_number,
                    strict_public_schema,
                )
                sample_id = record["id"]
                normalized_question = normalize_question(record["question"])
                if not normalized_question:
                    raise ValueError(f"line {line_number}: normalized question is empty")
                if sample_id in seen_ids:
                    raise ValueError(
                        f"line {line_number}: duplicate id first seen at line {seen_ids[sample_id]}"
                    )
                if normalized_question in seen_questions:
                    raise ValueError(
                        f"line {line_number}: duplicate normalized question first seen at "
                        f"line {seen_questions[normalized_question]}"
                    )
                seen_ids[sample_id] = line_number
                seen_questions[normalized_question] = line_number
                source_distribution[record["source_dataset"]] += 1
                pair_type_distribution[record["pair_type"]] += 1
                terminal_distribution["chosen"][chosen_layout.terminal] += 1
                terminal_distribution["rejected"][rejected_layout.terminal] += 1
                information_blocks["chosen"] += len(chosen_layout.information_spans)
                information_blocks["rejected"] += len(rejected_layout.information_spans)
                records.append(record)
                layouts.append((chosen_layout, rejected_layout))
                line_numbers.append(line_number)

        if expected_num_samples > 0 and len(records) != expected_num_samples:
            raise ValueError(
                f"expected {expected_num_samples} pairs, found {len(records)} in {train_file}"
            )
        if not records:
            raise ValueError("no preference pairs found")
        for position, record in enumerate(records, start=1):
            expected_id = f"{position:06d}"
            if record["id"] != expected_id:
                raise ValueError(
                    f"record {position}: canonical id must be {expected_id}, got {record['id']}"
                )

        self.samples: List[Dict[str, Any]] = []
        lengths = {"chosen": [], "rejected": []}
        prompt_masked = {"chosen": [], "rejected": []}
        information_masked = {"chosen": [], "rejected": []}
        supervised = {"chosen": [], "rejected": []}
        filtered_overlength = 0
        normalized_choices = 0
        backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
        tokenizer_normalizer = getattr(backend_tokenizer, "normalizer", None)

        for batch_start in range(0, len(records), tokenization_batch_size):
            batch_records = records[batch_start : batch_start + tokenization_batch_size]
            batch_layouts = layouts[batch_start : batch_start + tokenization_batch_size]
            batch_lines = line_numbers[batch_start : batch_start + tokenization_batch_size]

            flat_texts: List[str] = []
            flat_kinds: List[str] = []
            maps: List[Dict[str, Any]] = []
            for record, (chosen_layout, rejected_layout), line_number in zip(
                batch_records, batch_layouts, batch_lines
            ):
                prompt_index = len(flat_texts)
                flat_texts.append(record["prompt_text"])
                flat_kinds.append("prompt")
                item_map: Dict[str, Any] = {"prompt": prompt_index}
                for side, layout in (
                    ("chosen", chosen_layout),
                    ("rejected", rejected_layout),
                ):
                    segments, final_continuation = build_continuation_segments(
                        record[side], layout, line_number, side
                    )
                    start = len(flat_texts)
                    for segment_text, segment_kind in segments:
                        flat_texts.append(segment_text)
                        flat_kinds.append(segment_kind)
                    item_map[side] = {
                        "range": (start, len(flat_texts)),
                        "final_text": record["prompt_text"] + final_continuation,
                        "terminal": layout.terminal,
                    }
                maps.append(item_map)

            encoded = tokenizer(
                flat_texts,
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )

            for local_index, item_map in enumerate(maps):
                record_index = batch_start + local_index
                record = records[record_index]
                line_number = line_numbers[record_index]
                prompt_ids = list(encoded.input_ids[item_map["prompt"]])
                if not prompt_ids:
                    raise ValueError(f"line {line_number}: prompt tokenized to zero tokens")
                sample: Dict[str, Any] = {"id": record["id"]}
                pair_too_long = False

                for side in ("chosen", "rejected"):
                    segment_start, segment_end = item_map[side]["range"]
                    input_ids = list(prompt_ids)
                    labels = [IGNORE_INDEX] * len(prompt_ids)
                    information_count = 0
                    for segment_index in range(segment_start, segment_end):
                        segment_ids = list(encoded.input_ids[segment_index])
                        segment_kind = flat_kinds[segment_index]
                        if not segment_ids:
                            raise ValueError(
                                f"line {line_number}: {side} {segment_kind} segment has no tokens"
                            )
                        input_ids.extend(segment_ids)
                        if segment_kind == "supervised":
                            labels.extend(segment_ids)
                        elif segment_kind == "information":
                            labels.extend([IGNORE_INDEX] * len(segment_ids))
                            information_count += len(segment_ids)
                        else:
                            raise ValueError(
                                f"line {line_number}: unknown segment kind {segment_kind!r}"
                            )

                    decoded = tokenizer.decode(
                        input_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    source_text = item_map[side]["final_text"]
                    normalized_text = (
                        tokenizer_normalizer.normalize_str(source_text)
                        if tokenizer_normalizer is not None
                        else source_text
                    )
                    if normalized_text != source_text:
                        normalized_choices += 1
                    if decoded != normalized_text:
                        mismatch = next(
                            (
                                index
                                for index, (decoded_char, source_char) in enumerate(
                                    zip(decoded, normalized_text)
                                )
                                if decoded_char != source_char
                            ),
                            min(len(decoded), len(normalized_text)),
                        )
                        raise ValueError(
                            f"line {line_number}: {side} segmented-token roundtrip differs "
                            f"at character {mismatch}"
                        )

                    if len(input_ids) > max_seq_len:
                        pair_too_long = True
                    if all(label == IGNORE_INDEX for label in labels):
                        raise ValueError(f"line {line_number}: all {side} labels are masked")
                    if any(
                        label != IGNORE_INDEX and label != token_id
                        for label, token_id in zip(labels, input_ids)
                    ):
                        raise ValueError(
                            f"line {line_number}: supervised {side} labels differ from tokens"
                        )
                    if labels[0] != IGNORE_INDEX:
                        raise ValueError(f"line {line_number}: {side} prompt is not masked")
                    if item_map[side]["terminal"] == "answer":
                        if input_ids[-1] != tokenizer.eos_token_id:
                            raise ValueError(
                                f"line {line_number}: answer-terminal {side} lacks final EOT"
                            )
                        if labels[-1] != tokenizer.eos_token_id:
                            raise ValueError(
                                f"line {line_number}: answer-terminal {side} EOT is not supervised"
                            )
                    elif input_ids[-1] == tokenizer.eos_token_id:
                        raise ValueError(
                            f"line {line_number}: search-terminal {side} must not append EOT"
                        )

                    supervised_count = sum(label != IGNORE_INDEX for label in labels)
                    if len(prompt_ids) + information_count + supervised_count != len(
                        input_ids
                    ):
                        raise ValueError(
                            f"line {line_number}: {side} token mask accounting mismatch"
                        )
                    sample[f"{side}_input_ids"] = input_ids
                    sample[f"{side}_labels"] = labels
                    lengths[side].append(len(input_ids))
                    prompt_masked[side].append(len(prompt_ids))
                    information_masked[side].append(information_count)
                    supervised[side].append(supervised_count)

                if pair_too_long:
                    if long_sample_policy == "error":
                        raise ValueError(
                            f"line {line_number}: pair exceeds max_seq_len={max_seq_len}; "
                            "preference continuations must not be truncated"
                        )
                    for side in ("chosen", "rejected"):
                        lengths[side].pop()
                        prompt_masked[side].pop()
                        information_masked[side].pop()
                        supervised[side].pop()
                    filtered_overlength += 1
                    continue

                sample["length"] = max(
                    len(sample["chosen_input_ids"]),
                    len(sample["rejected_input_ids"]),
                )
                self.samples.append(sample)

        if not self.samples:
            raise ValueError("no preference pairs remained after length filtering")

        self.audit = DatasetAudit(
            source_file=os.path.abspath(train_file),
            source_sha256=source_sha256,
            input_pairs=len(records),
            kept_pairs=len(self.samples),
            filtered_overlength_pairs=filtered_overlength,
            unique_ids=len(seen_ids),
            unique_normalized_questions=len(seen_questions),
            decoded_roundtrip_choices=2 * len(records),
            tokenizer_normalized_choices=normalized_choices,
            masking_strategy="prompt_masked_and_information_masked_in_both_choices",
            source_distribution=dict(sorted(source_distribution.items())),
            pair_type_distribution=dict(sorted(pair_type_distribution.items())),
            terminal_distribution={
                side: dict(sorted(counter.items()))
                for side, counter in terminal_distribution.items()
            },
            information_blocks=dict(sorted(information_blocks.items())),
            total_tokens={side: distribution(values) for side, values in lengths.items()},
            prompt_masked_tokens={
                side: distribution(values) for side, values in prompt_masked.items()
            },
            information_masked_tokens={
                side: distribution(values) for side, values in information_masked.items()
            },
            supervised_tokens={
                side: distribution(values) for side, values in supervised.items()
            },
        )
        self._print_audit()

    def _print_audit(self) -> None:
        audit = self.audit
        rank0_print("========== AETHERSEARCH DPO DATA AUDIT ==========")
        rank0_print("[DATA] source:", audit.source_file)
        rank0_print("[DATA] sha256:", audit.source_sha256)
        rank0_print(
            "[DATA] input/kept/filtered:",
            audit.input_pairs,
            audit.kept_pairs,
            audit.filtered_overlength_pairs,
        )
        rank0_print(
            "[DATA] unique ids/questions:",
            audit.unique_ids,
            audit.unique_normalized_questions,
        )
        rank0_print("[DATA] mask policy:", audit.masking_strategy)
        rank0_print("[DATA] source distribution:", audit.source_distribution)
        rank0_print("[DATA] pair types:", audit.pair_type_distribution)
        rank0_print("[DATA] terminals:", audit.terminal_distribution)
        rank0_print("[DATA] information blocks:", audit.information_blocks)
        rank0_print("[DATA] total tokens:", audit.total_tokens)
        rank0_print("[DATA] prompt-masked tokens:", audit.prompt_masked_tokens)
        rank0_print(
            "[DATA] information-masked tokens:", audit.information_masked_tokens
        )
        rank0_print("[DATA] supervised tokens:", audit.supervised_tokens)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]


@dataclass
class DPOCollator:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("cannot collate an empty batch")
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("tokenizer has neither pad nor EOS token")

        max_length = max(
            len(feature[f"{side}_input_ids"])
            for feature in features
            for side in ("chosen", "rejected")
        )
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple

        batch: Dict[str, torch.Tensor] = {}
        for side in ("chosen", "rejected"):
            input_ids: List[List[int]] = []
            attention_mask: List[List[int]] = []
            labels: List[List[int]] = []
            for feature in features:
                ids = feature[f"{side}_input_ids"]
                sample_labels = feature[f"{side}_labels"]
                if len(ids) != len(sample_labels):
                    raise ValueError(f"{side} input_ids/labels length mismatch")
                padding = max_length - len(ids)
                input_ids.append(ids + [pad_id] * padding)
                attention_mask.append([1] * len(ids) + [0] * padding)
                labels.append(sample_labels + [IGNORE_INDEX] * padding)
            batch[f"{side}_input_ids"] = torch.tensor(input_ids, dtype=torch.long)
            batch[f"{side}_attention_mask"] = torch.tensor(
                attention_mask, dtype=torch.long
            )
            batch[f"{side}_labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def validate_collator(dataset: AetherSearchDPODataset, tokenizer) -> None:
    pair_lengths = [sample["length"] for sample in dataset.samples]
    short_index = min(range(len(pair_lengths)), key=pair_lengths.__getitem__)
    long_index = max(range(len(pair_lengths)), key=pair_lengths.__getitem__)
    features = [dataset[short_index], dataset[long_index]]
    batch = DPOCollator(tokenizer=tokenizer)(features)
    expected_length = ((max(pair_lengths) + 7) // 8) * 8

    for side in ("chosen", "rejected"):
        ids = batch[f"{side}_input_ids"]
        mask = batch[f"{side}_attention_mask"]
        labels = batch[f"{side}_labels"]
        if tuple(ids.shape) != (2, expected_length):
            raise ValueError(f"{side} collator shape mismatch: {tuple(ids.shape)}")
        if mask.shape != ids.shape or labels.shape != ids.shape:
            raise ValueError(f"{side} collator tensor shapes differ")
        for row, feature in enumerate(features):
            sample_ids = feature[f"{side}_input_ids"]
            sample_labels = feature[f"{side}_labels"]
            sample_length = len(sample_ids)
            if not torch.equal(
                ids[row, :sample_length], torch.tensor(sample_ids, dtype=torch.long)
            ):
                raise ValueError(f"collator changed {side} input tokens")
            if not torch.equal(
                labels[row, :sample_length],
                torch.tensor(sample_labels, dtype=torch.long),
            ):
                raise ValueError(f"collator changed {side} labels")
            if not torch.all(mask[row, :sample_length] == 1):
                raise ValueError(f"collator masked real {side} tokens")
            if sample_length < expected_length:
                if not torch.all(mask[row, sample_length:] == 0):
                    raise ValueError(f"collator exposed {side} padding")
                if not torch.all(labels[row, sample_length:] == IGNORE_INDEX):
                    raise ValueError(f"collator supervised {side} padding")
    rank0_print(
        "DPO_COLLATOR_CHECK_OK",
        f"min_pair_tokens={min(pair_lengths)}",
        f"max_pair_tokens={max(pair_lengths)}",
        f"padded_batch_tokens={expected_length}",
    )


def sequence_logps(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    loss_mask = shifted_labels.ne(IGNORE_INDEX)
    if torch.any(loss_mask.sum(dim=-1) == 0):
        raise ValueError("a preference sequence has no scoreable continuation token")
    safe_labels = shifted_labels.masked_fill(~loss_mask, 0)
    token_nll = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        safe_labels.reshape(-1),
        reduction="none",
    ).reshape_as(safe_labels)
    return (-token_nll * loss_mask).sum(dim=-1)


def dpo_objective(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if beta <= 0:
        raise ValueError("beta must be positive")
    shapes = {
        tuple(policy_chosen_logps.shape),
        tuple(policy_rejected_logps.shape),
        tuple(reference_chosen_logps.shape),
        tuple(reference_rejected_logps.shape),
    }
    if len(shapes) != 1:
        raise ValueError("policy and reference log-probability shapes differ")

    policy_margin = policy_chosen_logps - policy_rejected_logps
    reference_margin = reference_chosen_logps - reference_rejected_logps
    preference_logits = policy_margin - reference_margin
    losses = -F.logsigmoid(beta * preference_logits)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    return losses, chosen_rewards.detach(), rejected_rewards.detach()


class AetherSearchDPOTrainer(Trainer):
    def __init__(
        self,
        *args,
        reference_model=None,
        beta: float = 0.1,
        forward_mode: str = "sequential",
        **kwargs,
    ):
        if reference_model is None:
            raise ValueError("reference_model is required")
        if beta <= 0:
            raise ValueError("beta must be positive")
        if forward_mode not in {"sequential", "concatenated"}:
            raise ValueError("forward_mode must be sequential or concatenated")
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.forward_mode = forward_mode
        self.reference_model = self._prepare_reference_model(reference_model)

    def _prepare_reference_model(self, model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.config.use_cache = False

        if self.is_deepspeed_enabled:
            import deepspeed

            plugin = self.accelerator.state.deepspeed_plugin
            config = deepcopy(plugin.deepspeed_config)
            zero = config.setdefault("zero_optimization", {})
            stage = int(zero.get("stage", 0))
            hidden_size = getattr(model.config, "hidden_size", None)
            if stage == 3 and hidden_size is not None:
                zero["reduce_bucket_size"] = hidden_size * hidden_size
                zero["stage3_param_persistence_threshold"] = 10 * hidden_size
                zero["stage3_prefetch_bucket_size"] = int(
                    0.9 * hidden_size * hidden_size
                )
            elif stage != 3:
                zero["stage"] = 0
            model, *_ = deepspeed.initialize(model=model, config=config)
        else:
            model.to(self.args.device)
        model.train(False)
        return model

    def _pair_logps(self, model, inputs: Dict[str, torch.Tensor]):
        if self.forward_mode == "concatenated":
            batch_size = inputs["chosen_input_ids"].shape[0]
            all_logps = sequence_logps(
                model,
                torch.cat(
                    [inputs["chosen_input_ids"], inputs["rejected_input_ids"]], dim=0
                ),
                torch.cat(
                    [
                        inputs["chosen_attention_mask"],
                        inputs["rejected_attention_mask"],
                    ],
                    dim=0,
                ),
                torch.cat(
                    [inputs["chosen_labels"], inputs["rejected_labels"]], dim=0
                ),
            )
            return all_logps[:batch_size], all_logps[batch_size:]

        chosen_logps = sequence_logps(
            model,
            inputs["chosen_input_ids"],
            inputs["chosen_attention_mask"],
            inputs["chosen_labels"],
        )
        rejected_logps = sequence_logps(
            model,
            inputs["rejected_input_ids"],
            inputs["rejected_attention_mask"],
            inputs["rejected_labels"],
        )
        return chosen_logps, rejected_logps

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        del num_items_in_batch
        policy_chosen, policy_rejected = self._pair_logps(model, inputs)
        with torch.no_grad():
            reference_chosen, reference_rejected = self._pair_logps(
                self.reference_model, inputs
            )
        losses, chosen_rewards, rejected_rewards = dpo_objective(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            self.beta,
        )
        loss = losses.mean()
        diagnostics = {
            "loss": loss.detach(),
            "chosen_reward": chosen_rewards.mean(),
            "rejected_reward": rejected_rewards.mean(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean(),
            "policy_chosen_logps": policy_chosen.detach().mean(),
            "policy_rejected_logps": policy_rejected.detach().mean(),
        }
        if return_outputs:
            return loss, diagnostics
        return loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preference training for the public AetherSearch DPO dataset."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--ref_model_name_or_path", required=True)
    parser.add_argument("--ref_model_revision", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--deepspeed", default=None)

    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument(
        "--long_sample_policy", choices=["error", "filter"], default="error"
    )
    parser.add_argument("--expected_num_samples", type=int, default=CANONICAL_PAIR_COUNT)
    parser.add_argument("--expected_sha256", default=CANONICAL_DATA_SHA256)
    parser.add_argument("--allow_extra_fields", action="store_true")
    parser.add_argument("--tokenization_batch_size", type=int, default=64)
    parser.add_argument("--check_data_only", action="store_true")
    parser.add_argument("--audit_report_path", default=None)

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--forward_mode",
        choices=["sequential", "concatenated"],
        default="sequential",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--precision", choices=["bf16", "fp16", "fp32"], default="bf16"
    )
    parser.add_argument("--tf32", action="store_true")

    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument(
        "--save_strategy", choices=["no", "steps", "epoch"], default="no"
    )
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    return parser.parse_args()


def write_json(path: str, payload: Dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    validate_cli_args(args)
    set_seed(args.seed)

    rank0_print("[INFO] Loading tokenizer:", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("the Qwen fast tokenizer is required for segmented masking")
    if tokenizer.eos_token != IM_END:
        raise ValueError(f"expected Qwen eos_token={IM_END!r}, got {tokenizer.eos_token!r}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rank0_print("[INFO] Auditing and tokenizing dataset:", args.train_file)
    train_dataset = AetherSearchDPODataset(
        train_file=args.train_file,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        long_sample_policy=args.long_sample_policy,
        expected_num_samples=args.expected_num_samples,
        expected_sha256=args.expected_sha256,
        strict_public_schema=not args.allow_extra_fields,
        tokenization_batch_size=args.tokenization_batch_size,
    )
    audit_payload = asdict(train_dataset.audit)
    if args.audit_report_path and rank0():
        write_json(args.audit_report_path, audit_payload)
        rank0_print("[INFO] Wrote data audit:", args.audit_report_path)
    if args.check_data_only:
        validate_collator(train_dataset, tokenizer)
        rank0_print(f"DPO_DATA_CHECK_OK pairs={len(train_dataset)}")
        return

    final_model_dir = os.path.join(args.output_dir, "final_model")
    incomplete_model_dir = os.path.join(args.output_dir, ".final_model.incomplete")
    if os.path.exists(final_model_dir):
        raise FileExistsError(f"refusing to overwrite final model: {final_model_dir}")
    if os.path.exists(incomplete_model_dir):
        raise FileExistsError(
            f"an incomplete final-model export requires inspection: {incomplete_model_dir}"
        )

    if args.precision == "bf16":
        model_dtype = torch.bfloat16
    elif args.precision == "fp16":
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        bf16=args.precision == "bf16",
        fp16=args.precision == "fp16",
        tf32=args.tf32,
        deepspeed=args.deepspeed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        report_to=["tensorboard"],
        logging_dir=os.path.join(args.output_dir, "tb_logs"),
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        optim="adamw_torch",
        seed=args.seed,
        data_seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    rank0_print("[INFO] Loading policy model:", args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=model_dtype,
    )
    rank0_print("[INFO] Loading frozen reference model:", args.ref_model_name_or_path)
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.ref_model_name_or_path,
        revision=args.ref_model_revision,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=model_dtype,
    )

    for field in (
        "model_type",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "max_position_embeddings",
    ):
        if getattr(model.config, field, None) != getattr(
            reference_model.config, field, None
        ):
            raise ValueError(f"policy/reference config mismatch for {field}")
    model_max_positions = getattr(model.config, "max_position_embeddings", None)
    if model_max_positions is not None and args.max_seq_len > model_max_positions:
        raise ValueError(
            f"max_seq_len={args.max_seq_len} exceeds model capacity={model_max_positions}"
        )

    original_use_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = False
    reference_model.config.use_cache = False

    trainer = AetherSearchDPOTrainer(
        model=model,
        reference_model=reference_model,
        beta=args.beta,
        forward_mode=args.forward_mode,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DPOCollator(tokenizer=tokenizer),
        processing_class=tokenizer,
    )

    rank0_print("[INFO] Starting AetherSearch DPO training")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.model.config.use_cache = original_use_cache

    rank0_print("[INFO] Saving final model atomically to:", final_model_dir)
    trainer.save_model(incomplete_model_dir)
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(incomplete_model_dir)
        run_manifest = {
            "base_model": args.model_name_or_path,
            "base_model_revision": args.model_revision,
            "reference_model": args.ref_model_name_or_path,
            "reference_model_revision": args.ref_model_revision,
            "train_file": os.path.abspath(args.train_file),
            "data_audit": audit_payload,
            "training_arguments": vars(args),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "effective_global_batch_size": (
                args.per_device_train_batch_size
                * args.gradient_accumulation_steps
                * int(os.environ.get("WORLD_SIZE", "1"))
            ),
            "train_metrics": train_result.metrics,
            "final_model_dir": os.path.abspath(final_model_dir),
        }
        write_json(os.path.join(args.output_dir, "dpo_run_manifest.json"), run_manifest)
        os.replace(incomplete_model_dir, final_model_dir)
    trainer.accelerator.wait_for_everyone()
    rank0_print("[INFO] Done")


if __name__ == "__main__":
    main()
