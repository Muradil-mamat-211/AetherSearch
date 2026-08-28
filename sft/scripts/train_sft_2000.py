#!/usr/bin/env python3
"""Strict full-trajectory trainer for the canonical AetherSearch SFT-2000 data."""

import argparse
import collections
import hashlib
import json
import math
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
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
    "trajectory_type",
    "search_count",
    "full_trajectory_text",
)
PUBLIC_FIELDS = set(PUBLIC_FIELD_ORDER)
CANONICAL_DATA_SHA256 = (
    "fec609652d3832c7a6c0ee2861c6f946b6cf7c3d3d40fc5d9be9b75df6325dcb"
)

INFORMATION_RE = re.compile(r"<information>.*?</information>", flags=re.S)
SEARCH_RE = re.compile(r"<search>(.*?)</search>", flags=re.S)
THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.S)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.S)
STRUCTURAL_TAG_RE = re.compile(r"</?(?:think|search|information|answer)>")


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
    text = unicodedata.normalize("NFKC", text).casefold()
    text = " ".join(text.split()).strip()
    return re.sub(r"[?？]+$", "", text).strip()


def validate_cli_args(args: argparse.Namespace) -> None:
    """Reject ambiguous or unsafe training configurations before heavy work."""
    if not os.path.isfile(args.train_file):
        raise FileNotFoundError(f"training JSONL does not exist: {args.train_file}")
    if args.deepspeed and not os.path.isfile(args.deepspeed):
        raise FileNotFoundError(f"DeepSpeed config does not exist: {args.deepspeed}")
    if args.max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if args.expected_num_samples < 0:
        raise ValueError("expected_num_samples must be non-negative (zero disables the check)")
    if args.expected_sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", args.expected_sha256):
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
    if args.tokenization_batch_size <= 0:
        raise ValueError("tokenization_batch_size must be positive")
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
        raise ValueError("warmup_ratio must be between 0 and 1")
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


def percentile(values: Sequence[int], probability: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * probability) - 1))
    return ordered[index]


def ensure_nonempty_matches(
    matches: Sequence[re.Match],
    tag_name: str,
    line_number: int,
) -> None:
    for index, match in enumerate(matches, start=1):
        if not match.group(1).strip():
            raise ValueError(
                f"line {line_number}: empty <{tag_name}> body at occurrence {index}"
            )


@dataclass(frozen=True)
class RecordLayout:
    assistant_content_start: int
    information_spans: List[Tuple[int, int]]
    search_count: int
    think_count: int


@dataclass
class DatasetAudit:
    source_file: str
    source_sha256: str
    input_records: int
    kept_records: int
    filtered_overlength_records: int
    unique_ids: int
    unique_normalized_questions: int
    masking_strategy: str
    decoded_roundtrip_records: int
    tokenizer_normalized_records: int
    trajectory_types: Dict[str, int]
    search_depths: Dict[str, int]
    total_tokens: Dict[str, int]
    prompt_masked_tokens: Dict[str, int]
    information_masked_tokens: Dict[str, int]
    supervised_tokens: Dict[str, int]


def distribution(values: Sequence[int]) -> Dict[str, int]:
    return {
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "sum": sum(values),
    }


def validate_public_record(
    record: Dict[str, Any],
    line_number: int,
    strict_public_schema: bool,
) -> RecordLayout:
    missing = PUBLIC_FIELDS - set(record)
    if missing:
        raise ValueError(f"line {line_number}: missing public fields: {sorted(missing)}")
    if strict_public_schema and tuple(record.keys()) != PUBLIC_FIELD_ORDER:
        raise ValueError(
            f"line {line_number}: fields must be exactly {list(PUBLIC_FIELD_ORDER)} in "
            f"that order, got {list(record.keys())}"
        )

    sample_id = record["id"]
    question = record["question"]
    trajectory_type = record["trajectory_type"]
    expected_search_count = record["search_count"]
    text = record["full_trajectory_text"]

    if not isinstance(sample_id, str) or not re.fullmatch(r"\d{6}", sample_id):
        raise ValueError(f"line {line_number}: id must be a six-digit string, got {sample_id!r}")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"line {line_number}: question must be a non-empty string")
    if trajectory_type not in {"single_search", "multi_search"}:
        raise ValueError(
            f"line {line_number}: unsupported trajectory_type={trajectory_type!r}"
        )
    if not isinstance(expected_search_count, int) or expected_search_count < 1:
        raise ValueError(f"line {line_number}: invalid search_count={expected_search_count!r}")
    if trajectory_type == "single_search" and expected_search_count != 1:
        raise ValueError(
            f"line {line_number}: single_search must have search_count=1, "
            f"got {expected_search_count}"
        )
    if trajectory_type == "multi_search" and expected_search_count < 2:
        raise ValueError(
            f"line {line_number}: multi_search must have search_count>=2, "
            f"got {expected_search_count}"
        )
    if not isinstance(text, str) or not text:
        raise ValueError(f"line {line_number}: full_trajectory_text must be non-empty")

    if text.count(IM_START) != 3 or text.count(IM_END) != 3:
        raise ValueError(
            f"line {line_number}: expected exactly three ChatML starts and ends, got "
            f"start={text.count(IM_START)}, end={text.count(IM_END)}"
        )
    if text.count(SYSTEM_MARKER) != 1:
        raise ValueError(f"line {line_number}: expected exactly one system marker")
    if text.count(USER_MARKER) != 1:
        raise ValueError(f"line {line_number}: expected exactly one user marker")
    if text.count(ASSISTANT_MARKER) != 1:
        raise ValueError(f"line {line_number}: expected exactly one assistant marker")
    if not text.startswith(SYSTEM_MARKER):
        raise ValueError(f"line {line_number}: trajectory must begin with the system marker")
    if not text.endswith(f"</answer>{IM_END}"):
        raise ValueError(
            f"line {line_number}: trajectory must end exactly with </answer>{IM_END}"
        )

    system_eot_start = text.index(IM_END, len(SYSTEM_MARKER))
    expected_system_transition = f"{IM_END}\n{USER_MARKER}"
    if not text.startswith(expected_system_transition, system_eot_start):
        raise ValueError(
            f"line {line_number}: system message is not followed by the canonical user marker"
        )

    user_marker_start = system_eot_start + len(IM_END) + 1
    user_content_start = user_marker_start + len(USER_MARKER)
    user_eot_start = text.index(IM_END, user_content_start)
    expected_user_transition = f"{IM_END}\n{ASSISTANT_MARKER}"
    if not text.startswith(expected_user_transition, user_eot_start):
        raise ValueError(
            f"line {line_number}: user message is not followed by the canonical assistant marker"
        )

    assistant_marker_start = user_eot_start + len(IM_END) + 1
    assistant_content_start = assistant_marker_start + len(ASSISTANT_MARKER)
    prompt_text = text[:assistant_content_start]
    assistant_text = text[assistant_content_start : -len(IM_END)]
    user_text = text[user_content_start:user_eot_start]

    if SYSTEM_MARKER not in prompt_text or USER_MARKER not in prompt_text:
        raise ValueError(f"line {line_number}: incomplete system/user prompt")
    if prompt_text.index(SYSTEM_MARKER) > prompt_text.index(USER_MARKER):
        raise ValueError(f"line {line_number}: user marker occurs before system marker")
    if not user_text.endswith(f"Question: {question}"):
        raise ValueError(
            f"line {line_number}: user message must end with the exact public question"
        )
    if IM_START in assistant_text or IM_END in assistant_text:
        raise ValueError(f"line {line_number}: unexpected ChatML marker inside assistant content")
    if not assistant_text.startswith("<think>"):
        raise ValueError(f"line {line_number}: assistant content must begin with <think>")

    if assistant_text.count("<information>") != assistant_text.count("</information>"):
        raise ValueError(f"line {line_number}: unbalanced information tags")
    if assistant_text.count("<search>") != assistant_text.count("</search>"):
        raise ValueError(f"line {line_number}: unbalanced search tags")
    if assistant_text.count("<think>") != assistant_text.count("</think>"):
        raise ValueError(f"line {line_number}: unbalanced think tags")
    if assistant_text.count("<answer>") != assistant_text.count("</answer>"):
        raise ValueError(f"line {line_number}: unbalanced answer tags")

    information_matches = list(INFORMATION_RE.finditer(assistant_text))
    search_matches = list(SEARCH_RE.finditer(assistant_text))
    think_matches = list(THINK_RE.finditer(assistant_text))
    answer_matches = list(ANSWER_RE.finditer(assistant_text))

    expected_tag_sequence: List[str] = []
    for _ in range(expected_search_count):
        expected_tag_sequence.extend(
            [
                "<think>",
                "</think>",
                "<search>",
                "</search>",
                "<information>",
                "</information>",
            ]
        )
    expected_tag_sequence.extend(["<think>", "</think>", "<answer>", "</answer>"])
    actual_tag_sequence = [match.group(0) for match in STRUCTURAL_TAG_RE.finditer(assistant_text)]
    if actual_tag_sequence != expected_tag_sequence:
        raise ValueError(
            f"line {line_number}: assistant structural-tag sequence does not match "
            "(think, search, information)* then (think, answer)"
        )

    if len(search_matches) != expected_search_count:
        raise ValueError(
            f"line {line_number}: search_count field={expected_search_count}, "
            f"actual={len(search_matches)}"
        )
    if len(information_matches) != expected_search_count:
        raise ValueError(
            f"line {line_number}: expected one information block per search, got "
            f"searches={expected_search_count}, information={len(information_matches)}"
        )
    if len(think_matches) != expected_search_count + 1:
        raise ValueError(
            f"line {line_number}: expected one think per search plus one final-answer think, "
            f"got think={len(think_matches)}, search={expected_search_count}"
        )
    if len(answer_matches) != 1:
        raise ValueError(f"line {line_number}: expected exactly one answer block")

    ensure_nonempty_matches(search_matches, "search", line_number)
    ensure_nonempty_matches(think_matches, "think", line_number)
    ensure_nonempty_matches(answer_matches, "answer", line_number)
    for index, match in enumerate(information_matches, start=1):
        body_start = match.start() + len("<information>")
        body_end = match.end() - len("</information>")
        if not assistant_text[body_start:body_end].strip():
            raise ValueError(
                f"line {line_number}: empty <information> body at occurrence {index}"
            )

    for index, (think_match, search_match, information_match) in enumerate(
        zip(think_matches[:-1], search_matches, information_matches), start=1
    ):
        if think_match.end() != search_match.start():
            raise ValueError(
                f"line {line_number}: search {index} is not directly adjacent to think {index}"
            )
        if search_match.end() != information_match.start():
            raise ValueError(
                f"line {line_number}: information {index} is not directly adjacent to search {index}"
            )
        if information_match.end() != think_matches[index].start():
            raise ValueError(
                f"line {line_number}: next think is not directly adjacent to information {index}"
            )

    if think_matches[-1].end() != answer_matches[0].start():
        raise ValueError(f"line {line_number}: final answer is not directly adjacent to final think")
    if answer_matches[0].end() != len(assistant_text):
        raise ValueError(f"line {line_number}: content exists after the final answer")

    absolute_information_spans = [
        (
            assistant_content_start + match.start(),
            assistant_content_start + match.end(),
        )
        for match in information_matches
    ]
    return RecordLayout(
        assistant_content_start=assistant_content_start,
        information_spans=absolute_information_spans,
        search_count=expected_search_count,
        think_count=len(think_matches),
    )


def build_mask_segments(
    text: str,
    layout: RecordLayout,
    line_number: int,
) -> List[Tuple[str, str]]:
    """Split at loss-mask boundaries so no BPE token can straddle two policies."""
    segments: List[Tuple[str, str]] = [
        (text[: layout.assistant_content_start], "prompt"),
    ]
    cursor = layout.assistant_content_start
    for information_start, information_end in layout.information_spans:
        if not (cursor < information_start < information_end <= len(text)):
            raise ValueError(f"line {line_number}: invalid information span boundaries")
        segments.append((text[cursor:information_start], "supervised"))
        segments.append((text[information_start:information_end], "information"))
        cursor = information_end
    segments.append((text[cursor:], "supervised"))

    if any(not segment_text for segment_text, _ in segments):
        raise ValueError(f"line {line_number}: empty tokenization segment")
    if "".join(segment_text for segment_text, _ in segments) != text:
        raise ValueError(f"line {line_number}: mask segments do not reconstruct source text")
    return segments


class SearchSFT2000Dataset(Dataset):
    def __init__(
        self,
        train_file: str,
        tokenizer,
        max_seq_len: int = 4096,
        long_sample_policy: str = "error",
        expected_num_samples: int = 2000,
        expected_sha256: Optional[str] = None,
        strict_public_schema: bool = True,
        tokenization_batch_size: int = 64,
    ):
        if long_sample_policy not in {"error", "filter"}:
            raise ValueError("long_sample_policy must be 'error' or 'filter'; truncation is unsafe")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if tokenization_batch_size <= 0:
            raise ValueError("tokenization_batch_size must be positive")
        final_eot_ids = tokenizer(IM_END, add_special_tokens=False).input_ids
        if final_eot_ids != [tokenizer.eos_token_id]:
            raise ValueError(
                f"Qwen final EOT must tokenize to exactly the EOS id; got {final_eot_ids}"
            )

        source_sha256 = sha256_file(train_file)
        if expected_sha256 and source_sha256.lower() != expected_sha256.lower():
            raise ValueError(
                f"dataset SHA256 mismatch: expected={expected_sha256}, actual={source_sha256}"
            )

        records: List[Dict[str, Any]] = []
        layouts: List[RecordLayout] = []
        source_line_numbers: List[int] = []
        seen_ids: Dict[str, int] = {}
        seen_questions: Dict[str, int] = {}
        trajectory_types = collections.Counter()
        search_depths = collections.Counter()

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

                layout = validate_public_record(record, line_number, strict_public_schema)
                sample_id = record["id"]
                normalized_question = normalize_question(record["question"])
                if not normalized_question:
                    raise ValueError(
                        f"line {line_number}: question is empty after canonical normalization"
                    )
                if sample_id in seen_ids:
                    raise ValueError(
                        f"line {line_number}: duplicate id={sample_id!r}; first seen at "
                        f"line {seen_ids[sample_id]}"
                    )
                if normalized_question in seen_questions:
                    raise ValueError(
                        f"line {line_number}: duplicate normalized question; first seen at "
                        f"line {seen_questions[normalized_question]}"
                    )
                seen_ids[sample_id] = line_number
                seen_questions[normalized_question] = line_number
                trajectory_types[record["trajectory_type"]] += 1
                search_depths[str(record["search_count"])] += 1
                records.append(record)
                layouts.append(layout)
                source_line_numbers.append(line_number)

        if expected_num_samples > 0 and len(records) != expected_num_samples:
            raise ValueError(
                f"expected {expected_num_samples} records, found {len(records)} in {train_file}"
            )
        if not records:
            raise ValueError("no SFT records found")
        for position, record in enumerate(records, start=1):
            expected_id = f"{position:06d}"
            if record["id"] != expected_id:
                raise ValueError(
                    f"record {position}: canonical public id must be {expected_id}, "
                    f"got {record['id']}"
                )

        self.samples: List[Dict[str, List[int]]] = []
        total_lengths: List[int] = []
        prompt_masked_lengths: List[int] = []
        information_masked_lengths: List[int] = []
        supervised_lengths: List[int] = []
        filtered_overlength = 0
        tokenizer_normalized_records = 0
        backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
        tokenizer_normalizer = getattr(backend_tokenizer, "normalizer", None)

        texts = [record["full_trajectory_text"] for record in records]
        for batch_start in range(0, len(records), tokenization_batch_size):
            batch_texts = texts[batch_start : batch_start + tokenization_batch_size]
            batch_layouts = layouts[batch_start : batch_start + tokenization_batch_size]
            batch_line_numbers = source_line_numbers[
                batch_start : batch_start + tokenization_batch_size
            ]
            flat_segment_texts: List[str] = []
            flat_segment_kinds: List[str] = []
            record_segment_ranges: List[Tuple[int, int]] = []
            for text, layout, line_number in zip(
                batch_texts, batch_layouts, batch_line_numbers
            ):
                segments = build_mask_segments(text, layout, line_number)
                segment_start = len(flat_segment_texts)
                for segment_text, segment_kind in segments:
                    flat_segment_texts.append(segment_text)
                    flat_segment_kinds.append(segment_kind)
                record_segment_ranges.append((segment_start, len(flat_segment_texts)))

            encoded = tokenizer(
                flat_segment_texts,
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )

            for local_index, (segment_start, segment_end) in enumerate(record_segment_ranges):
                record_index = batch_start + local_index
                line_number = source_line_numbers[record_index]
                record = records[record_index]
                text = texts[record_index]

                input_ids: List[int] = []
                labels: List[int] = []
                prompt_masked = 0
                information_masked = 0
                for segment_index in range(segment_start, segment_end):
                    segment_ids = list(encoded.input_ids[segment_index])
                    segment_kind = flat_segment_kinds[segment_index]
                    if not segment_ids:
                        raise ValueError(
                            f"line {line_number}: {segment_kind} segment tokenized to zero tokens"
                        )
                    input_ids.extend(segment_ids)
                    if segment_kind == "supervised":
                        labels.extend(segment_ids)
                    else:
                        labels.extend([IGNORE_INDEX] * len(segment_ids))
                        if segment_kind == "prompt":
                            prompt_masked += len(segment_ids)
                        elif segment_kind == "information":
                            information_masked += len(segment_ids)
                        else:
                            raise ValueError(
                                f"line {line_number}: unknown mask segment kind={segment_kind!r}"
                            )

                decoded = tokenizer.decode(
                    input_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                normalized_text = (
                    tokenizer_normalizer.normalize_str(text)
                    if tokenizer_normalizer is not None
                    else text
                )
                if normalized_text != text:
                    tokenizer_normalized_records += 1
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
                        f"line {line_number}: segmented-token roundtrip differs from the "
                        f"tokenizer-normalized source at character {mismatch}"
                    )

                if len(input_ids) > max_seq_len:
                    if long_sample_policy == "error":
                        raise ValueError(
                            f"line {line_number}: token length {len(input_ids)} exceeds "
                            f"max_seq_len={max_seq_len}; id={record['id']}. Full trajectories "
                            "must not be truncated."
                        )
                    filtered_overlength += 1
                    continue

                final_eot_index = len(input_ids) - 1
                if input_ids[final_eot_index] != tokenizer.eos_token_id:
                    raise ValueError(f"line {line_number}: final input token is not EOS")
                if labels[final_eot_index] != tokenizer.eos_token_id:
                    raise ValueError(f"line {line_number}: final assistant EOT is not supervised")
                if all(label == IGNORE_INDEX for label in labels):
                    raise ValueError(f"line {line_number}: all labels are masked")
                if any(
                    label != IGNORE_INDEX and label != token_id
                    for label, token_id in zip(labels, input_ids)
                ):
                    raise ValueError(f"line {line_number}: supervised labels differ from input ids")

                supervised = sum(label != IGNORE_INDEX for label in labels)
                if prompt_masked + information_masked + supervised != len(input_ids):
                    raise ValueError(f"line {line_number}: token mask accounting mismatch")

                self.samples.append({"input_ids": list(input_ids), "labels": labels})
                total_lengths.append(len(input_ids))
                prompt_masked_lengths.append(prompt_masked)
                information_masked_lengths.append(information_masked)
                supervised_lengths.append(supervised)

        if not self.samples:
            raise ValueError("no SFT samples remained after length filtering")

        self.audit = DatasetAudit(
            source_file=os.path.abspath(train_file),
            source_sha256=source_sha256,
            input_records=len(records),
            kept_records=len(self.samples),
            filtered_overlength_records=filtered_overlength,
            unique_ids=len(seen_ids),
            unique_normalized_questions=len(seen_questions),
            masking_strategy="independent_prompt/information/supervised_segments",
            decoded_roundtrip_records=len(records),
            tokenizer_normalized_records=tokenizer_normalized_records,
            trajectory_types=dict(sorted(trajectory_types.items())),
            search_depths=dict(sorted(search_depths.items(), key=lambda item: int(item[0]))),
            total_tokens=distribution(total_lengths),
            prompt_masked_tokens=distribution(prompt_masked_lengths),
            information_masked_tokens=distribution(information_masked_lengths),
            supervised_tokens=distribution(supervised_lengths),
        )
        self._print_audit()

    def _print_audit(self) -> None:
        audit = self.audit
        rank0_print("========== SEARCH SFT 2000 DATASET AUDIT ==========")
        rank0_print("[DATA] source:", audit.source_file)
        rank0_print("[DATA] sha256:", audit.source_sha256)
        rank0_print("[DATA] input/kept/filtered:", audit.input_records, audit.kept_records, audit.filtered_overlength_records)
        rank0_print("[DATA] unique ids/questions:", audit.unique_ids, audit.unique_normalized_questions)
        rank0_print("[DATA] masking strategy:", audit.masking_strategy)
        rank0_print("[DATA] decoded roundtrips:", audit.decoded_roundtrip_records)
        rank0_print("[DATA] tokenizer-normalized records:", audit.tokenizer_normalized_records)
        rank0_print("[DATA] trajectory types:", audit.trajectory_types)
        rank0_print("[DATA] search depths:", audit.search_depths)
        rank0_print("[DATA] total tokens:", audit.total_tokens)
        rank0_print("[DATA] prompt-masked tokens:", audit.prompt_masked_tokens)
        rank0_print("[DATA] information-masked tokens:", audit.information_masked_tokens)
        rank0_print("[DATA] supervised tokens:", audit.supervised_tokens)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.samples[index]


@dataclass
class SearchSFT2000Collator:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("cannot collate an empty batch")
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")

        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple

        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        labels: List[List[int]] = []
        for feature in features:
            ids = feature["input_ids"]
            sample_labels = feature["labels"]
            if len(ids) != len(sample_labels):
                raise ValueError("input_ids/labels length mismatch in collator")
            padding = max_length - len(ids)
            input_ids.append(ids + [pad_id] * padding)
            attention_mask.append([1] * len(ids) + [0] * padding)
            labels.append(sample_labels + [IGNORE_INDEX] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def validate_collator(dataset: SearchSFT2000Dataset, tokenizer) -> None:
    lengths = [len(sample["input_ids"]) for sample in dataset.samples]
    short_index = min(range(len(lengths)), key=lengths.__getitem__)
    long_index = max(range(len(lengths)), key=lengths.__getitem__)
    features = [dataset[short_index], dataset[long_index]]
    batch = SearchSFT2000Collator(tokenizer=tokenizer)(features)
    expected_length = ((max(lengths) + 7) // 8) * 8

    if tuple(batch["input_ids"].shape) != (2, expected_length):
        raise ValueError(f"collator input shape mismatch: {tuple(batch['input_ids'].shape)}")
    if batch["labels"].shape != batch["input_ids"].shape:
        raise ValueError("collator labels shape differs from input shape")
    if batch["attention_mask"].shape != batch["input_ids"].shape:
        raise ValueError("collator attention-mask shape differs from input shape")

    for row, feature in enumerate(features):
        sample_length = len(feature["input_ids"])
        if not torch.equal(
            batch["input_ids"][row, :sample_length],
            torch.tensor(feature["input_ids"], dtype=torch.long),
        ):
            raise ValueError(f"collator changed input tokens in smoke row {row}")
        if not torch.equal(
            batch["labels"][row, :sample_length],
            torch.tensor(feature["labels"], dtype=torch.long),
        ):
            raise ValueError(f"collator changed labels in smoke row {row}")
        if not torch.all(batch["attention_mask"][row, :sample_length] == 1):
            raise ValueError(f"collator masked real tokens in smoke row {row}")
        if sample_length < expected_length:
            if not torch.all(batch["attention_mask"][row, sample_length:] == 0):
                raise ValueError(f"collator exposed padding in smoke row {row}")
            if not torch.all(batch["labels"][row, sample_length:] == IGNORE_INDEX):
                raise ValueError(f"collator supervises padding in smoke row {row}")
    rank0_print(
        "SFT_2000_COLLATOR_CHECK_OK",
        f"min_tokens={min(lengths)}",
        f"max_tokens={max(lengths)}",
        f"padded_batch_tokens={expected_length}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-trajectory SFT trainer for the public Search-SFT 2000 dataset."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument(
        "--model_revision",
        default=None,
        help="Optional immutable Hugging Face model revision (commit SHA recommended).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Opt in to executing custom code from a remote model repository.",
    )
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--deepspeed", default=None)

    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--long_sample_policy", choices=["error", "filter"], default="error")
    parser.add_argument("--expected_num_samples", type=int, default=2000)
    parser.add_argument(
        "--expected_sha256",
        default=CANONICAL_DATA_SHA256,
        help="Expected training-file SHA-256 (defaults to canonical SFT-2000 data).",
    )
    parser.add_argument("--allow_extra_fields", action="store_true")
    parser.add_argument("--tokenization_batch_size", type=int, default=64)
    parser.add_argument("--check_data_only", action="store_true")
    parser.add_argument("--audit_report_path", default=None)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--tf32", action="store_true")

    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_strategy", choices=["no", "steps", "epoch"], default="no")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument(
        "--group_by_length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Group similar sequence lengths to reduce dynamic-padding waste (default: true).",
    )
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
        raise ValueError("the Qwen fast tokenizer is required for reproducible segmented masking")
    if tokenizer.eos_token != IM_END:
        raise ValueError(
            f"expected Qwen eos_token={IM_END!r}, got {tokenizer.eos_token!r}"
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rank0_print("[INFO] Auditing and tokenizing dataset:", args.train_file)
    train_dataset = SearchSFT2000Dataset(
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
        rank0_print(f"SFT_2000_DATA_CHECK_OK samples={len(train_dataset)}")
        return

    final_model_dir = os.path.join(args.output_dir, "final_model")
    incomplete_model_dir = os.path.join(args.output_dir, ".final_model.incomplete")
    if os.path.exists(final_model_dir):
        raise FileExistsError(
            f"refusing to overwrite existing final model: {final_model_dir}"
        )
    if os.path.exists(incomplete_model_dir):
        raise FileExistsError(
            "an incomplete final-model export already exists; inspect or move it before "
            f"retrying: {incomplete_model_dir}"
        )

    if args.precision == "bf16":
        model_dtype = torch.bfloat16
    elif args.precision == "fp16":
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    # Build TrainingArguments before from_pretrained so Transformers can activate
    # ZeRO-3 initialization while the model is being loaded.
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
        group_by_length=args.group_by_length,
        optim="adamw_torch",
        seed=args.seed,
        data_seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    rank0_print("[INFO] Loading model:", args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=model_dtype,
    )
    model_max_positions = getattr(model.config, "max_position_embeddings", None)
    if model_max_positions is not None and args.max_seq_len > model_max_positions:
        raise ValueError(
            f"max_seq_len={args.max_seq_len} exceeds model max_position_embeddings="
            f"{model_max_positions}"
        )
    original_use_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=SearchSFT2000Collator(tokenizer=tokenizer),
        processing_class=tokenizer,
    )

    rank0_print("[INFO] Starting Search-SFT 2000 training")
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
        write_json(os.path.join(args.output_dir, "sft_2000_run_manifest.json"), run_manifest)
        os.replace(incomplete_model_dir, final_model_dir)
    trainer.accelerator.wait_for_everyone()
    rank0_print("[INFO] Done")


if __name__ == "__main__":
    main()
