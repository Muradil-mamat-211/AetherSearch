#!/usr/bin/env python3
"""Build gold-validated, zero-search DeepSeek trajectories for AetherSearch SFT."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sqlite3
import ssl
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised by the CLI preflight
    raise SystemExit("pyarrow is required to audit the Arrow/Parquet question sources") from exc

from answer_utils import canonicalize_answer, normalize_answer
from controlled_rollout import DIRECT_FINAL_THINK, Rejected, canonical_answer, parse_action
from deepseek_key import KEY_FILE, NoRedirect, load_key_file
from published_sft_format import (EOT, PUBLIC_FIELDS, PUBLIC_USER_PROMPT,
                                  public_prefix, validate_public_record)
from token_budget import student_budget


API_URL = "https://api.deepseek.com/beta/chat/completions"
PROMPT_VERSION = "direct_answer_teacher_v1"
GENERATOR_VERSION = "deepseek_direct_answer_gold_validated_v1"
OVERLAP_POLICY_VERSION = "sft_dpo_rl_exact_question_exclusion_v1"
EXPORT_POLICY_VERSION = "direct_answer_export_v2_preserve_acronyms"
MODEL = "deepseek-flash"
THINKING = "disabled"
MAX_API_TOKENS = 500
FINAL_ACTION = re.compile(r"^<think>([^<>]+)</think><answer>([^<>]+)</answer>$")
DOC_CITATION = re.compile(r"\b(?:Turn\s+\d+\s+)?Doc\s+\d+\b", re.I)
SYSTEM_PROMPT = (
    "You are a teacher generating one direct-answer SFT action. No retrieval tool, browser, network search, "
    "shell, or file access is available in this request. Use reliable prior knowledge only. Return exactly "
    "<think>brief basis</think><answer>minimal answer</answer>. Do not add whitespace, text, or other tags "
    "before <think>, between </think> and <answer>, or after </answer>. Keep the think summary brief. Inside "
    "<answer>, give only the shortest complete answer requested by the question, without explanation, citation, "
    "parenthetical detail, biography, or follow-up. Do not claim that retrieval, search, tools, or documents were used."
)
USER_PROMPT = "Answer directly with the shortest complete answer.\nQuestion: {question}"
EXPECTED_RETRIEVAL_SHA256 = "fec609652d3832c7a6c0ee2861c6f946b6cf7c3d3d40fc5d9be9b75df6325dcb"
EXPECTED_DPO_SHA256 = "c42adcb0f194cff3126134b37afd85e4b89aa9917e5c98dda4b09904509f61e9"
EXPECTED_RL_SHA256 = "c3cc21e862a8469105de666101578cbff23cdc77e91a803cef102622c89cc4f6"
EXPECTED_RL_ROWS = 169_615
EXPECTED_RL_NQ_ROWS = 79_168
RL_SELECTED_NQ_ROWS = 60_298
RL_SELECTION_SEED = 20_260_708
SOURCE_QUOTAS = {"nq": 300, "web_questions": 300}


class CandidateRejected(Exception):
    """A single model response is unsuitable for the released dataset."""


class APIError(Exception):
    """A retriable or terminal provider failure without response-body leakage."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def normalize_question(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = " ".join(text.split())
    return text.rstrip("?？").strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: row is not an object")
            rows.append(row)
    return rows


def read_arrow(path: Path, source: str) -> list[dict[str, Any]]:
    with pa.memory_map(str(path), "r") as mapped:
        table = ipc.open_stream(mapped).read_all().select(["id", "question", "golden_answers"])
    rows: list[dict[str, Any]] = []
    for raw in table.to_pylist():
        question = " ".join(str(raw.get("question") or "").split())
        aliases = [canonicalize_answer(x) for x in raw.get("golden_answers") or []]
        aliases = list(dict.fromkeys(x for x in aliases if normalize_answer(x)))
        normalized = normalize_question(question)
        rows.append({"source_id": str(raw.get("id") or ""), "data_source": source,
                     "split": "train", "question": question, "golden_answers": aliases,
                     "normalized_question": normalized})
    return rows


def checked_questions(path: Path, expected_sha256: str, expected_rows: int) -> tuple[set[str], str]:
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise RuntimeError(f"dataset_identity_mismatch:{path}:{digest}")
    rows = read_jsonl(path)
    if len(rows) != expected_rows:
        raise RuntimeError(f"dataset_row_count_mismatch:{path}:{len(rows)}")
    questions = {normalize_question(row.get("question")) for row in rows}
    questions.discard("")
    if len(questions) != expected_rows:
        raise RuntimeError(f"dataset_question_identity_mismatch:{path}:{len(questions)}")
    return questions, digest


def rl_exclusions(path: Path, nq_candidates: list[dict[str, Any]]) -> tuple[set[str], dict[str, Any]]:
    digest = sha256_file(path)
    if digest != EXPECTED_RL_SHA256:
        raise RuntimeError(f"rl_dataset_identity_mismatch:{digest}")
    table = pq.read_table(path, columns=["id", "question", "data_source"])
    if table.num_rows != EXPECTED_RL_ROWS:
        raise RuntimeError(f"rl_dataset_row_count_mismatch:{table.num_rows}")
    nq = table.filter(pc.equal(table["data_source"], "nq"))
    if nq.num_rows != EXPECTED_RL_NQ_ROWS:
        raise RuntimeError(f"rl_nq_row_count_mismatch:{nq.num_rows}")
    rl_ids = nq["id"].to_pylist()
    rl_questions = nq["question"].to_pylist()
    if len(nq_candidates) != EXPECTED_RL_NQ_ROWS:
        raise RuntimeError(f"raw_nq_row_count_mismatch:{len(nq_candidates)}")
    for position, candidate in enumerate(nq_candidates):
        if candidate["source_id"] != str(rl_ids[position]):
            raise RuntimeError(f"nq_source_id_order_mismatch:{position}")
        if candidate["normalized_question"] != normalize_question(rl_questions[position]):
            raise RuntimeError(f"nq_source_question_order_mismatch:{position}")
    selected = np.random.RandomState(RL_SELECTION_SEED).choice(
        EXPECTED_RL_NQ_ROWS, size=RL_SELECTED_NQ_ROWS, replace=False
    )
    excluded = {normalize_question(rl_questions[int(position)]) for position in selected}
    return excluded, {"sha256": digest, "source_rows": table.num_rows,
                      "nq_rows": nq.num_rows, "selected_nq_rows": RL_SELECTED_NQ_ROWS,
                      "selected_unique_nq_questions": len(excluded),
                      "selection_seed": RL_SELECTION_SEED}


def stable_candidates(rows: Iterable[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: hashlib.sha256(
        f"{seed}\0{row['data_source']}\0{row['source_id']}\0{row['normalized_question']}".encode("utf-8")
    ).digest())


class _NoProxyClient:
    def __init__(self, key: str, timeout: float, retries: int):
        self.key = key
        self.timeout = timeout
        self.retries = retries
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int, float]:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        for attempt in range(1, self.retries + 2):
            request = urllib.request.Request(
                API_URL, data=encoded, method="POST",
                headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"},
            )
            started = time.perf_counter()
            try:
                with self.opener.open(request, timeout=self.timeout) as stream:
                    raw = stream.read(2_097_153)
            except urllib.error.HTTPError as exc:
                code = exc.code
                exc.close()
                if code not in {429, 500, 502, 503, 504} or attempt > self.retries:
                    raise APIError(f"deepseek_http_{code}") from None
            except (urllib.error.URLError, TimeoutError, OSError, ssl.SSLError):
                if attempt > self.retries:
                    raise APIError("deepseek_connection_failed") from None
            else:
                if len(raw) > 2_097_152:
                    raise APIError("deepseek_response_too_large")
                try:
                    response = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    raise APIError("deepseek_invalid_json") from None
                if not isinstance(response, dict):
                    raise APIError("deepseek_invalid_response")
                return response, attempt, time.perf_counter() - started
            time.sleep(min(2 ** (attempt - 1), 8))
        raise APIError("deepseek_retry_limit")


_thread_local = threading.local()


def thread_client(key: str, timeout: float, retries: int) -> _NoProxyClient:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = _NoProxyClient(key, timeout, retries)
        _thread_local.client = client
    return client


def payload_for(question: str, model: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(question=question)},
        ],
        "thinking": {"type": THINKING},
        "max_tokens": MAX_API_TOKENS,
        "stream": False,
    }
    if "tools" in payload or "tool_choice" in payload:
        raise AssertionError("direct_answer_payload_must_not_register_tools")
    return payload


def parse_response(raw: dict[str, Any], aliases: list[str]) -> dict[str, Any]:
    if not isinstance(raw.get("id"), str) or not raw["id"]:
        raise CandidateRejected("missing_api_receipt")
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise CandidateRejected("invalid_model_response")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise CandidateRejected("incomplete_model_output")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise CandidateRejected("invalid_model_response")
    if message.get("tool_calls") not in (None, []):
        raise CandidateRejected("unexpected_tool_call")
    if message.get("reasoning_content") not in (None, ""):
        raise CandidateRejected("unexpected_reasoning_content")
    content = message.get("content")
    if not isinstance(content, str) or FINAL_ACTION.fullmatch(content) is None:
        raise CandidateRejected("invalid_direct_answer_action")
    reason, answer = FINAL_ACTION.fullmatch(content).groups()  # type: ignore[union-attr]
    if DOC_CITATION.search(reason) or re.search(r"\b(?:retrieved|retrieval|searched|tool result)\b", reason, re.I):
        raise CandidateRejected("false_retrieval_claim")
    try:
        parse_action(content, "answer")
    except Rejected as exc:
        raise CandidateRejected(str(exc)) from None
    normalized_answer = normalize_answer(answer)
    matches = [alias for alias in aliases if normalize_answer(alias) == normalized_answer]
    if not normalized_answer or not matches:
        raise CandidateRejected("answer_not_normalized_gold_match")
    clean_answer = canonical_answer(answer)
    training_action = f"<think>{DIRECT_FINAL_THINK}</think><answer>{clean_answer}</answer>"
    try:
        parse_action(training_action, "answer")
    except Rejected as exc:
        raise CandidateRejected(str(exc)) from None
    return {"raw_action": content, "raw_think": reason, "raw_answer": answer,
            "training_answer": clean_answer, "training_action": training_action,
            "matched_gold_alias": matches[0],
            "student_action_tokens": student_budget().check_action(training_action, "answer")}


def attempt(candidate: dict[str, Any], key: str, model: str, timeout: float, retries: int) -> dict[str, Any]:
    payload = payload_for(candidate["question"], model)
    started = time.perf_counter()
    try:
        raw, http_attempts, request_seconds = thread_client(key, timeout, retries).post(payload)
        parsed = parse_response(raw, candidate["golden_answers"])
    except (CandidateRejected, APIError) as exc:
        return {**candidate, "status": "rejected", "reason": str(exc),
                "seconds": time.perf_counter() - started, "request_sha256": sha256_json(payload)}
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    return {**candidate, **parsed, "status": "accepted", "reason": None,
            "seconds": time.perf_counter() - started, "request_seconds": request_seconds,
            "http_attempts": http_attempts, "response_id": raw["id"],
            "response_model": raw.get("model"), "usage": usage,
            "request_sha256": sha256_json(payload),
            "response_sha256": sha256_json(raw)}


def open_checkpoint(path: Path, identity: dict[str, Any]) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS attempts (candidate_key TEXT PRIMARY KEY, source TEXT NOT NULL, "
               "source_id TEXT NOT NULL, status TEXT NOT NULL, reason TEXT, result_json TEXT NOT NULL, created REAL NOT NULL)")
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True)
    old = db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
    if old is not None and old[0] != encoded:
        raise RuntimeError("checkpoint_identity_mismatch")
    db.execute("INSERT OR IGNORE INTO metadata(key,value) VALUES('identity',?)", (encoded,))
    db.commit()
    return db


def candidate_key(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        f"{row['data_source']}\0{row['source_id']}\0{row['normalized_question']}".encode("utf-8")
    ).hexdigest()


def checkpoint_rows(db: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    return {key: json.loads(value) for key, value in db.execute("SELECT candidate_key,result_json FROM attempts")}


def save_attempt(db: sqlite3.Connection, result: dict[str, Any]) -> None:
    key = candidate_key(result)
    db.execute("INSERT OR REPLACE INTO attempts(candidate_key,source,source_id,status,reason,result_json,created) "
               "VALUES(?,?,?,?,?,?,?)", (key, result["data_source"], result["source_id"], result["status"],
                                         result.get("reason"), json.dumps(result, ensure_ascii=False), time.time()))
    db.commit()


def detailed_record(result: dict[str, Any], public_id: str, serial: int, model: str) -> dict[str, Any]:
    answer = canonical_answer(result["raw_answer"])
    action = f"<think>{DIRECT_FINAL_THINK}</think><answer>{answer}</answer>"
    parse_action(action, "answer")
    action_tokens = student_budget().check_action(action, "answer")
    public_prompt = PUBLIC_USER_PROMPT.format(question=result["question"])
    return {
        "id": f"deepseek_direct_answer_{serial:06d}",
        "data_source": result["data_source"],
        "split": "train",
        "trajectory_type": "teacher_direct_answer_real_rollout",
        "question": result["question"],
        "golden_answers": result["golden_answers"],
        "prompt": public_prompt,
        "messages": [{"role": "user", "content": public_prompt},
                     {"role": "assistant", "content": action}],
        "response": action,
        "events": [{"role": "assistant", "text": action, "train_on_tokens": True}],
        "metadata": {
            "public_id": public_id,
            "search_count": 0,
            "answer_source": "deepseek_prior_knowledge",
            "teacher_model": model,
            "teacher_thinking": THINKING,
            "registered_tools": [],
            "prompt_version": PROMPT_VERSION,
            "generator_version": GENERATOR_VERSION,
            "export_policy_version": EXPORT_POLICY_VERSION,
            "overlap_policy": OVERLAP_POLICY_VERSION,
            "raw_teacher_action": result["raw_action"],
            "raw_teacher_action_sha256": hashlib.sha256(result["raw_action"].encode("utf-8")).hexdigest(),
            "matched_gold_alias": result["matched_gold_alias"],
            "normalized_answer_match": True,
            "student_action_tokens": action_tokens,
            "api_response_id": result["response_id"],
            "api_response_model": result.get("response_model"),
            "api_request_sha256": result["request_sha256"],
            "api_response_sha256": result["response_sha256"],
            "api_usage": result.get("usage") or {},
        },
    }


def public_from_detailed(record: dict[str, Any]) -> dict[str, Any]:
    row = {"id": record["metadata"]["public_id"], "question": record["question"],
           "trajectory_type": "direct_answer", "search_count": 0,
           "full_trajectory_text": public_prefix(record["question"]) + record["response"] + EOT}
    errors = validate_public_record(row)
    if errors:
        raise RuntimeError("public_record_invalid:" + ",".join(errors))
    return row


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_release(public: list[dict[str, Any]], detailed: list[dict[str, Any]], exclusions: dict[str, set[str]]) -> None:
    if len(public) != sum(SOURCE_QUOTAS.values()) or len(detailed) != len(public):
        raise RuntimeError("release_row_count_mismatch")
    ids, questions, counts = set(), set(), Counter()
    for row, audit in zip(public, detailed):
        if tuple(row) != PUBLIC_FIELDS or validate_public_record(row):
            raise RuntimeError("invalid_public_record")
        if row["id"] in ids or row["question"] != audit["question"]:
            raise RuntimeError("release_id_or_pairing_mismatch")
        ids.add(row["id"])
        normalized = normalize_question(row["question"])
        if normalized in questions or any(normalized in values for values in exclusions.values()):
            raise RuntimeError("release_question_overlap")
        questions.add(normalized)
        counts[audit["data_source"]] += 1
        if row["trajectory_type"] != "direct_answer" or row["search_count"] != 0:
            raise RuntimeError("release_not_direct_answer")
        body = row["full_trajectory_text"][len(public_prefix(row["question"])):-len(EOT)]
        _, answer = parse_action(body, "answer")
        if normalize_answer(answer) not in {normalize_answer(x) for x in audit["golden_answers"]}:
            raise RuntimeError("release_answer_not_gold_match")
        if any(tag in body for tag in ("<search>", "<information>")):
            raise RuntimeError("release_contains_retrieval")
    if dict(counts) != SOURCE_QUOTAS:
        raise RuntimeError(f"release_source_quota_mismatch:{dict(counts)}")


def write_review(path: Path, detailed: list[dict[str, Any]], count: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = sorted(detailed, key=lambda row: hashlib.sha256(
        f"20260929\0{row['data_source']}\0{row['question']}".encode("utf-8")
    ).digest())[:count]
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"DeepSeek direct-answer review packet ({len(detailed)} accepted)\n\n")
        for index, row in enumerate(selected, 1):
            stream.write(f"[{index}] public_id={row['metadata']['public_id']} source={row['data_source']}\n")
            stream.write(f"Question: {row['question']}\n")
            stream.write(f"Golden aliases: {json.dumps(row['golden_answers'], ensure_ascii=False)}\n")
            stream.write(f"Raw teacher action: {row['metadata']['raw_teacher_action']}\n")
            stream.write(f"Training action: {row['response']}\n\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(os.environ.get("AETHERSEARCH_SFT_WORKSPACE", Path(__file__).resolve().parents[4]))
    parser.add_argument("--workspace", type=Path, default=root)
    parser.add_argument("--nq-arrow", type=Path, default=root / "data/raw/flashrag/nq/train/data-00000-of-00001.arrow")
    parser.add_argument("--webq-arrow", type=Path, default=root / "data/raw/flashrag/web_questions/train/data-00000-of-00001.arrow")
    parser.add_argument(
        "--retrieval-input",
        type=Path,
        default=root / "data/release_inputs/retrieval_trajectories.jsonl",
    )
    parser.add_argument("--dpo-input", type=Path, default=root / "data/exclusion_sources/aethersearch_dpo_2126.jsonl")
    parser.add_argument("--rl-train", type=Path, default=root / "data/exclusion_sources/nq_hotpotqa_train.parquet")
    parser.add_argument("--output", type=Path, default=root / "data/search_sft_teacher/deepseek_direct_answer_600.jsonl")
    parser.add_argument("--audit-output", type=Path, default=root / "data/search_sft_teacher/deepseek_direct_answer_600_audit.jsonl")
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=root / "data/search_sft_teacher/AetherSearch_SFT_2600_unshuffled.jsonl",
    )
    parser.add_argument("--manifest", type=Path, default=root / "data/search_sft_teacher/deepseek_direct_answer_600_manifest.json")
    parser.add_argument("--review", type=Path, default=root / "data/search_sft_teacher/review_packet_deepseek_direct_answer_600.txt")
    parser.add_argument("--checkpoint", type=Path, default=root / "logs/search_sft_teacher/deepseek_direct_answer_600.sqlite")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--key-file", type=Path, default=KEY_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model != MODEL or not 1 <= args.concurrency <= 64 or args.max_attempts < 600:
        raise SystemExit("invalid generation limits or model")
    required = [args.nq_arrow, args.webq_arrow, args.retrieval_input, args.dpo_input, args.rl_train]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise SystemExit("missing required source files: " + ", ".join(missing))
    started = time.perf_counter()
    key = load_key_file(args.key_file)
    nq = read_arrow(args.nq_arrow, "nq")
    webq = read_arrow(args.webq_arrow, "web_questions")
    retrieval_questions, retrieval_sha = checked_questions(
        args.retrieval_input, EXPECTED_RETRIEVAL_SHA256, 2000
    )
    dpo_questions, dpo_sha = checked_questions(args.dpo_input, EXPECTED_DPO_SHA256, 2126)
    rl_questions, rl_identity = rl_exclusions(args.rl_train, nq)
    exclusions = {
        "retrieval_trajectories": retrieval_questions,
        "dpo": dpo_questions,
        "rl_train": rl_questions,
    }
    pools: dict[str, list[dict[str, Any]]] = {}
    overlap_counts: dict[str, dict[str, int]] = {}
    for source, rows in (("nq", nq), ("web_questions", webq)):
        overlap_counts[source] = {name: sum(row["normalized_question"] in values for row in rows)
                                  for name, values in exclusions.items()}
        filtered, seen = [], set()
        for row in rows:
            normalized = row["normalized_question"]
            if (not normalized or not row["golden_answers"] or normalized in seen
                    or any(normalized in values for values in exclusions.values())):
                continue
            seen.add(normalized)
            filtered.append(row)
        pools[source] = stable_candidates(filtered, args.seed)
        if len(pools[source]) < SOURCE_QUOTAS[source]:
            raise SystemExit(f"insufficient non-overlapping {source} candidates: {len(pools[source])}")
    identity = {"generator_version": GENERATOR_VERSION, "prompt_version": PROMPT_VERSION,
                "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(), "model": args.model,
                "thinking": THINKING, "tools": [], "seed": args.seed, "quotas": SOURCE_QUOTAS,
                "retrieval_sha256": retrieval_sha, "dpo_sha256": dpo_sha,
                "rl_sha256": rl_identity["sha256"]}
    db = open_checkpoint(args.checkpoint, identity)
    previous = checkpoint_rows(db)
    accepted: dict[str, list[dict[str, Any]]] = {source: [] for source in SOURCE_QUOTAS}
    for result in previous.values():
        if result.get("status") == "accepted" and result.get("data_source") in accepted:
            accepted[result["data_source"]].append(result)
    for source in accepted:
        accepted[source] = stable_candidates(accepted[source], args.seed)[:SOURCE_QUOTAS[source]]
    attempted = len(previous)
    rejected = Counter(result.get("reason") for result in previous.values() if result.get("status") != "accepted")
    cursors = {source: 0 for source in SOURCE_QUOTAS}
    print(json.dumps({"event": "preflight", "pool_sizes": {k: len(v) for k, v in pools.items()},
                      "overlap_counts": overlap_counts, "resumed_attempts": attempted,
                      "resumed_accepted": {k: len(v) for k, v in accepted.items()}}, sort_keys=True), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        while any(len(accepted[source]) < quota for source, quota in SOURCE_QUOTAS.items()):
            if attempted >= args.max_attempts:
                break
            wave: list[dict[str, Any]] = []
            active = [source for source, quota in SOURCE_QUOTAS.items() if len(accepted[source]) < quota]
            slots = max(1, args.concurrency // len(active))
            for source in active:
                needed = SOURCE_QUOTAS[source] - len(accepted[source])
                while len([x for x in wave if x["data_source"] == source]) < min(slots, needed):
                    if cursors[source] >= len(pools[source]):
                        break
                    candidate = pools[source][cursors[source]]
                    cursors[source] += 1
                    if candidate_key(candidate) in previous:
                        continue
                    wave.append(candidate)
            if not wave:
                break
            if attempted + len(wave) > args.max_attempts:
                wave = wave[:args.max_attempts - attempted]
            futures = [executor.submit(attempt, row, key, args.model, args.timeout, args.retries) for row in wave]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                attempted += 1
                previous[candidate_key(result)] = result
                save_attempt(db, result)
                if result["status"] == "accepted" and len(accepted[result["data_source"]]) < SOURCE_QUOTAS[result["data_source"]]:
                    accepted[result["data_source"]].append(result)
                elif result["status"] != "accepted":
                    rejected[result["reason"]] += 1
            print(json.dumps({"event": "progress", "attempted": attempted,
                              "accepted": {k: len(v) for k, v in accepted.items()},
                              "top_rejections": rejected.most_common(6)}, sort_keys=True), flush=True)
    first_attempt, last_attempt = db.execute("SELECT MIN(created),MAX(created) FROM attempts").fetchone()
    db.close()
    if any(len(accepted[source]) != quota for source, quota in SOURCE_QUOTAS.items()):
        raise SystemExit(f"target_not_reached attempted={attempted} accepted=" +
                         json.dumps({k: len(v) for k, v in accepted.items()}, sort_keys=True))
    chosen = accepted["nq"] + accepted["web_questions"]
    chosen = stable_candidates(chosen, args.seed + 1)
    detailed = [detailed_record(result, f"{2001 + index:06d}", index + 1, args.model)
                for index, result in enumerate(chosen)]
    public = [public_from_detailed(row) for row in detailed]
    validate_release(public, detailed, exclusions)
    retrieval = read_jsonl(args.retrieval_input)
    if [row.get("id") for row in retrieval] != [f"{n:06d}" for n in range(1, 2001)]:
        raise RuntimeError("retrieval_input_id_sequence_mismatch")
    combined = retrieval + public
    if len({normalize_question(row["question"]) for row in combined}) != len(combined):
        raise RuntimeError("combined_dataset_question_overlap")
    atomic_jsonl(args.audit_output, detailed)
    atomic_jsonl(args.output, public)
    atomic_jsonl(args.combined_output, combined)
    write_review(args.review, detailed)
    candidate_seconds = [float(result.get("seconds") or 0.0) for result in previous.values()]
    manifest = {
        "generator_version": GENERATOR_VERSION, "prompt_version": PROMPT_VERSION,
        "export_policy_version": EXPORT_POLICY_VERSION,
        "overlap_policy_version": OVERLAP_POLICY_VERSION, "teacher_model": args.model,
        "thinking": THINKING, "tools_registered": [], "api_max_tokens": MAX_API_TOKENS,
        "records": len(public), "source_distribution": dict(Counter(row["data_source"] for row in detailed)),
        "trajectory_distribution": {"direct_answer": len(public)}, "search_count_distribution": {"0": len(public)},
        "attempted": attempted, "skip_reasons": dict(sorted(rejected.items())),
        "input_pool_sizes": {"nq": len(nq), "web_questions": len(webq)},
        "eligible_pool_sizes": {k: len(v) for k, v in pools.items()}, "overlap_counts": overlap_counts,
        "exclusion_sources": {"retrieval_trajectories": {
                                  "rows": len(retrieval_questions), "sha256": retrieval_sha},
                              "dpo": {"rows": len(dpo_questions), "sha256": dpo_sha},
                              "rl_train": rl_identity},
        "student_budget": student_budget().specification(), "public_format_fields": list(PUBLIC_FIELDS),
        "outputs": {"direct_answer_trajectories": str(args.output),
                    "direct_answer_audit": str(args.audit_output),
                    "unshuffled_2600": str(args.combined_output), "review": str(args.review),
                    "checkpoint": str(args.checkpoint)},
        "sha256": {"direct_answer_trajectories": sha256_file(args.output),
                   "direct_answer_audit": sha256_file(args.audit_output),
                   "unshuffled_2600": sha256_file(args.combined_output)},
        "elapsed_seconds": (last_attempt - first_attempt) if first_attempt is not None else 0.0,
        "export_run_elapsed_seconds": time.perf_counter() - started,
        "sum_candidate_processing_seconds": sum(candidate_seconds),
        "mean_candidate_processing_seconds": (sum(candidate_seconds) / len(candidate_seconds)) if candidate_seconds else 0.0,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "records": len(public), "attempted": attempted,
                      "elapsed_seconds": manifest["elapsed_seconds"], "outputs": manifest["outputs"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
