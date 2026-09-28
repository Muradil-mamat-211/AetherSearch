#!/usr/bin/env python3
"""Real DeepSeek -> local Hybrid-RAG -> DeepSeek rollout with per-question checkpoints."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import signal
import sqlite3
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from controlled_rollout import (BM25_DB, CONTINUATION_POLICY_VERSION, CORPUS, DEEPSEEK_GENERATOR_VERSION, DENSE_URL, E5_MODEL, FAISS_INDEX,
                                FINAL_SUMMARY_POLICY_VERSION,
                                INSTRUCTIONS, PROMPT, PROMPT_VERSION, QUERY_POLICY_VERSION, SEARCH_ACTION_SOURCE, SEARCH_EXTRACTION_POLICY, TOOL,
                                Rejected, asset_signature, parse_action, resource_status, rollout, response_items)
from deepseek_client import (API_URL, DEFAULT_MAX_CUMULATIVE_REASONING_TOKENS,
                             DEFAULT_NONTHINKING_MAX_TOKENS, DEFAULT_THINKING_MAX_TOKENS,
                             MAX_API_OUTPUT_TOKENS, MAX_API_REQUEST_BYTES, MAX_ANSWER_REPAIRS, TOOL_POLICY_VERSION,
                             APIError, DeepSeekClient, SharedRequestBudget)
from batched_retriever import BatchedRetrieverCoordinator
from deepseek_key import KeyFileError, load_key_file
from published_sft_format import MAX_FULL_TRAJECTORY_TOKENS, PUBLIC_FORMAT_VERSION, public_record
from validate_teacher_rollout import validate_record
from token_budget import MAX_SEARCH_TURNS, student_budget


VERSION = DEEPSEEK_GENERATOR_VERSION
SCHEMA = "deepseek_teacher_checkpoint_v22_faiss_flat"
RETRIEVAL_DEADLINE_SECONDS = 120
DEFAULT_PUBLIC_ID_START = 500001


@contextlib.contextmanager
def retrieval_deadline(seconds: float = RETRIEVAL_DEADLINE_SECONDS) -> Iterator[None]:
    """Bound the complete synchronous retrieval, including a slow HTTP body."""
    if threading.current_thread() is not threading.main_thread() or signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError("retrieval_deadline_requires_main_thread_without_existing_timer")
    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum: int, frame: Any) -> None:
        raise TimeoutError("retrieval_deadline_exceeded")

    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def dense_command_matches(argv: list[str], cwd: Path) -> bool:
    """Only accept the official dense server bound to this FlatIP asset."""
    if not any(arg.endswith("search_r1/search/retrieval_server.py") for arg in argv):
        return False
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    for flag in ("index_path", "corpus_path", "retriever_model", "retriever_name"):
        parser.add_argument("--" + flag)
    parser.add_argument("--faiss_gpu", action="store_true")
    try:
        config, _ = parser.parse_known_args(argv[1:])
    except (argparse.ArgumentError, ValueError):
        return False
    def resolve(value: str | None) -> Path | None:
        return (cwd / value).resolve() if value else None
    return (config.retriever_name == "e5" and config.faiss_gpu
            and resolve(config.index_path) == FAISS_INDEX
            and resolve(config.corpus_path) == CORPUS.resolve()
            and resolve(config.retriever_model) == E5_MODEL.resolve())


def dense_process_verified() -> bool:
    """Check the process owning port 8000, not just an unrelated open TCP port."""
    inodes: set[str] = set()
    for name in ("tcp", "tcp6"):
        try:
            for line in Path("/proc/net", name).read_text().splitlines()[1:]:
                fields = line.split()
                if fields[3] == "0A" and fields[1].rsplit(":", 1)[1] == "1F40":
                    inodes.add(fields[9])
        except (OSError, IndexError):
            continue
    if not inodes:
        return False
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            owns_port = any(os.readlink(fd) in {"socket:[" + inode + "]" for inode in inodes} for fd in (proc / "fd").iterdir())
            if not owns_port:
                continue
            argv = (proc / "cmdline").read_bytes().decode("utf-8", "replace").split("\0")
            cwd = (proc / "cwd").resolve()
            if dense_command_matches(argv, cwd):
                return True
        except (OSError, argparse.ArgumentError, ValueError):
            continue
    return False


def doctor() -> dict[str, Any]:
    status = resource_status()
    status.pop("model_api_key_set", None)
    try:
        load_key_file()
        status["deepseek_key_file_readable"] = True
    except KeyFileError:
        status["deepseek_key_file_readable"] = False
    status["e5_config"] = (E5_MODEL / "config.json").is_file()
    status["e5_tokenizer"] = any((E5_MODEL / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json"))
    status["e5_weights"] = any((E5_MODEL / name).is_file() for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"))
    status["dense_process_config_verified"] = dense_process_verified()
    try:
        student_budget()
        status["student_tokenizer_ready"] = True
    except (RuntimeError, OSError, ValueError):
        status["student_tokenizer_ready"] = False
    probe_error = None
    status["hybrid_retrieval_probe"] = False
    rag_checks = ("uncompressed_wiki18_corpus", "bm25_index", "faiss_flat_index", "e5_model",
                  "retriever_env", "dense_server_listening", "dense_process_config_verified",
                  "e5_config", "e5_tokenizer", "e5_weights")
    if all(status.get(name) for name in rag_checks):
        try:
            probe_hybrid_retriever()
            status["hybrid_retrieval_probe"] = True
        except Exception as exc:
            probe_error = f"{type(exc).__name__}: {str(exc)[:200]}"
    return {"ready": all(status.values()), "checks": status, "api_called": False,
            "dense_url": DENSE_URL, "retriever_probe_error": probe_error,
            "training_data_created": False,
            "tool_policy": {"version": TOOL_POLICY_VERSION, "api_endpoint": API_URL,
                            "strict_requested": True, "allowed_tools": ["retrieve"],
                            "search_policy": "adaptive", "retrieval_deadline_seconds": RETRIEVAL_DEADLINE_SECONDS,
                            "os_network_isolation": "not_verified", "provider_internal_network": "not_auditable"}}


class CheckedDenseBranch:
    """Verify dense passage identity against the corpus-backed BM25 document table."""

    def __init__(self, branch: Any, connection: sqlite3.Connection):
        self.branch, self.connection = branch, connection

    def search(self, query: str, topn: int) -> list[dict[str, Any]]:
        docs = self.branch.search(query, topn)
        self._check(docs, topn)
        return docs

    def search_batch(self, queries: list[str], topn: int) -> list[list[dict[str, Any]]]:
        batches = self.branch.search_batch(queries, topn)
        if len(batches) != len(queries):
            raise RuntimeError("dense_batch_result_count_invalid")
        for docs in batches:
            self._check(docs, topn)
        return batches

    def _check(self, docs: list[dict[str, Any]], topn: int) -> None:
        if len(docs) != topn or len({doc["doc_id"] for doc in docs}) != topn:
            raise RuntimeError("dense_candidate_count_or_identity_invalid")
        for doc in docs:
            row = self.connection.execute("SELECT title, text FROM docs WHERE doc_id=?", (doc["doc_id"],)).fetchone()
            if row is None or tuple(row) != (doc["title"], doc["text"]):
                raise RuntimeError("dense_passage_does_not_match_wiki18")


class CheckedBM25Branch:
    def __init__(self, branch: Any):
        self.branch, self.conn = branch, branch.conn

    def search(self, query: str, topn: int) -> list[dict[str, Any]]:
        docs = self.branch.search(query, topn)
        if len(docs) != topn or len({doc["doc_id"] for doc in docs}) != topn:
            raise RuntimeError("bm25_candidate_count_or_identity_invalid")
        return docs

    def close(self) -> None:
        self.branch.close()


class VerifiedHybridRetriever:
    def __init__(self):
        from hybrid_retriever_v1 import HybridRetrieverV1
        self.hybrid = HybridRetrieverV1(auto_build_bm25=False, dense_url=DENSE_URL,
                                        candidate_topn=20, rrf_k=60, log_path=None, restricted_dense=True)
        self.hybrid.bm25 = CheckedBM25Branch(self.hybrid.bm25)
        self.original_dense = self.hybrid.dense
        self.hybrid.dense = CheckedDenseBranch(self.original_dense, self.hybrid.bm25.conn)
        self.trace: list[dict[str, Any]] = []

    def retrieve(self, query: str, topk: int) -> Any:
        if topk != 3:
            raise ValueError("restricted_retriever_topk_must_be_three")
        with retrieval_deadline():
            return self._retrieve(query, topk)

    def _retrieve(self, query: str, topk: int) -> Any:
        from hybrid_retriever_v1 import docs_to_jsonable, format_information
        started = time.perf_counter()
        docs = self.hybrid.retrieve(query, topk)
        self._check_docs(docs)
        self.trace.append({"query": query, "documents": docs_to_jsonable(docs),
                           "information": format_information(docs), "seconds": time.perf_counter() - started})
        return docs

    def retrieve_batch(self, queries: list[str], topk: int) -> list[Any]:
        if topk != 3 or not 1 <= len(queries) <= 8:
            raise ValueError("restricted_retriever_batch_invalid")
        if len(queries) == 1:
            return [self.retrieve(queries[0], topk)]
        with retrieval_deadline():
            started = time.perf_counter()
            batches = self.hybrid.retrieve_batch(queries, topk)
            if len(batches) != len(queries):
                raise RuntimeError("hybrid_batch_result_count_invalid")
            from hybrid_retriever_v1 import docs_to_jsonable, format_information
            for query, docs in zip(queries, batches, strict=True):
                self._check_docs(docs)
                self.trace.append({"query": query, "documents": docs_to_jsonable(docs),
                                   "information": format_information(docs), "seconds": time.perf_counter() - started})
            return batches

    def _check_docs(self, docs: Any) -> None:
        for doc in docs:
            ranks = [r for r in (doc.bm25_rank, doc.dense_rank) if r is not None]
            if not ranks or not math.isclose(doc.rrf_score, sum(1 / (60 + rank) for rank in ranks), rel_tol=1e-9):
                raise RuntimeError("rrf_score_invalid")
            row = self.hybrid.bm25.conn.execute("SELECT title, text FROM docs WHERE doc_id=?", (doc.doc_id,)).fetchone()
            if row is None or tuple(row) != (doc.title, doc.text):
                raise RuntimeError("fused_passage_does_not_match_wiki18")

    def close(self) -> None:
        self.original_dense.session.close()
        self.hybrid.close()


class LazyHybridRetriever:
    """Load the sole real backend once, only after a validated search decision."""

    def __init__(self):
        self.backend: VerifiedHybridRetriever | None = None
        self.empty_trace: list[dict[str, Any]] = []

    @property
    def trace(self) -> list[dict[str, Any]]:
        return self.backend.trace if self.backend is not None else self.empty_trace

    def retrieve(self, query: str, topk: int) -> Any:
        if self.backend is None:
            self.backend = VerifiedHybridRetriever()
        return self.backend.retrieve(query, topk)

    def retrieve_batch(self, queries: list[str], topk: int) -> list[Any]:
        if self.backend is None:
            self.backend = VerifiedHybridRetriever()
        return self.backend.retrieve_batch(queries, topk)

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()


def probe_hybrid_retriever() -> None:
    """Exercise both real branches and their provenance checks before paid API calls."""
    backend = VerifiedHybridRetriever()
    try:
        docs = backend.retrieve("United States", 3)
        if len(docs) != 3 or any(not doc.title.strip() or not doc.text.strip() for doc in docs):
            raise RuntimeError("hybrid_probe_invalid_top3")
    finally:
        backend.close()


def load_candidates(path: Path, limit: int,
                    public_id_start: int = DEFAULT_PUBLIC_ID_START) -> tuple[list[dict[str, Any]], str]:
    if type(public_id_start) is not int or not 1 <= public_id_start <= 999999 or public_id_start + limit - 1 > 999999:
        raise ValueError("public_id_range_invalid")
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    questions: set[str] = set()
    with path.open("rb") as stream:
        for number, line in enumerate(iter(lambda: stream.readline(1_048_577), b""), 1):
            if len(line) > 1_048_576:
                raise ValueError(f"input_line_too_large:{number}")
            digest.update(line)
            if not line.strip() or len(rows) >= limit:
                continue
            try:
                original = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                raise ValueError(f"invalid_input_json:{number}") from None
            if not isinstance(original, dict):
                raise ValueError(f"invalid_input_row:{number}")
            row = {key: original.get(key) for key in ("id", "question", "golden_answers", "data_source", "split")}
            if not isinstance(row["question"], str) or not row["question"].strip() or not isinstance(row["golden_answers"], list) or not row["golden_answers"] or any(not isinstance(a, str) or not a.strip() for a in row["golden_answers"]):
                raise ValueError(f"invalid_question_or_answers:{number}")
            if not isinstance(row["data_source"], str) or not row["data_source"] or not isinstance(row["split"], str) or not row["split"]:
                raise ValueError(f"missing_source_or_split:{number}")
            original_id = str(row["id"]) if row["id"] is not None and str(row["id"]) else hashlib.sha256(row["question"].encode()).hexdigest()[:20]
            row["id"] = f'{row["data_source"]}:{row["split"]}:{original_id}'
            row["public_id"] = f"{public_id_start + len(rows):06d}"
            if row["id"] in seen:
                raise ValueError(f"duplicate_input_id:{number}")
            question_key = " ".join(row["question"].casefold().split())
            if question_key in questions:
                raise ValueError(f"duplicate_input_question:{number}")
            seen.add(row["id"])
            questions.add(question_key)
            rows.append(row)
    if not rows:
        raise ValueError("empty_questions_file")
    return rows, digest.hexdigest()


@contextlib.contextmanager
def checkpoint(path: Path, *, readonly: bool = False) -> Iterator[sqlite3.Connection]:
    if readonly:
        db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            yield db
        finally:
            db.close()
        return
    if path.is_symlink():
        raise ValueError("checkpoint_symlink_not_allowed")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(str(path) + ".lock", os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("checkpoint_already_in_use") from None
        is_new = not path.exists()
        db = sqlite3.connect(path, timeout=30)
        if is_new:
            path.chmod(0o600)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            yield db
        finally:
            db.close()


def make_client(args: argparse.Namespace, *, smoke: bool = False,
                request_budget: SharedRequestBudget | None = None) -> DeepSeekClient:
    return DeepSeekClient(args.model, load_key_file(), thinking=args.thinking, max_tokens=args.max_tokens,
                          reasoning_effort=getattr(args, "reasoning_effort", None),
                          timeout=args.timeout, retries=0 if smoke else args.api_retries,
                          max_requests=1 if smoke else args.max_api_requests,
                          request_budget=request_budget,
                          max_cumulative_reasoning_tokens=getattr(args, "max_cumulative_reasoning_tokens",
                                                               DEFAULT_MAX_CUMULATIVE_REASONING_TOKENS))


def bind_config(db: sqlite3.Connection, config: dict[str, Any]) -> None:
    db.execute("CREATE TABLE IF NOT EXISTS run_config (id INTEGER PRIMARY KEY CHECK (id=1), config_json TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS rollouts (uid TEXT PRIMARY KEY, public_id TEXT NOT NULL UNIQUE, row_sha256 TEXT NOT NULL, data_source TEXT NOT NULL, status TEXT NOT NULL, record_json TEXT, audit_json TEXT NOT NULL, attempts INTEGER NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS attempt_history (uid TEXT NOT NULL, attempt INTEGER NOT NULL, data_source TEXT NOT NULL, status TEXT NOT NULL, record_json TEXT, audit_json TEXT NOT NULL, PRIMARY KEY(uid,attempt))")
    value = json.dumps(config, sort_keys=True)
    previous = db.execute("SELECT config_json FROM run_config WHERE id=1").fetchone()
    if previous and previous[0] != value:
        raise ValueError("checkpoint_config_changed_use_a_new_checkpoint")
    if previous is None:
        with db:
            db.execute("INSERT INTO run_config VALUES (1, ?)", (value,))


def summary(db: sqlite3.Connection) -> dict[str, Any]:
    statuses, sources, skips, branches = Counter(), defaultdict(Counter), Counter(), Counter()
    source_skips: dict[str, Counter] = defaultdict(Counter)
    searches = info_length = accepted = attempts = 0
    answer_sources, search_counts = Counter(), Counter()
    for source, status, record_json, audit_json, count in db.execute("SELECT data_source,status,record_json,audit_json,attempts FROM rollouts"):
        statuses[status] += 1
        sources[source][status] += 1
        attempts += count
        audit = json.loads(audit_json)
        if status in {"rejected", "error"}:
            reason = audit.get("reason", status)
            skips[reason] += 1
            source_skips[source][reason] += 1
        elif record_json:
            record = json.loads(record_json)
            accepted += 1
            searches += record["metadata"]["search_count"]
            search_counts[record["metadata"]["search_count"]] += 1
            answer_sources[record["metadata"].get("answer_source", "legacy_retrieved_evidence")] += 1
            info_length += sum(len(e["text"]) for e in record["events"] if e["role"] == "environment")
            branches.update(record["metadata"]["evidence_source_branches"])
    historical_skips, historical_sources = Counter(), defaultdict(Counter)
    for source, value in db.execute("SELECT data_source,audit_json FROM attempt_history WHERE status IN ('rejected','error')"):
        reason = json.loads(value).get("reason", "unknown")
        historical_skips[reason] += 1
        historical_sources[source][reason] += 1
    return {"stored_candidates": sum(statuses.values()), "candidate_attempts": attempts, "statuses": dict(statuses),
            "data_sources": {s: dict(c) for s, c in sources.items()}, "skip_reasons": dict(skips),
            "source_level_skip_statistics": {s: dict(c) for s, c in source_skips.items()},
            "historical_attempt_skip_reasons": dict(historical_skips),
            "source_level_attempt_skip_statistics": {s: dict(c) for s, c in historical_sources.items()},
            "evidence_source_branches": dict(branches), "average_search_count": searches / accepted if accepted else 0,
            "answer_sources": dict(answer_sources), "search_count_distribution": dict(search_counts),
            "average_information_chars_per_sample": info_length / accepted if accepted else 0}


def evaluate_candidate(args: argparse.Namespace, row: dict[str, Any], client: DeepSeekClient,
                       retriever: Any) -> tuple[str, dict[str, Any] | None, dict[str, Any], bool]:
    call_start = len(client.calls)
    started = time.perf_counter()
    record: dict[str, Any] | None = None
    fatal = False
    try:
        if row["split"] != "train":
            raise Rejected("non_training_split")
        record, audit = rollout(row, client, retriever, args.max_searches)
        if record["metadata"]["search_count"] == 0:
            raise Rejected("zero_search_not_training_eligible")
        audit["provenance_checked"] = True
        errors = validate_record(record, audit)
        if errors:
            raise Rejected("validator_failed:" + ",".join(errors))
        status = "needs_semantic_review"
    except Rejected as exc:
        status = "rejected"
        audit = {"reason": str(exc), "retrieval_trace": list(retriever.trace)}
        record = None
    except (APIError, RuntimeError, OSError, sqlite3.Error) as exc:
        status, fatal = "error", True
        audit = {"reason": "infrastructure_failure", "failure_type": type(exc).__name__,
                 "api_failure_code": str(exc) if isinstance(exc, APIError) else None,
                 "retrieval_trace": list(retriever.trace)}
        record = None
    audit["api_calls"] = client.calls[call_start:]
    audit["elapsed_seconds"] = time.perf_counter() - started
    audit["status"] = status
    return status, record, audit, fatal


def commit_candidate(db: sqlite3.Connection, row: dict[str, Any], row_sha: str,
                     existing: Any, outcome: tuple[str, dict[str, Any] | None, dict[str, Any], bool],
                     api_requests: int) -> bool:
    status, record, audit, fatal = outcome
    uid = row["id"]
    count = existing[2] + 1 if existing else 1
    with db:
        db.execute("INSERT INTO attempt_history VALUES (?,?,?,?,?,?)", (uid, count, row["data_source"], status,
                   json.dumps(record, ensure_ascii=False) if record else None, json.dumps(audit, ensure_ascii=False)))
        if existing:
            db.execute("UPDATE rollouts SET status=?,record_json=?,audit_json=?,attempts=? WHERE uid=?",
                       (status, json.dumps(record, ensure_ascii=False) if record else None,
                        json.dumps(audit, ensure_ascii=False), count, uid))
        else:
            db.execute("INSERT INTO rollouts VALUES (?,?,?,?,?,?,?,?)", (uid, row["public_id"], row_sha, row["data_source"], status,
                       json.dumps(record, ensure_ascii=False) if record else None, json.dumps(audit, ensure_ascii=False), count))
    print(json.dumps({"event": "candidate_complete", "uid": uid, "status": status,
                      "reason": audit.get("reason"), "attempt": count,
                      "search_count": record["metadata"]["search_count"] if record else None,
                      "answer_source": record["metadata"]["answer_source"] if record else None,
                      "seconds": audit["elapsed_seconds"], "api_requests": api_requests}), flush=True)
    return fatal


def run_concurrent(args: argparse.Namespace, db: sqlite3.Connection, work: list[Any]) -> int:
    backend = LazyHybridRetriever()
    coordinator = BatchedRetrieverCoordinator(backend, max_batch_queries=args.retrieval_batch_queries,
                                              batch_wait_ms=args.retrieval_batch_wait_ms)
    budget = SharedRequestBudget(args.max_api_requests)
    selected = iter(work)
    active: dict[Future[Any], tuple[dict[str, Any], str, Any, DeepSeekClient]] = {}
    fatal_seen = False
    try:
        clients = [make_client(args, request_budget=budget) for _ in range(args.concurrency)]
        with ThreadPoolExecutor(max_workers=args.concurrency, thread_name_prefix="teacher") as pool:
            def schedule(client: DeepSeekClient) -> bool:
                try:
                    row, row_sha, existing = next(selected)
                except StopIteration:
                    return False
                future = pool.submit(evaluate_candidate, args, row, client, coordinator.client())
                active[future] = (row, row_sha, existing, client)
                return True

            for client in clients:
                if not schedule(client):
                    break
            try:
                while active:
                    coordinator.dispatch()
                    for future in [item for item in active if item.done()]:
                        row, row_sha, existing, client = active.pop(future)
                        try:
                            outcome = future.result()
                        except Exception as exc:
                            outcome = ("error", None,
                                       {"reason": "internal_worker_failure", "failure_type": type(exc).__name__,
                                        "api_calls": client.calls[:], "elapsed_seconds": 0, "status": "error"}, True)
                        fatal = commit_candidate(db, row, row_sha, existing, outcome, budget.count)
                        client.calls.clear()
                        if fatal and not fatal_seen:
                            fatal_seen = True
                            budget.abort()
                        if not fatal_seen:
                            schedule(client)
            except BaseException:
                budget.abort()
                coordinator.fail_pending()
                raise
        if fatal_seen:
            print(json.dumps({"event": "stopped", "api_requests_this_run": budget.count,
                              "retrieval_batches": coordinator.batch_count,
                              "retrieval_queries": coordinator.query_count, **summary(db)}, sort_keys=True), flush=True)
            raise RuntimeError("infrastructure_failure_checkpoint_saved_no_fallback")
        print(json.dumps({"event": "retrieval_batching", "batches": coordinator.batch_count,
                          "queries": coordinator.query_count,
                          "max_observed_batch": coordinator.max_observed_batch}), flush=True)
        return budget.count
    finally:
        backend.close()


def run(args: argparse.Namespace) -> None:
    if not args.questions or not args.db:
        raise ValueError("run_requires_questions_and_db")
    health = doctor()
    if not health["ready"]:
        print(json.dumps({"event": "preflight_failed", **health}, sort_keys=True), flush=True)
        raise RuntimeError("rag_or_credentials_unavailable_no_api_call_no_generation")
    public_id_start = getattr(args, "public_id_start", DEFAULT_PUBLIC_ID_START)
    rows, input_sha = load_candidates(args.questions, args.max_examples, public_id_start)
    retry_ids = set(args.retry_ids.split(",")) if args.retry_ids else set()
    if retry_ids - {row["id"] for row in rows}:
        raise ValueError("retry_ids_not_in_selected_input_window")
    config = {"schema": SCHEMA, "generator": VERSION, "provider": "deepseek", "model": args.model,
              "continuation_policy": CONTINUATION_POLICY_VERSION,
              "public_format": PUBLIC_FORMAT_VERSION, "max_full_trajectory_tokens": MAX_FULL_TRAJECTORY_TOKENS,
              "public_id_start": public_id_start,
              "max_api_request_bytes": MAX_API_REQUEST_BYTES,
              "max_cumulative_reasoning_tokens": getattr(args, "max_cumulative_reasoning_tokens",
                                                      DEFAULT_MAX_CUMULATIVE_REASONING_TOKENS),
              "tool_policy": TOOL_POLICY_VERSION, "api_endpoint": API_URL, "strict_requested": True,
              "rollout_budget": student_budget().specification(), "max_answer_repairs": MAX_ANSWER_REPAIRS,
              "retrieval_deadline_seconds": RETRIEVAL_DEADLINE_SECONDS,
              "search_policy": "adaptive", "prompt_version": PROMPT_VERSION,
              "final_summary_policy": FINAL_SUMMARY_POLICY_VERSION,
              "search_action_source": SEARCH_ACTION_SOURCE,
              "search_extraction_policy": SEARCH_EXTRACTION_POLICY,
              "query_policy": QUERY_POLICY_VERSION,
              "concurrency": getattr(args, "concurrency", 1),
              "retrieval_batch_queries": getattr(args, "retrieval_batch_queries", 8),
              "retrieval_batch_wait_ms": getattr(args, "retrieval_batch_wait_ms", 5.0),
              "prompt_protocol_sha256": hashlib.sha256((INSTRUCTIONS + PROMPT + json.dumps(TOOL, sort_keys=True)).encode()).hexdigest(),
              "thinking": args.thinking, "max_tokens": args.max_tokens, "max_searches": args.max_searches,
              "reasoning_effort": (getattr(args, "reasoning_effort", None) or "high") if args.thinking == "enabled" else None,
              "questions_sha256": input_sha, "dense_url": DENSE_URL, "rrf_k": 60, "candidate_topn": 20,
              "assets": {"corpus": asset_signature(CORPUS), "bm25": asset_signature(BM25_DB),
                         "faiss": {"path": str(FAISS_INDEX), "signature": asset_signature(FAISS_INDEX)},
                         "e5_files": {name: asset_signature(E5_MODEL / name) for name in
                                      ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
                                       "model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")}}}
    started = time.perf_counter()
    with checkpoint(args.db) as db:
        bind_config(db, config)
        if retry_ids:
            for uid in retry_ids:
                existing = db.execute("SELECT status FROM rollouts WHERE uid=?", (uid,)).fetchone()
                if not existing or existing[0] not in {"rejected", "error"}:
                    raise ValueError("retry_only_allows_existing_failed_ids")
        work = []
        for row in rows:
            uid = row["id"]
            if retry_ids and uid not in retry_ids:
                continue
            row_sha = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
            existing = db.execute("SELECT row_sha256,status,attempts,public_id FROM rollouts WHERE uid=?", (uid,)).fetchone()
            if existing:
                if existing[0] != row_sha or existing[3] != row["public_id"]:
                    raise ValueError("checkpoint_question_changed")
                if uid not in retry_ids and existing[1] != "error":
                    continue
            work.append((row, row_sha, existing))
        if not work:
            print(json.dumps({"event": "resume_no_work", "api_requests_this_run": 0, **summary(db)}, sort_keys=True), flush=True)
            return
        if getattr(args, "concurrency", 1) > 1:
            requests = run_concurrent(args, db, work)
            print(json.dumps({"event": "complete", "elapsed_seconds": time.perf_counter() - started,
                              "api_requests_this_run": requests, **summary(db)}, sort_keys=True), flush=True)
            return
        retriever = LazyHybridRetriever()
        try:
            client = make_client(args)
            for row, row_sha, existing in work:
                retriever.trace.clear()
                outcome = evaluate_candidate(args, row, client, retriever)
                fatal = commit_candidate(db, row, row_sha, existing, outcome, client.request_count)
                if fatal:
                    print(json.dumps({"event": "stopped", **summary(db)}, sort_keys=True), flush=True)
                    raise RuntimeError("infrastructure_failure_checkpoint_saved_no_fallback")
            print(json.dumps({"event": "complete", "elapsed_seconds": time.perf_counter() - started,
                              "api_requests_this_run": client.request_count, **summary(db)}, sort_keys=True), flush=True)
        finally:
            retriever.close()


def approve(args: argparse.Namespace) -> None:
    if not args.db or not args.ids or not (args.reviewer or "").strip() or not (args.review_note or "").strip():
        raise ValueError("approve_requires_db_ids_reviewer_and_review_note")
    with checkpoint(args.db) as db:
        selected = []
        for uid in set(args.ids.split(",")):
            value = db.execute("SELECT status,record_json,audit_json FROM rollouts WHERE uid=?", (uid,)).fetchone()
            if not value or value[0] != "needs_semantic_review":
                raise ValueError("approval_only_allows_pending_ids")
            record, audit = json.loads(value[1]), json.loads(value[2])
            if record["metadata"]["search_count"] < 1:
                raise ValueError("zero_search_not_training_eligible")
            if validate_record(record, audit):
                raise ValueError("cannot_approve_invalid_record")
            if record["metadata"]["evidence_support_warning"] and not args.ack_evidence_warning:
                raise ValueError("review_requires_explicit_evidence_warning_acknowledgement")
            approval = {"reviewer": args.reviewer, "note": args.review_note,
                        "time": datetime.now(timezone.utc).isoformat(),
                        "evidence_warning_acknowledged": args.ack_evidence_warning}
            record["metadata"]["semantic_review_status"] = "approved"
            audit["approval"] = approval
            audit["status"] = "approved"
            selected.append((json.dumps(record, ensure_ascii=False), json.dumps(audit, ensure_ascii=False), uid))
        with db:
            db.executemany("UPDATE rollouts SET status='approved',record_json=?,audit_json=? WHERE uid=?", selected)
        print(json.dumps({"approved": len(selected)}))


def export(args: argparse.Namespace) -> None:
    if not args.db or not args.output:
        raise ValueError("export_requires_db_and_new_output")
    status = "needs_semantic_review" if args.export_candidates else "approved"
    with checkpoint(args.db, readonly=True) as db:
        db.execute("BEGIN")
        # Prevalidate the entire selection before creating any output file.
        count = 0
        for record_json, audit_json in db.execute("SELECT record_json,audit_json FROM rollouts WHERE status=?", (status,)):
            record = json.loads(record_json)
            if record["metadata"]["search_count"] < 1:
                raise ValueError("zero_search_not_training_eligible_no_output_written")
            if validate_record(record, json.loads(audit_json), require_approved=not args.export_candidates):
                raise ValueError("export_validation_failed_no_output_written")
            if not args.export_candidates:
                public_record(record)
            count += 1
        if not count:
            raise ValueError("no_rows_in_requested_export_status")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            for (value,) in db.execute("SELECT record_json FROM rollouts WHERE status=? ORDER BY public_id", (status,)):
                stream.write((value if args.export_candidates else
                              json.dumps(public_record(json.loads(value)), ensure_ascii=False)) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({"exported": count, "output": str(args.output), "status": status,
                      "public_format": PUBLIC_FORMAT_VERSION if not args.export_candidates else None,
                      "final_training_data": not args.export_candidates}))


def review_packet(args: argparse.Namespace) -> None:
    if not args.db or not args.output:
        raise ValueError("review_packet_requires_db_and_new_output")
    with checkpoint(args.db, readonly=True) as db:
        rows = db.execute("SELECT uid,status,record_json,audit_json FROM rollouts WHERE record_json IS NOT NULL ORDER BY uid LIMIT ?", (args.max_examples,)).fetchall()
        if not rows:
            raise ValueError("no_candidates_to_review")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write("SEMANTIC REVIEW PACKET - approval requires checking the asked relationship, not just answer overlap.\n\n")
            for uid, status, value, audit_json in rows:
                record = json.loads(value)
                audit = json.loads(audit_json)
                stream.write(f"ID: {uid}\nStatus: {status}\nSource: {record['data_source']}\nQuestion: {record['question']}\n")
                stream.write(f"Answer source: {record['metadata']['answer_source']}\nSearch count: {record['metadata']['search_count']}\n")
                stream.write("Golden answers (reviewer only): " + json.dumps(record["golden_answers"], ensure_ascii=False) + "\n")
                stream.write("Support warning: " + json.dumps({k: record["metadata"][k] for k in ("evidence_support_warning", "missing_question_entities")}, ensure_ascii=False) + "\n")
                for event in record["events"]:
                    stream.write(f"\n{event['role']} (train_on_tokens={event['train_on_tokens']}):\n{event['text']}\n")
                stream.write("\nRaw teacher FINAL (audit only):\n" + audit["raw_final_action"] + "\n")
                stream.write("\n" + "=" * 72 + "\n\n")
    print(json.dumps({"review_examples": len(rows), "output": str(args.output)}))


def smoke(args: argparse.Namespace) -> None:
    client = make_client(args, smoke=True)
    history = [{"role": "user", "content": "API protocol test only: search for the verification marker. Produce one retrieve search action. No real corpus is connected; do not answer."}]
    response = client.create(history, "required")
    call, _ = response_items(response)
    if call is None:
        raise RuntimeError("smoke_did_not_return_retrieve")
    parse_action(json.loads(call["arguments"])["action"], "search")
    print(json.dumps({"api_smoke_passed": True, "model_requested": client.model,
                      "model_reported": response["model"], "response_id": response["id"],
                      "thinking": client.thinking, "reasoning_effort": client.reasoning_effort,
                      "max_tokens": client.max_tokens, "usage": response["usage"], "api_requests": client.request_count,
                      "registered_tools": ["retrieve"], "tool_policy": TOOL_POLICY_VERSION,
                      "api_endpoint": API_URL, "strict_requested": True,
                      "rag_connected": False, "training_data_created": False}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("doctor", "api-smoke", "run", "stats", "review-packet", "approve", "export-candidates", "export-approved"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--model", choices=("deepseek-flash", "deepseek-v4-pro"), default="deepseek-flash")
    parser.add_argument("--thinking", choices=("disabled", "enabled"), default="disabled")
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"), help="Requires --thinking enabled; default effort is high")
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--db", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-examples", type=int, default=10, help="Total input window, including already checkpointed rows")
    parser.add_argument("--public-id-start", type=int, default=DEFAULT_PUBLIC_ID_START,
                        help="First six-digit public ID for the fixed input file; default 500001")
    parser.add_argument("--max-searches", type=int, choices=range(1, MAX_SEARCH_TURNS + 1), default=MAX_SEARCH_TURNS)
    parser.add_argument("--max-api-requests", type=int, default=100, help="Per-process HTTP attempt budget, including retries")
    parser.add_argument("--concurrency", type=int, default=8, help="Independent trajectories advanced concurrently (1-32)")
    parser.add_argument("--retrieval-batch-queries", type=int, default=8, help="Maximum queries per dense HTTP request (1-8)")
    parser.add_argument("--retrieval-batch-wait-ms", type=float, default=5.0, help="Short coalescing window (0-20 ms)")
    parser.add_argument("--max-tokens", type=int, help="Total API output budget: default 500 without thinking, 16384 with thinking")
    parser.add_argument("--max-cumulative-reasoning-tokens", type=int, default=DEFAULT_MAX_CUMULATIVE_REASONING_TOKENS,
                        help="Maximum provider reasoning tokens per candidate across all turns; default 32768")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--api-retries", type=int, choices=range(4), default=2)
    parser.add_argument("--retry-ids", help="Only retry these failed source:split:id checkpoint IDs")
    parser.add_argument("--ids", help="Comma-separated checkpoint IDs for manual semantic approval")
    parser.add_argument("--reviewer")
    parser.add_argument("--review-note")
    parser.add_argument("--ack-evidence-warning", action="store_true")
    args = parser.parse_args()
    if args.reasoning_effort and args.thinking != "enabled":
        parser.error("--reasoning-effort requires --thinking enabled")
    if args.max_tokens is None:
        args.max_tokens = DEFAULT_THINKING_MAX_TOKENS if args.thinking == "enabled" else DEFAULT_NONTHINKING_MAX_TOKENS
    if (not 1 <= args.max_examples <= 20000 or args.max_api_requests < 1
            or not 1 <= args.public_id_start <= 999999
            or args.public_id_start + args.max_examples - 1 > 999999
            or not 1 <= args.concurrency <= 32 or not 1 <= args.retrieval_batch_queries <= 8
            or not math.isfinite(args.retrieval_batch_wait_ms) or not 0 <= args.retrieval_batch_wait_ms <= 20
            or not 128 <= args.max_tokens <= MAX_API_OUTPUT_TOKENS
            or args.max_cumulative_reasoning_tokens < 1
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("invalid bounded run limits")
    try:
        if args.doctor:
            print(json.dumps(doctor(), indent=2, sort_keys=True))
        elif args.api_smoke:
            smoke(args)
        elif args.run:
            run(args)
        elif args.stats:
            if not args.db:
                raise ValueError("stats_requires_db")
            with checkpoint(args.db, readonly=True) as db:
                print(json.dumps(summary(db), indent=2, sort_keys=True))
        elif args.approve:
            approve(args)
        elif args.review_packet:
            review_packet(args)
        else:
            export(args)
    except (KeyFileError, APIError, ValueError, RuntimeError, OSError, sqlite3.Error, Rejected) as exc:
        # Never echo arbitrary server bodies, request headers, or credential values.
        safe = str(exc) if isinstance(exc, (KeyFileError, APIError, ValueError, Rejected)) else "operation_failed_check_preflight_and_checkpoint"
        parser.exit(1, "Error: " + safe + "\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; committed questions are preserved. Resume may repeat up to the concurrency limit of in-flight questions.\n")


if __name__ == "__main__":
    main()
