#!/usr/bin/env python
"""Hybrid-RAG retriever: BM25 + E5/FAISS FlatIP + RRF fusion."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

try:
    from build_bm25_index import DEFAULT_CORPUS, DEFAULT_INDEX_DIR, DB_NAME, build_index, validate_index
except ImportError:  # pragma: no cover
    from .build_bm25_index import DEFAULT_CORPUS, DEFAULT_INDEX_DIR, DB_NAME, build_index, validate_index


WORKSPACE = Path(os.environ.get("AETHERSEARCH_SFT_WORKSPACE", str(Path(__file__).resolve().parents[3]))).expanduser().resolve()
DEFAULT_DENSE_URL = "http://127.0.0.1:8000/retrieve"
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")
STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was", "were",
    "who", "what", "when", "where", "which", "whose", "whom", "why", "how", "did", "does", "do",
    "by", "with", "from", "that", "this", "it", "as", "at", "be", "been", "being", "into", "about",
    "answer", "answers", "actor", "actress", "album", "author", "book", "called", "capital",
    "character", "city", "company", "country", "date", "directed", "director", "episode", "film",
    "first", "game", "genre", "group", "held", "kind", "last", "located", "made", "many", "movie",
    "name", "named", "novel", "number", "person", "place", "played", "player", "plays", "released",
    "season", "series", "show", "singer", "song", "state", "team", "television", "title", "type",
    "used", "uses", "using", "written", "wrote", "year",
    "all", "among", "best", "can", "come", "control", "development", "middle", "members", "off",
    "other", "out", "people", "selling", "set", "takes", "things", "watch", "young",
}


@dataclass
class RetrievedDoc:
    doc_id: str
    title: str
    text: str
    source_branch: str
    bm25_rank: int | None
    dense_rank: int | None
    rrf_score: float


def clean_text(text: str) -> str:
    return " ".join(str(text).split())


def split_contents(doc: dict[str, Any]) -> tuple[str, str]:
    title = str(doc.get("title") or "").strip().strip('"')
    text = str(doc.get("text") or "").strip()
    contents = str(doc.get("contents") or "")
    if contents and (not title or not text):
        if "\n" in contents:
            raw_title, raw_text = contents.split("\n", 1)
            title = title or raw_title.strip().strip('"')
            text = text or raw_text.strip()
        else:
            text = text or contents.strip()
    return title, clean_text(text)


def query_terms(query: str, max_terms: int = 8) -> list[str]:
    tokens: list[str] = []
    for token in TOKEN_RE.findall(query.lower()):
        token = token.strip("_'-")
        if len(token) < 2 or token in STOPWORDS:
            continue
        tokens.append(token)
    if not tokens:
        tokens = [t.strip("_'-").lower() for t in TOKEN_RE.findall(query) if t.strip("_'-")]
    deduped = list(dict.fromkeys(tokens))[:max_terms]
    if not deduped:
        raise ValueError(f"cannot build BM25 query from: {query!r}")
    return deduped


def fts_expr_for_terms(terms: list[str], mode: str = "and") -> str:
    op = " OR " if mode == "or" else " "
    return op.join(f'"{term}"' for term in terms)


def fts_query(query: str, max_terms: int = 8, mode: str = "and") -> str:
    return fts_expr_for_terms(query_terms(query, max_terms=max_terms), mode=mode)


def fts_phrase_expr(phrase: str, max_terms: int = 8) -> str | None:
    tokens = []
    for token in TOKEN_RE.findall(phrase.lower()):
        token = token.strip("_'-")
        if len(token) < 2:
            continue
        tokens.append(token)
    deduped = list(dict.fromkeys(tokens))[:max_terms]
    if len(deduped) < 2:
        return None
    return '"' + " ".join(deduped) + '"'


def phrase_candidates(query: str, limit: int = 4) -> list[str]:
    phrases: list[str] = []
    phrases.extend(re.findall(r"['\"]([^'\"]{2,80})['\"]", query))
    phrases.extend(
        re.findall(
            r"\b(?:[A-Z][A-Za-z0-9'&.-]*|[A-Z0-9]{2,})(?:\s+(?:[A-Z][A-Za-z0-9'&.-]*|[A-Z0-9]{2,}))*",
            query,
        )
    )
    out: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        expr = fts_phrase_expr(phrase)
        if expr and expr not in seen:
            seen.add(expr)
            out.append(expr)
        if len(out) >= limit:
            break
    return out


def high_quality_terms(terms: list[str], max_terms: int = 4) -> list[str]:
    def score(term: str) -> tuple[int, int, int]:
        has_digit = int(any(ch.isdigit() for ch in term))
        return (len(term), has_digit, -terms.index(term))

    return sorted(terms, key=score, reverse=True)[:max_terms]


def fts_query_candidates(query: str) -> list[str]:
    terms = query_terms(query, max_terms=8)
    candidates: list[str] = []
    candidates.extend(phrase_candidates(query))
    if len(terms) == 1:
        candidates.append(fts_expr_for_terms(terms, mode="and"))
    for n in range(min(6, len(terms)), 1, -1):
        candidates.append(fts_expr_for_terms(terms[:n], mode="and"))
    if len(terms) > 1:
        for n in range(2, min(4, len(terms)) + 1):
            candidates.append(fts_expr_for_terms(high_quality_terms(terms, max_terms=n), mode="or"))
    out: list[str] = []
    seen: set[str] = set()
    for expr in candidates:
        if expr and expr not in seen:
            seen.add(expr)
            out.append(expr)
    return out


class BM25Retriever:
    def __init__(self, db_path: Path, max_expr_seconds: float = 2.0):
        self.db_path = Path(db_path)
        self.max_expr_seconds = max_expr_seconds
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA query_only=ON")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA cache_size=-1000000")
        self.conn.execute("PRAGMA mmap_size=30000000000")

    def close(self) -> None:
        self.conn.close()

    def search(self, query: str, topn: int = 20) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for expr in fts_query_candidates(query):
            for row in self._search_expr(expr, topn):
                if str(row["doc_id"]) not in seen:
                    rows.append(row)
                    seen.add(str(row["doc_id"]))
                if len(rows) >= topn:
                    break
            if len(rows) >= topn:
                break
        for i, row in enumerate(rows[:topn], 1):
            row["bm25_rank"] = i
        return rows[:topn]

    def _search_expr(self, expr: str, topn: int) -> list[dict[str, Any]]:
        deadline = time.perf_counter() + self.max_expr_seconds

        def should_interrupt() -> int:
            return 1 if time.perf_counter() > deadline else 0

        self.conn.set_progress_handler(should_interrupt, 1000)
        try:
            rows = self.conn.execute(
                """
                SELECT docs.rowid AS rowid, docs.doc_id AS doc_id, docs.title AS title, docs.text AS text,
                       docs_fts.rank AS bm25_score
                FROM docs_fts
                JOIN docs ON docs.rowid = docs_fts.rowid
                WHERE docs_fts MATCH ?
                ORDER BY docs_fts.rank
                LIMIT ?
                """,
                (expr, topn),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                return []
            raise
        finally:
            self.conn.set_progress_handler(None, 0)
        return [
            {
                "doc_id": str(row["doc_id"]),
                "title": row["title"],
                "text": row["text"],
                "bm25_score": float(row["bm25_score"]),
                "bm25_rank": i + 1,
            }
            for i, row in enumerate(rows)
        ]


class DenseServerRetriever:
    MAX_BATCH_QUERIES = 8

    def __init__(self, url: str = DEFAULT_DENSE_URL, timeout: float = 120.0, *, restricted_local: bool = False):
        if restricted_local and url != DEFAULT_DENSE_URL:
            raise ValueError("restricted_dense_endpoint_not_allowed")
        self.url = url
        self.timeout = timeout
        self.restricted_local = restricted_local
        self.session = requests.Session()
        self.session.trust_env = False

    def _request(self, query: str, topn: int) -> Any:
        return self._request_batch([query], topn)

    def _request_batch(self, queries: list[str], topn: int) -> Any:
        if (not isinstance(queries, list) or not 1 <= len(queries) <= self.MAX_BATCH_QUERIES
                or any(not isinstance(query, str) or not 1 <= len(query) <= 300 for query in queries)):
            raise ValueError("invalid_dense_query_batch")
        payload = {"queries": queries, "topk": topn, "return_scores": True}
        if self.restricted_local:
            if self.url != DEFAULT_DENSE_URL or topn != 20:
                raise ValueError("restricted_dense_request_not_allowed")
            limit = 2_097_152 * len(queries)
            # Never follow redirects or decompress an unbounded response from a tool backend.
            with self.session.post(DEFAULT_DENSE_URL, json=payload, timeout=self.timeout,
                                   allow_redirects=False, stream=True,
                                   headers={"Accept": "application/json", "Accept-Encoding": "identity"}) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(f"restricted_dense_http_{resp.status_code}")
                if (resp.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json"
                        or resp.headers.get("Content-Encoding", "identity").lower() != "identity"):
                    raise RuntimeError("restricted_dense_response_type_invalid")
                length = resp.headers.get("Content-Length")
                if length is not None and (not length.isascii() or not length.isdigit() or int(length) > limit):
                    raise RuntimeError("restricted_dense_response_size_invalid")
                raw = bytearray()
                for chunk in resp.iter_content(chunk_size=65536):
                    if len(raw) + len(chunk) > limit:
                        raise RuntimeError("restricted_dense_response_too_large")
                    raw.extend(chunk)
                try:
                    return json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    raise RuntimeError("restricted_dense_invalid_json") from None
        resp = self.session.post(self.url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def search(self, query: str, topn: int = 20) -> list[dict[str, Any]]:
        data = self._request(query, topn)
        return self._parse_batch(data, 1, topn)[0]

    def search_batch(self, queries: list[str], topn: int = 20) -> list[list[dict[str, Any]]]:
        return self._parse_batch(self._request_batch(queries, topn), len(queries), topn)

    def _parse_batch(self, data: Any, expected: int, topn: int) -> list[list[dict[str, Any]]]:
        if not isinstance(data, dict):
            raise RuntimeError("dense_response_must_be_object")
        result = data.get("result")
        if not isinstance(result, list) or len(result) != expected or any(not isinstance(items, list) for items in result):
            if self.restricted_local:
                raise RuntimeError("restricted_dense_result_invalid")
            raise RuntimeError(f"unexpected dense retriever response: {data}")
        batches: list[list[dict[str, Any]]] = []
        for items in result:
            docs: list[dict[str, Any]] = []
            for i, item in enumerate(items[:topn], 1):
                score = None
                doc = item
                if isinstance(item, dict) and "document" in item:
                    score = item.get("score")
                    doc = item.get("document") or {}
                if not isinstance(doc, dict):
                    continue
                doc_id = str(doc.get("id") or doc.get("doc_id") or "")
                if not doc_id:
                    continue
                title, text = split_contents(doc)
                docs.append({
                    "doc_id": doc_id,
                    "title": title,
                    "text": text,
                    "dense_score": float(score) if score is not None else None,
                    "dense_rank": i,
                })
            batches.append(docs)
        return batches


class HybridRetrieverV1:
    def __init__(
        self,
        bm25_db: Path | None = None,
        corpus_path: Path = DEFAULT_CORPUS,
        dense_url: str = DEFAULT_DENSE_URL,
        rrf_k: int = 60,
        candidate_topn: int = 20,
        dense_timeout: float = 120.0,
        bm25_max_expr_seconds: float = 2.0,
        auto_build_bm25: bool = True,
        log_path: Path | None = WORKSPACE / "logs/search_sft_gen/hybrid_v1_real_50.log",
        restricted_dense: bool = False,
    ):
        self.corpus_path = corpus_path
        self.rrf_k = rrf_k
        self.candidate_topn = candidate_topn
        db_path = bm25_db or (DEFAULT_INDEX_DIR / DB_NAME)
        ok, _ = validate_index(db_path, corpus_path)
        if not ok:
            if not auto_build_bm25:
                raise RuntimeError(f"BM25 index is unavailable or invalid: {db_path}")
            db_path = build_index(corpus_path=corpus_path, index_dir=DEFAULT_INDEX_DIR, log_path=log_path)
        self.bm25_db = db_path
        self.bm25 = BM25Retriever(db_path, max_expr_seconds=bm25_max_expr_seconds)
        self.dense = DenseServerRetriever(dense_url, timeout=dense_timeout, restricted_local=restricted_dense)

    def close(self) -> None:
        self.bm25.close()

    def retrieve(self, query: str, topk: int = 3) -> list[RetrievedDoc]:
        bm25_docs = self.bm25.search(query, self.candidate_topn)
        if len(bm25_docs) < min(3, self.candidate_topn):
            raise RuntimeError(f"BM25 returned too few candidates: {len(bm25_docs)} for query={query!r}")
        dense_docs = self.dense.search(query, self.candidate_topn)
        if len(dense_docs) < min(3, self.candidate_topn):
            raise RuntimeError(f"Dense E5-FAISS FlatIP returned too few candidates: {len(dense_docs)} for query={query!r}")
        return self._fuse(bm25_docs, dense_docs, topk)

    def retrieve_batch(self, queries: list[str], topk: int = 3) -> list[list[RetrievedDoc]]:
        if not isinstance(queries, list) or not 1 <= len(queries) <= DenseServerRetriever.MAX_BATCH_QUERIES:
            raise ValueError("invalid_hybrid_query_batch")
        sparse = [self.bm25.search(query, self.candidate_topn) for query in queries]
        if any(len(docs) < self.candidate_topn for docs in sparse):
            raise RuntimeError("BM25 returned too few batched candidates")
        dense = self.dense.search_batch(queries, self.candidate_topn)
        if len(dense) != len(queries) or any(len(docs) < self.candidate_topn for docs in dense):
            raise RuntimeError("Dense E5-FAISS FlatIP returned too few batched candidates")
        return [self._fuse(bm25_docs, dense_docs, topk) for bm25_docs, dense_docs in zip(sparse, dense, strict=True)]

    def _fuse(self, bm25_docs: list[dict[str, Any]], dense_docs: list[dict[str, Any]], topk: int) -> list[RetrievedDoc]:
        fused: dict[str, dict[str, Any]] = {}
        for doc in bm25_docs:
            doc_id = doc["doc_id"]
            entry = fused.setdefault(doc_id, {"doc_id": doc_id, "title": doc["title"], "text": doc["text"], "rrf_score": 0.0})
            entry["bm25_rank"] = doc["bm25_rank"]
            entry["rrf_score"] += 1.0 / (self.rrf_k + doc["bm25_rank"])
        for doc in dense_docs:
            doc_id = doc["doc_id"]
            entry = fused.setdefault(doc_id, {"doc_id": doc_id, "title": doc["title"], "text": doc["text"], "rrf_score": 0.0})
            if not entry.get("title") and doc["title"]:
                entry["title"] = doc["title"]
            if not entry.get("text") and doc["text"]:
                entry["text"] = doc["text"]
            entry["dense_rank"] = doc["dense_rank"]
            entry["rrf_score"] += 1.0 / (self.rrf_k + doc["dense_rank"])

        ranked = sorted(
            fused.values(),
            key=lambda d: (-float(d["rrf_score"]), d.get("bm25_rank") or math.inf, d.get("dense_rank") or math.inf, d["doc_id"]),
        )
        out: list[RetrievedDoc] = []
        for entry in ranked[:topk]:
            bm25_rank = entry.get("bm25_rank")
            dense_rank = entry.get("dense_rank")
            if bm25_rank is not None and dense_rank is not None:
                branch = "both"
            elif bm25_rank is not None:
                branch = "bm25"
            else:
                branch = "dense"
            out.append(
                RetrievedDoc(
                    doc_id=str(entry["doc_id"]),
                    title=str(entry.get("title") or ""),
                    text=clean_text(entry.get("text") or ""),
                    source_branch=branch,
                    bm25_rank=int(bm25_rank) if bm25_rank is not None else None,
                    dense_rank=int(dense_rank) if dense_rank is not None else None,
                    rrf_score=float(entry["rrf_score"]),
                )
            )
        if len(out) != topk:
            raise RuntimeError(f"Hybrid RRF returned {len(out)} docs, expected {topk}")
        return out


def format_information(docs: list[RetrievedDoc]) -> str:
    lines = ["<information>"]
    for i, doc in enumerate(docs, 1):
        title = doc.title.replace('"', "'")
        lines.append(f'Doc {i}(Title: "{title}") {doc.text}')
    lines.append("</information>")
    return "\n".join(lines)


def docs_to_jsonable(docs: list[RetrievedDoc]) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "text": doc.text,
            "source_branch": doc.source_branch,
            "bm25_rank": doc.bm25_rank,
            "dense_rank": doc.dense_rank,
            "rrf_score": doc.rrf_score,
        }
        for doc in docs
    ]
