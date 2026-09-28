#!/usr/bin/env python
"""Build or validate the reusable wiki18 BM25 index for Hybrid-RAG V1."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterator

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


WORKSPACE = Path(os.environ.get("AETHERSEARCH_SFT_WORKSPACE", str(Path(__file__).resolve().parents[3]))).expanduser().resolve()
DEFAULT_CORPUS = WORKSPACE / "data/wiki18_corpus/wiki-18.jsonl"
DEFAULT_INDEX_DIR = WORKSPACE / "data/bm25_index"
DEFAULT_LOG = WORKSPACE / "logs/search_sft_gen/hybrid_v1_real_50.log"
DB_NAME = "wiki18_bm25_fts5.db"
META_VERSION = "hybrid_bm25_fts5_v1"


def setup_logging(log_path: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("bm25_index")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def ensure_fts5_available() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE docs USING fts5(title, text)")
    finally:
        conn.close()


def split_contents(item: dict[str, Any]) -> tuple[str, str]:
    title = str(item.get("title") or "").strip().strip('"')
    text = str(item.get("text") or "").strip()
    contents = str(item.get("contents") or "")
    if contents and (not title or not text):
        if "\n" in contents:
            raw_title, raw_text = contents.split("\n", 1)
            title = title or raw_title.strip().strip('"')
            text = text or raw_text.strip()
        else:
            text = text or contents.strip()
    return title, " ".join(text.split())


def parse_json_line(raw: bytes) -> dict[str, Any] | None:
    raw = raw.strip()
    if not raw:
        return None
    if not raw.startswith(b"{"):
        start = raw.find(b'{"id"')
        if start < 0:
            start = raw.find(b"{")
        if start < 0:
            return None
        raw = raw[start:]
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return None


def iter_corpus(corpus_path: Path) -> Iterator[tuple[str, str, str]]:
    with corpus_path.open("rb") as f:
        for raw in f:
            item = parse_json_line(raw)
            if not item:
                continue
            doc_id = str(item.get("id", "")).strip()
            if not doc_id:
                continue
            title, text = split_contents(item)
            yield doc_id, title, text


def corpus_signature(corpus_path: Path) -> dict[str, str]:
    stat = corpus_path.stat()
    return {
        "corpus_path": str(corpus_path.resolve()),
        "corpus_size": str(stat.st_size),
        "corpus_mtime_ns": str(stat.st_mtime_ns),
        "index_version": META_VERSION,
    }


def equivalent_corpus_path(stored_path: str | None, expected_path: str) -> bool:
    if not stored_path:
        return False
    stored = Path(stored_path)
    if not stored.is_absolute():
        stored = WORKSPACE / stored
    expected = Path(expected_path)
    if not expected.is_absolute():
        expected = WORKSPACE / expected
    try:
        return stored.resolve() == expected.resolve()
    except OSError:
        return str(stored) == str(expected)


def get_meta(conn: sqlite3.Connection) -> dict[str, str]:
    try:
        return dict(conn.execute("SELECT key, value FROM meta").fetchall())
    except sqlite3.Error:
        return {}


def validate_index(db_path: Path, corpus_path: Path, logger: logging.Logger | None = None) -> tuple[bool, dict[str, str]]:
    if not db_path.is_file() or db_path.stat().st_size == 0:
        return False, {"reason": "missing_db"}
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            meta = get_meta(conn)
            sig = corpus_signature(corpus_path)
            docs = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
            fts = conn.execute("SELECT COUNT(*) FROM docs_fts").fetchone()[0]
            ok = (
                meta.get("status") == "complete"
                and equivalent_corpus_path(meta.get("corpus_path"), sig["corpus_path"])
                and all(meta.get(k) == v for k, v in sig.items() if k != "corpus_path")
                and docs > 0
                and fts == docs
                and meta.get("doc_count") == str(docs)
            )
            info = dict(meta)
            info.update({"doc_count_actual": str(docs), "fts_count_actual": str(fts)})
            if logger:
                logger.info("BM25 index validation: ok=%s docs=%s fts=%s db=%s", ok, docs, fts, db_path)
            return ok, info
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return False, {"reason": f"sqlite_error:{exc}"}


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        PRAGMA cache_size=-2000000;
        CREATE TABLE docs (
          rowid INTEGER PRIMARY KEY,
          doc_id TEXT NOT NULL UNIQUE,
          title TEXT NOT NULL,
          text TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE docs_fts USING fts5(
          title,
          text,
          content='docs',
          content_rowid='rowid',
          tokenize='porter unicode61'
        );
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """
    )


def build_index(
    corpus_path: Path = DEFAULT_CORPUS,
    index_dir: Path = DEFAULT_INDEX_DIR,
    log_path: Path | None = DEFAULT_LOG,
    batch_size: int = 10000,
    force: bool = False,
) -> Path:
    logger = setup_logging(log_path)
    ensure_fts5_available()
    if not corpus_path.is_file() or corpus_path.stat().st_size == 0:
        raise FileNotFoundError(f"corpus not found or empty: {corpus_path}")
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / DB_NAME
    ok, info = validate_index(db_path, corpus_path, logger)
    if ok and not force:
        logger.info("BM25 index reused: %s", db_path)
        return db_path
    if db_path.exists() and not force:
        logger.warning("Existing BM25 DB is not reusable (%s). Rebuilding in place: %s", info, db_path)
    if db_path.exists():
        db_path.unlink()
    for sidecar in db_path.parent.glob(db_path.name + "-*"):
        sidecar.unlink()

    logger.info("Building BM25 FTS5 index from corpus=%s", corpus_path)
    logger.info("Writing BM25 index to %s", db_path)
    started = time.time()
    conn = sqlite3.connect(str(db_path))
    try:
        create_schema(conn)
        batch_docs: list[tuple[str, str, str]] = []
        batch_fts: list[tuple[int, str, str]] = []
        rowid = 0
        skipped = 0
        iterator = iter_corpus(corpus_path)
        progress = tqdm(total=None, unit="docs", desc="BM25 indexing") if tqdm else None
        for doc_id, title, text in iterator:
            if not text:
                skipped += 1
                continue
            rowid += 1
            batch_docs.append((doc_id, title, text))
            batch_fts.append((rowid, title, text))
            if len(batch_docs) >= batch_size:
                conn.executemany("INSERT INTO docs(doc_id, title, text) VALUES (?, ?, ?)", batch_docs)
                conn.executemany("INSERT INTO docs_fts(rowid, title, text) VALUES (?, ?, ?)", batch_fts)
                conn.commit()
                if progress is not None:
                    progress.update(len(batch_docs))
                if rowid % 500000 == 0:
                    logger.info("BM25 indexed docs=%s skipped=%s elapsed=%.1fs", rowid, skipped, time.time() - started)
                batch_docs.clear()
                batch_fts.clear()
        if batch_docs:
            conn.executemany("INSERT INTO docs(doc_id, title, text) VALUES (?, ?, ?)", batch_docs)
            conn.executemany("INSERT INTO docs_fts(rowid, title, text) VALUES (?, ?, ?)", batch_fts)
            conn.commit()
            if progress is not None:
                progress.update(len(batch_docs))
        if progress is not None:
            progress.close()

        sig = corpus_signature(corpus_path)
        meta = {
            **sig,
            "status": "complete",
            "doc_count": str(rowid),
            "skipped_count": str(skipped),
            "created_at_unix": str(int(time.time())),
        }
        conn.executemany(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            sorted(meta.items()),
        )
        conn.commit()
        logger.info("BM25 index complete docs=%s skipped=%s elapsed=%.1fs", rowid, skipped, time.time() - started)
    finally:
        conn.close()

    ok, info = validate_index(db_path, corpus_path, logger)
    if not ok:
        raise RuntimeError(f"BM25 index validation failed after build: {info}")
    return db_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_index(args.corpus, args.index_dir, args.log, args.batch_size, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
