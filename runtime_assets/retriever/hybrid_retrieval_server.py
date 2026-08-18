#!/usr/bin/env python3
import argparse
import json
import logging
import os
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import datasets
import torch
import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer


LOGGER = logging.getLogger("hybrid_retriever")


def _release_clean_file_cache(path: str | Path) -> dict[str, int | bool]:
    """Release clean immutable-input pages after retriever initialization.

    The retriever keeps the corpus/index objects alive, but those source files
    do not need to remain fully resident in the 360-GiB cgroup.  Linux will
    fault hot pages back in on demand.  This is a cache hint only and never
    changes the corpus or index contents.
    """

    target = Path(path)
    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fadvise is None or dontneed is None or not target.exists():
        return {"files": 0, "bytes": 0, "supported": False}
    files = 0
    total_bytes = 0
    candidates = [target] if target.is_file() else (
        item for item in target.rglob("*") if item.is_file()
    )
    for item in candidates:
        try:
            size = int(item.stat().st_size)
            with item.open("rb") as handle:
                fadvise(handle.fileno(), 0, 0, dontneed)
            files += 1
            total_bytes += size
        except (FileNotFoundError, OSError):
            continue
    return {"files": files, "bytes": total_bytes, "supported": True}


def setup_logging(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def minmax(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi - lo < 1e-12:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.ones((1, 1), device="cuda")
        _ = (x + 1).cpu()
        return True
    except Exception as exc:
        LOGGER.warning("CUDA is visible but unusable for torch: %s", repr(exc))
        return False


class DenseEncoder:
    def __init__(self, model_name: str, model_path: str, device: str, max_length: int = 256, use_fp16: bool = True):
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        self.model.to(self.device)
        if use_fp16 and self.device.type == "cuda":
            self.model.half()

    @torch.no_grad()
    def encode(self, queries: List[str]) -> np.ndarray:
        if "e5" in self.model_name.lower():
            queries = [f"query: {q}" for q in queries]
        inputs = self.tokenizer(
            queries,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output = self.model(**inputs, return_dict=True)
        hidden = output.last_hidden_state.masked_fill(~inputs["attention_mask"][..., None].bool(), 0.0)
        emb = hidden.sum(dim=1) / inputs["attention_mask"].sum(dim=1)[..., None]
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.detach().float().cpu().numpy().astype(np.float32, order="C")


@dataclass
class RetrievalRequest:
    queries: List[str]
    topk: Optional[int]
    mode: str
    completed: threading.Event = field(default_factory=threading.Event)
    result: Optional[Tuple[List[List[Dict[str, str]]], List[List[float]]]] = None
    error: Optional[BaseException] = None


class HybridRetriever:
    def __init__(self, args):
        self.topk = args.topk
        self.alpha = args.alpha
        self.bm25_topn = args.bm25_topn
        self.dense_topn = args.dense_topn
        self.dense_query_batch_size = max(1, int(args.dense_query_batch_size))
        self.request_batch_max_queries = max(1, int(args.request_batch_max_queries))
        self.request_batch_wait_seconds = max(
            0.0,
            float(args.request_batch_wait_ms) / 1000.0,
        )
        self.request_wait_timeout_seconds = max(
            1.0,
            float(args.request_wait_timeout_seconds),
        )
        self.corpus = datasets.load_dataset("json", data_files=args.corpus_path, split="train", num_proc=4)
        LOGGER.info("Loaded corpus from %s with %d rows", args.corpus_path, len(self.corpus))

        from pyserini.search.lucene import LuceneSearcher
        self.bm25 = LuceneSearcher(args.bm25_index_path)
        self.bm25_pool = ThreadPoolExecutor(
            max_workers=max(1, int(args.bm25_workers)),
            thread_name_prefix="bm25",
        )
        self.bm25_workers = max(1, int(args.bm25_workers))
        LOGGER.info("Loaded BM25 index from %s", args.bm25_index_path)

        self.dense_index = faiss.read_index(args.dense_index_path)
        self.faiss_gpu_enabled = False
        if args.faiss_gpu:
            try:
                if args.faiss_gpu_stream_flat:
                    self.dense_index = self._stream_flat_index_to_gpu(args)
                    self.faiss_gpu_enabled = True
                    LOGGER.info("Dense FAISS flat index streamed to GPU as %s", type(self.dense_index).__name__)
                else:
                    cpu_index_type = type(self.dense_index).__name__
                    co = faiss.GpuMultipleClonerOptions()
                    co.useFloat16 = True
                    co.shard = True
                    converted = faiss.index_cpu_to_all_gpus(self.dense_index, co=co)
                    converted_type = type(converted).__name__
                    if converted_type == cpu_index_type and not converted_type.startswith("Gpu"):
                        raise RuntimeError(f"FAISS returned unchanged CPU index type {converted_type}")
                    self.dense_index = converted
                    self.faiss_gpu_enabled = True
                    LOGGER.info("Dense FAISS index moved to GPU as %s", converted_type)
            except Exception as exc:
                LOGGER.warning("Dense FAISS GPU index unavailable; falling back to CPU FAISS. error=%s", repr(exc))
                if args.require_faiss_gpu:
                    raise
        else:
            LOGGER.info("Dense FAISS GPU disabled by config; using CPU FAISS")
        if args.require_faiss_gpu and not self.faiss_gpu_enabled:
            raise RuntimeError("Dense FAISS GPU index is required but was not enabled")

        dense_device = args.dense_device
        if dense_device == "auto":
            dense_device = "cuda" if cuda_usable() else "cpu"
        self.encoder = DenseEncoder(
            model_name=args.retriever_name,
            model_path=args.retriever_model,
            device=dense_device,
            max_length=args.query_max_length,
            use_fp16=args.retrieval_use_fp16,
        )
        LOGGER.info("Dense encoder loaded on %s from %s", dense_device, args.retriever_model)
        probe = self.encoder.encode(["dimension check"])
        if probe.ndim != 2 or int(probe.shape[1]) != int(self.dense_index.d):
            raise RuntimeError(
                f"Query encoder/index dimension mismatch: {probe.shape} vs d={self.dense_index.d}"
            )
        if int(self.dense_index.ntotal) != len(self.corpus):
            raise RuntimeError(
                f"Dense index/corpus row mismatch: {self.dense_index.ntotal} != {len(self.corpus)}"
            )
        LOGGER.info(
            "Retriever integrity verified: corpus=%d ntotal=%d d=%d gpu=%s",
            len(self.corpus),
            int(self.dense_index.ntotal),
            int(self.dense_index.d),
            self.faiss_gpu_enabled,
        )
        for label, path in (
            ("corpus", args.corpus_path),
            ("bm25", args.bm25_index_path),
            ("dense_index", args.dense_index_path),
            ("encoder", args.retriever_model),
        ):
            cache_release = _release_clean_file_cache(path)
            LOGGER.info("Released clean %s input cache: %s", label, cache_release)
        self._request_queue: queue.Queue[Optional[RetrievalRequest]] = queue.Queue()
        self._closed = False
        self._stats_lock = threading.Lock()
        self._batch_stats = {
            "requests": 0,
            "queries": 0,
            "worker_batches": 0,
            "coalesced_batches": 0,
            "max_requests_per_batch": 0,
            "max_queries_per_batch": 0,
            "cumulative_batch_seconds": 0.0,
        }
        self._batch_worker = threading.Thread(
            target=self._request_batch_loop,
            name="retrieval-batch-worker",
            daemon=True,
        )
        self._batch_worker.start()
        LOGGER.info(
            "Retriever batching enabled: wait_ms=%.3f max_queries=%d "
            "dense_query_batch=%d bm25_workers=%d",
            self.request_batch_wait_seconds * 1000.0,
            self.request_batch_max_queries,
            self.dense_query_batch_size,
            self.bm25_workers,
        )

    def _stream_flat_index_to_gpu(self, args):
        if self.dense_index.d <= 0 or self.dense_index.ntotal <= 0:
            raise RuntimeError("Dense FAISS index is empty")
        if not type(self.dense_index).__name__.startswith("IndexFlat"):
            raise RuntimeError(f"Streaming GPU load requires IndexFlat*, got {type(self.dense_index).__name__}")

        res = faiss.StandardGpuResources()
        if hasattr(res, "setTempMemory"):
            res.setTempMemory(int(args.faiss_temp_memory_mb) * 1024 * 1024)

        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = int(args.faiss_gpu_device)
        cfg.useFloat16 = bool(args.faiss_gpu_use_fp16)
        if self.dense_index.metric_type == faiss.METRIC_INNER_PRODUCT:
            gpu_index = faiss.GpuIndexFlatIP(res, self.dense_index.d, cfg)
        elif self.dense_index.metric_type == faiss.METRIC_L2:
            gpu_index = faiss.GpuIndexFlatL2(res, self.dense_index.d, cfg)
        else:
            raise RuntimeError(f"Unsupported flat metric_type={self.dense_index.metric_type}")

        ntotal = int(self.dense_index.ntotal)
        batch_size = int(args.faiss_add_batch_size)
        if batch_size <= 0 or batch_size >= ntotal:
            LOGGER.info("Reconstructing all dense vectors for one-shot GPU add: %d/%d", ntotal, ntotal)
            vectors = self.dense_index.reconstruct_n(0, ntotal)
            vectors = np.asarray(vectors, dtype=np.float32, order="C")
            gpu_index.add(vectors)
            LOGGER.info("Streamed dense vectors to GPU: %d/%d", ntotal, ntotal)
            if int(gpu_index.ntotal) != ntotal:
                raise RuntimeError(f"GPU index row mismatch: {gpu_index.ntotal} != {ntotal}")
            return gpu_index

        started = 0
        for start in range(0, ntotal, batch_size):
            n = min(batch_size, ntotal - start)
            vectors = self.dense_index.reconstruct_n(start, n)
            vectors = np.asarray(vectors, dtype=np.float32, order="C")
            gpu_index.add(vectors)
            started += n
            if started == ntotal or started % max(batch_size * 20, 1) == 0:
                LOGGER.info("Streamed dense vectors to GPU: %d/%d", started, ntotal)
        if int(gpu_index.ntotal) != ntotal:
            raise RuntimeError(f"GPU index row mismatch: {gpu_index.ntotal} != {ntotal}")
        return gpu_index

    def _doc_from_corpus_idx(self, idx: int) -> Dict[str, str]:
        row = self.corpus[int(idx)]
        contents = row.get("contents") or row.get("text") or ""
        return {
            "id": str(row.get("id", idx)),
            "contents": contents,
            "title": contents.split("\n")[0].strip('"') if contents else "",
            "text": "\n".join(contents.split("\n")[1:]) if contents else "",
        }

    def _doc_from_bm25_hit(self, hit) -> Dict[str, str]:
        raw = self.bm25.doc(hit.docid).raw()
        if raw:
            payload = json.loads(raw)
            contents = payload.get("contents", "")
            doc_id = str(payload.get("id", hit.docid))
        else:
            row = self.corpus[int(hit.docid)]
            contents = row.get("contents", "")
            doc_id = str(row.get("id", hit.docid))
        return {
            "id": doc_id,
            "contents": contents,
            "title": contents.split("\n")[0].strip('"') if contents else "",
            "text": "\n".join(contents.split("\n")[1:]) if contents else "",
        }

    def _bm25_search_one(
        self,
        query: str,
    ) -> Tuple[List[Dict[str, str]], List[float]]:
        bm25_hits = self.bm25.search(query, self.bm25_topn)
        bm25_docs = [self._doc_from_bm25_hit(hit) for hit in bm25_hits]
        bm25_scores = [float(hit.score) for hit in bm25_hits]
        return bm25_docs, bm25_scores

    def _dense_search_many(
        self,
        queries: List[str],
    ) -> List[Tuple[List[Dict[str, str]], List[float]]]:
        results: List[Tuple[List[Dict[str, str]], List[float]]] = []
        for start in range(0, len(queries), self.dense_query_batch_size):
            chunk = queries[start : start + self.dense_query_batch_size]
            emb = self.encoder.encode(chunk)
            dense_scores_arr, dense_idxs_arr = self.dense_index.search(
                emb,
                self.dense_topn,
            )
            for row in range(len(chunk)):
                dense_idxs = [int(value) for value in dense_idxs_arr[row].tolist()]
                dense_scores = [
                    float(value) for value in dense_scores_arr[row].tolist()
                ]
                dense_docs = [
                    self._doc_from_corpus_idx(index) for index in dense_idxs
                ]
                results.append((dense_docs, dense_scores))
        return results

    def _fuse_one(
        self,
        dense_result: Tuple[List[Dict[str, str]], List[float]],
        bm25_result: Tuple[List[Dict[str, str]], List[float]],
        topk: int,
        mode: str,
    ) -> Tuple[List[Dict[str, str]], List[float]]:
        dense_docs, dense_scores = dense_result
        bm25_docs, bm25_scores = bm25_result

        if mode == "dense":
            return dense_docs[:topk], dense_scores[:topk]
        if mode == "sparse":
            return bm25_docs[:topk], bm25_scores[:topk]
        if mode != "hybrid":
            raise ValueError(f"Unsupported retrieval mode: {mode}")

        combined: Dict[str, Dict[str, object]] = {}
        for doc, score in zip(dense_docs, minmax(dense_scores)):
            doc_id = str(doc.get("id"))
            combined.setdefault(
                doc_id,
                {"document": doc, "dense": 0.0, "bm25": 0.0},
            )
            combined[doc_id]["dense"] = max(
                float(combined[doc_id]["dense"]),
                score,
            )
        for doc, score in zip(bm25_docs, minmax(bm25_scores)):
            doc_id = str(doc.get("id"))
            combined.setdefault(
                doc_id,
                {"document": doc, "dense": 0.0, "bm25": 0.0},
            )
            combined[doc_id]["bm25"] = max(
                float(combined[doc_id]["bm25"]),
                score,
            )

        ranked = []
        for item in combined.values():
            score = self.alpha * float(item["dense"]) + (
                1.0 - self.alpha
            ) * float(item["bm25"])
            ranked.append((item["document"], score))
        ranked.sort(key=lambda value: value[1], reverse=True)
        docs = [doc for doc, _ in ranked[:topk]]
        scores = [score for _, score in ranked[:topk]]
        return docs, scores

    def _batch_search_impl(
        self,
        queries: List[str],
        topk: Optional[int] = None,
        mode: str = "hybrid",
    ) -> Tuple[List[List[Dict[str, str]]], List[List[float]]]:
        topk = topk or self.topk
        if mode not in {"hybrid", "dense", "sparse"}:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
        if not queries:
            return [], []

        empty_result: Tuple[List[Dict[str, str]], List[float]] = ([], [])
        bm25_futures = []
        if mode in {"hybrid", "sparse"}:
            bm25_futures = [
                self.bm25_pool.submit(self._bm25_search_one, query)
                for query in queries
            ]

        dense_results = [empty_result for _ in queries]
        if mode in {"hybrid", "dense"}:
            dense_results = self._dense_search_many(queries)

        bm25_results = [empty_result for _ in queries]
        if bm25_futures:
            bm25_results = [future.result() for future in bm25_futures]

        all_docs = []
        all_scores = []
        for dense_result, bm25_result in zip(dense_results, bm25_results):
            docs, scores = self._fuse_one(
                dense_result,
                bm25_result,
                int(topk),
                mode,
            )
            all_docs.append(docs)
            all_scores.append(scores)
        return all_docs, all_scores

    def search_one(
        self,
        query: str,
        topk: Optional[int] = None,
        mode: str = "hybrid",
    ) -> Tuple[List[Dict[str, str]], List[float]]:
        docs, scores = self.batch_search([query], topk, mode=mode)
        return docs[0], scores[0]

    def _record_batch_stats(
        self,
        requests: int,
        queries: int,
        elapsed_seconds: float,
    ) -> None:
        with self._stats_lock:
            self._batch_stats["requests"] += int(requests)
            self._batch_stats["queries"] += int(queries)
            self._batch_stats["worker_batches"] += 1
            self._batch_stats["coalesced_batches"] += int(requests > 1)
            self._batch_stats["max_requests_per_batch"] = max(
                self._batch_stats["max_requests_per_batch"],
                int(requests),
            )
            self._batch_stats["max_queries_per_batch"] = max(
                self._batch_stats["max_queries_per_batch"],
                int(queries),
            )
            self._batch_stats["cumulative_batch_seconds"] += float(
                elapsed_seconds
            )

    def batching_stats(self) -> Dict[str, object]:
        with self._stats_lock:
            stats = dict(self._batch_stats)
        worker_batches = max(1, int(stats["worker_batches"]))
        stats.update(
            {
                "average_requests_per_batch": float(stats["requests"])
                / worker_batches,
                "average_queries_per_batch": float(stats["queries"])
                / worker_batches,
                "average_batch_seconds": float(
                    stats["cumulative_batch_seconds"]
                )
                / worker_batches,
                "request_batch_wait_ms": self.request_batch_wait_seconds
                * 1000.0,
                "request_batch_max_queries": self.request_batch_max_queries,
                "dense_query_batch_size": self.dense_query_batch_size,
                "bm25_workers": self.bm25_workers,
                "worker_alive": self._batch_worker.is_alive(),
            }
        )
        return stats

    def _request_batch_loop(self) -> None:
        deferred: Optional[RetrievalRequest] = None
        while True:
            if deferred is None:
                request = self._request_queue.get()
            else:
                request = deferred
                deferred = None
            if request is None:
                self._request_queue.task_done()
                break

            requests = [request]
            query_count = len(request.queries)
            deadline = time.perf_counter() + self.request_batch_wait_seconds
            stop_after_batch = False
            while query_count < self.request_batch_max_queries:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    candidate = self._request_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if candidate is None:
                    self._request_queue.task_done()
                    stop_after_batch = True
                    break
                if (
                    requests
                    and query_count + len(candidate.queries)
                    > self.request_batch_max_queries
                ):
                    deferred = candidate
                    break
                requests.append(candidate)
                query_count += len(candidate.queries)

            started = time.perf_counter()
            grouped: Dict[Tuple[int, str], List[RetrievalRequest]] = defaultdict(
                list
            )
            for item in requests:
                grouped[(int(item.topk or self.topk), item.mode)].append(item)

            for (topk, mode), group in grouped.items():
                flattened = [
                    query for item in group for query in item.queries
                ]
                try:
                    docs, scores = self._batch_search_impl(
                        flattened,
                        topk=topk,
                        mode=mode,
                    )
                    offset = 0
                    for item in group:
                        end = offset + len(item.queries)
                        item.result = (docs[offset:end], scores[offset:end])
                        offset = end
                except BaseException as exc:
                    for item in group:
                        item.error = exc

            elapsed = time.perf_counter() - started
            self._record_batch_stats(len(requests), query_count, elapsed)
            for item in requests:
                item.completed.set()
                self._request_queue.task_done()
            if stop_after_batch:
                break

        if deferred is not None:
            deferred.error = RuntimeError(
                "Retriever batch worker stopped before processing request"
            )
            deferred.completed.set()
            self._request_queue.task_done()

    def batch_search(
        self,
        queries: List[str],
        topk: Optional[int] = None,
        mode: str = "hybrid",
    ):
        if not queries:
            return [], []
        if self._closed:
            raise RuntimeError("Retriever is closed")
        request = RetrievalRequest(
            queries=list(queries),
            topk=topk,
            mode=mode,
        )
        self._request_queue.put(request)
        deadline = time.monotonic() + self.request_wait_timeout_seconds
        while not request.completed.wait(timeout=1.0):
            if not self._batch_worker.is_alive():
                raise RuntimeError("Retriever batch worker exited unexpectedly")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Retriever batch request exceeded internal wait timeout"
                )
        if request.error is not None:
            raise request.error
        if request.result is None:
            raise RuntimeError("Retriever completed request without a result")
        return request.result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._request_queue.put(None)
        self._batch_worker.join(timeout=self.request_wait_timeout_seconds)
        self.bm25_pool.shutdown(wait=True)
        if self._batch_worker.is_alive():
            raise RuntimeError("Retriever batch worker did not stop cleanly")


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False
    mode: str = "hybrid"


app = FastAPI()
retriever: Optional[HybridRetriever] = None


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    docs, scores = retriever.batch_search(
        request.queries,
        request.topk,
        mode=request.mode,
    )
    result = []
    for doc_list, score_list in zip(docs, scores):
        if request.return_scores:
            result.append([{"document": doc, "score": score} for doc, score in zip(doc_list, score_list)])
        else:
            result.append(doc_list)
    return {"result": result}


@app.get("/health")
def health_endpoint():
    if retriever is None:
        return {"ready": False}
    return {
        "ready": True,
        "corpus_passages": len(retriever.corpus),
        "index_ntotal": int(retriever.dense_index.ntotal),
        "embedding_dimension": int(retriever.dense_index.d),
        "index_type": type(retriever.dense_index).__name__,
        "faiss_gpu_enabled": bool(retriever.faiss_gpu_enabled),
        "fusion_alpha": float(retriever.alpha),
        "topk": int(retriever.topk),
        "batching": retriever.batching_stats(),
    }


@app.on_event("shutdown")
def shutdown_retriever():
    if retriever is not None:
        retriever.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25-index-path", required=True)
    parser.add_argument("--dense-index-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--retriever-name", default="e5")
    parser.add_argument("--retriever-model", required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--bm25-topn", type=int, default=20)
    parser.add_argument("--dense-topn", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--query-max-length", type=int, default=256)
    parser.add_argument("--dense-query-batch-size", type=int, default=64)
    parser.add_argument("--bm25-workers", type=int, default=16)
    parser.add_argument("--request-batch-wait-ms", type=float, default=5.0)
    parser.add_argument("--request-batch-max-queries", type=int, default=256)
    parser.add_argument(
        "--request-wait-timeout-seconds",
        type=float,
        default=180.0,
    )
    parser.add_argument("--retrieval-use-fp16", action="store_true")
    parser.add_argument("--faiss-gpu", action="store_true")
    parser.add_argument("--require-faiss-gpu", action="store_true")
    parser.add_argument("--faiss-gpu-stream-flat", action="store_true")
    parser.add_argument("--faiss-gpu-device", type=int, default=0)
    parser.add_argument("--faiss-gpu-use-fp16", action="store_true")
    parser.add_argument("--faiss-temp-memory-mb", type=int, default=256)
    parser.add_argument("--faiss-add-batch-size", type=int, default=131072)
    parser.add_argument("--dense-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-file", required=True)
    args = parser.parse_args()

    setup_logging(args.log_file)
    LOGGER.info("Starting hybrid retriever with args=%s", vars(args))
    global retriever
    retriever = HybridRetriever(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
