# Provenance for DeepSeek-distilled Search-SFT exports

This note documents the data-construction method for approved records exported
by `deepseek_rollout.py --export-approved`. AetherSearch Search-SFT
trajectories are built by distilling DeepSeek's visible search actions and
answers while grounding every `<information>` observation in the local
Hybrid-RAG retriever. Pair each released snapshot with its exact checkpoint,
generator version and dataset manifest.

The exported JSONL deliberately keeps that release's five-field schema:
`id`, `question`, `trajectory_type`, `search_count`, and
`full_trajectory_text`. Do not add service metadata inside individual training
rows. Keep this provenance note and the versioned checkpoint alongside any
new dataset release.
The export policy version is `aethersearch_full_trajectory_v2_numeric_ids`.
Public IDs are stable six-digit input-position IDs, defaulting to `500001`
for the first candidate. The detailed checkpoint keeps the original
source-based internal ID and the public-ID mapping. A separately generated
file needs a non-overlapping `--public-id-start` range before merging.

## Retrieval during rollout

The teacher can answer without retrieval, but zero-search trajectories are
recorded as skipped and cannot enter this Search-SFT export. Every exported
row has `search_count>=1`. When it searches, the controller
executes only the local Hybrid-RAG V1 path against the Search-R1 wiki18 corpus:

1. A separate, long-running Search-R1 E5-base-v2/FAISS FlatIP dense server
   serves HTTP `POST http://127.0.0.1:8000/retrieve`, returning 20 candidates.
2. The long-running teacher process keeps the persistent wiki18 BM25 index
   loaded and obtains 20 sparse candidates locally.
3. The controller checks dense documents against the same corpus, fuses both
   branches using Reciprocal Rank Fusion with `k=60`, and exposes only the
   top three real passages as a bounded `<information>` observation.

Independent teacher trajectories may run concurrently. Search queries that
arrive in a short window share one dense HTTP batch; BM25, RRF, evidence
identity checks, and each trajectory's continuation remain query-specific.
Batching changes throughput, not the retrieval algorithm or document source.

The dense server is a distinct process, but the **complete hybrid fusion is
not** served by that port: BM25, RRF, and provenance checks run in the teacher
process. The published AetherSearch RL retriever also uses an HTTP `/retrieve`
service and the same `e5_Flat.index` artifact, but its released hybrid server combines normalized scores using a
weight. It must not be substituted for this RRF pipeline or described as an
identical retriever. Neither service is live web search.
The released RL topology and implementation are documented in
[`configs/retriever_external.yaml`](https://github.com/Muradil-mamat-211/AetherSearch/blob/main/configs/retriever_external.yaml)
and [`hybrid_retrieval_server.py`](https://github.com/Muradil-mamat-211/AetherSearch/blob/main/runtime_assets/retriever/hybrid_retrieval_server.py).

The model-written SEARCH action is preserved as received after validation;
the controller does not synthesize search think or query text. For a retrieved
FINAL, the controller validates the raw answer against visible evidence and
then normalizes only its training `<think>` to
`The retrieved evidence now supports the answer.` The unmodified raw final
action and retrieval receipts remain in the checkpoint. Direct answers are
parseable for internal audit only; no fake `<information>` is added and they
are excluded from the training export. DeepSeek private `reasoning_content` is replayed for tool-call
continuation but never exported as student training text.

Structural validation and answer overlap do not prove that a passage supports
the requested relationship. Every new candidate remains pending until explicit
human semantic review and approval. Export only approved records, then run the
public validator with `--db --require-approved` to verify the exact JSONL text
against its checkpoint.
