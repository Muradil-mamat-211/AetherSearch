from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RetrievalDocument:
    document_id: str
    contents: str
    score: float | None = None

    @property
    def passage_id(self) -> str:
        """Stable corpus passage ID exposed by the production retriever."""

        return str(self.document_id)


@dataclass(frozen=True)
class RetrievalResponse:
    documents_by_query: tuple[tuple[RetrievalDocument, ...], ...]


def parse_retrieval_payload(
    payload: Mapping[str, Any],
    *,
    expected_query_count: int,
) -> RetrievalResponse:
    raw_results = payload.get("result")
    if not isinstance(raw_results, list) or len(raw_results) != expected_query_count:
        raise ValueError("Retriever result cardinality mismatch")
    parsed: list[tuple[RetrievalDocument, ...]] = []
    for query_results in raw_results:
        if not isinstance(query_results, list):
            raise ValueError("Each retriever query result must be a list")
        documents: list[RetrievalDocument] = []
        for item in query_results:
            if not isinstance(item, Mapping):
                raise ValueError("Retriever documents must be mappings")
            wrapped = item.get("document", item)
            if not isinstance(wrapped, Mapping):
                raise ValueError("Invalid retriever document wrapper")
            contents = str(wrapped.get("contents", wrapped.get("text", "")))
            if not contents:
                raise ValueError("Retriever returned an empty document")
            score = item.get("score")
            documents.append(
                RetrievalDocument(
                    document_id=str(wrapped.get("id", "")),
                    contents=contents,
                    score=float(score) if score is not None else None,
                )
            )
        parsed.append(tuple(documents))
    return RetrievalResponse(tuple(parsed))
