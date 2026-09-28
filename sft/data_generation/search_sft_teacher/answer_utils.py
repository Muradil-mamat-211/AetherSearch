"""Shared answer canonicalization and heuristic evidence warnings."""

from __future__ import annotations

import re
import string
from typing import Any


ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was", "were",
    "who", "what", "when", "where", "which", "whose", "whom", "why", "how", "did", "does", "do",
    "by", "with", "from", "that", "this", "it", "as", "at", "be", "been", "being", "into", "about",
}
QUESTION_GENERIC_WORDS = STOPWORDS | {
    "answer", "answers", "actor", "actress", "album", "author", "book", "called", "capital",
    "character", "city", "company", "country", "date", "directed", "director", "episode", "film",
    "first", "game", "genre", "group", "held", "kind", "last", "located", "made", "many", "movie",
    "name", "named", "novel", "number", "person", "place", "played", "player", "plays", "released",
    "season", "series", "show", "singer", "song", "state", "team", "television", "title", "type",
    "used", "uses", "using", "written", "wrote", "year",
}
ANSWER_STRIP_CHARS = " \t\r\n\"'“”‘’`[]{}()<>"
ANSWER_EDGE_PUNCT = ",;:!?，。；：！？、"


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = ARTICLES_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    return " ".join(text.split())


def is_all_upper_alias(text: str) -> bool:
    letters = [ch for ch in str(text) if ch.isalpha()]
    return bool(letters) and all(ch.isupper() for ch in letters)


def titlecase_upper_alias(text: str) -> str:
    def fix_token(token: str) -> str:
        compact = re.sub(r"[^A-Za-z]", "", token)
        if len(compact) <= 4 and token.replace(".", "").isalpha():
            return token
        return token[:1].upper() + token[1:].lower() if token else token

    return " ".join(fix_token(part) for part in text.split())


def canonicalize_answer(answer: Any) -> str:
    text = TAG_RE.sub(" ", str(answer or ""))
    text = text.replace("\u00a0", " ")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r"\s+", " ", text).strip(ANSWER_STRIP_CHARS)
    text = re.sub(rf"^[{re.escape(ANSWER_EDGE_PUNCT)}\s]+", "", text)
    text = re.sub(rf"[{re.escape(ANSWER_EDGE_PUNCT)}\s]+$", "", text)
    while text.endswith(".") and not re.search(r"(?:[A-Za-z]\.){2,}$", text):
        text = text[:-1].rstrip()
    text = re.sub(r"\s+", " ", text).strip(ANSWER_STRIP_CHARS)
    if is_all_upper_alias(text):
        compact = re.sub(r"[^A-Za-z]", "", text)
        if not (len(compact) <= 4 and " " not in text):
            text = titlecase_upper_alias(text)
    return text


def tokenize_normalized(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_answer(text))


def is_entity_present(entity: str, text: str) -> bool:
    norm_entity = normalize_answer(entity)
    if not norm_entity:
        return False
    return re.search(rf"(?<!\w){re.escape(norm_entity)}(?!\w)", normalize_answer(text)) is not None


def extract_question_key_entities(question: str, max_entities: int = 5) -> list[str]:
    question = " ".join(str(question or "").split())
    candidates: list[str] = []
    candidates.extend(re.findall(r"['\"]([^'\"]{2,80})['\"]", question))
    capitalized = re.findall(
        r"\b(?:[A-Z][A-Za-z0-9'&.-]*|[A-Z0-9]{2,})(?:\s+(?:[A-Z][A-Za-z0-9'&.-]*|[A-Z0-9]{2,}))*",
        question,
    )
    for phrase in capitalized:
        first = normalize_answer(phrase).split(" ")[0] if normalize_answer(phrase) else ""
        if first and first not in QUESTION_GENERIC_WORDS:
            candidates.append(phrase)
    for token in tokenize_normalized(question):
        if token not in QUESTION_GENERIC_WORDS and len(token) >= 4:
            candidates.append(token)
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        norm = normalize_answer(candidate)
        if not norm or norm in seen or norm in QUESTION_GENERIC_WORDS:
            continue
        seen.add(norm)
        deduped.append(candidate)
        if len(deduped) >= max_entities:
            break
    return deduped


def evidence_support_status(question: str, information: str) -> dict[str, Any]:
    entities = extract_question_key_entities(question)
    missing = [entity for entity in entities if not is_entity_present(entity, information)]
    return {
        "warning": len(entities) > 1 and bool(missing),
        "question_key_entities": entities,
        "missing_question_entities": missing,
    }
