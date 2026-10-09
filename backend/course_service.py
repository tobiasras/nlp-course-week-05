"""Dense search over DTU courses. Each text field is embedded on its own."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import simplemma

EMBEDDING_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v2"

_STOPWORDS = (
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "from", "at", "as", "is", "are", "be", "this", "that", "it", "its",
    "og", "i", "på", "paa", "af", "til", "en", "et", "den", "det", "som", "med", "er", "de", "at",
)
_FOLD = str.maketrans({"æ": "ae", "ø": "o", "å": "a", "ä": "a", "ö": "o", "ü": "u"})
_TOKEN = re.compile(r"[a-z0-9]+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
_PHONE = re.compile(r"\+?\d[\d\s().-]{6,}\d")

# name, json key, whether to strip emails and phone numbers
FIELDS = (
    ("title", "title", False),
    ("learning_objectives", "learning_objectives", False),
    ("danish_title", "Danish title", False),
    ("academic_prerequisites", "Academic prerequisites", False),
    ("responsible", "Responsible", True),
    ("course_co_responsible", "Course co-responsible", True),
)

_model: Any = None
Embedder = Callable[[list[str]], Any]


def tokenize(text: str, strip_contacts: bool = False) -> list[str]:
    """Lowercase, fold accents, and lemmatize."""
    text = text.lower().translate(_FOLD)
    text = "".join(char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char))
    if strip_contacts:
        text = _PHONE.sub(" ", _EMAIL.sub(" ", text))
    return [simplemma.lemmatize(token, lang=("en", "da")) for token in _TOKEN.findall(text)]


STOPWORDS = frozenset(token for word in _STOPWORDS for token in tokenize(word))


def drop_frequent(docs: list[list[str]], threshold: float = 0.5) -> tuple[list[list[str]], set[str]]:
    """Drop stopwords and tokens that occur in more than half of the documents."""
    counts: dict[str, int] = {}
    for doc in docs:
        for token in set(doc):
            counts[token] = counts.get(token, 0) + 1
    banned = set(STOPWORDS)
    if len(docs) > 1:
        banned.update(token for token, count in counts.items() if count / len(docs) > threshold)
    return [[token for token in doc if token not in banned] for doc in docs], banned


def _texts(value: Any) -> list[str]:
    """Turn a string or list into a list of text strings."""
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            texts.extend(_texts(item))
        return texts
    return []


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each vector to length 1."""
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _top(scores: np.ndarray, top_k: int) -> list[int]:
    """Return the indexes of the highest scores above zero."""
    chosen = [int(index) for index in np.argsort(-scores) if scores[index] > 0]
    return chosen[:top_k]


@dataclass
class Field:
    originals: list[str]
    cleaned: list[str]
    course_index: np.ndarray
    vectors: np.ndarray
    course_vectors: np.ndarray
    has_course: np.ndarray
    banned: set[str]
    strip_contacts: bool


class CourseSearchService:
    def __init__(self, embedder: Embedder | None = None) -> None:
        """Create an empty index. Pass embedder to skip the real model in tests."""
        self.embedder = embedder
        self.courses: list[dict[str, str]] = []
        self.by_id: dict[str, int] = {}
        self.indexes: dict[str, Field] = {}
        self.n_objectives = 0
        self.ready = False

    def init(self, path: str | Path) -> None:
        """Load the JSONL file and embed each search field separately."""
        data_path = Path(path)
        if not data_path.is_file():
            raise FileNotFoundError(f"course data not found: {data_path}")

        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for line in data_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            course_id = str(record.get("course_code", "")).strip()
            if not course_id or course_id in seen:
                continue
            seen.add(course_id)
            records.append(record)

        self.courses = [
            {"id": str(record["course_code"]).strip(), "title": str(record.get("title") or record["course_code"])}
            for record in records
        ]
        self.by_id = {course["id"]: index for index, course in enumerate(self.courses)}
        self.n_objectives = sum(len(_texts(record.get("learning_objectives"))) for record in records)
        self.indexes = {
            name: field
            for name, key, strip_contacts in FIELDS
            if (field := self._build_field(key, strip_contacts, records)) is not None
        }
        self.ready = True

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Find courses whose fields are closest to the query."""
        self._check()
        scores = np.zeros(len(self.courses))
        for field in self.indexes.values():
            query_vector = self._query_vector(field, query)
            if query_vector is None:
                continue
            np.maximum.at(scores, field.course_index, field.vectors @ query_vector)
        return [self._hit(index, scores[index]) for index in _top(scores, top_k)]

    def similar(self, course_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Find other courses closest to this one."""
        self._check()
        course_index = self.by_id[course_id]
        total = np.zeros(len(self.courses))
        count = np.zeros(len(self.courses))
        for field in self.indexes.values():
            if not field.has_course[course_index]:
                continue
            scores = field.course_vectors @ field.course_vectors[course_index]
            total[field.has_course] += scores[field.has_course]
            count[field.has_course] += 1
        count[course_index] = 0
        scores = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        return [self._hit(index, scores[index]) for index in _top(scores, top_k)]

    def search_objectives(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Find learning objectives closest to the query."""
        self._check()
        field = self.indexes.get("learning_objectives")
        if field is None:
            return []
        query_vector = self._query_vector(field, query)
        if query_vector is None:
            return []
        scores = field.vectors @ query_vector
        hits = []
        for doc_index in _top(scores, top_k):
            course = self.courses[int(field.course_index[doc_index])]
            hits.append(
                {
                    "course_id": course["id"],
                    "title": course["title"],
                    "objective": field.originals[doc_index],
                    "score": float(scores[doc_index]),
                }
            )
        return hits

    def health(self) -> dict[str, Any]:
        """Report how many courses and objectives are indexed."""
        self._check()
        return {"status": "ok", "index_sizes": {"courses": len(self.courses), "objectives": self.n_objectives}}

    def _check(self) -> None:
        """Fail if init has not been called."""
        if not self.ready:
            raise RuntimeError("course index is not loaded")

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Turn texts into normalized embedding vectors."""
        embed = self.embedder or _load_model()
        vectors = np.asarray(embed(texts), dtype=np.float64)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return _normalize(vectors)

    def _build_field(self, key: str, strip_contacts: bool, records: list[dict[str, Any]]) -> Field | None:
        """Clean and embed one JSON field for every course."""
        originals: list[str] = []
        course_index: list[int] = []
        token_docs: list[list[str]] = []
        for record in records:
            course_id = str(record["course_code"]).strip()
            value = record.get(key) if key in ("title", "learning_objectives") else (record.get("fields") or {}).get(key)
            for text in _texts(value):
                originals.append(text)
                course_index.append(self.by_id[course_id])
                token_docs.append(tokenize(text, strip_contacts=strip_contacts))
        if not token_docs:
            return None

        cleaned_docs, banned = drop_frequent(token_docs)
        cleaned = [" ".join(tokens) for tokens in cleaned_docs]
        if not any(cleaned):
            return None

        vectors = self._embed(cleaned)
        indexes = np.asarray(course_index, dtype=np.int32)
        course_vectors, has_course = _course_means(vectors, indexes, len(self.courses))
        return Field(originals, cleaned, indexes, vectors, course_vectors, has_course, banned, strip_contacts)

    def _query_vector(self, field: Field, query: str) -> np.ndarray | None:
        """Embed the query with the same cleaning as this field."""
        tokens = [token for token in tokenize(query, field.strip_contacts) if token not in field.banned]
        if not tokens:
            return None
        return self._embed([" ".join(tokens)])[0]

    def _hit(self, index: int, score: float) -> dict[str, Any]:
        """Build one course result."""
        course = self.courses[index]
        return {"course_id": course["id"], "title": course["title"], "score": float(score)}


def _load_model():
    """Load the sentence embedding model once."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL)
    return lambda texts: _model.encode(texts, normalize_embeddings=True)


def _course_means(vectors: np.ndarray, course_index: np.ndarray, n_courses: int) -> tuple[np.ndarray, np.ndarray]:
    """One mean vector per course for this field."""
    totals = np.zeros((n_courses, vectors.shape[1]))
    np.add.at(totals, course_index, vectors)
    counts = np.bincount(course_index, minlength=n_courses).astype(float)
    means = _normalize(totals / np.maximum(counts, 1)[:, None])
    means[counts == 0] = 0
    return means, counts > 0
