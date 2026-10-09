import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from course_service import CourseSearchService, drop_frequent, tokenize
from main import app, service

EMBED_DIM = 512
_TOKEN_INDEX: dict[str, int] = {}

FIXTURE_COURSES = [
    {
        "course_code": "02451",
        "title": "02451 Zebrafish genomics commonword",
        "learning_objectives": [
            "Study quantum entanglement in genomes",
            "Review linear algebra basics",
        ],
        "fields": {
            "Danish title": "Zebrafisk genomik",
            "Academic prerequisites": "introductory biology",
            "Responsible": "Bjørn Sand Jensen , uepe@dtu.dk , Ph. (+45) 4525 5203",
        },
    },
    {
        "course_code": "02460",
        "title": "02460 Particle physics commonword",
        "learning_objectives": [
            "Apply quantum entanglement to sensors",
            "Practice numerical methods",
        ],
        "fields": {
            "Danish title": "Partikelfysik",
            "Responsible": "Ada Lovelace",
        },
    },
    {
        "course_code": "01002",
        "title": "01002 Calculus methods commonword",
        "learning_objectives": [
            "Calculate Surface Integrals",
            "Solve differential equations",
            "Estimate fourier series",
        ],
        "fields": {
            "Danish title": "Matematik",
            "Academic prerequisites": "single variable calculus",
            "Responsible": "Ulrik Engelund Pedersen",
            "Course co-responsible": ["Jane Doe", "John Roe"],
        },
    },
]


def fake_embed(texts: list[str]) -> np.ndarray:
    """One dimension per token so tests do not depend on hash collisions."""
    vectors = np.zeros((len(texts), EMBED_DIM), dtype=np.float64)
    for row, text in enumerate(texts):
        for token in text.split():
            if token not in _TOKEN_INDEX:
                _TOKEN_INDEX[token] = len(_TOKEN_INDEX)
            index = _TOKEN_INDEX[token]
            if index < EMBED_DIM:
                vectors[row, index] += 1.0
    return vectors


@pytest.fixture
def client(tmp_path, monkeypatch):
    path = tmp_path / "courses.jsonl"
    path.write_text(
        "".join(json.dumps(course) + "\n" for course in FIXTURE_COURSES),
        encoding="utf-8",
    )
    monkeypatch.setenv("COURSES_PATH", str(path))
    service.embedder = fake_embed
    with TestClient(app) as test_client:
        yield test_client


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "app is running"}


def test_preprocess_lowercase_lemma_then_drops_frequent_words():
    assert tokenize("The Courses") == ["the", "course"]

    cleaned, banned = drop_frequent(
        [
            ["the", "course"],
            ["the", "alpha"],
            ["the", "beta"],
        ]
    )
    assert "the" in banned
    assert cleaned[0] == ["course"]

    cleaned, banned = drop_frequent(
        [
            ["quantum", "alpha"],
            ["quantum", "beta"],
            ["quantum", "gamma"],
        ]
    )
    assert "quantum" in banned
    assert all("quantum" not in doc for doc in cleaned)
    assert cleaned[0] == ["alpha"]


def test_tokenize_folds_accents_and_strips_contacts():
    tokens = tokenize(
        "Bjørn Sand Jensen , uepe@dtu.dk , Ph. (+45) 4525 5203",
        strip_contacts=True,
    )
    assert "bjorn" in tokens
    assert "jensen" in tokens
    assert "4525" not in tokens
    assert all("@" not in token for token in tokens)


def _indexed_tokens(field_name: str) -> set[str]:
    tokens: set[str] = set()
    for text in service.indexes[field_name].cleaned:
        tokens.update(text.split())
    return tokens


def test_fields_stay_separate(client):
    title_tokens = _indexed_tokens("title")
    objective_tokens = _indexed_tokens("learning_objectives")
    assert "zebrafish" in title_tokens
    assert "zebrafish" not in objective_tokens
    assert "quantum" in objective_tokens
    assert "quantum" not in title_tokens
    assert "commonword" not in title_tokens
    for text in service.indexes["title"].originals:
        assert "Surface Integrals" not in text


def test_search_returns_indexed_course(client):
    response = client.get("/v1/search", params={"query": "zebrafish", "mode": "sparse"})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "zebrafish"
    assert "mode" not in body
    assert "alpha" not in body
    assert body["top_k"] == 10
    assert body["results"][0]["course_id"] == "02451"
    assert body["results"][0]["title"] == "02451 Zebrafish genomics commonword"
    assert body["results"][0]["score"] > 0


def test_teacher_search_matches_folded_name(client):
    response = client.get("/v1/search", params={"query": "bjorn"})
    assert response.status_code == 200
    assert response.json()["results"][0]["course_id"] == "02451"


def test_similar_courses_skips_the_query_course(client):
    response = client.get("/v1/courses/02451/similar")
    assert response.status_code == 200
    body = response.json()
    assert body["query_course_id"] == "02451"
    assert "mode" not in body
    assert "alpha" not in body
    assert body["results"][0]["course_id"] == "02460"
    assert body["results"][0]["score"] > 0
    assert all(hit["course_id"] != "02451" for hit in body["results"])


def test_similar_unknown_course_is_not_found(client):
    response = client.get("/v1/courses/missing/similar")
    assert response.status_code == 404


def test_objective_search_returns_original_objective(client):
    response = client.get(
        "/v1/objectives/search", params={"query": "surface integrals"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "surface integrals"
    assert "mode" not in body
    assert "alpha" not in body
    hit = body["results"][0]
    assert hit["course_id"] == "01002"
    assert hit["title"] == "01002 Calculus methods commonword"
    assert hit["objective"] == "Calculate Surface Integrals"
    assert hit["score"] > 0


def test_health_counts_match_fixture(client):
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "index_sizes": {"courses": 3, "objectives": 7},
    }


def test_init_builds_separate_indexes_without_the_app(tmp_path):
    path = tmp_path / "courses.jsonl"
    path.write_text(json.dumps(FIXTURE_COURSES[0]) + "\n", encoding="utf-8")
    index = CourseSearchService(embedder=fake_embed)
    index.init(path)
    assert index.health()["index_sizes"]["courses"] == 1
    assert "title" in index.indexes
    assert "learning_objectives" in index.indexes
    assert index.indexes["title"].originals == ["02451 Zebrafish genomics commonword"]
    assert index.indexes["learning_objectives"].originals[0].startswith("Study quantum")
