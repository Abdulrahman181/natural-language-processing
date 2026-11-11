import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from helpers import make_documents  # noqa: E402

from rageval.api import create_app  # noqa: E402
from rageval.corpus import save_corpus  # noqa: E402
from rageval.index import build_index  # noqa: E402


def test_api_health_and_not_ready(tmp_path):
    client = TestClient(create_app(tmp_path / "missing"))
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json()["status"] == "not_ready"


def test_api_search_after_index_load(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    save_corpus(make_documents(), corpus)
    build_index(make_documents(), tmp_path / "idx", 60, 10, 20)
    with TestClient(create_app(tmp_path / "idx")) as client:
        assert client.get("/readyz").json()["status"] == "ready"
        response = client.post("/v1/search", json={"query": "زيتون", "k": 2})
        assert response.status_code == 200
        assert 1 <= len(response.json()["results"]) <= 2
