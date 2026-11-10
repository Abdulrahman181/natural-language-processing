import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from cwe_qlora.api import create_app  # noqa: E402


def test_health_and_readiness_endpoints():
    client = TestClient(create_app())
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "not_ready", "model_loaded": False}


def test_predict_validates_empty_description_before_model_load():
    client = TestClient(create_app())
    response = client.post("/v1/predict", json={"description": ""})
    assert response.status_code == 422
