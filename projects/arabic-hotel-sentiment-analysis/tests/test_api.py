from fastapi.testclient import TestClient
from sentiment_service.app import app
from sentiment_service.model import model


def test_healthz():
    response = TestClient(app).get('/healthz')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_validation_rejects_blank_text():
    response = TestClient(app).post('/v1/predict', json={'text': '   '})
    assert response.status_code == 422


def test_prediction_contract_when_model_loaded(monkeypatch):
    monkeypatch.setattr(model, 'pipeline', object())
    monkeypatch.setattr(model, 'predict_one', lambda text: {'sentiment':'positive','confidence':0.9,'probabilities':{'negative':0.02,'neutral':0.08,'positive':0.9},'model_version':'test'})
    response = TestClient(app).post('/v1/predict', json={'text':'الفندق ممتاز والخدمة رائعة'})
    assert response.status_code == 200
    assert set(response.json()) == {'sentiment','confidence','probabilities','model_version'}
