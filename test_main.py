# test_main.py
import pytest
from fastapi.testclient import TestClient

import main

VALID_ROW = [5.1, 3.5, 1.4, 0.2]

@pytest.fixture()
def client():
    with TestClient(main.app) as c:   # контекстный менеджер запускает lifespan
        yield c

def test_health(client):
    assert client.get("/health").status_code == 200

def test_info_contains_version(client):
    body = client.get("/info").json()
    assert "model_version" in body and "feature_names" in body

def test_predict_ok(client):
    r = client.post("/predict", json={"features": VALID_ROW})
    assert r.status_code == 200
    assert r.json()["prediction"] == 0
    assert len(r.json()["probabilities"]) == 3

def test_predict_rejects_wrong_shape(client):
    assert client.post("/predict", json={"features": [1.0, 2.0]}).status_code == 422

def test_batch(client):
    r = client.post("/predict/batch",
                    json={"features": [VALID_ROW, [6.7, 3.0, 5.2, 2.3]]})
    assert r.status_code == 200
    assert r.json()["predictions"] == [0, 2]

def test_batch_rejects_ragged_rows(client):
    r = client.post("/predict/batch", json={"features": [VALID_ROW, [1.0]]})
    assert r.status_code == 422

def test_api_key_when_enabled(client):
    main.API_KEY = "secret"            # имитируем прод-конфиг
    try:
        assert client.post("/predict", json={"features": VALID_ROW}).status_code == 401
        r = client.post("/predict", json={"features": VALID_ROW},
                        headers={"X-API-Key": "secret"})
        assert r.status_code == 200
    finally:
        main.API_KEY = None