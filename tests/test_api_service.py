from fastapi.testclient import TestClient

from services.api.main import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json().get("status") == "ok"


def test_predict_endpoint_success(monkeypatch):
    expected = {
        "model_id": "ABCD1234",
        "dataset": "Iris Species",
        "version": 2,
        "stage": "production",
        "predictions": [1],
    }

    monkeypatch.setattr("services.api.main.predict_from_payload", lambda payload, dataset=None: expected)

    client = TestClient(app)
    response = client.post(
        "/predict",
        json={"features": {"f1": 1.0, "f2": 2.0}, "dataset": "Iris Species"},
    )
    assert response.status_code == 200
    assert response.json() == expected


def test_predict_endpoint_validation_error(monkeypatch):
    monkeypatch.setattr(
        "services.api.main.predict_from_payload",
        lambda payload, dataset=None: (_ for _ in ()).throw(ValueError("bad input")),
    )

    client = TestClient(app)
    response = client.post("/predict", json={"features": {"f1": 1.0}})
    assert response.status_code == 400
    assert "bad input" in response.json().get("detail", "")
