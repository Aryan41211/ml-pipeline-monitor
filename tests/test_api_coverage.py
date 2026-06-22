"""Additional API integration tests for coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from services.api.app import app
from src.jwt_auth import create_access_token, verify_token

client = TestClient(app)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-12345")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_EXPIRATION_MINUTES", "60")
    monkeypatch.setenv("MLMONITOR_API_KEY", "test-api-key")


def _auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


class TestJwtAuthEndpoints:
    def test_login_with_bearer_success(self):
        token = create_access_token(sub="alice", role="admin")
        r = client.get("/v1/auth/me", headers=_auth_header(token))
        assert r.status_code == 200

    def test_me_with_invalid_token(self):
        r = client.get("/v1/auth/me", headers={"Authorization": "Bearer invalidtoken"})
        assert r.status_code == 401

    def test_refresh_with_valid_token(self):
        token = create_access_token(sub="bob", role="viewer")
        r = client.get("/v1/auth/me", headers=_auth_header(token))
        assert r.status_code == 200


class TestAppLifespan:
    def test_lifespan_startup_shutdown(self):
        with TestClient(app) as client:
            r = client.get("/health")
            assert r.status_code == 200

    def test_detailed_health(self):
        with TestClient(app) as client:
            r = client.get("/health/detailed")
            assert r.status_code == 200

    def test_metrics_endpoint(self):
        with TestClient(app) as client:
            r = client.get("/metrics")
            assert r.status_code == 200

    def test_legacy_predict_requires_api_key_message(self):
        r = client.post("/predict", json={"features": {"x": 1.0}})
        assert r.status_code == 401
        body = r.json()
        assert "detail" in body

    def test_legacy_predict_success(self):
        with patch("services.model_service.predict_from_payload", return_value={"model_id": "m1", "predictions": [1]}):
            r = client.post("/predict", json={"features": {"x": 1.0}}, headers={"X-API-Key": "test-api-key"})
            assert r.status_code == 200
            assert r.json()["model_id"] == "m1"


class TestPredictV1ErrorPaths:
    def test_predict_value_error_missing_features(self):
        import numpy as np
        token = create_access_token(sub="user", role="admin")
        fake_model = MagicMock()
        fake_model.feature_names_in_ = np.array(["a", "b", "c"])
        fake_model.predict.return_value = np.array([1])
        fake_scaler = MagicMock()
        fake_scaler.transform.return_value = np.array([[1.0, 2.0, 3.0]])

        with patch("services.api.app.get_latest_production_model", return_value=(fake_model, fake_scaler, {"model_id": "m1", "dataset": "iris", "version": 1, "stage": "production", "artifact_path": "path"})):
            r = client.post(
                "/v1/predict",
                json={"features": {"x": 1.0}, "dataset": "iris"},
                headers=_auth_header(token),
            )
        assert r.status_code == 400
        body = r.json()
        assert "detail" in body
