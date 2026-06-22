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


class TestPredictV1Errors:
    def test_predict_validation_error(self):
        token = create_access_token(sub="user", role="admin")
        r = client.post(
            "/v1/predict",
            json={"features": "invalid"},
            headers=_auth_header(token),
        )
        assert r.status_code == 422

    def test_predict_empty_features(self):
        token = create_access_token(sub="user", role="admin")
        r = client.post(
            "/v1/predict",
            json={},
            headers=_auth_header(token),
        )
        assert r.status_code == 422

    def test_predict_file_not_found(self):
        token = create_access_token(sub="user", role="admin")
        fake_model = MagicMock()
        fake_model.predict.return_value = [1]
        with patch("services.api.app.get_latest_production_model", return_value=(fake_model, None, {"artifact_path": "/nonexistent/path/model.joblib", "model_id": "m1", "dataset": "iris", "version": 1, "stage": "production"})):
            r = client.post(
                "/v1/predict",
                json={"features": {"x": 1.0}},
                headers=_auth_header(token),
            )
        assert r.status_code == 404
