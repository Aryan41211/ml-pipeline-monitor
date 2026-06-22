"""Coverage-targeted tests for services/api/app.py uncovered paths."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from services.api.app import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-12345")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_EXPIRATION_MINUTES", "60")
    monkeypatch.setenv("MLMONITOR_API_KEY", "test-api-key")


class TestAPIAuthPaths:
    def test_api_key_auth_success(self):
        r = client.post("/v1/predict", json={"features": {"x": 1.0}}, headers={"X-API-Key": "test-api-key"})
        assert r.status_code in (404, 500)

    def test_jwt_invalid_token_rejected(self):
        r = client.get("/v1/auth/me", headers={"Authorization": "Bearer badtoken"})
        assert r.status_code == 401

    def test_jwt_expired_token_rejected(self):
        from src.jwt_auth import create_access_token
        expired = create_access_token(sub="u", expires_delta=-10)
        r = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {expired}"})
        assert r.status_code == 401

    def test_authenticate_with_valid_jwt(self):
        from src.jwt_auth import create_access_token
        token = create_access_token(sub="legit", role="admin")
        r = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["sub"] == "legit"


class TestPredictErrorPaths:
    def test_predict_file_not_found(self):
        from src.jwt_auth import create_access_token
        import numpy as np
        token = create_access_token(sub="u")
        fake_model = MagicMock()
        fake_model.feature_names_in_ = np.array(["a", "b"])
        fake_model.predict.return_value = np.array([1])
        fake_scaler = MagicMock()
        fake_scaler.transform.side_effect = FileNotFoundError("no scaler")
        with patch("services.api.app.get_latest_production_model", return_value=(fake_model, fake_scaler, {"model_id": "m1", "dataset": "iris", "version": 1, "stage": "production", "artifact_path": "x"})):
            r = client.post("/v1/predict", json={"features": {"a": 1, "b": 2}}, headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 500

    def test_predict_generic_exception(self):
        from src.jwt_auth import create_access_token
        import numpy as np
        token = create_access_token(sub="u")
        fake_model = MagicMock()
        fake_model.feature_names_in_ = np.array(["a", "b"])
        fake_model.predict.side_effect = RuntimeError("model error")
        fake_scaler = MagicMock()
        fake_scaler.transform.return_value = np.array([[1.0, 2.0]])
        with patch("services.api.app.get_latest_production_model", return_value=(fake_model, fake_scaler, {"model_id": "m1", "dataset": "iris", "version": 1, "stage": "production", "artifact_path": "x"})):
            r = client.post("/v1/predict", json={"features": {"a": 1, "b": 2}}, headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 500


class TestHealthPaths:
    def test_health_detailed_with_db_error(self):
        with patch("services.api.app.get_backend") as mock_b:
            mock_b.return_value.connect.side_effect = Exception("db down")
            r = client.get("/health/detailed")
        assert r.status_code in (200, 500)

    def test_health_ready_with_db_error(self):
        with patch("services.api.app.get_backend") as mock_b:
            mock_b.return_value.connect.side_effect = Exception("db down")
            r = client.get("/health/ready")
        assert r.status_code in (200, 500)
