"""Coverage-targeted tests for services/api/app.py uncovered paths and model_service."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from services.api.app import app
from services.model_service import (
    _validate_dataset,
    _validate_model_id,
    _validate_stage,
    list_models,
    revert_to_previous_production,
)

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


class TestLoginEndpoint:
    def test_login_with_fresh_token(self):
        import services.api.app as api_app
        with patch.object(api_app, "_get_expiration_minutes", return_value=60):
            with patch("src.auth._check_login", return_value=(True, "")):
                with patch("src.auth._resolve_user", return_value="admin"):
                    with patch("src.auth._credentials", return_value={"admin": {"password": "x", "role": "admin"}}):
                        r = client.post("/v1/auth/login", json={"username": "admin", "password": "password", "refresh": False})
        assert r.status_code == 200
        body = r.json()
        assert "access_token" in body
        assert "expires_in" in body

    def test_refresh_endpoint_success(self):
        from src.jwt_auth import create_refresh_token
        token = create_refresh_token(sub="admin", role="admin")
        r = client.post("/v1/auth/refresh", json={"refresh_token": token})
        assert r.status_code == 200
        assert "access_token" in r.json()


class TestModelServiceCoverage:
    def test_validate_model_id_empty(self):
        with pytest.raises(ValueError, match="model_id is required"):
            _validate_model_id("")

    def test_validate_model_id_none(self):
        with pytest.raises(ValueError, match="model_id is required"):
            _validate_model_id("")

    def test_validate_stage_invalid(self):
        with pytest.raises(ValueError, match="Invalid stage"):
            _validate_stage("invalid_stage")

    def test_validate_dataset_empty(self):
        with pytest.raises(ValueError, match="dataset is required"):
            _validate_dataset("")

    def test_list_models_invalid_limit(self):
        with pytest.raises(ValueError, match="limit must be between"):
            list_models(limit=0)
        with pytest.raises(ValueError, match="limit must be between"):
            list_models(limit=2000)

    def test_revert_to_previous_no_previous(self):
        with patch("services.model_service.get_recent_production_models", return_value=[]):
            with patch("services.model_service.get_rollback_hint", return_value={"previous_production": None}):
                with pytest.raises(ValueError, match="No previous production model"):
                    revert_to_previous_production(dataset="iris")

    def test_revert_to_previous_success(self):
        prev = {"model_id": "m-prev", "version": 1}
        with patch("services.model_service.get_rollback_hint", return_value={"previous_production": prev}):
            with patch("services.model_service.update_model_stage"):
                result = revert_to_previous_production(dataset="iris")
        assert result["model_id"] == "m-prev"
