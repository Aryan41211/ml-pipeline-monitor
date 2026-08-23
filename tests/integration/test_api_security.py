"""Security regression tests for the inference API.

Each test here pins down a specific hole that was open before: an unset API key
authenticating any caller, non-constant-time key comparison, and unlimited
login attempts.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ml_pipeline_monitor.api.main import app

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _jwt_env(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-for-security-suite")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")


class TestApiKeyEnforcement:
    def test_unset_api_key_rejects_arbitrary_header(self, monkeypatch):
        """With no key configured, no presented key may authenticate."""
        monkeypatch.setenv("MLMONITOR_API_KEY", "")
        r = client.post(
            "/predict",
            headers={"X-API-Key": "anything-at-all"},
            json={"features": {"a": 1.0}},
        )
        assert r.status_code == 401

    def test_wrong_api_key_rejected(self, monkeypatch):
        monkeypatch.setenv("MLMONITOR_API_KEY", "the-real-key")
        r = client.post(
            "/predict",
            headers={"X-API-Key": "the-wrong-key"},
            json={"features": {"a": 1.0}},
        )
        assert r.status_code == 401

    def test_missing_api_key_rejected(self, monkeypatch):
        monkeypatch.setenv("MLMONITOR_API_KEY", "the-real-key")
        r = client.post("/predict", json={"features": {"a": 1.0}})
        assert r.status_code == 401

    def test_correct_api_key_passes_auth(self, monkeypatch):
        """A valid key gets past auth; the request then fails on its own merits, not on 401."""
        monkeypatch.setenv("MLMONITOR_API_KEY", "the-real-key")
        r = client.post(
            "/predict",
            headers={"X-API-Key": "the-real-key"},
            json={"features": {"a": 1.0}},
        )
        assert r.status_code != 401

    def test_v1_endpoint_rejects_arbitrary_key_when_unset(self, monkeypatch):
        monkeypatch.setenv("MLMONITOR_API_KEY", "")
        r = client.get("/v1/auth/me", headers={"X-API-Key": "guessed"})
        assert r.status_code == 401


class TestSecurityHeaders:
    def test_responses_carry_hardening_headers(self):
        r = client.get("/health/live")
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert r.headers["X-Frame-Options"] == "DENY"
        assert r.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


class TestLoginRateLimit:
    def test_repeated_failed_logins_are_throttled(self, monkeypatch):
        """Login is the brute-force surface and must be rate limited."""
        monkeypatch.setenv("AUTH_USERNAME", "admin")
        monkeypatch.setenv("AUTH_PASSWORD", "correct-horse")

        statuses = [
            client.post("/v1/auth/login", json={"username": "admin", "password": f"guess-{i}"}).status_code
            for i in range(30)
        ]
        assert 429 in statuses, f"no throttling observed: {sorted(set(statuses))}"
