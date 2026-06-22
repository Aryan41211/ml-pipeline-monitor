"""Test suite for JWT authentication module."""

from __future__ import annotations

import os
import time

import pytest

from src.jwt_auth import (
    TokenPayload,
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_token,
    _get_algorithm,
    ALGORITHM,
)


@pytest.fixture(autouse=True)
def _jwt_secret(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-12345")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_EXPIRATION_MINUTES", "60")
    monkeypatch.setenv("JWT_REFRESH_EXPIRATION_DAYS", "7")


def test_create_access_token():
    token = create_access_token(sub="alice", role="admin")
    assert isinstance(token, str)
    assert token.count(".") == 2


def test_create_refresh_token():
    token = create_refresh_token(sub="bob", role="viewer")
    assert isinstance(token, str)
    assert token.count(".") == 2


def test_access_token_roundtrip():
    token = create_access_token(sub="charlie", role="operator", jti="test-jti-001")
    payload = verify_token(token)
    assert payload.sub == "charlie"
    assert payload.role == "operator"
    assert payload.jti == "test-jti-001"
    assert payload.refresh is False


def test_refresh_token_roundtrip():
    token = create_refresh_token(sub="dave", role="admin")
    payload = verify_token(token)
    assert payload.sub == "dave"
    assert payload.role == "admin"
    assert payload.refresh is True


def test_expired_token_rejected():
    token = create_access_token(sub="eve", expires_delta=-60)
    with pytest.raises(ValueError, match="expired"):
        verify_token(token)


def test_tampered_signature_rejected():
    token = create_access_token(sub="frank")
    parts = token.split(".")
    tampered = parts[0] + "." + parts[1] + ".invalidsignature"
    with pytest.raises(ValueError, match="Invalid token signature"):
        verify_token(tampered)


def test_malformed_token_rejected():
    with pytest.raises(ValueError, match="Invalid token format"):
        verify_token("ab")


def test_different_algos():
    for algo in ("HS256", "HS384", "HS512"):
        os.environ["JWT_ALGORITHM"] = algo
        token = create_access_token(sub="grace")
        payload = verify_token(token)
        assert payload.sub == "grace"
    os.environ.pop("JWT_ALGORITHM", None)


def test_token_payload_dataclass():
    payload = TokenPayload(sub="user", role="viewer", exp=9999999999, iat=1000000000, jti="abc", refresh=True)
    assert payload.sub == "user"
    assert payload.refresh is True
