"""JWT authentication and token management for FastAPI."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from src.config_loader import load_config
from src.logger import get_app_logger

LOGGER = get_app_logger("auth_jwt")

ALGORITHM = "HS256"
DEFAULT_EXPIRATION_MINUTES = 60
DEFAULT_REFRESH_EXPIRATION_DAYS = 7


@dataclass
class TokenPayload:
    sub: str
    role: str
    exp: int
    iat: int
    jti: str
    refresh: bool = False


def _get_secret_key() -> str:
    secret = os.getenv("JWT_SECRET", "")
    if not secret:
        cfg = load_config().get("api", {})
        secret = cfg.get("jwt_secret", "")
    if not secret:
        raise RuntimeError("JWT_SECRET is not configured")
    return secret


def _get_algorithm() -> str:
    return os.getenv("JWT_ALGORITHM", load_config().get("api", {}).get("jwt_algorithm", ALGORITHM))


def _get_expiration_minutes() -> int:
    try:
        return int(os.getenv("JWT_EXPIRATION_MINUTES", load_config().get("api", {}).get("jwt_expiration_minutes", DEFAULT_EXPIRATION_MINUTES)))
    except Exception:
        return DEFAULT_EXPIRATION_MINUTES


def _get_refresh_expiration_days() -> int:
    try:
        return int(os.getenv("JWT_REFRESH_EXPIRATION_DAYS", load_config().get("api", {}).get("jwt_refresh_expiration_days", DEFAULT_REFRESH_EXPIRATION_DAYS)))
    except Exception:
        return DEFAULT_REFRESH_EXPIRATION_DAYS


def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _base64url_decode(data: str) -> bytes:
    padding = 4 - len(data) % 4
    data = data + ("=" * padding)
    return base64.urlsafe_b64decode(data)


def _sign(payload_b64: str, secret: str, algorithm: str) -> str:
    if algorithm == "HS256":
        signature = hmac.new(secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).digest()
    elif algorithm == "HS384":
        signature = hmac.new(secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha384).digest()
    elif algorithm == "HS512":
        signature = hmac.new(secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha512).digest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return _base64url_encode(signature)


def create_access_token(
    sub: str,
    role: str = "viewer",
    expires_delta: Optional[int] = None,
    jti: Optional[str] = None,
) -> str:
    now = int(time.time())
    exp = now + (expires_delta or _get_expiration_minutes() * 60)
    payload = TokenPayload(
        sub=sub,
        role=role,
        exp=exp,
        iat=now,
        jti=jti or str(uuid.uuid4())[:16],
        refresh=False,
    )
    header = {"alg": _get_algorithm(), "typ": "JWT"}
    payload_b64 = _base64url_encode(json.dumps(payload.__dict__).encode("utf-8"))
    header_b64 = _base64url_encode(json.dumps(header).encode("utf-8"))
    signature = _sign(f"{header_b64}.{payload_b64}", _get_secret_key(), _get_algorithm())
    return f"{header_b64}.{payload_b64}.{signature}"


def create_refresh_token(
    sub: str,
    role: str = "viewer",
    expires_delta: Optional[int] = None,
    jti: Optional[str] = None,
) -> str:
    now = int(time.time())
    exp = now + (expires_delta or _get_refresh_expiration_days() * 86400)
    payload = TokenPayload(
        sub=sub,
        role=role,
        exp=exp,
        iat=now,
        jti=jti or str(uuid.uuid4())[:16],
        refresh=True,
    )
    header = {"alg": _get_algorithm(), "typ": "JWT"}
    payload_b64 = _base64url_encode(json.dumps(payload.__dict__).encode("utf-8"))
    header_b64 = _base64url_encode(json.dumps(header).encode("utf-8"))
    signature = _sign(f"{header_b64}.{payload_b64}", _get_secret_key(), _get_algorithm())
    return f"{header_b64}.{payload_b64}.{signature}"


def decode_token(token: str) -> TokenPayload:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid token format")
    header_b64, payload_b64, signature = parts
    expected_sig = _sign(f"{header_b64}.{payload_b64}", _get_secret_key(), _get_algorithm())
    if not hmac.compare_digest(signature, expected_sig):
        raise ValueError("Invalid token signature")
    payload_data = json.loads(_base64url_decode(payload_b64).decode("utf-8"))
    now = int(time.time())
    if payload_data.get("exp", 0) < now:
        raise ValueError("Token expired")
    return TokenPayload(**payload_data)


def verify_token(token: str) -> TokenPayload:
    return decode_token(token)
