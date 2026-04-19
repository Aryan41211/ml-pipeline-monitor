"""Authentication and role helpers for Streamlit pages."""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import streamlit as st

from src.config_loader import load_config


ROLE_ORDER = {"viewer": 0, "operator": 1, "admin": 2}


def _auth_cfg() -> Dict[str, object]:
    return load_config().get("auth", {})


def _env_value(*keys: str) -> str:
    for key in keys:
        val = os.getenv(key)
        if val is not None and str(val).strip() != "":
            return str(val)
    return ""


def is_auth_enabled() -> bool:
    """Return whether authentication is enabled.

    Environment variable `MLMONITOR_AUTH_ENABLED` overrides config.
    """
    env_val = os.getenv("MLMONITOR_AUTH_ENABLED")
    if env_val is not None:
        return str(env_val).strip().lower() not in {"0", "false", "no", "off"}
    return bool(_auth_cfg().get("enabled", True))


def _users_from_env_json() -> Dict[str, Dict[str, str]]:
    """Load users from AUTH_USERS_JSON or MLMONITOR_AUTH_USERS_JSON."""
    raw = _env_value("AUTH_USERS_JSON", "MLMONITOR_AUTH_USERS_JSON").strip()
    if not raw:
        return {}

    try:
        payload = json.loads(raw)
    except Exception:
        return {}

    users: Dict[str, Dict[str, str]] = {}
    if isinstance(payload, dict):
        for username, meta in payload.items():
            if not isinstance(meta, dict):
                continue
            users[str(username)] = {
                "password": str(meta.get("password", "")),
                "role": str(meta.get("role", "viewer")).lower(),
            }
    elif isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            username = str(item.get("username", "")).strip()
            if not username:
                continue
            users[username] = {
                "password": str(item.get("password", "")),
                "role": str(item.get("role", "viewer")).lower(),
            }

    users = {
        u: meta
        for u, meta in users.items()
        if str(meta.get("password", "")).strip() and str(meta.get("role", "")).strip()
    }
    return users


def _credentials() -> Dict[str, Dict[str, str]]:
    """Resolve credentials from environment only."""
    users = _users_from_env_json()
    if users:
        return users

    username = _env_value("AUTH_USERNAME", "MLMONITOR_AUTH_USERNAME").strip()
    password = _env_value("AUTH_PASSWORD", "MLMONITOR_AUTH_PASSWORD")
    role = _env_value("AUTH_ROLE", "MLMONITOR_AUTH_ROLE").strip().lower() or "admin"
    if username and password:
        return {
            username: {
                "password": password,
                "role": role,
            }
        }
    return {}


def _auth_env_help() -> str:
    return (
        "Authentication is enabled but no credentials are configured. "
        "Set AUTH_USERNAME and AUTH_PASSWORD (optional AUTH_ROLE), "
        "or set AUTH_USERS_JSON."
    )


def is_authenticated() -> bool:
    if not is_auth_enabled():
        return True
    return bool(st.session_state.get("authenticated", False))


def current_user() -> str:
    if not is_auth_enabled():
        return "system"
    return str(st.session_state.get("auth_user", ""))


def current_role() -> str:
    if not is_auth_enabled():
        return "admin"
    role = str(st.session_state.get("auth_role", "viewer")).lower().strip()
    return role if role in ROLE_ORDER else "viewer"


def has_role(required_role: str) -> bool:
    required = str(required_role or "viewer").lower().strip()
    required = required if required in ROLE_ORDER else "viewer"
    return ROLE_ORDER.get(current_role(), 0) >= ROLE_ORDER.get(required, 0)


def can_run_pipeline() -> bool:
    return has_role("operator")


def can_administer() -> bool:
    return has_role("admin")


def require_role(role: str, action_name: str) -> bool:
    """Enforce role requirement for an action with consistent UX feedback."""
    if not is_authenticated():
        st.error(f"{action_name} requires login.")
        return False

    if not has_role(role):
        st.error(f"{action_name} requires {role} role. Current role: {current_role()}.")
        return False

    return True


def _check_login(username: str, password: str) -> Tuple[bool, str]:
    creds = _credentials()
    if not creds:
        return False, _auth_env_help()

    user = (username or "").strip()

    if user in creds and creds[user].get("password") == password:
        return True, ""

    lowered = {k.lower(): k for k in creds.keys()}
    resolved = lowered.get(user.lower())
    if resolved and creds[resolved].get("password") == password:
        return True, ""

    return False, "Invalid username or password"


def _resolve_user(username: str) -> str:
    creds = _credentials()
    user = (username or "").strip()
    if user in creds:
        return user
    lowered = {k.lower(): k for k in creds.keys()}
    return lowered.get(user.lower(), user)


def render_auth_controls() -> bool:
    """Render login/logout controls in current Streamlit container."""
    st.markdown("### Access")

    if not is_auth_enabled():
        st.info("Authentication disabled by config (auth.enabled=false).")
        st.caption("Role: admin")
        return True

    if is_authenticated():
        st.success(f"Signed in as {current_user()} ({current_role()})")
        if st.button("Logout", width="stretch"):
            st.session_state["authenticated"] = False
            st.session_state["auth_user"] = ""
            st.session_state["auth_role"] = "viewer"
            st.rerun()
        return True

    creds = _credentials()
    if not creds:
        st.error(_auth_env_help())
        st.caption("Example: AUTH_USERNAME=admin, AUTH_PASSWORD=secure_pass, AUTH_ROLE=admin")
        return False

    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login", type="primary", width="stretch"):
        ok, err = _check_login(username=username, password=password)
        if ok:
            resolved_user = _resolve_user(username)
            role = creds.get(resolved_user, {}).get("role", "viewer")
            st.session_state["authenticated"] = True
            st.session_state["auth_user"] = resolved_user
            st.session_state["auth_role"] = str(role).lower()
            st.rerun()
        st.error(err)

    st.caption("Login required for pipeline execution and admin operations.")
    st.caption("Credentials are environment-only for security hardening.")
    return False
