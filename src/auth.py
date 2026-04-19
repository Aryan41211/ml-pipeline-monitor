"""Simple config-driven authentication for Streamlit pages."""

from __future__ import annotations

from typing import Dict, Tuple

import streamlit as st

from src.config_loader import load_config


def _credentials() -> Dict[str, str]:
    cfg = load_config().get("auth", {})
    users = cfg.get("users")
    if isinstance(users, dict) and users:
        return {str(k): str(v) for k, v in users.items()}

    username = str(cfg.get("username", "admin"))
    password = str(cfg.get("password", "admin123"))
    return {username: password}


def is_authenticated() -> bool:
    return bool(st.session_state.get("authenticated", False))


def current_user() -> str:
    return str(st.session_state.get("auth_user", ""))


def _check_login(username: str, password: str) -> Tuple[bool, str]:
    creds = _credentials()
    if username in creds and creds[username] == password:
        return True, ""
    return False, "Invalid username or password"


def render_auth_controls() -> bool:
    """Render login/logout controls in current Streamlit container."""
    st.markdown("### Access")

    if is_authenticated():
        st.success(f"Signed in as {current_user()}")
        if st.button("Logout", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state["auth_user"] = ""
            st.rerun()
        return True

    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login", type="primary", use_container_width=True):
        ok, err = _check_login(username=username, password=password)
        if ok:
            st.session_state["authenticated"] = True
            st.session_state["auth_user"] = username
            st.rerun()
        st.error(err)

    st.caption("Login required for pipeline execution and model promotion.")
    return False
