"""Common database interfaces for pluggable backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DatabaseConnection(Protocol):
    """Minimal connection contract used by persistence services."""

    def execute(self, query: str, params: Any = None):
        ...

    def executescript(self, script: str) -> None:
        ...

    def commit(self) -> None:
        ...

    def rollback(self) -> None:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class DatabaseBackend(Protocol):
    """Backend contract for sqlite/postgres implementations."""

    name: str

    def connect(self) -> DatabaseConnection:
        ...
