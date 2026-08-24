"""Concurrency tests for scheduled-task claiming.

Before claim_schedule(), every worker replica read the same due row and ran
it, producing duplicate training runs and duplicate registered models. These
tests pin down that exactly one claimant wins.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from ml_pipeline_monitor.database import (
    claim_schedule,
    create_schedule,
    create_team,
    create_workspace,
    initialize_governance_registry,
    list_schedules,
)


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    """Point the whole DB layer at a throwaway SQLite file."""
    monkeypatch.setenv("PIPELINE_DB", str(tmp_path / "claims.db"))
    monkeypatch.setenv("MLMONITOR_DB_BACKEND", "sqlite")
    initialize_governance_registry()
    team_id = create_team("race-team")
    return create_workspace(workspace_name="race-ws", team_id=team_id)


def _make_schedule(workspace_id: int, name: str, next_run_at: str | None) -> int:
    schedule_id = create_schedule(
        workspace_id=workspace_id,
        schedule_name=name,
        schedule_type="pipeline_run",
        cron_expression="* * * * *",
        next_run_at=next_run_at,
    )
    return int(schedule_id)


def test_only_one_claimant_wins(workspace):
    """Two workers racing for the same schedule: exactly one may proceed."""
    past = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    schedule_id = _make_schedule(workspace, "contended", past)

    future = (datetime.now(UTC) + timedelta(minutes=1)).isoformat()
    now = datetime.now(UTC).isoformat()

    results = [
        claim_schedule(
            schedule_id=schedule_id,
            expected_next_run_at=past,
            next_run_at=future,
            last_run_at=now,
        )
        for _ in range(2)
    ]

    assert results.count(True) == 1, f"expected exactly one winner, got {results}"


def test_concurrent_threads_produce_one_winner(workspace):
    """The same guarantee under real thread contention."""
    past = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    schedule_id = _make_schedule(workspace, "threaded", past)

    future = (datetime.now(UTC) + timedelta(minutes=1)).isoformat()
    now = datetime.now(UTC).isoformat()

    def _claim(_: int) -> bool:
        return claim_schedule(
            schedule_id=schedule_id,
            expected_next_run_at=past,
            next_run_at=future,
            last_run_at=now,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(_claim, range(8)))

    assert outcomes.count(True) == 1, f"expected exactly one winner, got {outcomes.count(True)}"


def test_claim_advances_next_run_at(workspace):
    """A won claim must move the schedule forward so it is not immediately due again."""
    past = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    schedule_id = _make_schedule(workspace, "advances", past)
    future = (datetime.now(UTC) + timedelta(minutes=1)).isoformat()

    assert claim_schedule(
        schedule_id=schedule_id,
        expected_next_run_at=past,
        next_run_at=future,
        last_run_at=datetime.now(UTC).isoformat(),
    )

    row = next(s for s in list_schedules(limit=100) if int(s["id"]) == schedule_id)
    assert row["next_run_at"] == future
    assert row["last_run_at"] is not None


def test_stale_expectation_loses(workspace):
    """A worker holding an out-of-date next_run_at must not win the claim."""
    past = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    schedule_id = _make_schedule(workspace, "stale", past)
    future = (datetime.now(UTC) + timedelta(minutes=1)).isoformat()

    assert claim_schedule(
        schedule_id=schedule_id,
        expected_next_run_at=past,
        next_run_at=future,
        last_run_at=datetime.now(UTC).isoformat(),
    )
    # Second worker still believes next_run_at is the original value.
    assert not claim_schedule(
        schedule_id=schedule_id,
        expected_next_run_at=past,
        next_run_at=future,
        last_run_at=datetime.now(UTC).isoformat(),
    )


def test_never_scheduled_row_can_be_claimed_once(workspace):
    """A schedule with a NULL next_run_at is claimable exactly once."""
    schedule_id = _make_schedule(workspace, "fresh", None)
    future = (datetime.now(UTC) + timedelta(minutes=1)).isoformat()
    now = datetime.now(UTC).isoformat()

    first = claim_schedule(schedule_id=schedule_id, expected_next_run_at=None, next_run_at=future, last_run_at=now)
    second = claim_schedule(schedule_id=schedule_id, expected_next_run_at=None, next_run_at=future, last_run_at=now)
    assert first is True
    assert second is False
