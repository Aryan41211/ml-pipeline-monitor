"""Integration tests for the background worker's schedule claim and execution flow."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from ml_pipeline_monitor.database import (
    create_schedule,
    create_team,
    create_workspace,
    governance,
    initialize_db,
    initialize_governance_registry,
)
from ml_pipeline_monitor.services import worker


def _use_isolated_db() -> None:
    tmpdir = tempfile.mkdtemp(prefix="mlmonitor-worker-test-")
    os.environ["PIPELINE_DB"] = str(Path(tmpdir) / "test.db")


class TestScheduleClaim:
    def setup_method(self):
        _use_isolated_db()
        initialize_db()
        initialize_governance_registry()
        team_id = create_team("worker-team")
        self.workspace_id = create_workspace(workspace_name="worker-ws", team_id=team_id)

    def test_claim_due_schedule_and_advance_next_run(self):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        schedule_id = create_schedule(
            workspace_id=self.workspace_id,
            schedule_name="due-now",
            schedule_type="pipeline_run",
            cron_expression="*/5 * * * *",
            pipeline_dataset="iris",
            pipeline_model_type="Random Forest",
            next_run_at=past,
        )
        due = worker._claim_due_schedules()
        assert any(int(s["id"]) == schedule_id for s in due)
        updated = [s for s in governance.list_schedules() if s["id"] == schedule_id][0]
        assert updated["last_run_at"] is not None
        assert updated["next_run_at"] is not None
        next_run = worker._parse_dt(updated["next_run_at"])
        assert next_run is not None and next_run > datetime.now(timezone.utc)

    def test_not_due_when_next_run_in_future(self):
        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        schedule_id = create_schedule(
            workspace_id=self.workspace_id,
            schedule_name="not-due",
            schedule_type="pipeline_run",
            cron_expression="0 2 * * *",
            next_run_at=future,
        )
        due = worker._claim_due_schedules()
        assert all(int(s["id"]) != schedule_id for s in due)

    def test_disabled_schedule_not_claimed(self):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        schedule_id = create_schedule(
            workspace_id=self.workspace_id,
            schedule_name="disabled",
            schedule_type="pipeline_run",
            cron_expression="* * * * *",
            enabled=False,
            next_run_at=past,
        )
        due = worker._claim_due_schedules()
        assert all(int(s["id"]) != schedule_id for s in due)


class TestScheduleExecution:
    def setup_method(self):
        _use_isolated_db()
        initialize_db()
        initialize_governance_registry()
        team_id = create_team("worker-exec-team")
        self.workspace_id = create_workspace(workspace_name="worker-exec-ws", team_id=team_id)

    def test_run_schedule_success_records_run(self):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        schedule_id = create_schedule(
            workspace_id=self.workspace_id,
            schedule_name="exec-success",
            schedule_type="pipeline_run",
            cron_expression="* * * * *",
            pipeline_dataset="iris",
            pipeline_model_type="Random Forest",
            next_run_at=past,
        )
        with mock.patch.object(worker, "run_pipeline_and_persist") as run_mock:
            worker._run_schedule({"id": schedule_id, "schedule_name": "exec-success", "schedule_type": "pipeline_run"})
            run_mock.assert_called_once()
            run_mock.assert_called_with(dataset_label="iris", dataset_key="iris", model_type="Random Forest", task="classification", params={}, test_size=0.2, cv_folds=5, random_state=42)

        with governance._get_connection() as conn:
            rows = conn.execute(
                "SELECT status FROM schedule_runs WHERE schedule_id = ? ORDER BY id DESC LIMIT 1",
                (schedule_id,),
            ).fetchall()
        assert rows[0]["status"] == "success"

    def test_run_schedule_failure_records_failed(self):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        schedule_id = create_schedule(
            workspace_id=self.workspace_id,
            schedule_name="exec-fail",
            schedule_type="pipeline_run",
            cron_expression="* * * * *",
            next_run_at=past,
        )
        with mock.patch.object(worker, "run_pipeline_and_persist", side_effect=RuntimeError("boom")):
            worker._run_schedule({"id": schedule_id, "schedule_name": "exec-fail", "schedule_type": "pipeline_run"})

        with governance._get_connection() as conn:
            rows = conn.execute(
                "SELECT status FROM schedule_runs WHERE schedule_id = ? ORDER BY id DESC LIMIT 1",
                (schedule_id,),
            ).fetchall()
        assert rows[0]["status"] == "failed"
