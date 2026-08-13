"""Integration tests for governance, predictions, and lineage CRUD modules."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ml_pipeline_monitor.database import (
    governance,
    initialize_dataset_registry,
    initialize_db,
    initialize_governance_registry,
    initialize_prediction_registry,
    lineage,
    predictions,
)


def _use_isolated_db() -> None:
    """Point the persistence layer at a fresh temp DB for the current test.

    The persistence layer reads ``PIPELINE_DB`` on every backend connection,
    so a unique file per test keeps CRUD tests deterministic and repeatable
    (fixed request IDs / versions must not collide with previous runs).
    """
    tmpdir = tempfile.mkdtemp(prefix="mlmonitor-crud-test-")
    os.environ["PIPELINE_DB"] = str(Path(tmpdir) / "test.db")


class TestGovernance:
    def setup_method(self):
        _use_isolated_db()
        initialize_db()
        initialize_governance_registry()

    def test_create_team(self):
        team_id = governance.create_team("eng-team")
        assert team_id > 0

    def test_create_team_idempotent(self):
        id1 = governance.create_team("dup-team")
        id2 = governance.create_team("dup-team")
        assert id1 == id2

    def test_create_user(self):
        team_id = governance.create_team("user-team")
        user_id = governance.create_user(
            username="alice", password_hash="hash123", role="admin", team_id=team_id
        )
        assert user_id > 0

    def test_create_user_update(self):
        team_id = governance.create_team("upd-team")
        governance.create_user(username="bob", password_hash="old", role="viewer", team_id=team_id)
        user_id = governance.create_user(username="bob", password_hash="new", role="admin", team_id=team_id)
        assert user_id > 0

    def test_create_workspace(self):
        team_id = governance.create_team("ws-team")
        ws_id = governance.create_workspace(workspace_name="ws-prod", team_id=team_id)
        assert ws_id > 0

    def test_create_workspace_idempotent(self):
        team_id = governance.create_team("ws-dup-team")
        id1 = governance.create_workspace(workspace_name="ws-dup", team_id=team_id)
        id2 = governance.create_workspace(workspace_name="ws-dup", team_id=team_id)
        assert id1 == id2

    def test_log_user_activity(self):
        team_id = governance.create_team("act-team")
        user_id = governance.create_user(username="act-user", password_hash="h", role="viewer", team_id=team_id)
        governance.log_user_activity(user_id=user_id, workspace_id=None, action="login", metadata={"ip": "127.0.0.1"})

    def test_save_and_list_alert_events(self):
        governance.save_alert_event(
            workspace_id=None, alert_type="drift", severity="warning", message="Feature X drifted", metadata={"f": "X"}
        )
        events = governance.list_alert_events(limit=10)
        assert len(events) >= 1
        assert events[0]["alert_type"] == "drift"

    def test_list_alert_events_by_workspace(self):
        team_id = governance.create_team("alert-team")
        ws_id = governance.create_workspace(workspace_name="alert-ws", team_id=team_id)
        governance.save_alert_event(workspace_id=ws_id, alert_type="perf", severity="critical", message="down")
        events = governance.list_alert_events(workspace_id=ws_id, limit=10)
        assert len(events) >= 1

    def test_list_alert_events_invalid_limit(self):
        events = governance.list_alert_events(limit=0)
        assert isinstance(events, list)

    def test_create_and_list_schedules(self):
        team_id = governance.create_team("sched-team")
        ws_id = governance.create_workspace(workspace_name="sched-ws", team_id=team_id)
        sched_id = governance.create_schedule(
            workspace_id=ws_id,
            schedule_name="nightly-train",
            schedule_type="pipeline_run",
            cron_expression="0 2 * * *",
            pipeline_dataset="iris",
            pipeline_model_type="Random Forest",
        )
        assert sched_id > 0
        schedules = governance.list_schedules(workspace_id=ws_id)
        assert len(schedules) >= 1

    def test_list_schedules_invalid_limit(self):
        schedules = governance.list_schedules(limit=-1)
        assert isinstance(schedules, list)

    def test_record_schedule_run(self):
        team_id = governance.create_team("run-team")
        ws_id = governance.create_workspace(workspace_name="run-ws", team_id=team_id)
        sched_id = governance.create_schedule(
            workspace_id=ws_id, schedule_name="r1", schedule_type="pipeline_run", cron_expression="0 * * * *"
        )
        governance.record_schedule_run(schedule_id=sched_id, status="success")


class TestPredictions:
    def setup_method(self):
        _use_isolated_db()
        initialize_db()
        initialize_prediction_registry()

    def test_save_and_get_prediction_request(self):
        predictions.save_prediction_request(
            request_id="req-001",
            correlation_id="corr-001",
            model_id="m-001",
            dataset="iris",
            input_type="json",
            input_hash="abc123",
            num_predictions=5,
            status="success",
            duration_ms=12.3,
            error=None,
        )
        history = predictions.get_prediction_history(limit=10)
        assert len(history) >= 1

    def test_get_prediction_history_by_request_id(self):
        predictions.save_prediction_request(
            request_id="req-002", correlation_id=None, model_id="m-001",
            dataset=None, input_type="json", input_hash=None,
            num_predictions=3, status="success", duration_ms=5.0, error=None,
        )
        result = predictions.get_prediction_history_by_request_id("req-002")
        assert result is not None
        assert result["request_id"] == "req-002"

    def test_get_prediction_history_not_found(self):
        result = predictions.get_prediction_history_by_request_id("no-such-id")
        assert result is None

    def test_save_predictions_for_request(self):
        predictions.save_prediction_request(
            request_id="req-cov-003", correlation_id=None, model_id="m-cov-001",
            dataset=None, input_type="json", input_hash=None,
            num_predictions=3, status="success", duration_ms=5.0, error=None,
        )
        predictions.save_predictions_for_request(
            request_id="req-cov-003", predictions=["setosa", "versicolor", "virginica"],
            probabilities=[0.9, 0.7, 0.6],
        )
        result = predictions.get_prediction_history_by_request_id("req-cov-003")
        assert result is not None
        assert len(result["predictions"]) == 3

    def test_save_predictions_for_request_no_probs(self):
        predictions.save_prediction_request(
            request_id="req-cov-004", correlation_id=None, model_id="m-cov-001",
            dataset=None, input_type="json", input_hash=None,
            num_predictions=2, status="success", duration_ms=5.0, error=None,
        )
        predictions.save_predictions_for_request(
            request_id="req-cov-004", predictions=["cat", "dog"],
        )
        result = predictions.get_prediction_history_by_request_id("req-cov-004")
        assert result is not None
        assert len(result["predictions"]) == 2

    def test_get_prediction_history_invalid_limit(self):
        history = predictions.get_prediction_history(limit=0)
        assert isinstance(history, list)


class TestLineage:
    def setup_method(self):
        _use_isolated_db()
        initialize_db()
        initialize_dataset_registry()

    def test_create_dataset(self):
        ds_id = lineage.create_dataset("lds-001", "My Dataset")
        assert ds_id > 0

    def test_create_dataset_idempotent(self):
        id1 = lineage.create_dataset("lds-dup", "Dup Dataset")
        id2 = lineage.create_dataset("lds-dup", "Dup Dataset Updated")
        assert id1 == id2

    def test_create_dataset_version(self):
        lineage.create_dataset("lds-ver", "Versioned")
        ver_id = lineage.create_dataset_version("lds-ver", 1, "lhash-abc-unique", 1000, 10, "{}")
        assert ver_id > 0

    def test_save_schema_snapshot(self):
        lineage.create_dataset("lds-snap", "Snap")
        ver_id = lineage.create_dataset_version("lds-snap", 1, "lh1-unique", 100, 5, "{}")
        lineage.save_schema_snapshot(ver_id, "feature_a", "float64")
        lineage.save_schema_snapshot(ver_id, "feature_a", "int64")  # upsert

    def test_save_schema_change(self):
        lineage.create_dataset("lds-change", "Change")
        lineage.save_schema_change("lds-change", 1, 2, ["new_col"], ["old_col"], {"col_a": "int->float"})

    def test_get_dataset_versions(self):
        lineage.create_dataset("lds-get-ver", "GetVer")
        lineage.create_dataset_version("lds-get-ver", 1, "lh1-getver", 100, 5, "{}")
        versions = lineage.get_dataset_versions("lds-get-ver")
        assert len(versions) >= 1

    def test_get_schema_changes(self):
        lineage.create_dataset("lds-get-chg", "GetChg")
        lineage.save_schema_change("lds-get-chg", 1, 2, ["a"], ["b"], {"c": "x"})
        changes = lineage.get_schema_changes("lds-get-chg")
        assert len(changes) >= 1
        assert isinstance(changes[0]["added_columns"], list)

    def test_create_lineage_edge(self):
        lineage.create_dataset("lds-from", "From")
        lineage.create_dataset("lds-to", "To")
        edge_id = lineage.create_lineage_edge(
            edge_type="transform",
            from_dataset_id="lds-from", from_version=1,
            to_dataset_id="lds-to", to_version=1,
            note="cleaned",
        )
        assert edge_id > 0

    def test_get_lineage_edges_all(self):
        lineage.create_dataset("lds-e1", "E1")
        lineage.create_dataset("lds-e2", "E2")
        lineage.create_lineage_edge(edge_type="transform", from_dataset_id="lds-e1", to_dataset_id="lds-e2")
        edges = lineage.get_lineage_edges()
        assert len(edges) >= 1

    def test_get_lineage_edges_by_type(self):
        lineage.create_dataset("lds-f1", "F1")
        lineage.create_dataset("lds-f2", "F2")
        lineage.create_lineage_edge(edge_type="derived", from_dataset_id="lds-f1", to_dataset_id="lds-f2")
        edges = lineage.get_lineage_edges(edge_type="derived")
        assert len(edges) >= 1
        edges_other = lineage.get_lineage_edges(edge_type="nonexistent")
        assert len(edges_other) == 0

    def test_create_lineage_edge_optional_fields(self):
        edge_id = lineage.create_lineage_edge(
            edge_type="train",
            from_run_id="run-001",
            to_model_id="model-001",
        )
        assert edge_id > 0
