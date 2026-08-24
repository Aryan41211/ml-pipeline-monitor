"""Tests for the governance read service (alert history and lineage)."""

from __future__ import annotations

import pytest

from ml_pipeline_monitor.services import governance_service as gs


class TestCoerceMetadata:
    def test_parses_json_string(self):
        assert gs._coerce_metadata('{"dataset": "iris"}') == {"dataset": "iris"}

    def test_passes_through_dict(self):
        assert gs._coerce_metadata({"a": 1}) == {"a": 1}

    def test_empty_and_none_become_empty_dict(self):
        assert gs._coerce_metadata(None) == {}
        assert gs._coerce_metadata("") == {}

    def test_malformed_json_does_not_raise(self):
        """A bad row must not take down the whole governance page."""
        assert gs._coerce_metadata("{not json") == {}

    def test_non_object_json_becomes_empty_dict(self):
        assert gs._coerce_metadata("[1, 2, 3]") == {}


class TestListAlerts:
    def test_attaches_parsed_metadata(self, monkeypatch):
        monkeypatch.setattr(
            gs,
            "list_alert_events",
            lambda limit: [{"severity": "critical", "metadata_json": '{"dataset": "iris"}'}],
        )
        rows = gs.list_alerts(limit=10)
        assert rows[0]["metadata"] == {"dataset": "iris"}

    @pytest.mark.parametrize("limit", [0, -1, 1001])
    def test_rejects_out_of_range_limit(self, limit):
        with pytest.raises(ValueError, match="limit must be between"):
            gs.list_alerts(limit=limit)


class TestAlertSummary:
    def test_counts_by_severity(self):
        alerts = [
            {"severity": "critical"},
            {"severity": "critical"},
            {"severity": "warning"},
            {"severity": "info"},
        ]
        assert gs.alert_summary(alerts) == {"critical": 2, "warning": 1, "info": 1}

    def test_empty_input_gives_zeroes(self):
        assert gs.alert_summary([]) == {"critical": 0, "warning": 0, "info": 0}

    def test_unknown_severity_is_counted_not_dropped(self):
        assert gs.alert_summary([{"severity": "fatal"}])["fatal"] == 1

    def test_severity_is_case_insensitive(self):
        assert gs.alert_summary([{"severity": "CRITICAL"}])["critical"] == 1


class TestLineageReads:
    def test_lineage_edges_proxy(self, monkeypatch):
        monkeypatch.setattr(gs, "get_lineage_edges", lambda limit: [{"edge_type": "trained_on"}])
        assert gs.list_lineage_edges(limit=5)[0]["edge_type"] == "trained_on"

    @pytest.mark.parametrize("limit", [0, 1001])
    def test_lineage_rejects_bad_limit(self, limit):
        with pytest.raises(ValueError):
            gs.list_lineage_edges(limit=limit)

    def test_dataset_versions_requires_dataset_id(self):
        with pytest.raises(ValueError, match="dataset_id is required"):
            gs.list_dataset_versions("   ")

    def test_dataset_versions_trims_and_proxies(self, monkeypatch):
        seen = {}

        def _fake(dataset_id, limit):
            seen["dataset_id"] = dataset_id
            return [{"version": 1}]

        monkeypatch.setattr(gs, "get_dataset_versions", _fake)
        assert gs.list_dataset_versions("  iris  ")[0]["version"] == 1
        assert seen["dataset_id"] == "iris"

    def test_schema_changes_requires_dataset_id(self):
        with pytest.raises(ValueError, match="dataset_id is required"):
            gs.list_schema_changes("")

    def test_schema_changes_proxies(self, monkeypatch):
        monkeypatch.setattr(gs, "get_schema_changes", lambda dataset_id, limit: [{"from_version": 1}])
        assert gs.list_schema_changes("iris")[0]["from_version"] == 1

    @pytest.mark.parametrize("limit", [0, 501])
    def test_schema_changes_rejects_bad_limit(self, limit):
        with pytest.raises(ValueError):
            gs.list_schema_changes("iris", limit=limit)
