import numpy as np
import pytest

from ml_pipeline_monitor.services import model_service


def test_list_models_filters_by_dataset(monkeypatch):
    monkeypatch.setattr(
        model_service,
        "get_models",
        lambda limit=100: [
            {"model_id": "m1", "dataset": "iris"},
            {"model_id": "m2", "dataset": "wine"},
        ],
    )

    rows = model_service.list_models(dataset="iris")
    assert len(rows) == 1
    assert rows[0]["model_id"] == "m1"


def test_to_dataframe_valid_payloads():
    df_dict = model_service._to_dataframe({"a": 1, "b": 2})
    assert list(df_dict.columns) == ["a", "b"]

    df_list_dict = model_service._to_dataframe([{"a": 1}, {"a": 2}])
    assert len(df_list_dict) == 2

    df_vector = model_service._to_dataframe([1.0, 2.0, 3.0])
    assert df_vector.shape == (1, 3)

    df_matrix = model_service._to_dataframe([[1.0, 2.0], [3.0, 4.0]])
    assert df_matrix.shape == (2, 2)


def test_to_dataframe_invalid_payload_raises():
    with pytest.raises(ValueError):
        model_service._to_dataframe("bad payload")


def test_get_rollback_hint_handles_short_history(monkeypatch):
    monkeypatch.setattr(
        model_service,
        "get_recent_production_models",
        lambda dataset, limit=2: [{"model_id": "m1"}],
    )
    hint = model_service.get_rollback_hint("iris")
    assert hint["current_production"]["model_id"] == "m1"
    assert hint["previous_production"] is None


def test_revert_to_previous_production_updates_stage(monkeypatch):
    monkeypatch.setattr(
        model_service,
        "get_recent_production_models",
        lambda dataset, limit=2: [{"model_id": "new"}, {"model_id": "old"}],
    )

    calls = []

    def _fake_update(model_id: str, stage: str):
        calls.append((model_id, stage))

    monkeypatch.setattr(model_service, "update_model_stage", _fake_update)

    reverted = model_service.revert_to_previous_production("iris")
    assert reverted["model_id"] == "old"
    assert calls == [("old", "production")]


def test_revert_to_previous_production_no_previous(monkeypatch):
    monkeypatch.setattr(model_service, "get_recent_production_models", lambda dataset, limit=2: [])
    with pytest.raises(ValueError):
        model_service.revert_to_previous_production("iris")


def test_predict_from_payload_with_scaler(monkeypatch):
    class FakeScaler:
        def transform(self, x):
            return x

    class FakeModel:
        def predict(self, x):
            return np.array([1] * len(x))

        def predict_proba(self, x):
            return np.array([[0.2, 0.8]] * len(x))

    monkeypatch.setattr(
        model_service,
        "load_production_artifacts",
        lambda dataset=None: (
            FakeModel(),
            FakeScaler(),
            {"model_id": "m1", "dataset": "iris", "version": 3, "stage": "production"},
        ),
    )

    out = model_service.predict_from_payload([{"a": 1.0}, {"a": 2.0}], dataset="iris")
    assert out["model_id"] == "m1"
    assert out["dataset"] == "iris"
    assert out["predictions"] == [1, 1]
    assert "probabilities" in out


def test_load_production_artifacts_missing_model(monkeypatch):
    monkeypatch.setattr(model_service, "get_latest_production_model", lambda dataset=None: None)
    with pytest.raises(ValueError):
        model_service.load_production_artifacts()


def test_load_production_artifacts_missing_artifact_path(monkeypatch):
    monkeypatch.setattr(
        model_service,
        "get_latest_production_model",
        lambda dataset=None: {"model_id": "m1", "dataset": "iris", "artifact_path": ""},
    )
    with pytest.raises(ValueError):
        model_service.load_production_artifacts()


class TestPredictionHistory:
    """Covers the serving-history read path used by the Serving page."""

    def test_list_proxies_to_database(self, monkeypatch):
        monkeypatch.setattr(
            model_service, "get_prediction_history", lambda limit: [{"request_id": "r1", "status": "success"}]
        )
        assert model_service.list_prediction_history(limit=10)[0]["request_id"] == "r1"

    @pytest.mark.parametrize("limit", [0, -5, 1001])
    def test_list_rejects_out_of_range_limit(self, limit):
        with pytest.raises(ValueError, match="limit must be between"):
            model_service.list_prediction_history(limit=limit)

    def test_detail_requires_request_id(self):
        with pytest.raises(ValueError, match="request_id is required"):
            model_service.get_prediction_detail("  ")

    def test_detail_trims_and_proxies(self, monkeypatch):
        seen = {}

        def _fake(request_id):
            seen["request_id"] = request_id
            return {"request_id": request_id, "predictions": []}

        monkeypatch.setattr(model_service, "get_prediction_history_by_request_id", _fake)
        assert model_service.get_prediction_detail("  abc  ")["request_id"] == "abc"
        assert seen["request_id"] == "abc"


class TestPredictionStats:
    def test_empty_history_is_all_zeroes(self):
        stats = model_service.prediction_stats([])
        assert stats == {
            "total": 0,
            "success": 0,
            "failed": 0,
            "success_rate": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
        }

    def test_counts_successes_and_failures(self):
        history = [
            {"status": "success", "duration_ms": 10.0},
            {"status": "success", "duration_ms": 20.0},
            {"status": "failed", "duration_ms": 30.0},
            {"status": "failed", "duration_ms": None},
        ]
        stats = model_service.prediction_stats(history)
        assert stats["total"] == 4
        assert stats["success"] == 2
        assert stats["failed"] == 2
        assert stats["success_rate"] == 50.0

    def test_percentiles_ignore_missing_durations(self):
        """A row with no duration must not be counted as zero milliseconds."""
        history = [
            {"status": "success", "duration_ms": 100.0},
            {"status": "success", "duration_ms": None},
        ]
        stats = model_service.prediction_stats(history)
        assert stats["p50_ms"] == 100.0
        assert stats["p95_ms"] == 100.0

    def test_p95_tracks_the_slow_tail(self):
        history = [{"status": "success", "duration_ms": float(i)} for i in range(1, 101)]
        stats = model_service.prediction_stats(history)
        assert stats["p50_ms"] < stats["p95_ms"]
        assert stats["p95_ms"] >= 95.0
