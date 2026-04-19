import numpy as np
import pandas as pd
import pytest

from services import model_service


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
