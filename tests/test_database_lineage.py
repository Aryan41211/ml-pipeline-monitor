from src import database


def _seed_experiment(run_id: str, dataset: str):
    database.save_experiment(
        run_id=run_id,
        name=f"{dataset} run",
        dataset=dataset,
        model_type="Random Forest",
        task="classification",
        params={"n_estimators": 10},
        metrics={"accuracy": 0.9},
        duration=1.2,
    )


def test_save_model_tracks_version_and_lineage(monkeypatch, tmp_path):
    db_file = tmp_path / "lineage.db"
    monkeypatch.setenv("PIPELINE_DB", str(db_file))

    database.initialize_db()

    _seed_experiment("RUN001", "Iris Species")
    first = database.save_model(
        model_id="RUN001",
        run_id="RUN001",
        name="Random Forest",
        dataset="Iris Species",
        model_type="Random Forest",
        task="classification",
        metrics={"accuracy": 0.91},
        artifact_path="artifacts/models/RUN001_model.joblib",
        params={"n_estimators": 100},
        experiment_id="RUN001",
    )

    _seed_experiment("RUN002", "Iris Species")
    second = database.save_model(
        model_id="RUN002",
        run_id="RUN002",
        name="Random Forest",
        dataset="Iris Species",
        model_type="Random Forest",
        task="classification",
        metrics={"accuracy": 0.93},
        artifact_path="artifacts/models/RUN002_model.joblib",
        params={"n_estimators": 200},
        experiment_id="RUN002",
    )

    assert first["version"] == 1
    assert second["version"] == 2
    assert second["parent_model_id"] == "RUN001"

    lineage = database.get_model_lineage(limit=10, dataset="Iris Species")
    assert len(lineage) >= 2
    latest = lineage[0]
    assert latest["experiment_id"] in {"RUN001", "RUN002"}


def test_update_model_stage_records_events_and_single_production(monkeypatch, tmp_path):
    db_file = tmp_path / "stages.db"
    monkeypatch.setenv("PIPELINE_DB", str(db_file))

    database.initialize_db()

    _seed_experiment("M1", "Wine Recognition")
    database.save_model(
        model_id="M1",
        run_id="M1",
        name="Gradient Boosting",
        dataset="Wine Recognition",
        model_type="Gradient Boosting",
        task="classification",
        metrics={"accuracy": 0.88},
        artifact_path="artifacts/models/M1_model.joblib",
        params={},
        experiment_id="M1",
    )

    _seed_experiment("M2", "Wine Recognition")
    database.save_model(
        model_id="M2",
        run_id="M2",
        name="Gradient Boosting",
        dataset="Wine Recognition",
        model_type="Gradient Boosting",
        task="classification",
        metrics={"accuracy": 0.90},
        artifact_path="artifacts/models/M2_model.joblib",
        params={},
        experiment_id="M2",
    )

    database.update_model_stage("M1", "production")
    database.update_model_stage("M2", "production")

    latest_prod = database.get_latest_production_model(dataset="Wine Recognition")
    assert latest_prod is not None
    assert latest_prod["model_id"] == "M2"

    models = database.get_models(limit=10)
    model_stages = {m["model_id"]: m["stage"] for m in models if m["dataset"] == "Wine Recognition"}
    assert model_stages["M2"] == "production"
    assert model_stages["M1"] == "staging"

    events_m1 = database.get_model_stage_events("M1", limit=20)
    events_m2 = database.get_model_stage_events("M2", limit=20)
    assert len(events_m1) >= 2
    assert len(events_m2) >= 2
