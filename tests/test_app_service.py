from services import app_service


def test_initialize_application_calls_db_init(monkeypatch):
    called = {"ok": False}

    def _fake_init():
        called["ok"] = True

    monkeypatch.setattr(app_service, "initialize_db", _fake_init)
    app_service.initialize_application()
    assert called["ok"] is True


def test_get_dashboard_snapshot_smoke(monkeypatch):
    monkeypatch.setattr(app_service, "list_experiments", lambda limit=200: [{"run_id": "r1"}])
    monkeypatch.setattr(app_service, "list_models", lambda limit=100: [{"model_id": "m1"}])
    monkeypatch.setattr(app_service, "get_system_metrics", lambda: {"cpu_percent": 10.0})

    out = app_service.get_dashboard_snapshot(limit=5)
    assert "experiments" in out
    assert "models" in out
    assert "system" in out
    assert out["experiments"][0]["run_id"] == "r1"


def test_get_ui_settings_from_config(monkeypatch):
    monkeypatch.setattr(app_service, "load_config", lambda: {"ui": {"max_experiments_displayed": 123}})
    ui = app_service.get_ui_settings()
    assert ui["max_experiments_displayed"] == 123
