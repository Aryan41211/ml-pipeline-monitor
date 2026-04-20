from pathlib import Path

from src import alerts


def test_emit_email_alert_writes_sink(monkeypatch, tmp_path):
    sink_rel = "logs/test_alerts_sink.log"
    monkeypatch.setattr(alerts, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(alerts, "load_config", lambda: {"alerting": {"email_simulation_file": sink_rel}})

    out = alerts.emit_email_alert(
        "warning",
        subject="Drift Warning",
        message="Drift ratio exceeded threshold",
        metadata={"dataset": "Iris", "drift_ratio": 0.31},
    )

    sink = tmp_path / sink_rel
    assert sink.exists()
    content = sink.read_text(encoding="utf-8")
    assert "Drift Warning" in content
    assert "Iris" in content
    assert out["severity"] == "warning"
    assert Path(out["sink"]).name == "test_alerts_sink.log"
