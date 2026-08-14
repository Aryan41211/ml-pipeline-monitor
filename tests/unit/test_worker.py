"""Unit tests for the background worker's schedule polling and cron parsing."""

from datetime import datetime, timedelta, timezone

import pytest

from ml_pipeline_monitor.services.worker import (
    _build_task_config,
    _next_run_from_cron,
    _parse_cron_field,
    _parse_dt,
)


class TestParseCronField:
    def test_wildcard(self):
        assert _parse_cron_field("*", 0, 59) == set(range(0, 60))

    def test_step(self):
        assert _parse_cron_field("*/5", 0, 59) == {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55}

    def test_range(self):
        assert _parse_cron_field("10-20", 0, 59) == set(range(10, 21))

    def test_range_with_step(self):
        assert _parse_cron_field("0-30/10", 0, 59) == {0, 10, 20, 30}

    def test_comma_list(self):
        assert _parse_cron_field("1,15,30", 0, 59) == {1, 15, 30}

    def test_out_of_bounds_filtered(self):
        assert _parse_cron_field("0-70", 0, 59) == set(range(0, 60))


class TestNextRunFromCron:
    def test_every_minute(self):
        base = datetime(2026, 8, 14, 10, 0, 0, tzinfo=timezone.utc)
        assert _next_run_from_cron("* * * * *", base) == base + timedelta(minutes=1)

    def test_specific_minute(self):
        base = datetime(2026, 8, 14, 10, 0, 0, tzinfo=timezone.utc)
        assert _next_run_from_cron("30 * * * *", base) == datetime(2026, 8, 14, 10, 30, tzinfo=timezone.utc)

    def test_nightly_at_0200(self):
        base = datetime(2026, 8, 14, 9, 0, 0, tzinfo=timezone.utc)
        assert _next_run_from_cron("0 2 * * *", base) == datetime(2026, 8, 15, 2, 0, tzinfo=timezone.utc)

    def test_hourly(self):
        base = datetime(2026, 8, 14, 9, 30, 0, tzinfo=timezone.utc)
        assert _next_run_from_cron("0 * * * *", base) == datetime(2026, 8, 14, 10, 0, tzinfo=timezone.utc)

    def test_weekday_sunday_cron_zero(self):
        # 2026-08-16 is a Sunday (Python weekday 6). Cron 0 == Sunday.
        base = datetime(2026, 8, 14, 10, 0, 0, tzinfo=timezone.utc)
        assert _next_run_from_cron("0 0 * * 0", base) == datetime(2026, 8, 16, 0, 0, tzinfo=timezone.utc)

    def test_weekday_sunday_cron_seven(self):
        base = datetime(2026, 8, 14, 10, 0, 0, tzinfo=timezone.utc)
        assert _next_run_from_cron("0 0 * * 7", base) == datetime(2026, 8, 16, 0, 0, tzinfo=timezone.utc)

    def test_invalid_field_count_raises(self):
        base = datetime(2026, 8, 14, 10, 0, 0, tzinfo=timezone.utc)
        with pytest.raises(ValueError):
            _next_run_from_cron("0 2 * *", base)


class TestBuildTaskConfig:
    def test_uses_schedule_fields(self):
        schedule = {
            "schedule_type": "pipeline_run",
            "pipeline_dataset": "iris",
            "pipeline_model_type": "XGBoost",
        }
        config = _build_task_config(schedule)
        assert config["type"] == "pipeline_run"
        assert config["params"]["dataset"] == "iris"
        assert config["params"]["model_type"] == "XGBoost"

    def test_defaults_when_missing(self):
        config = _build_task_config({"schedule_type": "pipeline_run"})
        assert config["params"]["dataset"] == "iris"
        assert config["params"]["model_type"] == "Random Forest"


class TestParseDt:
    def test_iso_with_offset(self):
        dt = _parse_dt("2026-08-14T10:00:00+00:00")
        assert dt == datetime(2026, 8, 14, 10, 0, 0, tzinfo=timezone.utc)

    def test_iso_with_z(self):
        dt = _parse_dt("2026-08-14T10:00:00Z")
        assert dt == datetime(2026, 8, 14, 10, 0, 0, tzinfo=timezone.utc)

    def test_invalid_returns_none(self):
        assert _parse_dt("not-a-date") is None
