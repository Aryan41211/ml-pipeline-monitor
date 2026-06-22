"""Final coverage-targeted tests for system monitor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.system_monitor import _get_cpu_temperature_c, get_host_process_count


def test_get_cpu_temperature_empty_result():
    fake_psutil = MagicMock()
    fake_psutil.sensors_temperatures.return_value = {}
    with patch.dict("sys.modules", {"psutil": fake_psutil}):
        import importlib
        import src.system_monitor as sm
        importlib.reload(sm)
        assert sm._get_cpu_temperature_c() is None


def test_get_cpu_temperature_no_current():
    fake_entry = MagicMock()
    fake_entry.current = None
    fake_psutil = MagicMock()
    fake_psutil.sensors_temperatures.return_value = {"coretemp": [fake_entry]}
    with patch.dict("sys.modules", {"psutil": fake_psutil}):
        import importlib
        import src.system_monitor as sm
        importlib.reload(sm)
        assert sm._get_cpu_temperature_c() is None
