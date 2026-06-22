"""Additional tests targeting coverage gaps."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.system_monitor import _get_cpu_temperature_c, get_host_process_count


def test_get_cpu_temperature_empty_temps():
    with patch("psutil.sensors_temperatures", return_value={}):
        assert _get_cpu_temperature_c() is None


def test_get_cpu_temperature_no_current_values():
    fake = {"coretemp": [MagicMock(current=None)]}
    with patch("psutil.sensors_temperatures", return_value=fake):
        assert _get_cpu_temperature_c() is None


def test_get_host_process_count_exception():
    with patch("psutil.pids", side_effect=Exception("access denied")):
        assert get_host_process_count() == 0
