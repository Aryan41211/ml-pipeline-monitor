"""Tests for system monitor and metrics modules."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.system_monitor import (
    _get_cpu_temperature_c,
    get_host_process_count,
    get_process_metrics,
    get_system_metrics,
)


def test_get_cpu_temperature_unavailable():
    with patch("psutil.sensors_temperatures", side_effect=Exception("no sensors")):
        assert _get_cpu_temperature_c() is None


def test_get_cpu_temperature_empty():
    with patch("psutil.sensors_temperatures", return_value={}):
        assert _get_cpu_temperature_c() is None


def test_get_cpu_temperature_available():
    fake_temps = {"coretemp": [MagicMock(current=55.0)]}
    with patch("psutil.sensors_temperatures", return_value=fake_temps):
        assert _get_cpu_temperature_c() == 55.0


def test_get_host_process_count():
    with patch("psutil.pids", return_value=[1, 2, 3]):
        assert get_host_process_count() == 3


def test_get_process_metrics():
    mock_proc = MagicMock()
    mock_proc.pid = 1234
    mock_proc.memory_info.return_value = MagicMock(rss=1024 * 1024 * 100, vms=1024 * 1024 * 200)
    mock_proc.num_threads.return_value = 4
    mock_proc.status.return_value = "running"
    with patch("psutil.Process", return_value=mock_proc):
        metrics = get_process_metrics()
    assert metrics["pid"] == 1234
    assert metrics["num_threads"] == 4
    assert metrics["rss_mb"] == 100.0


def test_get_system_metrics():
    mock_cpu = MagicMock()
    mock_cpu.percent.return_value = 25.5
    mock_cpu.count.return_value = 8
    mock_mem = MagicMock()
    mock_mem.total = 16 * 1024**3
    mock_mem.used = 8 * 1024**3
    mock_mem.available = 8 * 1024**3
    mock_mem.percent = 50.0
    mock_disk = MagicMock()
    mock_disk.total = 500 * 1024**3
    mock_disk.used = 200 * 1024**3
    mock_disk.free = 300 * 1024**3
    mock_disk.percent = 40.0

    with patch("psutil.cpu_percent", return_value=25.5):
        with patch("psutil.cpu_count", side_effect=[8, 4]):
            with patch("psutil.virtual_memory", return_value=mock_mem):
                with patch("psutil.disk_usage", return_value=mock_disk):
                    with patch("src.system_monitor._get_cpu_temperature_c", return_value=None):
                        metrics = get_system_metrics()
    assert metrics["cpu_percent"] == 25.5
    assert metrics["memory_percent"] == 50.0
    assert metrics["disk_percent"] == 40.0
