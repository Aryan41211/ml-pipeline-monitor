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
    with patch("src.system_monitor._get_cpu_temperature_c", return_value=None):
        metrics = get_system_metrics()
    assert metrics["cpu_temperature_c"] is None


def test_get_cpu_temperature_available():
    with patch("src.system_monitor._get_cpu_temperature_c", return_value=55.0):
        metrics = get_system_metrics()
    assert metrics["cpu_temperature_c"] == 55.0


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
    with patch("src.system_monitor.psutil.cpu_percent", return_value=25.5):
        with patch("src.system_monitor.psutil.cpu_count", side_effect=[8, 4]):
            with patch("src.system_monitor.psutil.virtual_memory") as mock_mem:
                mock_mem.return_value = MagicMock(total=16 * 1024**3, used=8 * 1024**3, available=8 * 1024**3, percent=50.0)
                with patch("src.system_monitor.psutil.disk_usage") as mock_disk:
                    mock_disk.return_value = MagicMock(total=500 * 1024**3, used=200 * 1024**3, free=300 * 1024**3, percent=40.0)
                    with patch("src.system_monitor._get_cpu_temperature_c", return_value=None):
                        metrics = get_system_metrics()
    assert metrics["cpu_percent"] == 25.5
    assert metrics["memory_percent"] == 50.0
    assert metrics["disk_percent"] == 40.0
