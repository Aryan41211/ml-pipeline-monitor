"""
System resource snapshot using psutil.

Collected metrics are displayed on the home dashboard so operators
can correlate pipeline latency with host resource pressure.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import psutil


def get_system_metrics() -> Dict[str, Any]:
    """Return current CPU, memory, and disk utilisation."""
    cpu = psutil.cpu_percent(interval=0.3)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    return {
        "cpu_percent": cpu,
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "memory_total_gb": round(mem.total / 2**30, 2),
        "memory_used_gb": round(mem.used / 2**30, 2),
        "memory_available_gb": round(mem.available / 2**30, 2),
        "memory_percent": mem.percent,
        "disk_total_gb": round(disk.total / 2**30, 1),
        "disk_used_gb": round(disk.used / 2**30, 1),
        "disk_free_gb": round(disk.free / 2**30, 1),
        "disk_percent": disk.percent,
    }


def get_process_metrics() -> Dict[str, Any]:
    """Return resource usage for the current Python process."""
    proc = psutil.Process(os.getpid())
    mem_info = proc.memory_info()
    return {
        "pid": proc.pid,
        "rss_mb": round(mem_info.rss / 2**20, 1),
        "vms_mb": round(mem_info.vms / 2**20, 1),
        "num_threads": proc.num_threads(),
        "status": proc.status(),
    }
