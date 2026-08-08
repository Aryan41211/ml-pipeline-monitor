


"""
Prometheus metrics for ML Pipeline Monitor.

This module defines all Prometheus metrics exposed by the application.
Metrics are registered with the default Prometheus registry and exposed
via the /metrics endpoint on the FastAPI service.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry

# Create a custom registry for this application
registry = CollectorRegistry()

# ---------------------------------------------------------------------------
# Pipeline Metrics
# ---------------------------------------------------------------------------

pipeline_runs_total = Counter(
    "ml_pipeline_runs_total",
    "Total number of pipeline runs",
    ["status", "dataset", "model_type"],
    registry=registry,
)

pipeline_duration_seconds = Histogram(
    "ml_pipeline_duration_seconds",
    "Pipeline run duration in seconds",
    ["dataset", "model_type"],
    buckets=(10, 30, 60, 120, 300, 600, 1200, 1800, 3600),
    registry=registry,
)

pipeline_stage_duration_seconds = Histogram(
    "ml_pipeline_stage_duration_seconds",
    "Pipeline stage duration in seconds",
    ["stage", "dataset", "model_type"],
    buckets=(1, 5, 10, 30, 60, 120, 300),
    registry=registry,
)

# ---------------------------------------------------------------------------
# API / Prediction Metrics
# ---------------------------------------------------------------------------

api_requests_total = Counter(
    "ml_api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"],
    registry=registry,
)

api_request_duration_seconds = Histogram(
    "ml_api_request_duration_seconds",
    "API request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
    registry=registry,
)

predictions_total = Counter(
    "ml_predictions_total",
    "Total number of predictions made",
    ["model_id", "dataset", "status"],
    registry=registry,
)

prediction_latency_seconds = Histogram(
    "ml_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["model_id", "dataset"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
    registry=registry,
)

api_errors_total = Counter(
    "ml_api_errors_total",
    "Total number of API errors",
    ["method", "endpoint", "error_type"],
    registry=registry,
)

# ---------------------------------------------------------------------------
# Drift Detection Metrics
# ---------------------------------------------------------------------------

drift_detections_total = Counter(
    "ml_drift_detections_total",
    "Total number of drift detection runs",
    ["dataset", "severity", "drift_detected"],
    registry=registry,
)

drift_score = Gauge(
    "ml_drift_score",
    "Current drift score (PSI) for dataset_d)",
    ["dataset"],
    registry=registry,
)

drift_features_count = Gauge(
    "ml_drift_features_count",
    "Number of features with detected drift",
    ["dataset"],
    registry=registry,
)

drift_analysis_duration_seconds = Histogram(
    "ml_drift_analysis_duration_seconds",
    "Drift analysis duration in seconds",
    ["dataset"],
    buckets=(0.1, 0.5, 1, 5, 10, 30, 60),
    registry=registry,
)

# ---------------------------------------------------------------------------
# System Resource Metrics
# ---------------------------------------------------------------------------

system_cpu_percent = Gauge(
    "ml_system_cpu_percent",
    "Current CPU utilization percentage",
    registry=registry,
)

system_memory_percent = Gauge(
    "ml_system_memory_percent",
    "Current memory utilization percentage",
    registry=registry,
)

system_memory_used_bytes = Gauge(
    "ml_system_memory_used_bytes",
    "Memory used in bytes",
    registry=registry,
)

system_memory_available_bytes = Gauge(
    "ml_system_memory_available_bytes",
    "Memory available in bytes",
    registry=registry,
)

system_disk_percent = Gauge(
    "ml_system_disk_percent",
    "Current disk utilization percentage",
    registry=registry,
)

system_disk_used_bytes = Gauge(
    "ml_system_disk_used_bytes",
    "Disk used in bytes",
    registry=registry,
)

system_disk_free_bytes = Gauge(
    "ml_system_disk_free_bytes",
    "Disk free in bytes",
    registry=registry,
)

# ---------------------------------------------------------------------------
# Process Metrics
# ---------------------------------------------------------------------------

process_cpu_percent = Gauge(
    "ml_process_cpu_percent",
    "Current process CPU utilization percentage",
    registry=registry,
)

process_memory_rss_bytes = Gauge(
    "ml_process_memory_rss_bytes",
    "Process resident set size in bytes",
    registry=registry,
)

process_memory_vms_bytes = Gauge(
    "ml_process_memory_vms_bytes",
    "Process virtual memory size in bytes",
    registry=registry,
)

process_num_threads = Gauge(
    "ml_process_num_threads",
    "Number of threads in the process",
    registry=registry,
)

# System-level active threads and process count
system_active_threads = Gauge(
    "ml_system_active_threads",
    "Active threads on the host (best-effort)",
    registry=registry,
)

system_process_count = Gauge(
    "ml_system_process_count",
    "Process count on the host (best-effort)",
    registry=registry,
)

system_cpu_temperature_c = Gauge(
    "ml_system_cpu_temperature_c",
    "CPU temperature in Celsius (best-effort; NaN if unavailable)",
    registry=registry,
)

# ---------------------------------------------------------------------------
# Experiment / Model Registry Metrics
# ---------------------------------------------------------------------------

experiments_total = Counter(
    "ml_experiments_total",
    "Total number of experiments created",
    ["status", "dataset", "model_type"],
    registry=registry,
)

models_registered_total = Counter(
    "ml_models_registered_total",
    "Total number of models registered",
    ["dataset", "model_type", "stage"],
    registry=registry,
)

model_promotions_total = Counter(
    "ml_model_promotions_total",
    "Total number of model promotions/demotions",
    ["dataset", "model_type", "from_stage", "to_stage", "status"],
    registry=registry,
)

# ---------------------------------------------------------------------------
# Data Health / Validation Metrics
# ---------------------------------------------------------------------------

dataset_validations_total = Counter(
    "ml_dataset_validations_total",
    "Total number of dataset validations",
    ["dataset", "status"],
    registry=registry,
)

dataset_rows = Gauge(
    "ml_dataset_rows",
    "Number of rows in dataset",
    ["dataset"],
    registry=registry,
)

dataset_columns = Gauge(
    "ml_dataset_columns",
    "Number of columns in dataset",
    ["dataset"],
    registry=registry,
)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def update_system_metrics() -> None:
    """Update all system-level metrics from psutil.

    Note: Some values (like CPU temperature) are best-effort and may be NaN if
    sensors aren't available in the current environment.
    """
    import psutil

    # CPU
    system_cpu_percent.set(psutil.cpu_percent(interval=0.1))

    cpu_temp_c = float("nan")
    if hasattr(psutil, "sensors_temperatures"):
        try:
            temps = psutil.sensors_temperatures(fahrenheit=False)
            if temps:
                for _, entries in temps.items():
                    for entry in entries:
                        if entry.current is not None:
                            cpu_temp_c = float(entry.current)
                            break
                    if not (cpu_temp_c != cpu_temp_c):  # not-NaN
                        break
        except Exception:
            cpu_temp_c = float("nan")

    system_cpu_temperature_c.set(cpu_temp_c)

    # Memory
    mem = psutil.virtual_memory()
    system_memory_percent.set(mem.percent)
    system_memory_used_bytes.set(mem.used)
    system_memory_available_bytes.set(mem.available)

    # Disk
    disk = psutil.disk_usage("/")
    system_disk_percent.set(disk.percent)
    system_disk_used_bytes.set(disk.used)
    system_disk_free_bytes.set(disk.free)

    # Process + thread info (for both process and host)
    proc = psutil.Process()
    process_cpu_percent.set(proc.cpu_percent(interval=0.1))
    mem_info = proc.memory_info()
    process_memory_rss_bytes.set(mem_info.rss)
    process_memory_vms_bytes.set(mem_info.vms)
    process_num_threads.set(proc.num_threads())

    # Active threads: use current process thread count (best-effort)
    system_active_threads.set(proc.num_threads())

    # Process count: number of PIDs visible to the OS (best-effort)
    try:
        system_process_count.set(len(psutil.pids()))
    except Exception:
        system_process_count.set(0)


def record_pipeline_run(
    status: str,
    dataset: str,
    model_type: str,
    duration_seconds: float,
) -> None:
    """Record a pipeline run completion."""
    pipeline_runs_total.labels(status=status, dataset=dataset, model_type=model_type).inc()
    pipeline_duration_seconds.labels(dataset=dataset, model_type=model_type).observe(duration_seconds)


def record_pipeline_stage(
    stage: str,
    dataset: str,
    model_type: str,
    duration_seconds: float,
) -> None:
    """Record a pipeline stage completion."""
    pipeline_stage_duration_seconds.labels(
        stage=stage, dataset=dataset, model_type=model_type
    ).observe(duration_seconds)


def record_api_request(
    method: str,
    endpoint: str,
    status_code: int,
    duration_seconds: float,
) -> None:
    """Record an API request."""
    api_requests_total.labels(method=method, endpoint=endpoint, status_code=str(status_code)).inc()
    api_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration_seconds)


def record_api_error(
    method: str,
    endpoint: str,
    error_type: str,
) -> None:
    """Record an API error."""
    api_errors_total.labels(method=method, endpoint=endpoint, error_type=error_type).inc()


def record_prediction(
    model_id: str,
    dataset: str,
    status: str,
    latency_seconds: float,
) -> None:
    """Record a prediction request."""
    predictions_total.labels(model_id=model_id, dataset=dataset, status=status).inc()
    if status == "success":
        prediction_latency_seconds.labels(model_id=model_id, dataset=dataset).observe(latency_seconds)


def record_drift_detection(
    dataset: str,
    severity: str,
    drift_detected: bool,
    drift_score_value: float,
    features_drifted: int,
    duration_seconds: float,
) -> None:
    """Record a drift detection run."""
    drift_detections_total.labels(
        dataset=dataset,
        severity=severity,
        drift_detected=str(drift_detected).lower(),
    ).inc()
    drift_score.labels(dataset=dataset).set(drift_score_value)
    drift_features_count.labels(dataset=dataset).set(features_drifted)
    drift_analysis_duration_seconds.labels(dataset=dataset).observe(duration_seconds)


def record_experiment(
    status: str,
    dataset: str,
    model_type: str,
) -> None:
    """Record an experiment creation."""
    experiments_total.labels(status=status, dataset=dataset, model_type=model_type).inc()


def record_model_registration(
    dataset: str,
    model_type: str,
    stage: str,
) -> None:
    """Record a model registration."""
    models_registered_total.labels(dataset=dataset, model_type=model_type, stage=stage).inc()


def record_model_promotion(
    dataset: str,
    model_type: str,
    from_stage: str,
    to_stage: str,
    status: str,
) -> None:
    """Record a model promotion/demotion."""
    model_promotions_total.labels(
        dataset=dataset,
        model_type=model_type,
        from_stage=from_stage,
        to_stage=to_stage,
        status=status,
    ).inc()


def record_dataset_validation(
    dataset: str,
    status: str,
    rows: int,
    columns: int,
) -> None:
    """Record a dataset validation."""
    dataset_validations_total.labels(dataset=dataset, status=status).inc()
    dataset_rows.labels(dataset=dataset).set(rows)
    dataset_columns.labels(dataset=dataset).set(columns)