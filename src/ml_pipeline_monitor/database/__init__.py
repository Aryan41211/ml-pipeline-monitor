"""
Database package for ML Pipeline Monitor.

Provides a modular persistence layer with clear separation of concerns:
- Schema management and migrations
- Experiment tracking
- Model registry
- Drift detection reports
- Governance (teams, users, workspaces)
- Prediction history
- Dataset lineage
"""

from ml_pipeline_monitor.database.schema import (
    initialize_db,
    initialize_dataset_registry,
    initialize_prediction_registry,
    initialize_governance_registry,
)
from ml_pipeline_monitor.database.experiments import (
    save_experiment,
    get_experiments,
    get_experiment_by_run_id,
)
from ml_pipeline_monitor.database.models import (
    save_model,
    get_models,
    get_latest_production_model,
    get_recent_production_models,
    get_model_stage_events,
    get_model_lineage,
    update_model_stage,
)
from ml_pipeline_monitor.database.drift import (
    save_drift_report,
    get_drift_reports,
    save_drift_reference,
    get_drift_reference,
)
from ml_pipeline_monitor.database.predictions import (
    save_prediction_request,
    save_predictions_for_request,
    get_prediction_history,
    get_prediction_history_by_request_id,
)
from ml_pipeline_monitor.database.governance import (
    create_team,
    create_user,
    create_workspace,
    log_user_activity,
    save_alert_event,
    list_alert_events,
    create_schedule,
    list_schedules,
    record_schedule_run,
    update_schedule,
)
from ml_pipeline_monitor.database.lineage import (
    create_dataset,
    create_dataset_version,
    save_schema_snapshot,
    save_schema_change,
    create_lineage_edge,
    get_dataset_versions,
    get_schema_changes,
    get_lineage_edges,
)

__all__ = [
    # Schema initialization
    "initialize_db",
    "initialize_dataset_registry",
    "initialize_prediction_registry",
    "initialize_governance_registry",
    # Experiments
    "save_experiment",
    "get_experiments",
    "get_experiment_by_run_id",
    # Models
    "save_model",
    "get_models",
    "get_latest_production_model",
    "get_recent_production_models",
    "get_model_stage_events",
    "get_model_lineage",
    "update_model_stage",
    # Drift
    "save_drift_report",
    "get_drift_reports",
    "save_drift_reference",
    "get_drift_reference",
    # Predictions
    "save_prediction_request",
    "save_predictions_for_request",
    "get_prediction_history",
    "get_prediction_history_by_request_id",
    # Governance
    "create_team",
    "create_user",
    "create_workspace",
    "log_user_activity",
    "save_alert_event",
    "list_alert_events",
    "create_schedule",
    "list_schedules",
    "record_schedule_run",
    "update_schedule",
    # Lineage
    "create_dataset",
    "create_dataset_version",
    "save_schema_snapshot",
    "save_schema_change",
    "create_lineage_edge",
    "get_dataset_versions",
    "get_schema_changes",
    "get_lineage_edges",
]