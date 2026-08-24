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

from ml_pipeline_monitor.database.drift import (
    get_drift_reference,
    get_drift_reports,
    save_drift_reference,
    save_drift_report,
)
from ml_pipeline_monitor.database.experiments import (
    get_experiment_by_run_id,
    get_experiments,
    save_experiment,
)
from ml_pipeline_monitor.database.governance import (
    claim_schedule,
    create_schedule,
    create_team,
    create_user,
    create_workspace,
    list_alert_events,
    list_schedules,
    log_user_activity,
    record_schedule_run,
    save_alert_event,
    update_schedule,
)
from ml_pipeline_monitor.database.lineage import (
    create_dataset,
    create_dataset_version,
    create_lineage_edge,
    get_dataset_versions,
    get_lineage_edges,
    get_schema_changes,
    save_schema_change,
    save_schema_snapshot,
)
from ml_pipeline_monitor.database.models import (
    get_latest_production_model,
    get_model_by_id,
    get_model_lineage,
    get_model_stage_events,
    get_models,
    get_recent_production_models,
    save_model,
    update_model_stage,
)
from ml_pipeline_monitor.database.predictions import (
    get_prediction_history,
    get_prediction_history_by_request_id,
    save_prediction_request,
    save_predictions_for_request,
)
from ml_pipeline_monitor.database.schema import (
    initialize_dataset_registry,
    initialize_db,
    initialize_governance_registry,
    initialize_prediction_registry,
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
    "get_model_by_id",
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
    "claim_schedule",
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
