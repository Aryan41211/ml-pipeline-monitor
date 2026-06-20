# CLAUDE.md — ML Pipeline Monitor (Permanent AI Instructions)

Use this file as the **source of truth** for AI coding assistant behavior when modifying this repository.

## 1) Project Overview & Goals
ML Pipeline Monitor is a production-oriented observability and operations UI for ML pipelines. It provides:
- Streamlit dashboards for pipeline runs, experiment tracking, model registry, data drift, data health, and governance.
- Service-layer orchestration (Streamlit UI → services → core logic → persistence).
- An optional FastAPI API surface for model inference.

Primary goals:
- Reliability: deterministic orchestration and consistent persistence.
- Traceability: stage-level progress, lineage, and structured logging.
- Maintainability: clear modular boundaries and assistant-friendly conventions.

## 2) Existing Architecture (UI → Services → Core → Persistence)
- **UI (Streamlit)**: `app.py` and `pages/*.py`
- **Services**: `services/*.py` (use-cases and orchestration)
  - API services: `services/api/*.py`
- **Core**: `src/*.py` (domain logic: pipeline, drift, loaders, auth, etc.)
- **Persistence**: `src/database.py`, `src/db_engine.py`, `src/db_interface.py`

Data flow guideline:
`Streamlit UI -> services -> src core -> persistence/db layer`

## 3) Folder Structure & Responsibilities
- `pages/`: Streamlit page modules.
  - Keep UI concerns here (widgets, layout, page-local state).
  - Call `services/*` for operations.
- `services/`: orchestration and integration.
  - Validate inputs, translate UI needs to core calls.
  - Coordinate persistence writes/reads via core/persistence layer.
- `src/`: core business logic and utilities.
  - Implement algorithms, validations, and domain workflows.
  - Keep side effects minimal; delegate persistence to persistence layer.
- `tests/`: unit and e2e tests.
- `logs/` & `artifacts/`: runtime outputs (do not hardcode write paths in logic).

## 4) Coding Standards (Non-Negotiable)
- **Type hints**: Use explicit type hints for public functions/classes.
- **Docstrings**: Every module-level public function/class must have a docstring.
- **Modular design**: small, composable functions; single responsibility.
- **Avoid God classes**: no large all-in-one classes/modules.
- **Avoid duplicate code**: factor shared behavior into reusable helpers.
- **Correctness over cleverness**: prefer explicit, readable implementations.

## 5) Database Rules
- Use the existing persistence abstractions (do not bypass `db_interface`/`db_engine`).
- Prefer parameterized queries / safe ORM patterns used by the project.
- Ensure schema/lineage operations remain consistent with existing lineage tracking.
- Keep migrations/DB init consistent with current backend setup.

## 6) API Design Standards
- Follow the established FastAPI patterns in `services/api/`.
- Validate request bodies and types early (use Pydantic models already present).
- Return consistent response shapes; avoid leaking internal exceptions.
- Keep endpoints thin: route to `services/`.

## 7) Streamlit Page Conventions (`pages/*.py`)
- Page functions should primarily:
  - define widgets,
  - call corresponding services,
  - render results.
- Do not implement business logic directly in pages.
- Keep UI state localized; do not rely on implicit global mutation.
- Any expensive computation should be delegated to services/core and cached if applicable.

## 8) Logging Conventions
- Use the repository logger abstraction (see `src/logger.py`).
- Log with structured context where possible (IDs: run_id, experiment_id, dataset name, stage).
- Avoid logging secrets or raw credentials.
- Use log levels consistently: debug/info/warning/error.

## 9) Testing Requirements
- Add/extend **unit tests** under `tests/` for core and services.
- Add/extend **e2e tests** under `tests/e2e/` when user-facing flows change.
- Maintain coverage expectations enforced by `pytest.ini`.
- Tests must be deterministic; avoid time-based flakiness.

## 10) Git Commit Format
Use Conventional-like prefixes exactly as below:
- `feat:` new feature
- `fix:` bug fix
- `refactor:` structural improvement without behavior change
- `docs:` documentation-only changes

## 11) Dependency Policy
- Prefer existing libraries already used in `requirements.txt`.
- Avoid unnecessary new packages.
- Do not add heavy dependencies unless clearly required.

## 12) Performance Rules
- Cache expensive operations (use Streamlit caching or existing caching mechanisms in services/core).
- Avoid repeated disk I/O (read once, reuse; memoize in-process when appropriate).
- Minimize unnecessary DB roundtrips.

## 13) Security Rules
- **bcrypt only** for password hashing and verification.
- Secrets must come from **environment variables** (no hardcoded credentials).
- Input validation everywhere:
  - validate request payloads,
  - validate UI form inputs,
  - validate any persistence-layer inputs.

## 14) “Automatic Compliance” Expectation
All future modifications must follow this document’s rules automatically:
- If you must deviate, document the reason in a `docs:` commit and update the relevant section.
- Do not introduce silent behavior changes without tests.

