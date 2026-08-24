-- ML Pipeline Monitor - PostgreSQL initialization
-- =============================================================================
-- Runs ONCE, on first startup of an empty Postgres data volume.
--
-- Application tables are NOT created here. The application creates and migrates
-- its own schema at startup via initialize_db() (see database/schema.py), which
-- both the API and the worker call. Adding table DDL here would duplicate that
-- and drift from it.
--
-- This file previously created a login role `mlmonitor_app` with a password
-- hardcoded in the repository and granted it ALL PRIVILEGES on the database.
-- Nothing ever connected as that role -- the services authenticate with the
-- credentials in PIPELINE_DB_DSN -- so it was an unused account with a
-- publicly known password. It has been removed. If you later need a separate
-- least-privilege application role, create it with a password supplied from
-- the environment, not from a file in git.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

DO $$
BEGIN
    RAISE NOTICE 'ML Pipeline Monitor database initialized';
END
$$;
