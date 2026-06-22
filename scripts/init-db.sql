-- ML Pipeline Monitor Database Initialization Script
-- Runs on PostgreSQL container startup

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create application user if not exists (for multi-tenant setups)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'mlmonitor_app') THEN
        CREATE ROLE mlmonitor_app WITH LOGIN PASSWORD 'mlmonitor_app_password';
    END IF;
END
$$;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE mlmonitor TO mlmonitor_app;
GRANT ALL ON SCHEMA public TO mlmonitor_app;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO mlmonitor_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO mlmonitor_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO mlmonitor_app;

-- Create indexes for better performance (will be created by application migrations)
-- This script ensures the database is ready for the application

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'ML Pipeline Monitor database initialized successfully';
END
$$;