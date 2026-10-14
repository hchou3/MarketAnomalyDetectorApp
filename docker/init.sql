-- Run once at postgres container init time.
-- The mad database is created by POSTGRES_DB in docker-compose.yml.
-- This script creates the mad_test database used by the DB test suite.
SELECT 'CREATE DATABASE mad_test OWNER mad'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'mad_test'
)\gexec
