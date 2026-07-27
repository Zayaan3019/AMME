-- Enable TimescaleDB before the application creates its tables. The services
-- run their own migrations on startup (see amme_shared.db.TickStore.migrate),
-- so this file only has to guarantee the extension exists -- keeping schema
-- ownership in one place rather than split between SQL and Python.
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
