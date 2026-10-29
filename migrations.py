"""
migrations.py
-------------
MODULE: Database Migration (SQLite, additive-only)
INPUT:  the existing `data/history.sqlite3` database (created by
        history.py / citizen_report.py, possibly from an older version of
        RoadScan with none of the location/segment/dedup columns yet)
OUTPUT: the same database, with the additional columns/tables the four new
        features need, added in place. Existing rows and existing columns
        are never touched or dropped.
WHY:    the enhancement spec requires the app to "automatically
        create/migrate the necessary fields/tables for a fresh
        installation" and to keep "existing citizen reports and existing
        history... working" - this module is the single place that does
        that, run once at app startup, so history.py and citizen_report.py
        don't need their own migration logic duplicated.
ALGORITHM: for each (table, column, type) in the spec below, check
        `PRAGMA table_info(table)`; if the column is missing, `ALTER TABLE
        ... ADD COLUMN`. SQLite's ALTER TABLE ADD COLUMN is safe on
        existing data (new column is NULL for all existing rows).
        New tables are created with `CREATE TABLE IF NOT EXISTS`.
LIMITATIONS: additive only - this does not rename or remove columns, and
        does not attempt to backfill area/city/segment_id for rows that
        existed before this migration (their old free-text `location`
        field is preserved as-is; they simply have NULL for the new
        columns until re-inspected).
INTERACTION: `run_migrations()` is called once from app.py at startup,
        before any request is served.
"""

from __future__ import annotations
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "history.sqlite3")

# table -> [(column_name, sql_type), ...]
NEW_COLUMNS = {
    "assessments": [
        ("latitude", "REAL"),
        ("longitude", "REAL"),
        ("area", "TEXT"),
        ("city", "TEXT"),
        ("state", "TEXT"),
        ("road_name", "TEXT"),
        ("landmark", "TEXT"),
        ("segment_id", "TEXT"),
    ],
    "citizen_reports": [
        ("area", "TEXT"),
        ("city", "TEXT"),
        ("state", "TEXT"),
        ("segment_id", "TEXT"),
        ("duplicate_group_id", "TEXT"),
        # --- citizen-first redesign: identity, assignment, SLA/escalation,
        # --- resolution metadata (all additive, NULL for pre-existing rows) ---
        ("citizen_id", "INTEGER"),
        ("image_hash", "TEXT"),
        ("target_date", "REAL"),
        ("department", "TEXT"),
        ("officer_name", "TEXT"),
        ("officer_id", "TEXT"),
        ("assigned_at", "REAL"),
        ("escalation_level", "INTEGER DEFAULT 0"),
        ("escalated_at", "REAL"),
        ("overdue_flagged_at", "REAL"),
        ("verified_at", "REAL"),
        ("verified_by", "TEXT"),
        ("resolved_by", "TEXT"),
        ("resolution_notes", "TEXT"),
        ("resolution_date", "REAL"),
        ("citizen_followup_path", "TEXT"),
        ("followup_at", "REAL"),
    ],
}

NEW_TABLES = """
CREATE TABLE IF NOT EXISTS duplicate_groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id TEXT UNIQUE NOT NULL,
    damage_class TEXT,
    damage_label TEXT,
    location_label TEXT,
    confidence TEXT NOT NULL,
    member_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_duplicate_groups_group_id ON duplicate_groups(group_id);
"""

NEW_TABLES += """
CREATE TABLE IF NOT EXISTS citizens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    is_demo INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
-- Reward ledger: one row per (report, event type), so a report can never
-- earn the same reward twice. Points are always SUM(points) of this table.
CREATE TABLE IF NOT EXISTS reward_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    citizen_id INTEGER NOT NULL,
    report_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    points INTEGER NOT NULL,
    note TEXT,
    created_at REAL NOT NULL,
    UNIQUE(report_id, event_type)
);
CREATE INDEX IF NOT EXISTS idx_reward_events_citizen ON reward_events(citizen_id);
CREATE TABLE IF NOT EXISTS certificates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    verification_id TEXT UNIQUE NOT NULL,
    citizen_id INTEGER NOT NULL,
    citizen_name TEXT NOT NULL,
    badge TEXT NOT NULL,
    points INTEGER NOT NULL,
    verified_reports INTEGER NOT NULL,
    resolved_reports INTEGER NOT NULL,
    issued_at REAL NOT NULL
);
"""

NEW_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_assessments_segment ON assessments(segment_id);",
    "CREATE INDEX IF NOT EXISTS idx_assessments_area ON assessments(area);",
    "CREATE INDEX IF NOT EXISTS idx_citizen_reports_segment ON citizen_reports(segment_id);",
    "CREATE INDEX IF NOT EXISTS idx_citizen_reports_area ON citizen_reports(area);",
    "CREATE INDEX IF NOT EXISTS idx_citizen_reports_dup_group ON citizen_reports(duplicate_group_id);",
    "CREATE INDEX IF NOT EXISTS idx_citizen_reports_citizen ON citizen_reports(citizen_id);",
    "CREATE INDEX IF NOT EXISTS idx_citizen_reports_hash ON citizen_reports(image_hash);",
]


def _existing_columns(conn: sqlite3.Connection, table: str):
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.OperationalError:
        return set()
    return {r[1] for r in rows}  # column name is index 1


def run_migrations() -> list:
    """
    Idempotent - safe to call on every app startup. Returns a list of
    human-readable strings describing what (if anything) was changed, for
    startup logging.
    """
    # Import here (not at module load) so history.py / citizen_report.py's
    # own SCHEMA scripts run first and create the base tables if this is a
    # brand-new database.
    import history
    import citizen_report as creport

    changes = []
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(history.SCHEMA)
        conn.executescript(creport.SCHEMA)
        conn.executescript(NEW_TABLES)

        for table, columns in NEW_COLUMNS.items():
            existing = _existing_columns(conn, table)
            for col_name, col_type in columns:
                if col_name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                    changes.append(f"{table}.{col_name} ({col_type})")

        for stmt in NEW_INDEXES:
            conn.execute(stmt)

        conn.commit()
    finally:
        conn.close()
    return changes
