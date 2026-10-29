"""
history.py
----------
MODULE: Historical Tracking (SQLite)
INPUT:  a completed assessment (health score, damage list, etc.) tagged with
        a user-supplied location/road label
OUTPUT: persisted rows in `data/history.sqlite3`; on read, a list of
        `AssessmentRecord` and an optional trend comparison between the two
        most recent records for a location
WHY:    lets the same location be inspected repeatedly over time and shows
        whether it is improving, stable, or deteriorating
ALGORITHM: plain SQLite table, one row per assessment; trend is a threshold
        rule over (health_score delta, damage_count delta) - see
        `compare_latest_two`
LIMITATIONS: the location label is user-entered free text - the system does
        NOT verify that two images with the same label are actually the
        same physical place (no GPS/geo-matching). With fewer than two
        records for a location, no trend is computed (never fabricated).
INTERACTION: written to by `app.py` after each analysis; read by the
        History dashboard and by `report.py` for the "historical comparison"
        report section.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import List, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "history.sqlite3")

SCHEMA = """
CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    location TEXT NOT NULL,
    image_name TEXT NOT NULL,
    image_path TEXT NOT NULL,
    annotated_path TEXT,
    created_at REAL NOT NULL,
    health_score REAL NOT NULL,
    condition TEXT NOT NULL,
    damage_count INTEGER NOT NULL,
    priority TEXT NOT NULL,
    damaged_area_ratio REAL NOT NULL,
    detections_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_location ON assessments(location);
"""


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


@dataclass
class AssessmentRecord:
    id: int
    location: str
    image_name: str
    image_path: str
    annotated_path: str
    created_at: float
    health_score: float
    condition: str
    damage_count: int
    priority: str
    damaged_area_ratio: float
    detections: list
    # --- location-aware additions (GPS + area scorecard / segment history) ---
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    area: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    road_name: Optional[str] = None
    landmark: Optional[str] = None
    segment_id: Optional[str] = None


def _col(row, name, default=None):
    """Read a column that may not exist yet on a pre-migration row/connection."""
    try:
        return row[name]
    except (IndexError, KeyError):
        return default


def save_assessment(location: str, image_name: str, image_path: str, annotated_path: str,
                     health_score: float, condition: str, damage_count: int, priority: str,
                     damaged_area_ratio: float, detections_summary: list,
                     latitude: Optional[float] = None, longitude: Optional[float] = None,
                     area: Optional[str] = None, city: Optional[str] = None,
                     state: Optional[str] = None, road_name: Optional[str] = None,
                     landmark: Optional[str] = None, segment_id: Optional[str] = None) -> int:
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO assessments
           (location, image_name, image_path, annotated_path, created_at, health_score,
            condition, damage_count, priority, damaged_area_ratio, detections_json,
            latitude, longitude, area, city, state, road_name, landmark, segment_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (location, image_name, image_path, annotated_path, time.time(), health_score,
         condition, damage_count, priority, damaged_area_ratio, json.dumps(detections_summary),
         latitude, longitude, area, city, state, road_name, landmark, segment_id),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def _row_to_record(row) -> AssessmentRecord:
    return AssessmentRecord(
        id=row["id"], location=row["location"], image_name=row["image_name"],
        image_path=row["image_path"], annotated_path=row["annotated_path"],
        created_at=row["created_at"], health_score=row["health_score"],
        condition=row["condition"], damage_count=row["damage_count"],
        priority=row["priority"], damaged_area_ratio=row["damaged_area_ratio"],
        detections=json.loads(row["detections_json"]),
        latitude=_col(row, "latitude"), longitude=_col(row, "longitude"),
        area=_col(row, "area"), city=_col(row, "city"), state=_col(row, "state"),
        road_name=_col(row, "road_name"), landmark=_col(row, "landmark"),
        segment_id=_col(row, "segment_id"),
    )


def list_locations() -> List[str]:
    conn = get_conn()
    rows = conn.execute("SELECT DISTINCT location FROM assessments ORDER BY location").fetchall()
    conn.close()
    return [r["location"] for r in rows]


def get_history(location: str) -> List[AssessmentRecord]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM assessments WHERE location = ? ORDER BY created_at ASC", (location,)
    ).fetchall()
    conn.close()
    return [_row_to_record(r) for r in rows]


def get_assessment(assessment_id: int) -> Optional[AssessmentRecord]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM assessments WHERE id = ?", (assessment_id,)).fetchone()
    conn.close()
    return _row_to_record(row) if row else None


def list_all() -> List[AssessmentRecord]:
    """All inspection records (used by area_health.py / segment.py aggregation)."""
    conn = get_conn()
    rows = conn.execute("SELECT * FROM assessments ORDER BY created_at ASC").fetchall()
    conn.close()
    return [_row_to_record(r) for r in rows]


def get_by_segment(segment_id: str) -> List[AssessmentRecord]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM assessments WHERE segment_id = ? ORDER BY created_at ASC", (segment_id,)
    ).fetchall()
    conn.close()
    return [_row_to_record(r) for r in rows]


def compare_latest_two(location: str) -> Optional[dict]:
    """
    Compare the two most recent assessments for a location.

    Returns None if fewer than two assessments exist for this location -
    we never fabricate a trend from a single data point.

    `trend_status` is always exactly one of: "improving", "stable",
    "deteriorating" - rule: a >=8 point health-score swing, or a >=2
    change in damage count, decides the direction; otherwise "stable".
    """
    records = get_history(location)
    if len(records) < 2:
        return None
    prev, curr = records[-2], records[-1]

    score_delta = round(curr.health_score - prev.health_score, 1)
    count_delta = curr.damage_count - prev.damage_count
    area_delta = round(curr.damaged_area_ratio - prev.damaged_area_ratio, 4)

    if score_delta <= -8 or count_delta >= 2:
        trend_status = "deteriorating"
        trend = "Road condition has deteriorated."
    elif score_delta >= 8 or count_delta <= -2:
        trend_status = "improving"
        trend = "Road condition has improved."
    else:
        trend_status = "stable"
        trend = "Road condition is roughly unchanged."

    return {
        "previous": prev, "current": curr,
        "score_delta": score_delta, "count_delta": count_delta, "area_delta": area_delta,
        "trend": trend, "trend_status": trend_status,
        "priority_changed": prev.priority != curr.priority,
    }
