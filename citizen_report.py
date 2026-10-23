"""
citizen_report.py
------------------
MODULE: Citizen Road Damage Reporting / Complaint Tracking (prototype)
INPUT:  a citizen-submitted road photo + location/details, already run
        through the EXISTING RoadScan pipeline (detector.py -> severity.py
        -> health_score.py -> severity.compute_repair_priority) by app.py
OUTPUT: a persisted report row (SQLite) with a stable, human-readable
        Report ID, and a status lifecycle a "maintenance" user can advance
WHY:    adds a civic-reporting workflow (conceptually similar to apps like
        Swachhata-MoHUA: report -> categorize -> track -> resolve) on top
        of RoadScan's existing AI assessment, without duplicating any of
        the detection/scoring logic that already lives elsewhere.
ALGORITHM: plain SQLite table (same `data/history.sqlite3` database file
        `history.py` already uses - a new table, not a new storage
        mechanism). Report IDs are derived from the row's autoincrement id
        (RS-<year>-<6 digits>), so they are unique, persistent, and always
        reproducible from stored data - never a random value that could
        change on reload.
LIMITATIONS: this is a self-contained prototype demo, not a connection to
        any real municipal system. There is no real authentication: reports
        are linked to a lightweight local citizen identity (citizens.py) that
        can later be swapped for real accounts, and authority actions
        (authority_views.py) are manual, by design (see README).
REDESIGN NOTE: assignment (department/officer), the 14-day resolution
        target, escalation level, verification and resolution metadata now
        live on the same row (additive columns, see migrations.py). Every
        status/assignment/escalation change is appended to
        `status_history_json` with who did it and an optional note, which is
        the audit trail shown on the report page.
INTERACTION: written to by app.py's /report/submit route right after the
        shared `run_pipeline()` call; read by the Track Report, My Reports,
        Report Detail, and Maintenance Dashboard routes/templates.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import config

DB_PATH = "data/history.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS citizen_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT UNIQUE,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    image_path TEXT,
    annotated_image_path TEXT,
    latitude REAL,
    longitude REAL,
    location TEXT,
    road_name TEXT,
    landmark TEXT,
    description TEXT,
    reporter_name TEXT,
    reporter_contact TEXT,
    detected_damage TEXT,
    detections_json TEXT,
    severity_score REAL,
    severity_level TEXT,
    health_score REAL,
    health_condition TEXT,
    repair_priority TEXT,
    priority_reason TEXT,
    status TEXT NOT NULL DEFAULT 'REPORTED',
    resolution_image_path TEXT,
    resolution_annotated_path TEXT,
    resolved_at REAL,
    comparison_json TEXT,
    status_history_json TEXT NOT NULL,
    json_report_path TEXT,
    pdf_report_path TEXT,
    html_report_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_citizen_reports_report_id ON citizen_reports(report_id);
CREATE INDEX IF NOT EXISTS idx_citizen_reports_status ON citizen_reports(status);
"""

# Fixed lifecycle - deliberately linear/simple for a prototype demo.
STATUSES = ["REPORTED", "UNDER_REVIEW", "ASSIGNED", "IN_PROGRESS", "RESOLVED"]
STATUS_LABELS = {
    "REPORTED": "Reported",
    "UNDER_REVIEW": "Under Review",
    "ASSIGNED": "Assigned",
    "IN_PROGRESS": "In Progress",
    "RESOLVED": "Resolved",
    "ESCALATION_REQUIRED": "Escalation Required",
}
# ESCALATION_REQUIRED is not a step in the linear repair workflow: it is the
# *displayed* status of any open report that has passed its resolution
# target (see CitizenReport.display_status). STATUSES stays the 5 real steps.
ALL_STATUSES = STATUSES + ["ESCALATION_REQUIRED"]


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


@dataclass
class CitizenReport:
    id: int
    report_id: str
    created_at: float
    updated_at: float
    image_path: Optional[str]
    annotated_image_path: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    location: Optional[str]
    road_name: Optional[str]
    landmark: Optional[str]
    description: Optional[str]
    reporter_name: Optional[str]
    reporter_contact: Optional[str]
    detected_damage: Optional[str]
    detections: list
    severity_score: Optional[float]
    severity_level: Optional[str]
    health_score: Optional[float]
    health_condition: Optional[str]
    repair_priority: Optional[str]
    priority_reason: Optional[str]
    status: str
    resolution_image_path: Optional[str]
    resolution_annotated_path: Optional[str]
    resolved_at: Optional[float]
    comparison: Optional[dict]
    json_report_path: Optional[str] = None
    pdf_report_path: Optional[str] = None
    html_report_path: Optional[str] = None
    status_history: list = field(default_factory=list)
    # --- location-aware additions (GPS + area scorecard / segment history / dedup) ---
    area: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    segment_id: Optional[str] = None
    duplicate_group_id: Optional[str] = None
    # --- citizen-first redesign additions ---
    citizen_id: Optional[int] = None
    image_hash: Optional[str] = None
    target_date: Optional[float] = None
    department: Optional[str] = None
    officer_name: Optional[str] = None
    officer_id: Optional[str] = None
    assigned_at: Optional[float] = None
    escalation_level: int = 0
    escalated_at: Optional[float] = None
    overdue_flagged_at: Optional[float] = None
    verified_at: Optional[float] = None
    verified_by: Optional[str] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    resolution_date: Optional[float] = None
    citizen_followup_path: Optional[str] = None
    followup_at: Optional[float] = None

    @property
    def status_label(self) -> str:
        return STATUS_LABELS.get(self.status, self.status)

    # ---- SLA / escalation (computed, never stored as a guess) ----
    @property
    def target_at(self) -> float:
        """Resolution target = submission + SLA_DAYS (stored at creation; derived for older rows)."""
        return self.target_date or (self.created_at + config.SLA_DAYS * 86400)

    def is_overdue(self, now: Optional[float] = None) -> bool:
        return self.status != "RESOLVED" and (now or time.time()) > self.target_at

    @property
    def display_status(self) -> str:
        return "ESCALATION_REQUIRED" if self.is_overdue() else self.status

    @property
    def display_status_label(self) -> str:
        return STATUS_LABELS.get(self.display_status, self.display_status)

    @property
    def days_elapsed(self) -> int:
        end = self.resolved_at if (self.status == "RESOLVED" and self.resolved_at) else time.time()
        return max(0, int((end - self.created_at) // 86400))

    @property
    def days_remaining(self) -> int:
        """Whole days left before the target (negative once overdue)."""
        return int((self.target_at - time.time()) // 86400) if self.status != "RESOLVED" else 0

    @property
    def days_overdue(self) -> int:
        return max(0, -int((self.target_at - time.time()) // 86400)) if self.is_overdue() else 0

    @property
    def is_assigned(self) -> bool:
        return bool(self.department or self.officer_name)

    @property
    def is_verified(self) -> bool:
        return self.verified_at is not None

    @property
    def escalation_level_name(self) -> str:
        lv = min(max(self.escalation_level or 0, 0), len(config.ESCALATION_LEVELS) - 1)
        return config.ESCALATION_LEVELS[lv]

    @property
    def next_escalation_name(self) -> Optional[str]:
        nxt = (self.escalation_level or 0) + 1
        return config.ESCALATION_LEVELS[nxt] if nxt < len(config.ESCALATION_LEVELS) else None

    def step_times(self) -> dict:
        """status -> timestamp it was first reached (REPORTED = submission time)."""
        times = {"REPORTED": self.created_at}
        for h in self.status_history:
            st = h.get("status")
            if st in STATUSES and not h.get("event") and st not in times:
                times[st] = h.get("at")
        return times

    @property
    def status_step_index(self) -> int:
        """0-based position of the current status in STATUSES, for timeline rendering."""
        try:
            return STATUSES.index(self.status)
        except ValueError:
            return 0


def _col(row, name, default=None):
    """Read a column that may not exist yet on a pre-migration row/connection."""
    try:
        return row[name]
    except (IndexError, KeyError):
        return default


def _row_to_report(row) -> CitizenReport:
    return CitizenReport(
        id=row["id"], report_id=row["report_id"],
        created_at=row["created_at"], updated_at=row["updated_at"],
        image_path=row["image_path"], annotated_image_path=row["annotated_image_path"],
        latitude=row["latitude"], longitude=row["longitude"],
        location=row["location"], road_name=row["road_name"], landmark=row["landmark"],
        description=row["description"], reporter_name=row["reporter_name"],
        reporter_contact=row["reporter_contact"],
        detected_damage=row["detected_damage"],
        detections=json.loads(row["detections_json"]) if row["detections_json"] else [],
        severity_score=row["severity_score"], severity_level=row["severity_level"],
        health_score=row["health_score"], health_condition=row["health_condition"],
        repair_priority=row["repair_priority"], priority_reason=row["priority_reason"],
        status=row["status"],
        resolution_image_path=row["resolution_image_path"],
        resolution_annotated_path=row["resolution_annotated_path"],
        resolved_at=row["resolved_at"],
        comparison=json.loads(row["comparison_json"]) if row["comparison_json"] else None,
        json_report_path=row["json_report_path"], pdf_report_path=row["pdf_report_path"],
        html_report_path=row["html_report_path"],
        status_history=json.loads(row["status_history_json"]) if row["status_history_json"] else [],
        area=_col(row, "area"), city=_col(row, "city"), state=_col(row, "state"),
        segment_id=_col(row, "segment_id"), duplicate_group_id=_col(row, "duplicate_group_id"),
        citizen_id=_col(row, "citizen_id"), image_hash=_col(row, "image_hash"),
        target_date=_col(row, "target_date"), department=_col(row, "department"),
        officer_name=_col(row, "officer_name"), officer_id=_col(row, "officer_id"),
        assigned_at=_col(row, "assigned_at"), escalation_level=_col(row, "escalation_level") or 0,
        escalated_at=_col(row, "escalated_at"), overdue_flagged_at=_col(row, "overdue_flagged_at"),
        verified_at=_col(row, "verified_at"), verified_by=_col(row, "verified_by"),
        resolved_by=_col(row, "resolved_by"), resolution_notes=_col(row, "resolution_notes"),
        resolution_date=_col(row, "resolution_date"),
        citizen_followup_path=_col(row, "citizen_followup_path"), followup_at=_col(row, "followup_at"),
    )


def _new_report_id(row_id: int) -> str:
    """RS-<year>-<6 digit row id>. Deterministic from the stored row id, never random."""
    return f"RS-{time.strftime('%Y')}-{row_id:06d}"


def create_report(*, image_path: str, annotated_image_path: str,
                   latitude: Optional[float], longitude: Optional[float],
                   location: Optional[str], road_name: Optional[str], landmark: Optional[str],
                   description: Optional[str], reporter_name: Optional[str],
                   reporter_contact: Optional[str], detected_damage: str,
                   detections_summary: list, severity_score: float, severity_level: str,
                   health_score: float, health_condition: str,
                   repair_priority: str, priority_reason: str,
                   json_report_path: Optional[str] = None,
                   pdf_report_path: Optional[str] = None,
                   html_report_path: Optional[str] = None,
                   area: Optional[str] = None, city: Optional[str] = None,
                   state: Optional[str] = None, segment_id: Optional[str] = None,
                   citizen_id: Optional[int] = None, image_hash: Optional[str] = None) -> str:
    """Insert a new citizen report and return its generated report_id."""
    now = time.time()
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO citizen_reports
           (report_id, created_at, updated_at, image_path, annotated_image_path,
            latitude, longitude, location, road_name, landmark, description,
            reporter_name, reporter_contact, detected_damage, detections_json,
            severity_score, severity_level, health_score, health_condition,
            repair_priority, priority_reason, status, status_history_json,
            json_report_path, pdf_report_path, html_report_path,
            area, city, state, segment_id, citizen_id, image_hash, target_date)
           VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'REPORTED', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (now, now, image_path, annotated_image_path, latitude, longitude, location,
         road_name, landmark, description, reporter_name, reporter_contact,
         detected_damage, json.dumps(detections_summary), severity_score, severity_level,
         health_score, health_condition, repair_priority, priority_reason,
         json.dumps([{"status": "REPORTED", "at": now, "by": "Citizen", "note": None}]),
         json_report_path, pdf_report_path, html_report_path,
         area, city, state, segment_id, citizen_id, image_hash,
         now + config.SLA_DAYS * 86400),
    )
    row_id = cur.lastrowid
    report_id = _new_report_id(row_id)
    conn.execute("UPDATE citizen_reports SET report_id = ? WHERE id = ?", (report_id, row_id))
    conn.commit()
    conn.close()
    return report_id


def set_duplicate_group(report_id: str, group_id: Optional[str]) -> None:
    conn = get_conn()
    conn.execute("UPDATE citizen_reports SET duplicate_group_id = ? WHERE report_id = ?",
                 (group_id, report_id))
    conn.commit()
    conn.close()


def list_all() -> List[CitizenReport]:
    """All citizen reports, oldest first (used by area_health.py / segment.py / dedup.py)."""
    conn = get_conn()
    rows = conn.execute("SELECT * FROM citizen_reports ORDER BY created_at ASC").fetchall()
    conn.close()
    return [_row_to_report(r) for r in rows]


def list_by_segment(segment_id: str) -> List[CitizenReport]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM citizen_reports WHERE segment_id = ? ORDER BY created_at ASC", (segment_id,)
    ).fetchall()
    conn.close()
    return [_row_to_report(r) for r in rows]


def list_by_group(group_id: str) -> List[CitizenReport]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM citizen_reports WHERE duplicate_group_id = ? ORDER BY created_at ASC", (group_id,)
    ).fetchall()
    conn.close()
    return [_row_to_report(r) for r in rows]


def get_report(report_id: str) -> Optional[CitizenReport]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM citizen_reports WHERE report_id = ?", (report_id,)).fetchone()
    conn.close()
    return _row_to_report(row) if row else None


def list_reports(limit: Optional[int] = None) -> List[CitizenReport]:
    conn = get_conn()
    q = "SELECT * FROM citizen_reports ORDER BY created_at DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    rows = conn.execute(q).fetchall()
    conn.close()
    return [_row_to_report(r) for r in rows]


def list_reports_by_ids(report_ids: List[str]) -> List[CitizenReport]:
    """Legacy helper (pre-identity 'My Reports'): reports for an explicit list of IDs."""
    if not report_ids:
        return []
    conn = get_conn()
    placeholders = ",".join("?" for _ in report_ids)
    rows = conn.execute(
        f"SELECT * FROM citizen_reports WHERE report_id IN ({placeholders}) ORDER BY created_at DESC",
        report_ids,
    ).fetchall()
    conn.close()
    return [_row_to_report(r) for r in rows]


def list_by_citizen(citizen_id: int) -> List[CitizenReport]:
    """'My Reports': every report linked to one citizen identity, newest first."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM citizen_reports WHERE citizen_id = ? ORDER BY created_at DESC", (citizen_id,)
    ).fetchall()
    conn.close()
    return [_row_to_report(r) for r in rows]


def adopt_reports(report_ids: List[str], citizen_id: int) -> int:
    """Attach not-yet-owned reports (from an older session's list) to a citizen. Never steals owned ones."""
    if not report_ids:
        return 0
    conn = get_conn()
    placeholders = ",".join("?" for _ in report_ids)
    cur = conn.execute(
        f"UPDATE citizen_reports SET citizen_id = ? WHERE citizen_id IS NULL AND report_id IN ({placeholders})",
        [citizen_id, *report_ids],
    )
    conn.commit()
    n = cur.rowcount
    conn.close()
    return n


def find_by_hash(image_hash: str) -> List[CitizenReport]:
    if not image_hash:
        return []
    conn = get_conn()
    rows = conn.execute("SELECT * FROM citizen_reports WHERE image_hash = ? ORDER BY created_at ASC",
                        (image_hash,)).fetchall()
    conn.close()
    return [_row_to_report(r) for r in rows]


# Columns the workflow layer may change through update_report() - a whitelist,
# so a typo can never write to an arbitrary column.
_UPDATABLE = {
    "status", "department", "officer_name", "officer_id", "assigned_at", "escalation_level",
    "escalated_at", "overdue_flagged_at", "verified_at", "verified_by", "resolved_by",
    "resolution_notes", "resolution_date", "resolved_at", "resolution_image_path",
    "resolution_annotated_path", "comparison_json", "citizen_followup_path", "followup_at",
}


def update_report(report_id: str, fields: Optional[dict] = None,
                  history_entry: Optional[dict] = None) -> bool:
    """
    The one write path for post-submission changes. Applies `fields` and
    appends `history_entry` ({status, event?, by, note}) to the audit trail
    in a single statement. Returns False if the report does not exist.
    """
    fields = dict(fields or {})
    bad = set(fields) - _UPDATABLE
    if bad:
        raise ValueError(f"Not updatable: {sorted(bad)}")
    report = get_report(report_id)
    if not report:
        return False
    now = time.time()
    history = list(report.status_history)
    if history_entry:
        entry = {"status": fields.get("status", report.status), "at": now, "by": None, "note": None}
        entry.update(history_entry)
        history.append(entry)
    fields["updated_at"] = now
    fields["status_history_json"] = json.dumps(history)
    sets = ", ".join(f"{k} = ?" for k in fields)
    conn = get_conn()
    conn.execute(f"UPDATE citizen_reports SET {sets} WHERE report_id = ?", [*fields.values(), report_id])
    conn.commit()
    conn.close()
    return True


def update_status(report_id: str, new_status: str, by: Optional[str] = None,
                  note: Optional[str] = None) -> bool:
    if new_status not in STATUSES:
        return False
    return update_report(report_id, {"status": new_status},
                         {"status": new_status, "by": by, "note": note})


def resolve_report(report_id: str, resolution_image_path: Optional[str],
                    resolution_annotated_path: Optional[str],
                    comparison: Optional[dict], *, by: Optional[str] = None,
                    notes: Optional[str] = None, resolution_date: Optional[float] = None) -> bool:
    """Mark RESOLVED and attach the after-repair evidence + before/after comparison."""
    now = time.time()
    return update_report(report_id, {
        "status": "RESOLVED", "resolved_at": now, "resolution_image_path": resolution_image_path,
        "resolution_annotated_path": resolution_annotated_path,
        "comparison_json": json.dumps(comparison) if comparison else None,
        "resolved_by": by, "resolution_notes": notes, "resolution_date": resolution_date or now,
    }, {"status": "RESOLVED", "by": by, "note": notes})


def stats() -> Dict[str, int]:
    """Counts for dashboards. `escalation_required` = open reports past their target."""
    reports = list_all()
    now = time.time()
    counts = {s: 0 for s in STATUSES}
    for r in reports:
        counts[r.status] = counts.get(r.status, 0) + 1
    open_reports = [r for r in reports if r.status != "RESOLVED"]
    return {
        "total": len(reports),
        "open": len(open_reports),
        "high_priority": sum(1 for r in open_reports if r.repair_priority in ("HIGH", "URGENT")),
        "escalation_required": sum(1 for r in open_reports if r.is_overdue(now)),
        "resolved": counts["RESOLVED"],
        **counts,
    }
