"""
rewards.py
----------
MODULE: Citizen rewards, levels/badges and the prototype certificate
INPUT:  events that really happened (report submitted, verified, resolved,
        repair follow-up) - fired from workflow.py
OUTPUT: a reward ledger (`reward_events`), point totals, level/badge
        progress, contribution history and a downloadable PDF certificate
WHY:    Encourage genuine reporting without letting anyone farm points.
ALGORITHM: points are NEVER stored as a running total - they are always
        SUM(points) over the ledger, and the ledger has a UNIQUE(report_id,
        event_type) key, so the same report can earn each reward only once
        no matter how many times a code path runs. Point values, badge
        thresholds and the certificate threshold all come from config.py.
LIMITATIONS: badges and the certificate are RoadScan recognition only - not
        government-issued. Abuse protection is basic (unique photo, daily cap
        on rewarded reports, one reward per event) and would need review
        moderation in a real deployment.
"""

from __future__ import annotations

import hashlib
import io
import time
from typing import Dict, List, Optional

import citizen_report as creport
import config


# ---------------------------------------------------------------- ledger --

def award(citizen_id: Optional[int], report_id: str, event_type: str, note: Optional[str] = None) -> int:
    """Award points once per (report, event). Returns points added (0 if not awarded)."""
    if not citizen_id or event_type not in config.POINTS:
        return 0
    conn = creport.get_conn()
    cur = conn.execute(
        "INSERT OR IGNORE INTO reward_events (citizen_id, report_id, event_type, points, note, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (citizen_id, report_id, event_type, config.POINTS[event_type], note, time.time()),
    )
    conn.commit()
    added = config.POINTS[event_type] if cur.rowcount else 0
    conn.close()
    return added


def has_award(report_id: str, event_type: str) -> bool:
    conn = creport.get_conn()
    r = conn.execute("SELECT 1 FROM reward_events WHERE report_id = ? AND event_type = ?",
                     (report_id, event_type)).fetchone()
    conn.close()
    return r is not None


def total_points(citizen_id: Optional[int]) -> int:
    if not citizen_id:
        return 0
    conn = creport.get_conn()
    r = conn.execute("SELECT COALESCE(SUM(points), 0) AS p FROM reward_events WHERE citizen_id = ?",
                     (citizen_id,)).fetchone()
    conn.close()
    return int(r["p"])


def rewarded_today(citizen_id: int) -> int:
    lt = time.localtime()
    midnight = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
    conn = creport.get_conn()
    r = conn.execute("SELECT COUNT(*) AS c FROM reward_events WHERE citizen_id = ? AND event_type = ? "
                     "AND created_at >= ?", (citizen_id, "REPORT_SUBMITTED", midnight)).fetchone()
    conn.close()
    return int(r["c"])


def history(citizen_id: Optional[int], limit: int = 50) -> List[dict]:
    if not citizen_id:
        return []
    conn = creport.get_conn()
    rows = conn.execute(
        "SELECT e.*, r.location AS location FROM reward_events e "
        "LEFT JOIN citizen_reports r ON r.report_id = e.report_id "
        "WHERE e.citizen_id = ? ORDER BY e.created_at DESC, e.id DESC LIMIT ?", (citizen_id, limit)
    ).fetchall()
    conn.close()
    return [{**dict(r), "label": config.POINT_LABELS.get(r["event_type"], r["event_type"])} for r in rows]


# ---------------------------------------------------------- levels/badges --

def level_for(points: int) -> Dict:
    """Current badge, next badge and progress toward it (extensible via config.BADGES)."""
    badges = sorted(config.BADGES, key=lambda b: b["min_points"])
    current = badges[0]
    for b in badges:
        if points >= b["min_points"]:
            current = b
    nxt = next((b for b in badges if b["min_points"] > points), None)
    if nxt:
        span = nxt["min_points"] - current["min_points"]
        pct = round(100 * (points - current["min_points"]) / span) if span else 100
    else:
        pct = 100
    return {"points": points, "current": current, "next": nxt, "progress_pct": max(0, min(100, pct)),
            "points_to_next": (nxt["min_points"] - points) if nxt else 0,
            "earned": [b for b in badges if points >= b["min_points"]]}


def citizen_stats(citizen_id: Optional[int]) -> Dict:
    reports = creport.list_by_citizen(citizen_id) if citizen_id else []
    return {
        "submitted": len(reports),
        "verified": sum(1 for r in reports if r.is_verified),
        "resolved": sum(1 for r in reports if r.status == "RESOLVED"),
    }


def profile_summary(citizen_id: Optional[int]) -> Dict:
    pts = total_points(citizen_id)
    return {"level": level_for(pts), "stats": citizen_stats(citizen_id), "points": pts}


# ------------------------------------------------------------ certificate --

def certificate_min_points() -> int:
    return next(b["min_points"] for b in config.BADGES if b["key"] == config.CERTIFICATE_BADGE_KEY)


def certificate_eligible(points: int) -> bool:
    return points >= certificate_min_points()


def _verification_id(citizen_id: int, points: int, issued_at: float) -> str:
    digest = hashlib.sha1(f"roadscan|{citizen_id}|{points}|{int(issued_at)}".encode()).hexdigest()
    return "RSC-" + digest[:10].upper()


def issue_certificate(citizen) -> Optional[dict]:
    """
    Create (or re-use) a certificate record for an eligible citizen with a
    name. Re-issued only if their points or counts changed since the last one.
    """
    pts = total_points(citizen.id)
    if not certificate_eligible(pts) or not citizen.name:
        return None
    stats = citizen_stats(citizen.id)
    badge = level_for(pts)["current"]["name"]
    conn = creport.get_conn()
    last = conn.execute("SELECT * FROM certificates WHERE citizen_id = ? ORDER BY issued_at DESC LIMIT 1",
                        (citizen.id,)).fetchone()
    if last and (last["points"], last["verified_reports"], last["resolved_reports"], last["citizen_name"]) == \
            (pts, stats["verified"], stats["resolved"], citizen.name):
        conn.close()
        return dict(last)
    now = time.time()
    vid = _verification_id(citizen.id, pts, now)
    conn.execute(
        "INSERT INTO certificates (verification_id, citizen_id, citizen_name, badge, points, "
        "verified_reports, resolved_reports, issued_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (vid, citizen.id, citizen.name, badge, pts, stats["verified"], stats["resolved"], now))
    conn.commit()
    row = conn.execute("SELECT * FROM certificates WHERE verification_id = ?", (vid,)).fetchone()
    conn.close()
    return dict(row)


def get_certificate(verification_id: str) -> Optional[dict]:
    conn = creport.get_conn()
    row = conn.execute("SELECT * FROM certificates WHERE verification_id = ?", (verification_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def certificate_pdf(cert: dict) -> bytes:
    """One-page landscape A4 certificate (reportlab, already a project dependency)."""
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.colors import HexColor
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    w, h = landscape(A4)
    c = canvas.Canvas(buf, pagesize=landscape(A4))
    navy, yellow, grey = HexColor("#12324A"), HexColor("#F2B418"), HexColor("#5B6B79")

    c.setStrokeColor(navy); c.setLineWidth(3); c.rect(24, 24, w - 48, h - 48)
    c.setStrokeColor(yellow); c.setLineWidth(1.5); c.rect(34, 34, w - 68, h - 68)

    c.setFillColor(navy); c.setFont("Helvetica-Bold", 14); c.drawCentredString(w / 2, h - 80, "ROADSCAN")
    c.setFont("Helvetica-Bold", 26); c.drawCentredString(w / 2, h - 122, config.CERTIFICATE_TITLE)
    c.setFillColor(grey); c.setFont("Helvetica", 12)
    c.drawCentredString(w / 2, h - 158, "is proudly presented to")
    c.setFillColor(navy); c.setFont("Helvetica-Bold", 34)
    c.drawCentredString(w / 2, h - 210, cert["citizen_name"])
    c.setFillColor(grey); c.setFont("Helvetica", 12)
    c.drawCentredString(w / 2, h - 245, "for active contribution to safer roads, recognized with the badge")
    c.setFillColor(HexColor("#B8860B")); c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(w / 2, h - 282, cert["badge"].upper())

    c.setFillColor(navy); c.setFont("Helvetica", 13)
    y = h - 335
    for label, val in (("Points earned", f"{cert['points']:,}"),
                       ("Verified reports", str(cert["verified_reports"])),
                       ("Resolved reports", str(cert["resolved_reports"])),
                       ("Issued on", time.strftime("%d %b %Y", time.localtime(cert["issued_at"])))):
        c.setFont("Helvetica", 12); c.setFillColor(grey); c.drawRightString(w / 2 - 10, y, label + ":")
        c.setFont("Helvetica-Bold", 13); c.setFillColor(navy); c.drawString(w / 2 + 10, y, val)
        y -= 22

    c.setFillColor(navy); c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(w / 2, 92, f"RoadScan verification ID: {cert['verification_id']}")
    c.setFillColor(grey); c.setFont("Helvetica-Oblique", 9)
    c.drawCentredString(w / 2, 70, config.CERTIFICATE_NOTE)
    c.showPage(); c.save()
    return buf.getvalue()
