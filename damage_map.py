"""
damage_map.py
-------------
MODULE: Road Damage Heatmap (data layer)
INPUT:  every stored Inspect-Road assessment and citizen report that has a
        valid latitude/longitude pair
OUTPUT: a flat list of map points, each with everything the /damage-map
        page's popup needs (id, area, road, damage type, severity, health,
        priority) plus a `color_class` bucket for the marker
WHY:    keeps the "only plot what's real" rule (never invented coordinates)
        and the marker-coloring rule in ONE place, shared by both the
        Damage Map page and (potentially) any future map surface, instead
        of duplicating filtering logic in app.py.
ALGORITHM: no clustering/heatmap math here - this is a point layer.
        Marker color bucket is derived straight from the existing
        `repair_priority` (LOW/MEDIUM/HIGH/URGENT), reusing severity.py's
        vocabulary rather than inventing a second severity scale for maps.
LIMITATIONS: a plain lightweight lat/long scatter, not a tiled real-world
        map (see templates/damage_map.html for why, and what would change
        to swap in a real tile-based map later without touching this data
        layer).
INTERACTION: read by app.py's /damage-map route.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import config
import geo
import history
import citizen_report as creport

PRIORITY_COLOR = {
    "LOW": "green",
    "MEDIUM": "yellow",
    "HIGH": "orange",
    "URGENT": "red",
}


def get_map_points() -> List[Dict]:
    points = []

    for rec in history.list_all():
        if rec.latitude is None or rec.longitude is None:
            continue
        damage_types = sorted({d.get("class_name", "Damage") for d in (rec.detections or [])}) or ["No damage detected"]
        points.append({
            "id": f"insp-{rec.id}", "kind": "inspection",
            "lat": rec.latitude, "lon": rec.longitude,
            "area": rec.area, "city": rec.city, "road": rec.road_name or rec.location,
            "damage_type": ", ".join(damage_types),
            "severity": None, "health_score": rec.health_score,
            "priority": rec.priority, "color": PRIORITY_COLOR.get(rec.priority, "yellow"),
            "detail_url": None,
        })

    for rep in creport.list_all():
        if rep.latitude is None or rep.longitude is None:
            continue
        damage_types = sorted({d.get("class_name", "Damage") for d in (rep.detections or [])}) or [rep.detected_damage or "Reported damage"]
        points.append({
            "id": rep.report_id, "kind": "citizen_report",
            "lat": rep.latitude, "lon": rep.longitude,
            "area": rep.area, "city": rep.city, "road": rep.road_name or rep.location,
            "damage_type": ", ".join(damage_types),
            "severity": rep.severity_level, "health_score": rep.health_score,
            "priority": rep.repair_priority, "color": PRIORITY_COLOR.get(rep.repair_priority, "yellow"),
            "detail_url": rep.report_id,
        })

    return points


# ---------------------------------------------------------------------------
# Citizen-facing Road Map (Leaflet) - severity-coloured, filterable points.
# Same rule as above: ONLY stored coordinates are ever returned.
# ---------------------------------------------------------------------------

def _report_damage_types(rep) -> List[str]:
    names = sorted({d.get("class_name", "Damage") for d in (rep.detections or [])})
    return names or [rep.detected_damage or "Reported damage"]


def _matches_text(q: str, *parts) -> bool:
    hay = " ".join(p for p in parts if p).lower()
    return all(tok in hay for tok in q.lower().split())


def get_report_points(*, severity: Optional[str] = None, status: Optional[str] = None,
                      damage: Optional[str] = None, days: Optional[int] = None,
                      q: Optional[str] = None, include_inspections: bool = False) -> List[Dict]:
    """
    Points for the Road Map. Filters (all optional):
      severity: LOW|MODERATE|HIGH|CRITICAL      status: any creport.ALL_STATUSES key
      damage:   a damage class name (substring)  days:   only reports newer than N days
      q:        free-text area/city/road search  (every word must match)
    Status filtering uses the *displayed* status, so ESCALATION_REQUIRED works.
    """
    now = time.time()
    cutoff = now - days * 86400 if days else None
    sev = (severity or "").upper() or None
    st = (status or "").upper() or None
    dmg = (damage or "").strip().lower() or None
    points = []

    for rep in creport.list_all():
        if rep.latitude is None or rep.longitude is None:
            continue
        level = rep.severity_level if rep.severity_level in config.SEVERITY_COLORS else "NONE"
        types = _report_damage_types(rep)
        if sev and level != sev:
            continue
        if st and rep.display_status != st:
            continue
        if dmg and not any(dmg in t.lower() for t in types):
            continue
        if cutoff and rep.created_at < cutoff:
            continue
        if q and not _matches_text(q, rep.area, rep.city, rep.state, rep.road_name, rep.location,
                                   rep.landmark, rep.report_id):
            continue
        points.append({
            "id": rep.report_id, "kind": "report", "lat": rep.latitude, "lon": rep.longitude,
            "severity": level, "color": config.SEVERITY_COLORS[level],
            "damage": ", ".join(types), "health": rep.health_score,
            "status": rep.display_status, "status_label": rep.display_status_label,
            "location": rep.location or geo.display_area_label(rep.area, rep.city),
            "area": rep.area, "city": rep.city, "state": rep.state,
            "area_key": geo.area_key(rep.area, rep.city),
            "created": time.strftime("%d %b %Y", time.localtime(rep.created_at)),
            "created_ts": rep.created_at, "url": f"/reports/{rep.report_id}",
        })

    if include_inspections and not (st or sev or dmg):
        for rec in history.list_all():
            if rec.latitude is None or rec.longitude is None:
                continue
            if cutoff and rec.created_at < cutoff:
                continue
            if q and not _matches_text(q, rec.area, rec.city, rec.state, rec.road_name, rec.location):
                continue
            points.append({
                "id": f"insp-{rec.id}", "kind": "inspection", "lat": rec.latitude, "lon": rec.longitude,
                "severity": "NONE", "color": config.SEVERITY_COLORS["NONE"],
                "damage": ", ".join(sorted({d.get("class_name", "Damage") for d in rec.detections})) or "No damage detected",
                "health": rec.health_score, "status": "INSPECTION", "status_label": "Inspection record",
                "location": rec.location, "area": rec.area, "city": rec.city, "state": rec.state,
                "area_key": geo.area_key(rec.area, rec.city),
                "created": time.strftime("%d %b %Y", time.localtime(rec.created_at)),
                "created_ts": rec.created_at, "url": None,
            })
    return points
