"""
segment.py
----------
MODULE: Road Segment Health History
INPUT:  every stored Inspect-Road assessment (history.py) and citizen
        report (citizen_report.py) that carries the same `segment_id`
OUTPUT: a Road Segment Profile - road/area label, current health +
        priority, total inspections, first/last inspected dates, a damage
        summary, and the full ordered health-score history for that
        segment
WHY:    the four-stage detection pipeline (detector -> severity -> health
        -> priority) is per-photo; this module is what turns repeated
        per-photo results for the SAME physical road segment into a
        history a maintenance team can act on ("this segment has gone
        78 -> 71 -> 63 -> 54, it's getting worse").
ALGORITHM: `segment_id` is computed once, at submission time, by
        `geo.make_segment_id` (road_name/location + area/city, slugified -
        see geo.py docstring for why this is a deliberately simple,
        clearly-labeled prototype identity, not real GIS segmentation).
        This module just groups by that id and sorts by time; it invents
        nothing.
LIMITATIONS: two real-world segments that happen to slugify to the same
        text (e.g. two different cities each with a "Main Street" and no
        area/city given) would incorrectly merge - this is a named
        limitation of the prototype identifier, not a bug in the
        aggregation logic itself.
INTERACTION: read by app.py's /road/<segment_id> route.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import history
import citizen_report as creport


def list_segments() -> List[Dict]:
    """One summary row per known segment_id, newest-activity first."""
    by_segment = defaultdict(list)
    for rec in history.list_all():
        if rec.segment_id:
            by_segment[rec.segment_id].append({
                "created_at": rec.created_at, "health_score": rec.health_score,
                "road_name": rec.road_name, "area": rec.area, "city": rec.city,
                "location": rec.location, "source": "inspection",
            })
    for rep in creport.list_all():
        if rep.segment_id and rep.health_score is not None:
            by_segment[rep.segment_id].append({
                "created_at": rep.created_at, "health_score": rep.health_score,
                "road_name": None,
                "area": rep.area, "city": rep.city, "location": rep.location,
                "source": "citizen_report",
            })

    rows = []
    for seg_id, items in by_segment.items():
        items.sort(key=lambda x: x["created_at"])
        latest = items[-1]
        rows.append({
            "segment_id": seg_id,
            "road_label": latest.get("road_name") or latest.get("location") or seg_id,
            "area_label": latest.get("area") or latest.get("city") or "",
            "inspections": len(items),
            "current_health": latest["health_score"],
            "last_inspected": latest["created_at"],
        })
    rows.sort(key=lambda r: r["last_inspected"], reverse=True)
    return rows


def get_segment_profile(segment_id: str) -> Optional[Dict]:
    events = []

    for rec in history.get_by_segment(segment_id):
        damage_counts = defaultdict(int)
        for d in (rec.detections or []):
            damage_counts[d.get("class_name") or d.get("class_id") or "Unknown"] += 1
        events.append({
            "created_at": rec.created_at, "health_score": rec.health_score,
            "condition": rec.condition, "priority": rec.priority,
            "road_name": rec.road_name, "area": rec.area, "city": rec.city,
            "location": rec.location, "damage_counts": dict(damage_counts),
            "source": "inspection", "source_label": "Inspect Road",
            "detail_url": None,
        })

    for rep in creport.list_by_segment(segment_id):
        if rep.health_score is None:
            continue
        damage_counts = defaultdict(int)
        for d in (rep.detections or []):
            damage_counts[d.get("class_name") or d.get("class_id") or "Unknown"] += 1
        events.append({
            "created_at": rep.created_at, "health_score": rep.health_score,
            "condition": rep.health_condition, "priority": rep.repair_priority,
            "road_name": None, "area": rep.area, "city": rep.city,
            "location": rep.location, "damage_counts": dict(damage_counts),
            "source": "citizen_report", "source_label": f"Citizen report {rep.report_id}",
            "detail_url": rep.report_id,
        })

    if not events:
        return None

    events.sort(key=lambda e: e["created_at"])
    latest = events[-1]
    first = events[0]

    road_label = next((e["road_name"] for e in reversed(events) if e["road_name"]), None) \
        or next((e["location"] for e in reversed(events) if e["location"]), None) or segment_id
    area_label = next((e["area"] or e["city"] for e in reversed(events) if (e["area"] or e["city"])), "")

    damage_summary = defaultdict(int)
    for e in events:
        for cls, n in e["damage_counts"].items():
            damage_summary[cls] += n

    return {
        "segment_id": segment_id,
        "road_label": road_label, "area_label": area_label,
        "current_health": latest["health_score"], "current_condition": latest["condition"],
        "current_priority": latest["priority"],
        "total_inspections": len(events),
        "first_inspected": first["created_at"], "last_inspected": latest["created_at"],
        "damage_summary": dict(damage_summary),
        "history": events,
        "health_trend": [e["health_score"] for e in events],
    }
