"""
area_health.py
---------------
MODULE: Area Health Scorecard (aggregation layer, NOT a new ML model)
INPUT:  every stored Inspect-Road assessment (history.py) and every
        citizen report (citizen_report.py) that has an `area`/`city`
        recorded
OUTPUT: for a given area, a scorecard: Area Health Score, condition band,
        number of inspected road segments, damage-type distribution,
        condition distribution, priority/status counts, and (if enough
        historical data exists) a month-by-month trend
WHY:    turns many individual road observations into one honest,
        transparent area-level picture, WITHOUT pretending a single photo
        represents an entire area.
ALGORITHM: Area Health Score = mean of the most-recent health score for
        each distinct road segment observed in that area (see
        `get_area_scorecard` docstring for exactly what counts as an
        "observation" and why). This is a clearly documented aggregation
        of real stored data - not a prediction, not a new model, and never
        fabricated when data is insufficient (`insufficient_data=True` is
        returned instead of a number).
LIMITATIONS: prototype-grade grouping. "Area" is whatever free text (or
        reverse-geocoded label) the reporter/inspector entered, not a
        verified administrative boundary. Different phrasing for the same
        real-world area is grouped only if it slugifies to the same key
        (see geo.area_key) - this is documented, not hidden.
INTERACTION: read by app.py's /area-health routes and by the compact
        "Your Area" widget on the dashboard (index.html).
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional

import geo
import history
import citizen_report as creport

MIN_SEGMENTS_FOR_SCORE = 1  # a single inspected segment is enough to show *something*,
                             # but the UI must always say how many segments it's based on


def _condition_from_score(score: float) -> str:
    from health_score import condition_from_score
    return condition_from_score(score)


def _all_located_observations():
    """
    Every (area_key, display_label, road_key, health_score, condition,
    priority, status, damage_class_counts, created_at, source) tuple drawn
    from BOTH the Inspect-Road history and citizen reports - never
    fabricated, always traceable back to a real stored row.
    """
    obs = []
    for rec in history.list_all():
        ak = geo.area_key(rec.area, rec.city)
        if not ak:
            continue
        road_key = rec.segment_id or geo.slugify(rec.road_name or rec.location)
        damage_counts = defaultdict(int)
        for d in (rec.detections or []):
            damage_counts[d.get("class_name") or d.get("class_id") or "Unknown"] += 1
        obs.append({
            "area_key": ak, "label": geo.display_area_label(rec.area, rec.city),
            "road_key": road_key, "health_score": rec.health_score, "condition": rec.condition,
            "priority": rec.priority, "status": None, "damage_counts": dict(damage_counts),
            "created_at": rec.created_at, "source": "inspection",
        })
    for rep in creport.list_all():
        ak = geo.area_key(rep.area, rep.city)
        if not ak:
            continue
        if rep.health_score is None:
            continue
        road_key = rep.segment_id or geo.slugify(rep.road_name or rep.location)
        damage_counts = defaultdict(int)
        for d in (rep.detections or []):
            damage_counts[d.get("class_name") or d.get("class_id") or "Unknown"] += 1
        obs.append({
            "area_key": ak, "label": geo.display_area_label(rep.area, rep.city),
            "road_key": road_key, "health_score": rep.health_score, "condition": rep.health_condition,
            "priority": rep.repair_priority, "status": rep.status, "damage_counts": dict(damage_counts),
            "created_at": rep.created_at, "source": "citizen_report",
        })
    return obs


def list_areas() -> List[Dict]:
    """
    Summary row per known area: label, number of distinct road segments
    observed, number of citizen reports, latest score. Only areas with at
    least one real observation are listed - nothing is invented.
    """
    obs = _all_located_observations()
    by_area = defaultdict(list)
    for o in obs:
        by_area[o["area_key"]].append(o)

    rows = []
    for area_key, items in by_area.items():
        label = items[-1]["label"]
        segments = {i["road_key"] for i in items if i["road_key"]}
        n_reports = sum(1 for i in items if i["source"] == "citizen_report")
        latest_by_segment = _latest_per_segment(items)
        score = round(sum(latest_by_segment.values()) / len(latest_by_segment), 1) if latest_by_segment else None
        high_priority = sum(1 for i in items if (i["priority"] or "").upper() in ("HIGH", "URGENT"))
        rows.append({
            "area_key": area_key, "label": label,
            "segment_count": len(segments), "report_count": n_reports,
            "high_priority": high_priority,
            "score": score, "condition": _condition_from_score(score) if score is not None else None,
        })
    rows.sort(key=lambda r: r["label"])
    return rows


def _latest_per_segment(items: List[dict]) -> Dict[str, float]:
    """The most recent health score observed for each distinct road segment/key in `items`."""
    latest = {}
    for i in sorted(items, key=lambda x: x["created_at"]):
        key = i["road_key"] or f"__unkeyed__{i['created_at']}"
        latest[key] = i["health_score"]
    return latest


def get_area_scorecard(area_key: str) -> Optional[Dict]:
    """
    Full scorecard for one area, or None if the area key is entirely
    unknown (no stored observation matches it at all).

    Area Health Score = mean of health scores, one per distinct road
    segment in the area (the LATEST inspection/report for that segment),
    so a segment reported five times doesn't get weighted 5x versus a
    segment reported once. This is the "clearly documented aggregation
    method (mean of valid road-segment health scores)" the spec calls for.
    """
    obs = [o for o in _all_located_observations() if o["area_key"] == area_key]
    if not obs:
        return None

    label = obs[-1]["label"]
    latest_by_segment = _latest_per_segment(obs)
    segment_count = len(latest_by_segment)

    insufficient = segment_count < MIN_SEGMENTS_FOR_SCORE
    score = round(sum(latest_by_segment.values()) / segment_count, 1) if not insufficient else None
    condition = _condition_from_score(score) if score is not None else None

    reports = [o for o in obs if o["source"] == "citizen_report"]
    # Severity-based counts for the citizen "Area view" (real reports only).
    area_reports = [r for r in creport.list_all() if geo.area_key(r.area, r.city) == area_key]
    open_reports = sum(1 for r in area_reports if r.status != "RESOLVED")
    high_critical = sum(1 for r in area_reports if r.severity_level in ("HIGH", "CRITICAL"))
    scored = [r.health_score for r in area_reports if r.health_score is not None]
    avg_report_health = round(sum(scored) / len(scored), 1) if scored else None
    priority_counts = {"HIGH": 0, "URGENT": 0, "MEDIUM": 0, "LOW": 0}
    for o in obs:
        p = (o["priority"] or "").upper()
        if p in priority_counts:
            priority_counts[p] += 1
    high_priority = priority_counts["HIGH"] + priority_counts["URGENT"]

    resolved = sum(1 for r in reports if r["status"] == "RESOLVED")
    pending = sum(1 for r in reports if r["status"] and r["status"] != "RESOLVED")

    # damage-type distribution, across the latest observation per segment only
    # (mirrors the score methodology: one vote per segment, not per re-visit)
    latest_obs_by_segment = {}
    for o in sorted(obs, key=lambda x: x["created_at"]):
        key = o["road_key"] or f"__unkeyed__{o['created_at']}"
        latest_obs_by_segment[key] = o
    damage_totals = defaultdict(int)
    for o in latest_obs_by_segment.values():
        for cls, n in o["damage_counts"].items():
            damage_totals[cls] += n
    total_damage = sum(damage_totals.values())
    damage_distribution = {
        cls: round(100.0 * n / total_damage, 1) for cls, n in damage_totals.items()
    } if total_damage else {}

    condition_counts = defaultdict(int)
    for o in latest_obs_by_segment.values():
        if o["condition"]:
            condition_counts[o["condition"]] += 1
    n_conditions = sum(condition_counts.values())
    condition_distribution = {
        c: round(100.0 * n / n_conditions, 1) for c, n in condition_counts.items()
    } if n_conditions else {}

    # trend uses EVERY observation (not just latest-per-segment), so a
    # segment revisited across months actually shows the area's trajectory
    trend = _monthly_trend(obs)

    return {
        "area_key": area_key, "label": label,
        "score": score, "condition": condition, "insufficient_data": insufficient,
        "segment_count": segment_count, "report_count": len(reports),
        "high_priority": high_priority, "medium_priority": priority_counts["MEDIUM"],
        "low_priority": priority_counts["LOW"],
        "resolved": resolved, "pending": pending,
        "open_reports": open_reports, "high_critical": high_critical,
        "avg_report_health": avg_report_health,
        "damage_distribution": damage_distribution,
        "condition_distribution": condition_distribution,
        "trend": trend,
    }


def _monthly_trend(observations) -> Optional[Dict]:
    """
    Average health score per calendar month, across all observations
    (every re-visit counts here, since the question is "how did this
    month look", not "what's the segment's current score"). Only
    returned if there are at least 2 distinct months of data - otherwise
    None (never fabricate a trend from one data point).
    """
    by_month = defaultdict(list)
    for o in observations:
        month_key = time.strftime("%Y-%m", time.localtime(o["created_at"]))
        by_month[month_key].append(o["health_score"])
    if len(by_month) < 2:
        return None
    months = sorted(by_month.keys())
    series = [{"month": time.strftime("%b", time.strptime(m, "%Y-%m")),
               "month_key": m, "score": round(sum(by_month[m]) / len(by_month[m]), 1)}
              for m in months]
    delta = series[-1]["score"] - series[0]["score"]
    if delta >= 5:
        direction = "improving"
    elif delta <= -5:
        direction = "declining"
    else:
        direction = "relatively stable"
    return {"series": series, "direction": direction, "delta": round(delta, 1)}
