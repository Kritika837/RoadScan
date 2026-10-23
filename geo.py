"""
geo.py
------
MODULE: Location Utilities (GPS + Area/Segment identity)
INPUT:  raw latitude/longitude and/or free-text location fields (area, city,
        state, road name, landmark) coming from either the Inspect Road form
        or the citizen Report Damage form
OUTPUT: (a) a great-circle distance between two GPS points, used by
        dedup.py for "possible duplicate" grouping and by area_health.py /
        segment.py for proximity-based lookups;
        (b) a deterministic, human-readable "segment id" that stands in for
        a real GIS road-segment identifier in this prototype;
        (c) best-effort reverse geocoding (GPS -> area/city/state), which is
        entirely optional and NEVER required for the app to function.
WHY:    RoadScan's core detection pipeline has no concept of "where" - this
        module is the thin, dependency-light layer that turns a GPS point
        or a set of free-text fields into a stable identity RoadScan can
        group repeat observations by (an area, or a road segment), without
        pulling in a GIS stack.
ALGORITHM:
    - `haversine_m`: standard haversine great-circle distance formula.
    - `slugify`: lowercase, alnum-only, hyphen-joined token id.
    - `make_segment_id`: road_name (or location) + area (or city), slugged.
      Documented, deterministic "prototype road-segment grouping method" -
      see README/PART 4 of the enhancement spec. NOT real road-network
      segmentation.
    - `reverse_geocode`: calls the free OpenStreetMap Nominatim API with a
      short timeout, IF the `requests` package is available and the network
      call succeeds. Any failure (no internet, package missing, timeout,
      bad response) is caught and turns into `None` - callers must treat
      `None` as "no automatic estimate available" and fall back to manual
      area/city/state entry. This function is never called anywhere the
      app depends on its result.
LIMITATIONS: this is NOT a GIS system. "Segments" and "areas" are text
    identities derived from what the user typed (or a coordinate-derived
    estimate), not verified cadastral road segments. Reverse geocoding is
    best-effort and optional.
INTERACTION: used by app.py (both /analyze and /report/submit), by
    area_health.py (area grouping), by segment.py (segment grouping), and
    by dedup.py (GPS-proximity evidence for duplicate citizen reports).
"""

from __future__ import annotations

import math
import re
from typing import Optional, Dict


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two GPS points, in meters."""
    R = 6371000.0  # Earth radius, meters
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def slugify(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def make_segment_id(road_name: Optional[str], area: Optional[str], city: Optional[str],
                     fallback_location: Optional[str] = None) -> Optional[str]:
    """
    Deterministic prototype road-segment identity: slug(road_name or
    fallback_location) + "__" + slug(area or city).

    Returns None if there isn't enough information to identify anything
    (no road name, no fallback location, AND no area/city) - we never
    invent a segment id from nothing.
    """
    road_part = slugify(road_name) or slugify(fallback_location)
    area_part = slugify(area) or slugify(city)
    if not road_part and not area_part:
        return None
    if not area_part:
        return road_part
    if not road_part:
        return area_part
    return f"{road_part}__{area_part}"


def area_key(area: Optional[str], city: Optional[str]) -> Optional[str]:
    """Grouping key for the Area Health Scorecard - area+city, or city alone."""
    a, c = slugify(area), slugify(city)
    if not a and not c:
        return None
    if a and c:
        return f"{a}__{c}"
    return a or c


def display_area_label(area: Optional[str], city: Optional[str]) -> str:
    parts = [p for p in (area, city) if p]
    return ", ".join(parts) if parts else "(unspecified area)"


def reverse_geocode(lat: float, lon: float) -> Optional[Dict[str, str]]:
    """
    Best-effort GPS -> {area, city, state} lookup via OpenStreetMap's free
    Nominatim API. Returns None on ANY failure (no internet, timeout,
    library missing, unexpected response shape) - this is optional
    enrichment, never a dependency of the core workflow.
    """
    try:
        import requests  # local import: app must run fine even without it
    except ImportError:
        return None

    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 16},
            headers={"User-Agent": "RoadScan-Prototype/1.0 (college project)"},
            timeout=3.0,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        address = data.get("address", {}) if isinstance(data, dict) else {}
        area = (address.get("suburb") or address.get("neighbourhood") or
                address.get("residential") or address.get("quarter") or
                address.get("road"))
        city = (address.get("city") or address.get("town") or address.get("village") or
                address.get("county"))
        state = address.get("state")
        if not any([area, city, state]):
            return None
        return {
            "area": area or "",
            "city": city or "",
            "state": state or "",
        }
    except Exception:
        return None
