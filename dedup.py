"""
dedup.py
--------
MODULE: Duplicate-Damage Detection
INPUT:  detections from two uploads of the same declared location
OUTPUT: for each new detection, either a matching prior `DamageSignature`
        (likely the same physical damage) or None (treated as new)
WHY:    avoids double-counting the same crack/pothole as "new damage" every
        time the same spot is re-photographed
ALGORITHM: perceptual hash (dHash, 64-bit) of each detection's cropped
        region + its relative position in the frame; a match requires both
        the hash (Hamming distance <= HAMMING_THRESHOLD) and position
        (<= POSITION_THRESHOLD of the image diagonal) to agree.
LIMITATIONS: conservative by design - only merges on strong combined
        evidence; different camera angle/zoom between uploads will likely
        cause real repeats to be (safely) counted as new rather than merged.
INTERACTION: called from `app.py` right after a repeat-location upload,
        using the previous assessment's stored image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from detector import Detection

HASH_SIZE = 8  # 8x8 -> 64-bit dHash
HAMMING_THRESHOLD = 10   # out of 64 bits; lower = stricter match
POSITION_THRESHOLD = 0.12  # fraction of image diagonal


def dhash(crop_bgr: np.ndarray, hash_size: int = HASH_SIZE) -> int:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = 0
    for i, v in enumerate(diff.flatten()):
        if v:
            bits |= (1 << i)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass
class DamageSignature:
    class_id: str
    hash_value: int
    rel_center: tuple  # (cx/w, cy/h) at time of capture
    source_image: str
    detection_index: int


def compute_signatures(image_bgr: np.ndarray, detections: List[Detection], source_image: str) -> List[DamageSignature]:
    h, w = image_bgr.shape[:2]
    sigs = []
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d.bbox
        crop = image_bgr[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
        if crop.size == 0:
            continue
        hv = dhash(crop)
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        sigs.append(DamageSignature(
            class_id=d.class_id, hash_value=hv, rel_center=(cx, cy),
            source_image=source_image, detection_index=i,
        ))
    return sigs


def match_against_history(new_sigs: List[DamageSignature], known_sigs: List[DamageSignature]) -> List[Optional[DamageSignature]]:
    """For each new signature, return the best matching known signature, or None if it's new damage."""
    matches: List[Optional[DamageSignature]] = []
    for ns in new_sigs:
        best = None
        best_dist = None
        for ks in known_sigs:
            if ks.class_id != ns.class_id:
                continue
            pos_dist = ((ns.rel_center[0] - ks.rel_center[0]) ** 2 + (ns.rel_center[1] - ks.rel_center[1]) ** 2) ** 0.5
            if pos_dist > POSITION_THRESHOLD:
                continue
            hd = hamming_distance(ns.hash_value, ks.hash_value)
            if hd <= HAMMING_THRESHOLD and (best_dist is None or hd < best_dist):
                best, best_dist = ks, hd
        matches.append(best)
    return matches


def deduplicate_counts(detections: List[Detection], matches: List[Optional[DamageSignature]]):
    """Split detections into (new_damages, likely_repeat_damages) based on match results."""
    new_d, repeat_d = [], []
    for det, m in zip(detections, matches):
        (repeat_d if m is not None else new_d).append(det)
    return new_d, repeat_d


# ---------------------------------------------------------------------------
# PART 3 ADDITION: Duplicate Complaint Grouping (citizen reports)
#
# This extends the SAME module rather than creating a second, unrelated
# duplicate-detection system, per the enhancement spec. It reuses the same
# "conservative, evidence-based, never auto-delete" philosophy as the
# per-image dHash matching above, but works across DIFFERENT citizens'
# reports using the evidence actually available for a citizen report:
# GPS proximity, damage-class match, and free-text location/road-name
# similarity (via stdlib difflib - no new dependency). It does NOT reuse
# dHash image matching, because two different citizens' photos of the same
# pothole are taken from different angles/distances/lighting, which is
# exactly the case the dHash matcher is deliberately conservative about
# (see module LIMITATIONS above) - GPS + class + text evidence is the
# reliable signal here instead.
#
# Duplicate reports are only ever GROUPED (via `duplicate_group_id`),
# never merged or deleted. Group membership is recomputed from scratch
# each time `recompute_duplicate_groups()` runs (cheap at prototype scale),
# so it always reflects the current set of reports; existing Group IDs are
# preserved across recomputes so links/bookmarks to a group stay valid.
# ---------------------------------------------------------------------------

import difflib
import time as _time
from typing import Tuple

import geo

# Two reports are linked as "possible duplicates" if they share a damage
# class AND satisfy at least one of:
#   - both have GPS and are within GPS_CLOSE_M of each other  -> "likely related" (strong evidence)
#   - both have GPS and are within GPS_POSSIBLE_M              -> "possible duplicate" (medium evidence)
#   - either lacks GPS, but their location/road-name text is very similar -> "possible duplicate" (weaker evidence)
GPS_CLOSE_M = 40.0
GPS_POSSIBLE_M = 120.0
TEXT_SIMILARITY_THRESHOLD = 0.72


def _text_similarity(a: Optional[str], b: Optional[str]) -> float:
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _location_text(report) -> str:
    return " ".join(str(x) for x in (report.road_name, report.location, report.area) if x)


def _report_damage_class(report) -> Optional[str]:
    """Best-effort dominant damage class_id for a citizen report, from its stored detections."""
    if not report.detections:
        return None
    counts = {}
    for d in report.detections:
        cid = d.get("class_id")
        if cid:
            counts[cid] = counts.get(cid, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _link_evidence(r1, r2) -> Optional[Tuple[str, str]]:
    """
    Returns (confidence, reason) if r1 and r2 look like the same physical
    issue, else None. confidence is 'likely' or 'possible'.
    Requires the same dominant damage class as a baseline requirement -
    a pothole report and a crack report are never grouped, regardless of
    proximity, since they cannot be "the same road issue".
    """
    c1, c2 = _report_damage_class(r1), _report_damage_class(r2)
    if not c1 or not c2 or c1 != c2:
        return None

    if r1.latitude is not None and r1.longitude is not None and \
       r2.latitude is not None and r2.longitude is not None:
        dist = geo.haversine_m(r1.latitude, r1.longitude, r2.latitude, r2.longitude)
        if dist <= GPS_CLOSE_M:
            return "likely", f"Reports are only {dist:.0f} m apart with matching damage type."
        if dist <= GPS_POSSIBLE_M:
            return "possible", f"Reports are {dist:.0f} m apart with matching damage type."
        return None  # GPS present on both but far apart -> not the same spot, regardless of text

    sim = _text_similarity(_location_text(r1), _location_text(r2))
    if sim >= TEXT_SIMILARITY_THRESHOLD:
        return "possible", f"Similar location text ('{r1.location}' vs '{r2.location}') and matching damage type."
    return None


def _union_find_groups(reports) -> List[List[int]]:
    """Cluster report indices using union-find over pairwise `_link_evidence`."""
    n = len(reports)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    pair_evidence = {}
    for i in range(n):
        for j in range(i + 1, n):
            ev = _link_evidence(reports[i], reports[j])
            if ev:
                union(i, j)
                pair_evidence[(i, j)] = ev

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return [idx_list for idx_list in clusters.values() if len(idx_list) >= 2], pair_evidence


def recompute_duplicate_groups():
    """
    Re-clusters ALL open (non-resolved) citizen reports that have at least
    a damage class recorded, persists group membership to
    `citizen_reports.duplicate_group_id`, and upserts the `duplicate_groups`
    table. Existing group ids are kept stable where a cluster still
    contains a report that already belonged to that group; brand-new
    clusters get a freshly minted `DG-000NNN` id.

    Never deletes or merges the underlying reports - grouping only.
    Safe to call repeatedly (idempotent membership, cheap at prototype
    scale: O(n^2) pairwise comparisons over the report set).
    """
    import citizen_report as creport

    reports = [r for r in creport.list_all()]
    eligible = [r for r in reports if _report_damage_class(r) is not None]
    clusters, pair_evidence = _union_find_groups(eligible)

    conn = creport.get_conn()
    now = _time.time()

    # existing group_id -> members currently recorded, so new clusters can
    # inherit a stable id when they overlap an old group
    existing_group_of_report = {r.report_id: r.duplicate_group_id for r in reports if r.duplicate_group_id}

    kept_group_ids = set()
    assigned = {}  # report_id -> (group_id, confidence)

    for member_idxs in clusters:
        members = [eligible[i] for i in member_idxs]
        # confidence for the group = the strongest pairwise evidence found in it
        confidences = []
        for a in range(len(member_idxs)):
            for b in range(a + 1, len(member_idxs)):
                key = tuple(sorted((member_idxs[a], member_idxs[b])))
                if key in pair_evidence:
                    confidences.append(pair_evidence[key][0])
        confidence = "likely" if "likely" in confidences else "possible"

        # inherit an existing group id if any member already has one
        group_id = None
        for m in members:
            gid = existing_group_of_report.get(m.report_id)
            if gid:
                group_id = gid
                break
        if not group_id:
            row = conn.execute("SELECT COUNT(*) as c FROM duplicate_groups").fetchone()
            group_id = f"DG-{row['c'] + 1:05d}"
            while conn.execute("SELECT 1 FROM duplicate_groups WHERE group_id = ?", (group_id,)).fetchone():
                row = conn.execute("SELECT COUNT(*) as c FROM duplicate_groups").fetchone()
                group_id = f"DG-{row['c'] + 1 + len(kept_group_ids):05d}"

        kept_group_ids.add(group_id)
        damage_class = _report_damage_class(members[0])
        location_label = members[0].location or members[0].road_name or "(unspecified location)"

        conn.execute(
            """INSERT INTO duplicate_groups (group_id, damage_class, damage_label, location_label,
                                              confidence, member_count, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(group_id) DO UPDATE SET
                   damage_class=excluded.damage_class, damage_label=excluded.damage_label,
                   location_label=excluded.location_label, confidence=excluded.confidence,
                   member_count=excluded.member_count, updated_at=excluded.updated_at""",
            (group_id, damage_class, members[0].detected_damage, location_label,
             confidence, len(members), now, now),
        )
        for m in members:
            assigned[m.report_id] = group_id

    # apply / clear duplicate_group_id on every report
    for r in reports:
        new_gid = assigned.get(r.report_id)
        if new_gid != r.duplicate_group_id:
            conn.execute("UPDATE citizen_reports SET duplicate_group_id = ? WHERE report_id = ?",
                         (new_gid, r.report_id))

    # drop groups that no longer have >=2 members
    stale = conn.execute("SELECT group_id FROM duplicate_groups").fetchall()
    for row in stale:
        if row["group_id"] not in kept_group_ids:
            conn.execute("DELETE FROM duplicate_groups WHERE group_id = ?", (row["group_id"],))

    conn.commit()
    conn.close()
    return kept_group_ids


def list_duplicate_groups():
    import citizen_report as creport
    conn = creport.get_conn()
    rows = conn.execute("SELECT * FROM duplicate_groups ORDER BY updated_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_duplicate_group(group_id: str):
    import citizen_report as creport
    conn = creport.get_conn()
    row = conn.execute("SELECT * FROM duplicate_groups WHERE group_id = ?", (group_id,)).fetchone()
    conn.close()
    return dict(row) if row else None
