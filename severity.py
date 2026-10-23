"""
severity.py
-----------
MODULE: Severity Assessment
INPUT:  a list of `Detection` objects (from detector.py) + the source image shape
OUTPUT: a list of `DamageAssessment` objects (one per detection), each with a
        deterministic severity level and a plain-language `reason` string
WHY:    turns raw detector output (class + box + confidence) into a
        decision-relevant judgement ("how bad is this one damage?")
ALGORITHM: weighted rule-based scoring - see `_severity_score` docstring
LIMITATIONS: area/confidence weights are hand-set constants tuned for
        plausibility, not fit to labeled severity ground truth (RDD2022 has
        no severity labels). Treat outputs as decision-support, not a
        calibrated engineering measurement.
INTERACTION: consumed by `health_score.py` (aggregates these into the
        overall Road Health Score) and by `report.py` / the UI (displayed
        per-detection).

Repair priority (Low/Medium/High/Urgent) also lives in this module, since it
is a direct function of the severity assessments + health score - see
`compute_repair_priority` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from detector import Detection

# ---------------------------------------------------------------------------
# Configurable thresholds - change these to retune the whole system
# ---------------------------------------------------------------------------

# Relative structural/safety weight per class. A pothole of the same pixel
# area as a hairline crack is a bigger deal, so it gets a higher weight.
CLASS_WEIGHT = {
    "D00": 1.0,   # Longitudinal crack
    "D10": 1.0,   # Transverse crack
    "D20": 1.6,   # Alligator crack (structural fatigue, water infiltration)
    "D40": 2.2,   # Pothole (immediate driving hazard)
    "D43": 1.3,   # Surface / patch damage
}

# Severity score (0-100) -> label boundaries. Configurable.
SEVERITY_THRESHOLDS = {"LOW_MAX": 25, "MODERATE_MAX": 50, "HIGH_MAX": 75}  # >HIGH_MAX = CRITICAL
SEVERITY_LEVELS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]

# Area ratio (damage area / image area) at which the area factor saturates.
AREA_SATURATION_RATIO = 0.08


@dataclass
class DamageAssessment:
    detection: Detection
    area_ratio: float          # detected damage area / total image area
    severity_score: float      # 0-100 internal composite score for this damage
    severity: str               # LOW / MODERATE / HIGH / CRITICAL
    reason: str                  # plain-language explanation of the severity


def _severity_score(detection: Detection, image_area: int) -> float:
    """
    Composite 0-100 score for ONE detected damage:

        score = 100 * (0.55 * area_component + 0.25 * confidence_component)
                     * (class_weight / 2.2)
                + 8 * (class_weight - 1.0)

    - area_component: area_ratio / AREA_SATURATION_RATIO, capped at 1.0
        (a damage covering >=8% of the visible surface is already "maximal"
        for this factor)
    - confidence_component: (confidence - 0.25) / 0.75, capped at [0, 1]
        (low-confidence detections are pulled toward LOW rather than
        inflating severity on a weak guess)
    - class_weight: from CLASS_WEIGHT above; also adds a direct floor so a
        real pothole is never "trivially" scored even when small
    """
    area_ratio = detection.mask_area_px / max(1, image_area)
    weight = CLASS_WEIGHT.get(detection.class_id, 1.0)

    # Area contribution is now graduated instead of saturating at 8%.
    # This prevents a severely large detected region from being treated
    # almost the same as a small 8% region.
    if area_ratio >= 0.70:
        area_component = 1.00
    elif area_ratio >= 0.50:
        area_component = 0.90
    elif area_ratio >= 0.30:
        area_component = 0.75
    elif area_ratio >= 0.15:
        area_component = 0.60
    elif area_ratio >= 0.08:
        area_component = 0.45
    elif area_ratio >= 0.03:
        area_component = 0.30
    elif area_ratio >= 0.01:
        area_component = 0.20
    else:
        area_component = 0.10

    conf_component = max(0.0, (detection.confidence - 0.25) / 0.75)

    raw = 100 * (0.75 * area_component + 0.25 * conf_component) * (weight / 2.2)
    raw += 8 * (weight - 1.0)

    # Extensive visible deterioration should not be classified
    # as only MODERATE, especially for alligator cracking.
    if area_ratio >= 0.70:
        raw = max(raw, 55.0)
    elif area_ratio >= 0.50:
        raw = max(raw, 50.0)
    elif area_ratio >= 0.30:
        raw = max(raw, 45.0)

    return float(max(0.0, min(100.0, raw)))


def _severity_label(score: float) -> str:
    if score < SEVERITY_THRESHOLDS["LOW_MAX"]:
        return "LOW"
    if score < SEVERITY_THRESHOLDS["MODERATE_MAX"]:
        return "MODERATE"
    if score < SEVERITY_THRESHOLDS["HIGH_MAX"]:
        return "HIGH"
    return "CRITICAL"


def _reason(detection: Detection, area_ratio: float, score: float, label: str) -> str:
    weight = CLASS_WEIGHT.get(detection.class_id, 1.0)
    area_pct = area_ratio * 100
    parts = []
    if area_pct >= 3:
        parts.append(f"large affected area ({area_pct:.1f}% of the image)")
    elif area_pct >= 0.5:
        parts.append(f"moderate affected area ({area_pct:.1f}% of the image)")
    else:
        parts.append(f"small affected area ({area_pct:.2f}% of the image)")

    if weight >= 2.0:
        parts.append(f"{detection.class_name.lower()} carries a high repair-risk weighting")
    elif weight >= 1.4:
        parts.append(f"{detection.class_name.lower()} carries an elevated repair-risk weighting")

    if detection.confidence < 0.5:
        parts.append(f"detection confidence is only {detection.confidence*100:.0f}%, which limits certainty")

    return f"{label.title()} severity: " + "; ".join(parts) + "."


def assess_damages(detections: List[Detection], image_shape) -> List[DamageAssessment]:
    """Score every detection independently. Pure function of (detections, image_shape)."""
    h, w = image_shape[:2]
    image_area = h * w
    out = []
    for d in detections:
        area_ratio = d.mask_area_px / max(1, image_area)
        score = _severity_score(d, image_area)
        label = _severity_label(score)
        out.append(DamageAssessment(
            detection=d,
            area_ratio=area_ratio,
            severity_score=round(score, 1),
            severity=label,
            reason=_reason(d, area_ratio, score, label),
        ))
    return out


# ---------------------------------------------------------------------------
# Repair priority
# MODULE: Repair Priority
# INPUT:  DamageAssessment list + HealthResult (from health_score.py)
# OUTPUT: PriorityResult(level, reason, recommended_action)
# WHY:    turns "how bad is each damage + overall road health" into a single
#         actionable decision-support label
# ALGORITHM: deterministic rule cascade, evaluated top to bottom (see code)
# LIMITATIONS: this is decision support, not a certified inspection; it does
#         not account for traffic volume, road class, or budget
# ---------------------------------------------------------------------------

PRIORITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "URGENT"]


@dataclass
class PriorityResult:
    level: str
    reason: str
    recommended_action: str


_RECOMMENDED_ACTIONS = {
    "URGENT": "Inspect and schedule repair immediately; consider temporary warning signage or lane "
              "closure until the hazard is addressed.",
    "HIGH": "Schedule repair within the next maintenance cycle (days to a few weeks) and re-inspect if "
            "conditions worsen.",
    "MEDIUM": "Add to the routine maintenance queue; re-assess at the next scheduled inspection.",
    "LOW": "No immediate action required; continue periodic monitoring.",
}


def compute_repair_priority(assessments: List[DamageAssessment], health) -> PriorityResult:
    if not assessments:
        return PriorityResult(
            level="LOW",
            reason="No road damage was detected in this image, so no repair action is indicated at "
                   "this time.",
            recommended_action=_RECOMMENDED_ACTIONS["LOW"],
        )

    worst = max(assessments, key=lambda a: a.severity_score)
    critical_count = sum(1 for a in assessments if a.severity == "CRITICAL")
    high_or_critical_count = sum(1 for a in assessments if a.severity in ("HIGH", "CRITICAL"))
    has_critical_pothole = any(a.detection.class_id == "D40" and a.severity == "CRITICAL" for a in assessments)
    has_large_alligator = any(a.detection.class_id == "D20" and a.area_ratio > 0.05 for a in assessments)

    if has_critical_pothole or health.score < 30 or critical_count >= 2:
        level = "URGENT"
        if has_critical_pothole:
            reason = ("A pothole classified as CRITICAL severity occupies a significant portion of the "
                       "visible driving surface, posing an immediate hazard to vehicles.")
        elif critical_count >= 2:
            reason = f"{critical_count} separate detections were classified as CRITICAL severity."
        else:
            reason = f"Overall Road Health Score is {health.score}/100, indicating extensive, severe damage."
    # elif worst.severity == "CRITICAL" or has_large_alligator or health.score < 50 or high_or_critical_count >= 2:
    #     level = "HIGH"
    #     reason = (f"Significant damage detected (worst case: {worst.detection.class_name}, "
    #                f"'{worst.severity}' severity) with {high_or_critical_count} high-or-critical "
    #                f"finding(s); Road Health Score is {health.score}/100.")
    
    elif (
        worst.severity in ("HIGH", "CRITICAL")
        or has_large_alligator
        or health.score < 50
        or high_or_critical_count >= 2
    ):
        level = "HIGH"
        reason = (f"Significant damage detected (worst case: {worst.detection.class_name}, "
                   f"'{worst.severity}' severity) with {high_or_critical_count} high-or-critical "
                   f"finding(s); Road Health Score is {health.score}/100.")

    
    elif worst.severity == "MODERATE" or health.score < 75:

        level = "MEDIUM"
        reason = (f"Moderate damage detected (worst case: {worst.detection.class_name}, "
                   f"'{worst.severity}' severity); Road Health Score is {health.score}/100.")
    else:
        level = "LOW"
        reason = (f"Only minor, isolated damage was detected (worst case: {worst.detection.class_name}, "
                   f"'{worst.severity}' severity); Road Health Score is {health.score}/100.")

    return PriorityResult(level=level, reason=reason, recommended_action=_RECOMMENDED_ACTIONS[level])
