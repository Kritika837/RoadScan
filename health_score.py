"""
health_score.py
----------------
MODULE: Road Health Score
INPUT:  list of `DamageAssessment` (from severity.py) + source image shape
OUTPUT: `HealthResult(score, condition, breakdown, penalty_by_class)`
WHY:    aggregates all per-damage severities into one number a non-expert
        can act on ("64/100"), with a penalty breakdown that shows exactly
        where the points went - e.g. "Potholes: -16" - rather than a black
        box.
ALGORITHM: additive penalty model, see `compute_health_score` docstring.
LIMITATIONS: like severity.py, the weights are hand-set for plausibility,
        not fit to a labeled "true" road condition scale (PCI, etc.). This
        is a proxy score, not a certified pavement-condition index.
INTERACTION: consumed by `severity.compute_repair_priority`, `history.py`
        (tracked over time), `report.py`, and the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict

from detector import CLASS_NAMES
from severity import DamageAssessment, CLASS_WEIGHT

CONDITION_BANDS = [
    (90, 100, "Excellent"),
    (75, 90, "Good"),
    (55, 75, "Fair"),
    (35, 55, "Poor"),
    (0, 35, "Severe"),
]


def condition_from_score(score: float) -> str:
    for lo, hi, label in CONDITION_BANDS:
        if lo <= score <= hi:
            return label
    return "Severe"


@dataclass
class HealthResult:
    score: float
    condition: str
    breakdown: dict                    # damage_count / damage_penalty / count_penalty / total_penalty
    penalty_by_class: Dict[str, float] = field(default_factory=dict)  # e.g. {"Pothole": -16.0}


def compute_health_score(assessments: List[DamageAssessment], image_shape) -> HealthResult:
    """
    Road Health Score = 100 - total_penalty, where:

        total_penalty = damage_penalty + count_penalty

        damage_penalty = sum over each detection of:
            (severity_score / 100) * class_weight * area_multiplier * 6.0
        where area_multiplier = 1.0 + min(1.5, area_ratio / 0.03)
        (bigger damage costs proportionally more, capped so one giant
        detection can't single-handedly zero the score from area alone)

        count_penalty = min(20.0, 2.5 * max(0, damage_count - 1))
        (many small damages together indicate general deterioration even
        if none individually is severe; capped at 20 points)

    `penalty_by_class` re-aggregates `damage_penalty` per damage class, e.g.
    {"Pothole": -16.2, "Longitudinal Crack": -4.1}, so the UI/report can show
    a breakdown like:

        Road Health Score: 64/100
        Penalty breakdown:
        - Potholes: -16
        - Longitudinal cracks: -10
        - Alligator cracks: -10
    """
    if not assessments:
        return HealthResult(
            score=100.0, condition=condition_from_score(100.0),
            breakdown={"damage_count": 0, "damage_penalty": 0.0, "count_penalty": 0.0, "total_penalty": 0.0},
            penalty_by_class={},
        )

    damage_penalty = 0.0
    penalty_by_class: Dict[str, float] = {}
    for a in assessments:
        weight = CLASS_WEIGHT.get(a.detection.class_id, 1.0)
        area_multiplier = 1.0 + min(1.5, a.area_ratio / 0.03)
        this_penalty = (a.severity_score / 100.0) * weight * area_multiplier * 6.0
        damage_penalty += this_penalty

        class_label = CLASS_NAMES.get(a.detection.class_id, a.detection.class_id)
        penalty_by_class[class_label] = penalty_by_class.get(class_label, 0.0) + this_penalty

    count_penalty = min(20.0, 2.5 * max(0, len(assessments) - 1))
    total_penalty = damage_penalty + count_penalty
    score = max(0.0, 100.0 - total_penalty)

    return HealthResult(
        score=round(score, 1),
        condition=condition_from_score(score),
        breakdown={
            "damage_count": len(assessments),
            "damage_penalty": round(damage_penalty, 1),
            "count_penalty": round(count_penalty, 1),
            "total_penalty": round(total_penalty, 1),
        },
        penalty_by_class={k: round(v, 1) for k, v in penalty_by_class.items()},
    )


def recommended_action(priority, health: HealthResult) -> str:
    """Kept for backward compatibility - priority.recommended_action is now the primary source."""
    return getattr(priority, "recommended_action", None) or "Continue periodic monitoring."
