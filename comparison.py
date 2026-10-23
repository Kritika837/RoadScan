"""
comparison.py
-------------
MODULE: Before / After Comparison
INPUT:  two images (before-repair, after-repair)
OUTPUT: `RepairComparison(verdict, explanation, health_delta, count_delta,
        area_delta, scene_similarity, low_confidence_comparison)`
WHY:    lets a user verify whether a repair actually worked, numerically,
        instead of eyeballing two photos.
ALGORITHM: runs the full detect->severity->health pipeline independently on
        each image, diffs the results, and separately checks whether the
        two photos plausibly depict the same scene (HSV color-histogram
        correlation). If they don't look alike, the verdict is downgraded
        to a stated "cannot reliably determine" rather than a false
        improvement/decline claim.
LIMITATIONS: histogram correlation is a coarse same-scene proxy - it cannot
        confirm GPS location or camera angle. Treat `low_confidence_comparison`
        as a caution flag, not a hard guarantee either way.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def scene_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """0-1 rough similarity of overall scene based on color histogram correlation."""
    def hist(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(h, h)
        return h

    ha, hb = hist(img_a), hist(img_b)
    corr = cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)
    return float(max(0.0, min(1.0, corr)))


@dataclass
class RepairComparison:
    verdict: str          # "Improved" / "Remained similar" / "Appears worse" / "Cannot reliably determine"
    explanation: str
    health_delta: float
    count_delta: int
    area_delta: float
    scene_similarity: float
    low_confidence_comparison: bool
    reliable: bool


# Below this scene-similarity score, the two photos are different enough
# that we refuse to call an improvement/decline at all.
SCENE_MISMATCH_THRESHOLD = 0.20
# Below this (but above the hard mismatch threshold), we still answer but
# flag the comparison as lower-confidence.
SCENE_CAUTION_THRESHOLD = 0.35


def compare_before_after(before_health, before_count, before_area_ratio,
                          after_health, after_count, after_area_ratio,
                          img_before: np.ndarray, img_after: np.ndarray) -> RepairComparison:
    health_delta = round(after_health - before_health, 1)
    count_delta = after_count - before_count
    area_delta = round(after_area_ratio - before_area_ratio, 4)
    sim = scene_similarity(img_before, img_after)

    if sim < SCENE_MISMATCH_THRESHOLD:
        return RepairComparison(
            verdict="Cannot reliably determine",
            explanation=(
                "Comparison cannot reliably determine repair improvement because the scenes differ "
                f"significantly (scene similarity: {sim*100:.0f}%). Health score and damage-count "
                "numbers are still reported below for reference, but should not be interpreted as a "
                "before/after repair outcome."
            ),
            health_delta=health_delta, count_delta=count_delta, area_delta=area_delta,
            scene_similarity=round(sim, 2), low_confidence_comparison=True, reliable=False,
        )

    low_conf = sim < SCENE_CAUTION_THRESHOLD

    if health_delta >= 10 and count_delta <= 0:
        verdict = "Improved"
        explanation = (
            f"The Road Health Score increased by {health_delta} points and detected damage count "
            f"changed by {count_delta:+d}, consistent with a successful repair."
        )
    elif health_delta <= -10 or count_delta >= 2:
        verdict = "Appears worse"
        explanation = (
            f"The Road Health Score decreased by {abs(health_delta)} points and/or damage count "
            f"increased ({count_delta:+d}), suggesting the surface condition is worse in the 'after' image."
        )
    else:
        verdict = "Remained similar"
        explanation = (
            f"Health score changed by {health_delta:+.1f} points and damage count by {count_delta:+d} - "
            "not a large enough change to confidently call this an improvement or a decline."
        )

    if low_conf:
        explanation += (
            " Note: the two images have noticeably different overall appearance (lighting/angle/"
            "surroundings), so this comparison carries lower confidence - we cannot fully confirm both "
            "photos show the same physical spot."
        )

    return RepairComparison(
        verdict=verdict, explanation=explanation, health_delta=health_delta,
        count_delta=count_delta, area_delta=area_delta, scene_similarity=round(sim, 2),
        low_confidence_comparison=low_conf, reliable=True,
    )
