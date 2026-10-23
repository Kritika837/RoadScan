"""
tests/run_pipeline_tests.py
-----------------------------
Runs the ACTUAL detect -> severity -> health -> priority pipeline against
every image in `tests/images/` and writes the real output to
`tests/test_results.md`. It never fabricates a result: if `tests/images/`
is empty, it says so and lists what's needed instead of inventing numbers.

Usage:
    python tests/run_pipeline_tests.py
"""

import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from detector import get_default_detector
import severity as sev
import health_score as hsc

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "test_results.md")

REQUIRED_CASES = [
    "01_no_or_minimal_damage.jpg",
    "02_pothole.jpg",
    "03_longitudinal_crack.jpg",
    "04_transverse_crack.jpg",
    "05_multiple_damages.jpg",
    "06_severe_damage.jpg",
]


def main():
    status = get_default_detector()
    detector = status.detector
    image_paths = sorted(
        glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) +
        glob.glob(os.path.join(IMAGES_DIR, "*.png")) +
        glob.glob(os.path.join(IMAGES_DIR, "*.jpeg"))
    )

    lines = [
        "# Roadscan pipeline test results",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Detector backend used: `{status.backend_used}` "
        f"({'PRIMARY - YOLOv7' if status.is_primary else 'FALLBACK - classical CV, not a trained model'})",
        "",
    ]

    if not image_paths:
        lines += [
            "## No test images found",
            "",
            f"`{IMAGES_DIR}` is empty, so no results are recorded below - none were fabricated.",
            "",
            "To produce real test results, add real road photographs to `tests/images/` covering "
            "these cases (ideally named as below so they sort in order):",
            "",
        ]
        for case in REQUIRED_CASES:
            lines.append(f"- `{case}` - " + case.replace(".jpg", "").split("_", 1)[1].replace("_", " "))
        lines += [
            "",
            "Image guidance: daylight, road surface filling most of the frame, JPG/PNG, any resolution "
            "(the app downsamples long-side to 1600px). Then re-run `python tests/run_pipeline_tests.py`.",
        ]
        with open(RESULTS_PATH, "w") as f:
            f.write("\n".join(lines))
        print(f"No images found in {IMAGES_DIR}. Wrote requirements to {RESULTS_PATH}.")
        return

    lines += ["| Image | Detected classes | # Detections | Confidence range | Severity levels present | Health score | Condition | Priority |",
              "|---|---|---|---|---|---|---|---|"]

    for path in image_paths:
        name = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            lines.append(f"| {name} | (could not read file) | - | - | - | - | - | - |")
            continue

        detections = detector.detect(img)
        assessments = sev.assess_damages(detections, img.shape)
        health = hsc.compute_health_score(assessments, img.shape)
        priority = sev.compute_repair_priority(assessments, health)

        classes = ", ".join(sorted(set(d.class_id for d in detections))) or "(none)"
        n = len(detections)
        if detections:
            confs = [d.confidence for d in detections]
            conf_range = f"{min(confs)*100:.0f}%-{max(confs)*100:.0f}%"
        else:
            conf_range = "-"
        sev_levels = ", ".join(sorted(set(a.severity for a in assessments))) or "-"

        lines.append(
            f"| {name} | {classes} | {n} | {conf_range} | {sev_levels} | "
            f"{health.score} | {health.condition} | {priority.level} |"
        )

    with open(RESULTS_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {len(image_paths)} result(s) to {RESULTS_PATH}.")


if __name__ == "__main__":
    main()
