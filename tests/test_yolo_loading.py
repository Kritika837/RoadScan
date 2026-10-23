"""
tests/test_yolo_loading.py
---------------------------
Run this AFTER completing the steps in `models/README.md` (torch installed,
yolov7 repo cloned to `models/yolov7_repo/`, both checkpoints placed under
`models/weights/`).

This sandbox could not execute this test: no `torch`, no GPU, and network
access to install torch / clone the repo is blocked here. The test is
provided so you can verify the integration on your own machine before
trusting its output - do not assume it passes without running it yourself.

Usage:
    pip install pytest
    pytest tests/test_yolo_loading.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from detector import YoloV7EnsembleDetector, DetectorUnavailableError


def test_yolov7_reports_missing_dependencies_clearly_if_absent():
    """Even on a machine without the model set up, the detector must fail
    with a clear, specific reason rather than a cryptic stack trace."""
    det = YoloV7EnsembleDetector(
        weights_640="models/weights/does_not_exist_640.pt",
        weights_512="models/weights/does_not_exist_512.pt",
    )
    if not det.ready:
        assert det.unavailable_reason is not None
        assert len(det.unavailable_reason) > 10


@pytest.mark.skipif(
    not os.path.isfile("models/weights/yolov7all6_best.pt") or
    not os.path.isfile("models/weights/yolov7all61_best.pt"),
    reason="Trained YOLOv7 weights not present - see models/README.md",
)
def test_yolov7_loads_and_infers_on_blank_image():
    det = YoloV7EnsembleDetector()
    assert det.ready, f"Detector did not load: {det.unavailable_reason}"

    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    detections = det.detect(blank)
    assert isinstance(detections, list)
    for d in detections:
        assert d.class_id in {"D00", "D10", "D20", "D40"}
        assert 0.0 <= d.confidence <= 1.0
        x1, y1, x2, y2 = d.bbox
        assert x2 > x1 and y2 > y1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
