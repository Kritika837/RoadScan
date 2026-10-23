"""
explain.py
----------
MODULE: Region Visualization + Explainability (two distinct outputs)
INPUT:  the original image + `Detection` list (+ severity labels for display)
OUTPUT: (a) an annotated image: boxes, region masks, class/confidence/severity
            labels drawn on the image (`draw_annotations`)
        (b) a separate evidence heatmap: which pixels most influenced
            detection (`evidence_heatmap`)
WHY:    (a) is what a human checks the detections against; (b) is a
        model-transparency aid - kept visually and conceptually distinct
        from (a), and from raw detection, per project requirements.
ALGORITHM: (a) alpha-blended mask overlay + cv2 box/label drawing.
        (b) Canny-edge density + local-darkness contrast, gated to the
        padded area around each detection so it stays interpretable.

Note on Grad-CAM: Grad-CAM is specific to trained CNNs (it back-propagates
gradients through convolutional feature maps). The current backend
(`HeuristicCVDetector`) is a classical computer-vision pipeline with no
learned gradients to inspect, so we provide an equivalent, honestly-labeled
explanation instead: a per-detection "evidence heatmap" built from the same
edge/contrast/texture signals the detector actually used to find and score
each region. If `YoloV7EnsembleDetector` is activated in the future, true
Grad-CAM (or EigenCAM) over its convolutional layers can be substituted here
without changing any downstream code. This heatmap is never labeled
"Grad-CAM" in the UI or reports.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from detector import Detection

CLASS_COLORS = {
    "D00": (255, 140, 0),    # longitudinal crack - orange
    "D10": (0, 165, 255),    # transverse crack - amber
    "D20": (0, 0, 255),      # alligator crack - red
    "D40": (128, 0, 128),    # pothole - purple
    "D43": (0, 128, 255),    # surface/patch - light orange
}


def draw_annotations(image_bgr: np.ndarray, detections: List[Detection], severities: dict) -> np.ndarray:
    """Overlay bounding boxes, segmentation masks, and labels on a copy of the image."""
    out = image_bgr.copy()
    overlay = out.copy()

    for d in detections:
        color = CLASS_COLORS.get(d.class_id, (0, 255, 0))
        if d.mask is not None:
            overlay[d.mask] = color
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)

    for d in detections:
        color = CLASS_COLORS.get(d.class_id, (0, 255, 0))
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        sev = severities.get(id(d), "")
        label = f"{d.class_id} {d.class_name} {d.confidence*100:.0f}% [{sev}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty1 = max(0, y1 - th - 8)
        cv2.rectangle(out, (x1, ty1), (x1 + tw + 6, ty1 + th + 8), color, -1)
        text_color = (255, 255, 255)
        cv2.putText(out, label, (x1 + 3, ty1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    return out


def evidence_heatmap(image_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """
    Build a heatmap over the image highlighting, per pixel, how strongly it
    contributed to a detection: combines local edge density (crack evidence)
    and local darkness contrast (pothole/patch evidence), gated to detected
    regions so it stays interpretable rather than noisy across the full image.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120).astype(np.float32) / 255.0
    edges = cv2.GaussianBlur(edges, (9, 9), 0)

    blurred = cv2.GaussianBlur(gray, (15, 15), 0).astype(np.float32)
    darkness = np.clip((blurred.mean() - gray.astype(np.float32)) / 60.0, 0, 1)

    signal = np.clip(0.6 * edges + 0.6 * darkness, 0, 1)

    gate = np.zeros((h, w), dtype=np.float32)
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        pad_x, pad_y = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
        gx1, gy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        gx2, gy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        gate[gy1:gy2, gx1:gx2] = np.maximum(gate[gy1:gy2, gx1:gx2], d.confidence)

    heat = signal * gate
    if heat.max() > 0:
        heat = heat / heat.max()
    heat_u8 = (heat * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    dim_base = (gray.astype(np.float32) * 0.5).astype(np.uint8)
    dim_base_bgr = cv2.cvtColor(dim_base, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(heat_color, 0.65, dim_base_bgr, 0.55, 0)
    return blended
