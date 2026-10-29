"""
detector.py
-----------
Pluggable road-damage detection backend.

The original project (sivakanth1/Detecting_Road_Damage) uses an ensemble of two
custom-trained YOLOv7 models (640x640 and 512x512) over 4 RDD2022 classes:

    D00 - Longitudinal Crack
    D10 - Transverse Crack
    D20 - Alligator Crack
    D40 - Pothole

Those trained weights (~hundreds of MB, produced by ~28h of GPU training) are
NOT included in the GitHub repo and could not be downloaded in this offline,
no-GPU environment (no `torch`/`ultralytics`, no network access).

To keep the project genuinely runnable on a normal student laptop without any
large downloads, this module defines a `RoadDamageDetector` interface with:

  * `HeuristicCVDetector` - a real, working, classical computer-vision
    detector (OpenCV contour/edge/blob analysis) that produces the same
    output contract (class, bbox, confidence) as the YOLO models would.
    It is intentionally labeled as a *baseline* detector, not a trained
    deep model, and its confidence scores are derived from measurable
    shape/texture features (never random).

  * `YoloV7EnsembleDetector` - a drop-in slot that mirrors the exact
    ensemble pipeline described in `index.ipynb` (two YOLOv7 models + NMS
    merge). If `torch`/`ultralytics` and the two trained checkpoints are
    made available later, this class becomes the real detector without any
    change to the rest of the application (severity, health score, report,
    history, comparison, dedup all consume the same `Detection` objects).

Adding a new backend only requires implementing `.detect(image_bgr)`.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Shared class scheme (kept identical to the existing project's RDD2022 setup)
# ---------------------------------------------------------------------------

CLASS_NAMES = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
    "D43": "Surface / Patch Damage",  # extra class the heuristic detector can flag
}

CLASS_ORDER = ["D00", "D10", "D20", "D40", "D43"]


@dataclass
class Detection:
    class_id: str                 # e.g. "D40"
    class_name: str                # e.g. "Pothole"
    confidence: float              # 0.0 - 1.0
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 (pixels, in original image)
    mask: Optional[np.ndarray] = None  # bool array, same size as original image
    features: dict = field(default_factory=dict)  # raw measurements used for scoring

    @property
    def area_px(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def mask_area_px(self) -> int:
        if self.mask is None:
            return self.area_px
        return int(self.mask.sum())


class RoadDamageDetector:
    """Abstract interface every detection backend must implement."""

    name = "base"

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Backend 1: classical computer-vision baseline (actually runs, no downloads)
# ---------------------------------------------------------------------------

def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / float(area_a + area_b - inter)


def _nms(dets: List[Detection], iou_thresh: float = 0.35) -> List[Detection]:
    dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
    keep: List[Detection] = []
    for d in dets:
        if all(_iou(d.bbox, k.bbox) < iou_thresh for k in keep):
            keep.append(d)
    return keep


class HeuristicCVDetector(RoadDamageDetector):
    """
    Classical CV pipeline for road-surface damage candidates.

    Pipeline:
      1. CLAHE contrast enhancement on grayscale image (road photos are often
         low-contrast / unevenly lit).
      2. Two parallel candidate branches:
           a) Dark-blob branch (adaptive threshold + morphology) tuned for
              compact, roughly-round dark regions -> pothole candidates.
           b) Ridge/edge branch (Canny + directional morphological closing)
              tuned for thin, elongated dark structures -> crack candidates.
      3. Shape/texture features per contour (elongation, solidity, extent,
         local intensity contrast, branching density) drive both the class
         assignment and a measurable confidence score.
      4. Minimum-area filtering + custom NMS across both branches.

    This is NOT a trained neural network and will not match YOLOv7-level
    accuracy. It exists so the full application (severity, scoring,
    reporting, history, comparison, dedup, explainability) is genuinely
    exercised end-to-end on real uploaded images without any model download.
    """

    name = "heuristic_cv_v1"

    def __init__(
        self,
        min_area_frac: float = 0.0006,   # ignore specks smaller than this fraction of image area
        max_area_frac: float = 0.35,     # ignore near-whole-image blobs (likely shadow/asphalt edge)
        conf_threshold: float = 0.28,
    ):
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.conf_threshold = conf_threshold

    # -- public API ---------------------------------------------------
    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        h, w = image_bgr.shape[:2]
        img_area = h * w
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        candidates: List[Detection] = []
        candidates += self._detect_blobs(blurred, gray, img_area)
        candidates += self._detect_cracks(blurred, gray, img_area)

        candidates = [c for c in candidates if c.confidence >= self.conf_threshold]
        candidates = _nms(candidates, iou_thresh=0.3)
        # keep results sorted left-to-right, top-to-bottom for stable display
        candidates.sort(key=lambda d: (d.bbox[1], d.bbox[0]))
        return candidates

    # -- pothole / patch branch ---------------------------------------
    def _detect_blobs(self, blurred, gray, img_area) -> List[Detection]:
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 7
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: List[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            area_frac = area / img_area
            if area_frac < self.min_area_frac or area_frac > self.max_area_frac:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 8 or bh < 8:
                continue
            aspect = max(bw, bh) / max(1, min(bw, bh))
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * math.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            region = gray[y:y + bh, x:x + bw]
            surround = self._surrounding_mean(gray, x, y, bw, bh)
            contrast = max(0.0, (surround - float(region.mean())) / 255.0)

            # Pothole-like: fairly round, compact, aspect ratio close to 1,
            # solid fill, and noticeably darker than the surrounding road.
            if aspect <= 2.4 and circularity > 0.25 and solidity > 0.6:
                class_id = "D40"  # Pothole
                score = 0.35 + 0.35 * circularity + 0.20 * min(1.0, contrast * 3) + 0.10 * solidity
            else:
                class_id = "D43"  # Surface / patch damage (irregular blob, not a clean pothole)
                score = 0.25 + 0.25 * solidity + 0.25 * min(1.0, contrast * 3) + 0.15 * min(1.0, area_frac * 40)

            score = float(np.clip(score, 0.0, 0.97))
            mask = np.zeros(gray.shape, dtype=bool)
            cv2.drawContours(mask.view(np.uint8), [cnt], -1, 1, thickness=-1)

            out.append(Detection(
                class_id=class_id,
                class_name=CLASS_NAMES[class_id],
                confidence=score,
                bbox=(x, y, x + bw, y + bh),
                mask=mask,
                features={
                    "area_frac": round(area_frac, 5),
                    "circularity": round(circularity, 3),
                    "solidity": round(solidity, 3),
                    "aspect_ratio": round(aspect, 2),
                    "contrast": round(contrast, 3),
                    "branch": "blob",
                },
            ))
        return out

    # -- crack branch ---------------------------------------------------
    def _detect_cracks(self, blurred, gray, img_area) -> List[Detection]:
        edges = cv2.Canny(blurred, 40, 120)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        dilated = cv2.dilate(closed, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: List[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            area_frac = area / img_area
            if area_frac < self.min_area_frac or area_frac > self.max_area_frac:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 10 or bh < 10:
                continue
            if len(cnt) < 5:
                continue

            rect = cv2.minAreaRect(cnt)
            (rw, rh) = rect[1]
            angle = rect[2]
            if rw == 0 or rh == 0:
                continue
            elongation = max(rw, rh) / max(1.0, min(rw, rh))
            rect_area = rw * rh
            extent = area / rect_area if rect_area > 0 else 0

            # Elongated, thin, low-fill-ratio structures = crack-like.
            if elongation < 2.2 or extent > 0.55:
                continue  # too blob-like -> leave to the blob branch

            # branching density: count of local contour direction changes,
            # used as a rough proxy for alligator (interconnected) cracking
            branching = self._branching_score(cnt)

            region = gray[y:y + bh, x:x + bw]
            surround = self._surrounding_mean(gray, x, y, bw, bh)
            contrast = max(0.0, (surround - float(region.mean())) / 255.0)

            long_side_horizontal = rw >= rh
            norm_angle = angle % 180

            if branching > 0.35 and area_frac > 0.0015:
                class_id = "D20"  # Alligator crack (interconnected network)
                score = 0.30 + 0.35 * min(1.0, branching) + 0.20 * min(1.0, contrast * 3) + 0.10 * min(1.0, area_frac * 30)
            elif 60 <= norm_angle <= 120:
                class_id = "D10"  # Transverse crack (runs across the road ~perpendicular)
                score = 0.30 + 0.30 * min(1.0, elongation / 8) + 0.25 * min(1.0, contrast * 3) + 0.10
            else:
                class_id = "D00"  # Longitudinal crack (runs along the road)
                score = 0.30 + 0.30 * min(1.0, elongation / 8) + 0.25 * min(1.0, contrast * 3) + 0.10

            score = float(np.clip(score, 0.0, 0.97))
            mask = np.zeros(gray.shape, dtype=bool)
            cv2.drawContours(mask.view(np.uint8), [cnt], -1, 1, thickness=-1)
            # cracks are thin: dilate the mask slightly so it visibly covers the crack line
            mask_u8 = mask.astype(np.uint8)
            mask_u8 = cv2.dilate(mask_u8, np.ones((3, 3), np.uint8), iterations=1)
            mask = mask_u8.astype(bool)

            out.append(Detection(
                class_id=class_id,
                class_name=CLASS_NAMES[class_id],
                confidence=score,
                bbox=(x, y, x + bw, y + bh),
                mask=mask,
                features={
                    "area_frac": round(area_frac, 5),
                    "elongation": round(elongation, 2),
                    "extent": round(extent, 3),
                    "branching": round(branching, 3),
                    "angle_deg": round(norm_angle, 1),
                    "contrast": round(contrast, 3),
                    "branch": "crack",
                },
            ))
        return out

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _surrounding_mean(gray, x, y, bw, bh, margin_frac: float = 0.6) -> float:
        h, w = gray.shape[:2]
        mx, my = int(bw * margin_frac), int(bh * margin_frac)
        x1, y1 = max(0, x - mx), max(0, y - my)
        x2, y2 = min(w, x + bw + mx), min(h, y + bh + my)
        outer = gray[y1:y2, x1:x2]
        inner = gray[y:y + bh, x:x + bw]
        if outer.size == 0:
            return float(inner.mean()) if inner.size else 128.0
        outer_sum = float(outer.sum()) - float(inner.sum())
        outer_count = outer.size - inner.size
        if outer_count <= 0:
            return float(inner.mean())
        return outer_sum / outer_count

    # @staticmethod
    # def _branching_score(cnt) -> float:
    #     """Rough proxy for how 'net-like' a contour is, via convexity-defect density."""
    #     try:
    #         hull_idx = cv2.convexHull(cnt, returnPoints=False)
    #         if hull_idx is None or len(hull_idx) < 4:
    #             return 0.0
    #         defects = cv2.convexityDefects(cnt, hull_idx)
    #         if defects is None:
    #             return 0.0
    #         depths = defects[:, 0, 3] / 256.0
    #         significant = np.sum(depths > 2.0)
    #         return float(min(1.0, significant / 12.0))
    #     except cv2.error:
    #         return 0.0
    @staticmethod
    def _branching_score(cnt) -> float:
        """Rough proxy for how 'net-like' a contour is, via convexity-defect density."""
        try:
            if cnt is None or len(cnt) < 4:
                return 0.0

            hull_idx = cv2.convexHull(cnt, returnPoints=False)
            if hull_idx is None or len(hull_idx) < 4:
                return 0.0

            defects = cv2.convexityDefects(cnt, hull_idx)
            if defects is None or defects.size == 0:
                return 0.0

            # OpenCV may return different array shapes; normalize before indexing.
            defects = np.asarray(defects).reshape(-1, 4)
            depths = defects[:, 3].astype(float) / 256.0

            significant = np.sum(depths > 2.0)
            return float(min(1.0, significant / 12.0))

        except (cv2.error, ValueError, IndexError, TypeError):
            return 0.0



# ---------------------------------------------------------------------------
# Backend 2: real YOLOv7 ensemble (PRIMARY detector when its dependencies exist)
# ---------------------------------------------------------------------------

# RDD2022 class order as declared in the original project's data/crdd22.yaml:
#   names: ['D00', 'D10', 'D20', 'D40']
YOLO_CLASS_MAP = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}

DEFAULT_YOLOV7_REPO = "models/yolov7_repo"
DEFAULT_WEIGHTS_640 = "models/weights/yolov7all6_best.pt"
DEFAULT_WEIGHTS_512 = "models/weights/yolov7all61_best.pt"


class DetectorUnavailableError(RuntimeError):
    """Raised when YoloV7EnsembleDetector cannot run, with a precise reason."""


class YoloV7EnsembleDetector(RoadDamageDetector):
    """
    The project's real detector: an ensemble of two custom-trained YOLOv7
    models (640x640 and 512x512), merged with NMS - the exact pipeline
    described in the original repo's `index.ipynb`.

    THIS IS THE PRIMARY DETECTOR. `HeuristicCVDetector` only runs as a
    fallback when this class cannot be activated (see `get_default_detector`
    below), and the app always labels, in the UI and in every report, which
    of the two actually produced a given result - it never presents
    heuristic output as YOLOv7 output.

    To activate, all three must be true:
      1. `pip install torch torchvision` in this environment.
      2. The `WongKinYiu/yolov7` repository checked out and importable at
         `yolov7_repo_path` (default: `models/yolov7_repo/`) - the trained
         checkpoints were pickled against its custom `models/experimental.py`
         classes, so plain `torch.load` without that repo on `sys.path` will
         fail to unpickle them.
      3. Both trained checkpoints present on disk:
           - 640x640 model -> `weights_640` (default `models/weights/yolov7all6_best.pt`)
           - 512x512 model -> `weights_512` (default `models/weights/yolov7all61_best.pt`)

    NOTE ON VERIFICATION: this class was written to this sandbox's best
    understanding of the yolov7 repo's public loading/inference API
    (`models.experimental.attempt_load`, `utils.datasets.letterbox`,
    `utils.general.non_max_suppression` + `scale_coords`,
    `utils.torch_utils.select_device`), matching the pattern used in
    `index.ipynb`. It has NOT been executed in this environment, because
    `torch` is not installed here and network access to install it or fetch
    the yolov7 repo is blocked. Run `pytest tests/test_yolo_loading.py`
    (see `tests/`) after installing the dependencies above to confirm it
    loads and infers correctly on your machine before trusting its output.
    """

    name = "yolov7_ensemble"

    def __init__(
        self,
        weights_640: str = DEFAULT_WEIGHTS_640,
        weights_512: str = DEFAULT_WEIGHTS_512,
        yolov7_repo_path: str = DEFAULT_YOLOV7_REPO,
        conf_640: float = 0.10,
        conf_512: float = 0.25,
        img_size_640: int = 640,
        img_size_512: int = 512,
        iou_thres: float = 0.45,
        device: str = "cpu",
    ):
        self.weights_640 = weights_640
        self.weights_512 = weights_512
        self.yolov7_repo_path = yolov7_repo_path
        self.conf_640 = conf_640
        self.conf_512 = conf_512
        self.img_size_640 = img_size_640
        self.img_size_512 = img_size_512
        self.iou_thres = iou_thres
        self.device_str = device

        self.ready = False
        self.unavailable_reason: Optional[str] = None
        self._model_640 = None
        self._model_512 = None
        self._device = None
        self._letterbox = None
        self._non_max_suppression = None
        self._scale_coords = None

        self._try_load()

    # -- loading ----------------------------------------------------------
    def _try_load(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.unavailable_reason = (
                "`torch` is not installed. Run: pip install torch torchvision"
            )
            return

        if not os.path.isdir(self.yolov7_repo_path):
            self.unavailable_reason = (
                f"yolov7 repo not found at '{self.yolov7_repo_path}'. "
                f"Clone https://github.com/WongKinYiu/yolov7 into that path."
            )
            return

        missing = [p for p in (self.weights_640, self.weights_512) if not os.path.isfile(p)]
        if missing:
            self.unavailable_reason = (
                "Missing trained weight file(s): " + ", ".join(missing) +
                ". Place yolov7all6's best.pt at weights_640 and "
                "yolov7all61's best.pt at weights_512."
            )
            return

        try:
            import torch
            if self.yolov7_repo_path not in sys.path:
                sys.path.insert(0, self.yolov7_repo_path)
            from models.experimental import attempt_load          # type: ignore
            from utils.datasets import letterbox                  # type: ignore
            from utils.general import non_max_suppression, scale_coords  # type: ignore
            from utils.torch_utils import select_device            # type: ignore

            self._device = select_device(self.device_str)
            self._model_640 = attempt_load(self.weights_640, map_location=self._device)
            self._model_512 = attempt_load(self.weights_512, map_location=self._device)
            self._model_640.eval()
            self._model_512.eval()
            self._letterbox = letterbox
            self._non_max_suppression = non_max_suppression
            self._scale_coords = scale_coords
            self.ready = True
        except Exception as e:  # pragma: no cover - cannot execute in this sandbox
            self.unavailable_reason = f"Failed to load YOLOv7 models: {e!r}"

    # -- inference ----------------------------------------------------------
    def _infer_one(self, model, image_bgr: np.ndarray, img_size: int, conf_thres: float):
        import torch
        img0 = image_bgr
        img = self._letterbox(img0, img_size, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img)
        img_t = torch.from_numpy(img).to(self._device).float() / 255.0
        if img_t.ndimension() == 3:
            img_t = img_t.unsqueeze(0)

        with torch.no_grad():
            pred = model(img_t, augment=False)[0]
        pred = self._non_max_suppression(pred, conf_thres, self.iou_thres)[0]

        results = []
        if pred is not None and len(pred):
            pred[:, :4] = self._scale_coords(img_t.shape[2:], pred[:, :4], img0.shape).round()
            for *xyxy, conf, cls in pred.tolist():
                results.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), float(conf), int(cls)))
        return results

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        if not self.ready:
            raise DetectorUnavailableError(self.unavailable_reason or "YOLOv7 detector not initialized.")

        raw_640 = self._infer_one(self._model_640, image_bgr, self.img_size_640, self.conf_640)
        raw_512 = self._infer_one(self._model_512, image_bgr, self.img_size_512, self.conf_512)

        merged: List[Detection] = []
        for (x1, y1, x2, y2, conf, cls) in raw_640 + raw_512:
            class_id = YOLO_CLASS_MAP.get(cls)
            if class_id is None:
                continue
            merged.append(Detection(
                class_id=class_id,
                class_name=CLASS_NAMES.get(class_id, class_id),
                confidence=float(conf),
                bbox=(x1, y1, x2, y2),
                mask=None,  # YOLOv7 here is a detector, not a segmentation model -
                            # see segmentation note in explain.py / README
                features={"source_model": "yolov7_ensemble"},
            ))
        return _nms(merged, iou_thresh=self.iou_thres)



class YOLO26RoadDamageDetector(RoadDamageDetector):
    """
    Pretrained YOLO26 road-damage detector.

    Uses the YOLO26s RDD-FRDC Distilled model.
    Classes:
        0 -> D00 Longitudinal Crack
        1 -> D10 Transverse Crack
        2 -> D20 Alligator Crack
        3 -> D40 Pothole
    """

    name = "yolo26_rdd"

    def __init__(
        self,
        weights_path: str = "models/weights/road_damage_yolo26s.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "",
    ):
        self.weights_path = weights_path
        self.conf = conf
        self.iou = iou
        self.device = device
        self.model = None
        self.ready = False
        self.unavailable_reason = None

        self._try_load()

    def _try_load(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            self.unavailable_reason = (
                "Ultralytics is not installed. "
                "Run: pip install ultralytics"
            )
            return

        if not os.path.isfile(self.weights_path):
            self.unavailable_reason = (
                f"YOLO26 weights not found at '{self.weights_path}'."
            )
            return

        try:
            self.model = YOLO(self.weights_path)
            self.ready = True
        except Exception as e:
            self.unavailable_reason = (
                f"Failed to load YOLO26 model: {e!r}"
            )

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        if not self.ready:
            raise DetectorUnavailableError(
                self.unavailable_reason or
                "YOLO26 detector not initialized."
            )

        results = self.model.predict(
            source=image_bgr,
            conf=self.conf,
            iou=self.iou,
            device=self.device if self.device else None,
            verbose=False,
        )

        result = results[0]
        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        names = result.names

        class_map = {
            0: 0,  # D00 Longitudinal Crack
            1: 1,  # D10 Transverse Crack
            2: 2,  # D20 Alligator Crack
            3: 3,  # D40 Pothole
        }

        for box in result.boxes:
            cls_id = int(box.cls.item())
            confidence = float(box.conf.item())

            if cls_id not in class_map:
                continue

            class_id = class_map[cls_id]

            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=CLASS_NAMES.get(
                        class_id,
                        names.get(cls_id, str(cls_id))
                    ),
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    mask=None,
                    features={
                        "source_model": "yolo26_rdd",
                        "model_name": "YOLO26s_RDD_FRDC_Distilled_v2",
                    },
                )
            )

        return _nms(detections, iou_thresh=self.iou)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

@dataclass
class DetectorStatus:
    detector: RoadDamageDetector
    backend_used: str            # "yolov7_ensemble" or "heuristic_cv_v1"
    is_primary: bool              # True only if YOLOv7 actually loaded
    message: str


def get_default_detector(
    weights_path: str = "models/weights/road_damage_yolo26s.pt",
) -> DetectorStatus:
    """
    Select the primary RoadScan detector.

    YOLO26 is the primary trained deep-learning detector.
    The classical OpenCV detector is used only if YOLO26 cannot load.
    """

    yolo26 = YOLO26RoadDamageDetector(
        weights_path=weights_path,
    )

    if yolo26.ready:
        return DetectorStatus(
            detector=yolo26,
            backend_used=yolo26.name,
            is_primary=True,
            message=(
                "YOLO26 road-damage detector loaded and active. "
                "Using the pretrained YOLO26s RDD-FRDC Distilled model."
            ),
        )

    fallback = HeuristicCVDetector()

    return DetectorStatus(
        detector=fallback,
        backend_used=fallback.name,
        is_primary=False,
        message=(
            "YOLO26 detector is NOT active - "
            + (yolo26.unavailable_reason or "unknown reason")
            + f" Falling back to '{fallback.name}' "
            "(classical computer-vision baseline, not a trained "
            "neural network). Every result produced while this "
            "fallback is active is labeled as such in the UI "
            "and in exported reports."
        ),
    )

