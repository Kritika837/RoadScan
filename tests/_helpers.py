"""Shared test helpers: an isolated copy of the app + synthetic road photos."""
import io
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def isolated_app_copy():
    """
    Copy the project to a temp dir, chdir into it and put it first on sys.path,
    so tests never touch the real data/history.sqlite3 or static/ outputs.
    Returns the temp dir path.
    """
    tmp = tempfile.mkdtemp(prefix="roadscan_test_")
    dst = os.path.join(tmp, "app")

    def ignore(d, names):
        skip = {"__pycache__"}
        rel = os.path.relpath(d, SRC_ROOT)
        if rel in ("static/uploads", "static/results", "static/reports"):
            skip |= {n for n in names if n != ".gitkeep"}
        return skip

    shutil.copytree(SRC_ROOT, dst, ignore=ignore)
    os.chdir(dst)
    sys.path.insert(0, dst)
    return dst


def road_image_bytes(seed: int = 1, damaged: bool = True) -> io.BytesIO:
    """Synthetic asphalt photo (different seed -> different bytes/hash)."""
    rng = np.random.default_rng(seed)
    img = np.full((720, 960, 3), 120, np.uint8)
    img = cv2.add(img, rng.integers(0, 40, img.shape, dtype=np.uint8))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    if damaged:
        cx, cy = int(300 + rng.integers(0, 300)), int(400 + rng.integers(0, 150))
        cv2.ellipse(img, (cx, cy), (90, 55), int(rng.integers(0, 90)), 0, 360, (25, 25, 25), -1)
        cv2.ellipse(img, (cx, cy), (100, 65), 0, 0, 360, (70, 70, 70), 4)
        p = np.array([[80, 100], [300, 180], [420, 150], [700, 260]], np.int32)
        cv2.polylines(img, [p + int(rng.integers(0, 40))], False, (15, 15, 15), 3)
    ok, buf = cv2.imencode(".jpg", img)
    return io.BytesIO(buf.tobytes())


def ensure_sample_uploads(n: int = 4):
    """Older tests pick sample photos from static/uploads; write synthetic ones if none exist."""
    up = os.path.join("static", "uploads")
    os.makedirs(up, exist_ok=True)
    have = [f for f in os.listdir(up) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for i in range(len(have), n):
        with open(os.path.join(up, f"sample{i + 1}.jpg"), "wb") as f:
            f.write(road_image_bytes(i + 1).getvalue())
