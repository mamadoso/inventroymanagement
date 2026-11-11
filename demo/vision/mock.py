"""Heuristic detector so the demo works with no API key at all.

This is NOT computer vision research - it is a grid activity detector: cells with
enough local contrast/colour variance are treated as occupied. It responds to a
watch physically being removed, which is enough to demonstrate the workflow and
the report UI offline. Real counting needs AHADI_PROVIDER=gemini or openai.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
from PIL import Image, ImageFilter

from vision.base import Detection, build_detection

_SHADES = [
    (20, "very dark dial"),
    (45, "dark dial"),
    (75, "mid-tone dial"),
    (120, "light dial"),
    (256, "very light dial"),
]


def _grid_dims(w: int, h: int) -> tuple[int, int]:
    if w >= h:
        return 3, 6
    return 6, 3


def _desc_for(cell: np.ndarray) -> str:
    hsv = Image.fromarray(cell).convert("HSV")
    a = np.asarray(hsv)
    v = float(np.median(a[..., 2]))
    s = float(np.median(a[..., 1]))
    dial = next(lbl for thr, lbl in _SHADES if v < thr)
    strap = "metal bracelet" if s < 45 else "coloured strap"
    return f"{dial}, {strap}"


def detect(img: Image.Image, model: str = "heuristic-grid") -> Detection:
    started = time.perf_counter()
    g = img.convert("L").filter(ImageFilter.GaussianBlur(1))
    arr = np.asarray(g, dtype=np.float64)
    rgb = np.asarray(img.convert("RGB"))
    rows, cols = _grid_dims(img.width, img.height)
    ch, cw = arr.shape[0] // rows, arr.shape[1] // cols

    activity: list[tuple[float, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * ch, (r + 1) * ch if r < rows - 1 else arr.shape[0]
            x0, x1 = c * cw, (c + 1) * cw if c < cols - 1 else arr.shape[1]
            cell = arr[y0:y1, x0:x1]
            if cell.size < 400:
                continue
            gy, gx = np.gradient(cell)
            edge = float(np.mean(np.hypot(gx, gy)))
            contrast = float(np.std(cell))
            activity.append((0.6 * edge + 0.4 * contrast, r, c))

    if not activity:
        return build_detection({"items": [], "quality": "other"}, "mock", model, 0)

    vals = np.array([a[0] for a in activity])
    mx, mn = float(vals.max()), float(vals.min())
    if mx - mn < 4.0:
        # Flat frame (no texture anywhere): report nothing rather than everything.
        floor = mx + 1.0
    else:
        floor = mn + 0.35 * (mx - mn)

    # Stable jitter so repeated identical files give identical answers.
    seed = int(hashlib.sha256(img.tobytes()[:65536]).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    items = []
    for score, r, c in activity:
        if score < floor:
            continue
        y0, y1 = r * ch, (r + 1) * ch if r < rows - 1 else arr.shape[0]
        x0, x1 = c * cw, (c + 1) * cw if c < cols - 1 else arr.shape[1]
        inset_x, inset_y = (x1 - x0) * 0.16, (y1 - y0) * 0.16
        bx = (x0 + inset_x) / img.width
        by = (y0 + inset_y) / img.height
        bw = (x1 - x0 - 2 * inset_x) / img.width
        bh = (y1 - y0 - 2 * inset_y) / img.height
        cell = rgb[y0:y1, x0:x1]
        items.append(
            {
                "box": [round(bx, 4), round(by, 4), round(bw, 4), round(bh, 4)],
                "desc": _desc_for(cell),
                "conf": round(min(0.95, 0.55 + float(rng.random()) * 0.35), 2),
            }
        )

    parsed = {
        "items": items,
        "total": len(items),
        "quality": "good",
        "notes": "offline heuristic grid detector - set AHADI_AI_API_KEY for real AI",
    }
    ms = int((time.perf_counter() - started) * 1000)
    return build_detection(parsed, "mock", model, ms)
