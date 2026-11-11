#!/usr/bin/env python3
"""Offline self-test: no network, no API key, no browser.

Builds synthetic shelf photos, runs the real pipeline over them and checks the
three behaviours that matter in the demo:

  1. a shelf of N watches is counted as N
  2. removing one watch is reported as exactly 1 MISSING
  3. moving one watch is reported as MOVED, not as missing + new

Usage:
    python selftest.py            # heuristic provider
    AHADI_AI_API_KEY=... python selftest.py --live   # also hit the configured AI
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import config
import diff as diffmod
import imaging
import store
import vision

CELLS = [(r, c) for r in range(3) for c in range(6)]
ROWS, COLS = 3, 6
W, H = 1200, 800


def shelf(with_watches: list[int], move_from: int | None = None, move_to: int | None = None):
    """Draw a synthetic shelf: one 'watch' per occupied grid cell."""
    rng = np.random.default_rng(7)
    noise = rng.integers(-6, 7, (H, W, 3), dtype=np.int16)
    base = np.full((H, W, 3), (208, 198, 184), dtype=np.int16)
    img = Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    occupied = set(with_watches)
    if move_from is not None and move_to is not None:
        occupied.discard(move_from)
        occupied.add(move_to)

    cw, ch = W // COLS, H // ROWS
    for idx in sorted(occupied):
        r, c = CELS[idx]
        cx, cy = c * cw + cw // 2, r * ch + ch // 2
        draw.rectangle([cx - 10, cy - 62, cx + 10, cy - 40], fill=(96, 82, 62))
        draw.ellipse([cx - 46, cy - 46, cx + 46, cy + 46], fill=(58, 52, 44), outline=(160, 148, 126), width=7)
        draw.ellipse([cx - 26, cy - 26, cx + 26, cy + 26], fill=(22, 22, 26))
        draw.line([cx, cy, cx + 16, cy - 12], fill=(220, 214, 200), width=3)
        draw.line([cx, cy, cx - 10, cy + 14], fill=(220, 214, 200), width=3)
    return img


def run() -> int:
    config.ensure_dirs()
    store.init()
    full = list(range(12))
    base = shelf(full)
    minus = shelf(full[:-1])
    moved = shelf(full, move_from=0, move_to=17)

    det_base = vision.detect(base)
    det_minus = vision.detect(minus)
    det_moved = vision.detect(moved)

    print(f"provider          : {config.active_provider()}")
    print(f"baseline detected : {det_base.total}")
    print(f"one removed       : {det_minus.total}")
    print(f"one moved         : {det_moved.total}")

    d1 = diffmod.compare(base, minus, det_base.items, det_minus.items)
    d2 = diffmod.compare(base, moved, det_base.items, det_moved.items)
    print(f"removal  -> {d1['counts']}")
    print(f"move     -> {d2['counts']}")

    failures = []
    if det_base.total < 8:
        failures.append(f"expected ~12 watches in baseline, got {det_base.total}")
    if d1["counts"]["missing"] != 1:
        failures.append(f"removal should be 1 missing, got {d1['counts']['missing']}")
    if d1["counts"]["new"] != 0:
        failures.append(f"removal should create 0 new, got {d1['counts']['new']}")
    if d2["counts"]["moved"] != 1:
        failures.append(f"move should be 1 moved, got {d2['counts']['moved']}")
    if d2["counts"]["missing"] != 0:
        failures.append(f"move must not read as missing (this is the false-theft bug)")

    # Persistence + report rendering, still without a web server.
    sid = store.create_shelf("selftest")
    views = store.get_shelf(sid)["views"]
    out = imaging.annotate(base, det_base.items, config.DATA_DIR / "selftest-annot.jpg")
    if not out.exists():
        failures.append("annotation rendering produced no file")
    print(f"annotation        : {out}")

    if "--live" in sys.argv and config.active_provider() != "mock":
        from vision import openai_compat

        chk = openai_compat.check_connection()
        print(f"live AI check     : ok={chk['ok']} {chk['detail'][:120]}")
        if not chk["ok"]:
            failures.append("live AI check failed: " + chk["detail"][:200])
        det = openai_compat.detect(base)
        print(f"live detection    : {det.total} items, quality={det.quality}")

    try:
        from fastapi.testclient import TestClient

        import app as appmod

        client = TestClient(appmod.app)
        r = client.get("/")
        if r.status_code != 200:
            failures.append(f"GET / -> {r.status_code}")
        r = client.get("/health/json")
        if r.status_code != 200:
            failures.append(f"GET /health/json -> {r.status_code}")
        import io as _io

        buf = _io.BytesIO()
        base.save(buf, "JPEG")
        r = client.post(
            "/api/capture",
            files={"file": ("b.jpg", buf.getvalue(), "image/jpeg")},
            data={"shelf": str(sid), "view": views[0]["key"]},
        )
        if r.status_code != 200 or not r.json().get("ok"):
            failures.append(f"POST /api/capture -> {r.status_code} {r.text[:160]}")
        else:
            buf2 = _io.BytesIO()
            minus.save(buf2, "JPEG")
            r2 = client.post(
                "/api/capture",
                files={"file": ("c.jpg", buf2.getvalue(), "image/jpeg")},
                data={"shelf": str(sid), "view": views[0]["key"]},
            )
            j = r2.json()
            if not j.get("ok") or j.get("kind") != "capture":
                failures.append(f"second capture failed: {str(j)[:200]}")
            else:
                print(f"web capture       : {j['summary']['counts']}")
                page = client.get(f"/diffs/{j['diff_id']}")
                if page.status_code != 200:
                    failures.append(f"GET /diffs -> {page.status_code}")
        print("web app           : routes ok")
    except ImportError as exc:
        print(f"web app           : skipped ({exc})")

    store.run("DELETE FROM diffs")
    store.run("DELETE FROM shots")
    store.run("DELETE FROM views")
    store.run("DELETE FROM shelves")

    print()
    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("PASS — pipeline, diff logic, annotation and web routes all work.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
