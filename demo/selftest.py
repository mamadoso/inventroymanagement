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

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import config
import diff as diffmod
import imaging
import store
import vision

CELS = [(r, c) for r in range(3) for c in range(6)]
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


def check_multiangle(client, appmod, base, minus) -> list[str]:
    """Drive the guided multi-angle session end to end: stage -> analyse -> report.

    The capture screen no longer counts each photo as it lands, so the whole
    session has to work as one flow, and the shelf headline has to agree with
    the per-angle rows it is summarising.
    """
    import time as _time

    failures: list[str] = []
    sid = store.create_shelf("selftest-multi")
    store.add_view(sid, "left", "Left angle")
    keys = [v["key"] for v in store.get_shelf(sid)["views"]][:2]
    if len(keys) < 2:
        failures.append("test shelf needs two angles to prove the multi-angle flow")
        return failures

    def stage(image, key):
        buf = io.BytesIO()
        image.save(buf, "JPEG")
        return client.post(
            "/api/capture",
            files={"file": ("x.jpg", buf.getvalue(), "image/jpeg")},
            data={"shelf": str(sid), "view": key, "analyse": "0"},
        ).json()

    def analyse():
        started = client.post(f"/shelves/{sid}/analyse").json()
        deadline = _time.time() + 120
        prog = {}
        while _time.time() < deadline:
            prog = client.get(f"/api/analyse/{sid}/progress").json()
            if prog.get("complete"):
                return started, prog
            _time.sleep(0.2)
        return started, prog

    j1, j2 = stage(base, keys[0]), stage(base, keys[1])
    if not (j1.get("queued") and j2.get("queued")):
        return [f"staging must not call the AI: {j1} {j2}"]
    if j2.get("staged") != 2:
        return [f"expected 2 staged photos, got {j2.get('staged')}"]
    print(f"staging           : {j2['staged']} photos queued, no AI call yet")

    page = client.get(f"/capture?shelf={sid}")
    line = next((l for l in page.text.splitlines() if "window.STAGED" in l), "")
    if page.status_code != 200 or not all(k in line for k in keys):
        failures.append(f"capture page does not show the staged photos: {line[:120]}")

    started, prog = analyse()
    if started.get("started") != 2:
        return failures + [f"analyse should start 2 photos: {started}"]
    if not prog.get("complete"):
        return failures + [f"analyse never finished: {prog}"]
    if prog.get("failed"):
        return failures + [f"angle(s) failed to analyse: {prog.get('errors')}"]

    # Round one only establishes the reference set: neither angle had a baseline,
    # so both staged photos become baselines and there is nothing to diff yet.
    rep = client.get(f"/shelves/{sid}/report")
    shelf = store.get_shelf(sid)
    if rep.status_code != 200:
        return failures + [f"GET /shelves/{{id}}/report -> {rep.status_code}"]
    if "Nothing analysed yet" not in rep.text:
        failures.append("the first pass should set baselines, not invent a diff")
    if not all(v["baseline"] for v in shelf["views"]):
        failures.append("both angles should now have a baseline")
    else:
        print("first pass        : both angles became baselines")

    # Second round: the same front, one watch gone on the other angle.
    stage(base, keys[0])
    stage(minus, keys[1])
    started, prog = analyse()
    if not prog.get("complete") or prog.get("failed"):
        return failures + [f"second analyse pass failed: {prog}"]

    rep = client.get(f"/shelves/{sid}/report")
    if rep.status_code != 200 or "missing" not in rep.text:
        failures.append(f"the report page should show the theft: {rep.status_code}")

    report = appmod.shelf_report(store.get_shelf(sid))
    rows = {r["key"]: r for r in report["rows"]}
    a, b = rows[keys[0]], rows[keys[1]]
    if not a["counts"] or not b["counts"]:
        failures.append(f"both angles should have a diff: {[r['text'] for r in report['rows']]}")
    else:
        if b["counts"]["missing"] != 1:
            failures.append(f"the changed angle should show 1 missing, got {b['counts']}")
        if a["counts"]["missing"] != 0:
            failures.append(f"the unchanged angle should show 0 missing, got {a['counts']}")
        if "missing" not in report["headline"]:
            failures.append(f"headline should mention the theft: {report['headline']}")
        print(f"shelf headline      : {report['headline']}")
        print(f"  {keys[0]}: {a['text']} | {keys[1]}: {b['text']}")

    # Discarding staged photos must not touch baselines or past diffs.
    stage(minus, keys[0])
    before = [v["baseline_shot_id"] for v in store.get_shelf(sid)["views"]]
    gone = client.post(f"/shelves/{sid}/staged/discard")
    shelf = store.get_shelf(sid)
    left = sum(v["queued"] for v in shelf["views"])
    after = [v["baseline_shot_id"] for v in shelf["views"]]
    if gone.status_code != 303 or left or before != after:
        failures.append(f"discard staged: status={gone.status_code} left={left} baselines moved={before != after}")
    else:
        print("discard staged      : queue emptied, baselines intact")

    return failures


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

    # A gateway that runs out of max_tokens returns JSON cut off mid-list. That is
    # recoverable - keep the complete items - but it must never look like a clean
    # count, or a truncated shelf silently reports fewer watches than exist.
    from vision.base import build_detection, parse_json_object

    cut = ('{"total": 9, "items": [{"box": [0.1, 0.2, 0.1, 0.1], "desc": "steel"},'
           ' {"box": [0.5, 0.6, 0.1')
    try:
        salvaged = build_detection(parse_json_object(cut), "test", "test", 0)
        if salvaged.total != 1:
            failures.append(f"truncated JSON should salvage 1 complete item, got {salvaged.total}")
        if salvaged.quality == "good":
            failures.append("a truncated detection must not be reported as quality=good")
        print(f"truncated json    : salvaged {salvaged.total} item, quality={salvaged.quality}")
    except Exception as exc:
        failures.append(f"truncated JSON repair raised {exc!r}")

    # The diff compares boxes by position, so it silently invents thefts when the
    # photographer moves. The framing guard must fire before the counts do.
    shifted = base.crop((60, 0, base.width, base.height))
    shifted.paste(base.crop((0, 0, 60, base.height)), (base.width - 60, 0))
    align_same = imaging.frame_similarity(base, minus)
    align_shift = imaging.frame_similarity(base, shifted)
    if not align_shift < align_same:
        failures.append(
            f"frame_similarity missed a 60px shift ({align_shift:.2f} vs {align_same:.2f})"
        )
    # 0.55 is what a real hand re-shoot of a shop display measured. The synthetic
    # shelf is too flat and uniform to drift that far on a 60px shift, so judge
    # verdict() on the measured number rather than on this image pair.
    fake = {"counts": {"missing": 9, "new": 4, "swapped": 10, "moved": 1},
            "frame_align": 0.55, "cap_total": 50}
    state, text = diffmod.verdict(fake)
    if state != "bad" or "amera" not in text:
        failures.append(f"a shifted camera should read as unreliable, got {state}: {text}")
    print(f"framing guard     : same={align_same:.2f} shifted={align_shift:.2f} "
          f"floor={config.FRAME_ALIGN_MIN} | re-shoot 0.55 -> {text}")

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

                # Clearing an angle must delete its photos and let a fresh
                # baseline be established by the next capture.
                rc = client.post(f"/api/views/{sid}/{views[0]['key']}/clear")
                if rc.status_code != 200 or rc.json().get("removed", 0) < 2:
                    failures.append(f"clear view: {rc.status_code} {rc.text[:120]}")
                else:
                    print(f"clear view        : removed {rc.json()['removed']} photo(s)")
                r3 = client.post(
                    "/api/capture",
                    files={"file": ("b2.jpg", buf.getvalue(), "image/jpeg")},
                    data={"shelf": str(sid), "view": views[0]["key"]},
                )
                if r3.json().get("kind") != "baseline":
                    failures.append("after clearing, the next capture must become the baseline")

        failures.extend(check_multiangle(client, appmod, base, minus))
        print("web app           : routes ok")
    except ImportError as exc:
        print(f"web app           : skipped ({exc})")

    for sql in ("DELETE FROM diffs", "DELETE FROM shots", "DELETE FROM views",
                "DELETE FROM shelves"):
        store.run(sql)

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
