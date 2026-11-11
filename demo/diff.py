"""Change detection: match detections between two photos of the same shelf view.

Strategy (demo-grade, honest about its limits):
  1. Hungarian assignment between baseline items and current items, scored by
     spatial proximity + appearance similarity of the crops.
  2. Unmatched baseline item  -> MISSING
     Unmatched current item   -> NEW
  3. Second pass: a MISSING and a NEW that look like the same object -> MOVED.
     Without this, staff tidying a shelf would report as theft.
  4. Matched pair whose appearance diverged -> SWAPPED (different watch in slot).

Angle note: each view (front / left / right) is diffed against the baseline of
the *same* view. Cross-angle matching is not attempted here - see MVP-PLAN.md.
"""
from __future__ import annotations

import math
from collections import Counter  # noqa: F401  (kept for future per-kind rollups)

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

import config
import imaging


def _spatial(a: list[float], b: list[float]) -> float:
    """0..1 proximity. IoU when boxes overlap, else decays with centre distance."""
    ov = imaging.iou(a, b)
    if ov > 0.05:
        return min(1.0, 0.55 + ov)
    ax, ay = imaging.center(a)
    bx, by = imaging.center(b)
    dist = math.hypot(ax - bx, ay - by)
    return max(0.0, 1.0 - dist / 0.22)


def compare(
    base_img: Image.Image,
    cap_img: Image.Image,
    base_items: list[dict],
    cap_items: list[dict],
) -> dict:
    """Return a summary dict with per-item kinds and counts."""
    n, m = len(base_items), len(cap_items)
    base = [dict(it) for it in base_items]
    cap = [dict(it) for it in cap_items]
    for i, it in enumerate(base):
        it["n"] = i + 1
        it["kind"] = "unknown"
    for i, it in enumerate(cap):
        it["n"] = i + 1
        it["kind"] = "unknown"

    pairs: list[tuple[int, int, float, float]] = []
    if n and m:
        cost = np.zeros((n, m), dtype=np.float64)
        appear_cache: dict[tuple[int, int], float] = {}
        for i, bi in enumerate(base):
            bcrop = imaging.crop_norm(base_img, bi["box"])
            for j, ci in enumerate(cap):
                ap = imaging.appearance_similarity(bcrop, imaging.crop_norm(cap_img, ci["box"]))
                appear_cache[(i, j)] = ap
                cost[i, j] = -(0.55 * _spatial(bi["box"], ci["box"]) + 0.45 * ap)
        r, c = linear_sum_assignment(cost)
        for i, j in zip(r, c):
            pairs.append((int(i), int(j), float(-cost[i, j]), appear_cache[(int(i), int(j))]))

    pairs.sort(key=lambda t: -t[2])
    used_b: set[int] = set()
    used_c: set[int] = set()
    for i, j, score, ap in pairs:
        if i in used_b or j in used_c:
            continue
        if score < config.MATCH_MIN_SCORE:
            continue
        used_b.add(i)
        used_c.add(j)
        if ap >= config.SAME_OBJECT_APPEAR:
            base[i]["kind"] = cap[j]["kind"] = "unchanged"
        else:
            base[i]["kind"] = cap[j]["kind"] = "swapped"
        base[i]["match"] = cap[j]["match"] = {"score": round(score, 3), "appear": round(ap, 3)}

    unmatched_b = [i for i in range(n) if i not in used_b]
    unmatched_c = [j for j in range(m) if j not in used_c]

    # Pass 2: moved = same object, different place.
    moves: list[dict] = []
    for i in list(unmatched_b):
        best, best_ap = None, 0.0
        bcrop = imaging.crop_norm(base_img, base[i]["box"])
        for j in list(unmatched_c):
            ap = imaging.appearance_similarity(bcrop, imaging.crop_norm(cap_img, cap[j]["box"]))
            if ap > best_ap:
                best, best_ap = j, ap
        if best is not None and best_ap >= config.MOVED_MIN_APPEAR:
            base[i]["kind"] = "moved_out"
            cap[best]["kind"] = "moved_in"
            base[i]["match"] = cap[best]["match"] = {"appear": round(best_ap, 3)}
            moves.append(
                {
                    "from": base[i]["n"],
                    "to": cap[best]["n"],
                    "appear": round(best_ap, 3),
                    "desc": base[i].get("desc", ""),
                    "box": base[i]["box"],
                }
            )
            unmatched_b.remove(i)
            unmatched_c.remove(best)

    for i in unmatched_b:
        base[i]["kind"] = "missing"
    for j in unmatched_c:
        cap[j]["kind"] = "new"

    changes = [it for it in base if it["kind"] in ("missing", "swapped")] + [
        it for it in cap if it["kind"] == "new"
    ] + [
        {"kind": "moved", "n": mv["from"], "to": mv["to"],
         "desc": mv["desc"], "box": mv["box"]}
        for mv in moves
    ]

    base_total = sum(int(it.get("count", 1)) for it in base)
    cap_total = sum(int(it.get("count", 1)) for it in cap)

    return {
        "base_total": base_total,
        "cap_total": cap_total,
        "delta": cap_total - base_total,
        "counts": {
            "missing": sum(1 for it in base if it["kind"] == "missing"),
            "new": sum(1 for it in cap if it["kind"] == "new"),
            "moved": len(moves),
            "swapped": sum(1 for it in base if it["kind"] == "swapped"),
            "unchanged": sum(1 for it in base if it["kind"] == "unchanged"),
        },
        "changes": sorted(
            [
                {
                    "kind": c["kind"],
                    "n": c.get("n"),
                    "to": c.get("to"),
                    "desc": (c.get("desc") or "")[:90],
                    "box": [round(float(v), 4) for v in c["box"]],
                }
                for c in changes
            ],
            key=lambda c: str(c["kind"]),
        ),
        "moves": moves,
        "base_items": base,
        "cap_items": cap,
        "clean": not changes,
    }


def verdict(summary: dict) -> tuple[str, str]:
    """(css_class, human headline) for the UI."""
    c = summary["counts"]
    if summary.get("error"):
        return "bad", "Could not analyse this photo"
    if not c["missing"] and not c["new"] and not c["swapped"]:
        if c["moved"]:
            return "warn", f"Same {summary['cap_total']} watches, {c['moved']} moved"
        return "ok", f"No change - {summary['cap_total']} watches"
    parts = []
    if c["missing"]:
        parts.append(f"{c['missing']} missing")
    if c["new"]:
        parts.append(f"{c['new']} new")
    if c["swapped"]:
        parts.append(f"{c['swapped']} swapped")
    if c["moved"]:
        parts.append(f"{c['moved']} moved")
    return ("bad" if c["missing"] else "warn"), ", ".join(parts)
