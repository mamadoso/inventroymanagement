"""Compare vision models on one real shelf photo.

    .venv/bin/python modeltest.py data/images/5/front/somephoto.jpg
    .venv/bin/python modeltest.py photo.jpg --models sfnx,flash-next --runs 2
    .venv/bin/python modeltest.py photo.jpg --pair second.jpg   # check a real diff

Reports count, latency, whether the JSON survived, and - with --pair - what the
diff says. Run it from the demo directory so it picks up .env.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

import config
import diff as diffmod
import vision
from vision import openai_compat


def detect_ms(model: str, img: Image.Image) -> tuple[int, float, str, str]:
    config.AI_MODEL = model
    t0 = time.perf_counter()
    try:
        det = openai_compat.detect(img)
        return det.total, (time.perf_counter() - t0) * 1000, det.quality, ""
    except Exception as exc:  # noqa: BLE001 - a failing model is a result, not a crash
        return -1, (time.perf_counter() - t0) * 1000, "error", str(exc).replace("\n", " ")[:120]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("photo", type=Path, help="shelf photo to test")
    ap.add_argument("--pair", type=Path, default=None,
                    help="second photo; reports missing/new/moved against the first")
    ap.add_argument("--models", default="", help="comma-separated ids (default: gateway list)")
    ap.add_argument("--runs", type=int, default=2, help="repeats per model, to check stability")
    args = ap.parse_args()

    img = Image.open(args.photo).convert("RGB")
    pair = Image.open(args.pair).convert("RGB") if args.pair else None
    print(f"photo      : {args.photo}  {img.width}x{img.height}")
    print(f"base_url   : {config.normalized_base_url()}")
    print(f"max_tokens : {config.AI_MAX_TOKENS} (cap {config.AI_MAX_TOKENS_CAP})")
    print(f"deadline   : {config.AI_DEADLINE:.0f}s per detection\n")

    models = [m for m in args.models.split(",") if m.strip()]
    if not models:
        import httpx

        r = httpx.get(config.models_url(), timeout=20,
                      headers={"Authorization": f"Bearer {config.AI_API_KEY}"})
        names = [m.get("id", "") for m in r.json().get("data", [])]
        models = [m for m in names if m and m != "auto"] or names

    hdr = f"{'model':<30} {'run':<4} {'count':<6} {'ms':<6} {'quality':<9} notes"
    print(hdr)
    print("-" * len(hdr))
    for m in models:
        counts = []
        for run in range(1, args.runs + 1):
            n, ms, q, err = detect_ms(m, img)
            counts.append(n)
            print(f"{m:<30} {run:<4} {n:<6} {ms:<6.0f} {q:<9} {err or ''}", flush=True)
        stable = len(set(counts)) == 1
        if args.runs > 1:
            print(f"{m:<30} {'':<4} counts={counts} {'STABLE' if stable else 'UNSTABLE - do not trust'}",
                  flush=True)
        if pair is not None and stable and counts[0] > 0:
            config.AI_MODEL = m
            d1, d2 = openai_compat.detect(img), openai_compat.detect(pair)
            c = diffmod.compare(img, pair, d1.items, d2.items)["counts"]
            print(f"{m:<30} {'':<4} diff {d1.total}->{d2.total}: "
                  f"missing={c['missing']} new={c['new']} moved={c['moved']} "
                  f"swapped={c['swapped']}", flush=True)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
