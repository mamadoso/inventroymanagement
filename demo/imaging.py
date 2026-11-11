"""Image handling: normalisation, crops, perceptual signatures, annotation.

Phone photos arrive with EXIF rotation and 12MP dimensions. Everything here
works on the *normalised* RGB image so that normalised bounding boxes from the
model always map back to the stored file correctly.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import config

STYLE_COLORS = {
    "unchanged": (60, 200, 130),
    "missing": (235, 70, 70),
    "new": (70, 150, 255),
    "moved_out": (255, 170, 40),
    "moved_in": (255, 170, 40),
    "unknown": (170, 170, 170),
}


def open_image(path: str | Path) -> Image.Image:
    img = Image.open(str(path))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def save_upload(data: bytes, dest: Path, max_side: int | None = None) -> tuple[int, int]:
    """Decode, auto-rotate, downscale and re-encode as JPEG. Returns (w, h)."""
    max_side = max_side or config.MAX_SIDE
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / float(max(img.size))
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.LANCZOS,
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(dest), "JPEG", quality=config.JPEG_QUALITY, optimize=True)
    return img.width, img.height


def box_to_px(box: list[float], w: int, h: int, pad: float = 0.04) -> tuple[int, int, int, int]:
    """Normalised [x, y, w, h] -> pixel box, slightly padded, clamped."""
    x, y, bw, bh = (float(v) for v in box[:4])
    if bw <= 0 or bh <= 0:
        bw = bh = 0.05
    x0, y0 = x - bw * pad, y - bh * pad
    x1, y1 = x + bw * (1 + pad), y + bh * (1 + pad)
    x0, y0 = max(0, min(w - 1, x0 * w)), max(0, min(h - 1, y0 * h))
    x1, y1 = max(x0 + 1, min(w, x1 * w)), max(y0 + 1, min(h, y1 * h))
    return int(x0), int(y0), int(x1), int(y1)


def center(box: list[float]) -> tuple[float, float]:
    return float(box[0]) + float(box[2]) / 2.0, float(box[1]) + float(box[3]) / 2.0


def iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = (a[2] * a[3]) + (b[2] * b[3]) - inter
    return inter / union if union > 0 else 0.0


def crop_norm(img: Image.Image, box: list[float]) -> Image.Image:
    px = box_to_px(box, img.width, img.height)
    return img.crop(px)


def dhash(img: Image.Image, size: int = 12) -> int:
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = a[:, 1:] > a[:, :-1]
    out = 0
    for b in bits.flatten():
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _hist(img: Image.Image) -> np.ndarray:
    hsv = img.convert("HSV").resize((64, 64), Image.BILINEAR)
    arr = np.asarray(hsv)
    q = np.stack(
        [
            (arr[..., 0] * 6) // 256,
            (arr[..., 1] * 5) // 256,
            (arr[..., 2] * 4) // 256,
        ],
        axis=-1,
    )
    flat = (q[..., 0] * 20 + q[..., 1] * 4 + q[..., 2]).ravel()
    h = np.bincount(flat, minlength=120).astype(np.float64)
    s = h.sum()
    return h / s if s > 0 else h


def appearance_similarity(a: Image.Image, b: Image.Image) -> float:
    """Blend of structural hash and colour histogram. 0 = unrelated, 1 = same object."""
    bits = 144  # 12*12
    s_hash = 1.0 - hamming(dhash(a), dhash(b)) / float(bits)
    ha, hb = _hist(a), _hist(b)
    s_hist = float(np.minimum(ha, hb).sum())
    return 0.5 * s_hash + 0.5 * s_hist


def _font(size: int):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        try:
            return ImageFont.load_default(size)
        except TypeError:  # older Pillow
            return ImageFont.load_default()


def annotate(
    img: Image.Image,
    items: list[dict],
    out_path: Path,
    show_desc: bool = True,
) -> Path:
    """Draw numbered boxes coloured by change kind. Returns the written path."""
    canvas = img.copy()
    d = ImageDraw.Draw(canvas)
    lw = max(2, int(min(canvas.size) / 320))
    fs = max(13, int(min(canvas.size) / 42))
    font = _font(fs)
    for it in items:
        color = STYLE_COLORS.get(it.get("kind", "unknown"), STYLE_COLORS["unknown"])
        x0, y0, x1, y1 = box_to_px(it["box"], canvas.width, canvas.height)
        dashed = it.get("kind") in ("missing", "moved_out", "moved_in")
        if dashed:
            _dash_rect(d, x0, y0, x1, y1, color, lw)
        else:
            d.rectangle([x0, y0, x1, y1], outline=color, width=lw)
        label = str(it.get("n", it.get("slot", "")))
        if show_desc and it.get("desc"):
            label = f"{label} {it['desc']}"
        label = label.strip()
        if label:
            tw = d.textlength(label, font=font)
            ty = max(0, y0 - fs - 6)
            d.rectangle([x0, ty, x0 + tw + 8, ty + fs + 6], fill=color)
            d.text((x0 + 4, ty + 2), label, fill=(15, 15, 15), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path), "JPEG", quality=config.JPEG_QUALITY)
    return out_path


def _dash_rect(d: ImageDraw.ImageDraw, x0, y0, x1, y1, color, lw, dash: int = 10):
    for x in range(int(x0), int(x1), dash * 2):
        d.line([(x, y0), (min(x + dash, x1), y0)], fill=color, width=lw)
        d.line([(x, y1), (min(x + dash, x1), y1)], fill=color, width=lw)
    for y in range(int(y0), int(y1), dash * 2):
        d.line([(x0, y), (x0, min(y + dash, y1))], fill=color, width=lw)
        d.line([(x1, y), (x1, min(y + dash, y1))], fill=color, width=lw)


def contact_sheet(
    img: Image.Image,
    boxes: list[list[float]],
    labels: list[str],
    cell: int = 220,
) -> Image.Image:
    """Slot crops tiled into one labelled sheet (used by the grid-based path)."""
    per_row = max(1, int(np.ceil(np.sqrt(max(len(boxes), 1)))))
    rows = int(np.ceil(len(boxes) / per_row)) if boxes else 1
    sheet = Image.new("RGB", (per_row * cell, rows * (cell + 26)), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    font = _font(20)
    for i, box in enumerate(boxes):
        c = crop_norm(img, box).resize((cell - 8, cell - 8), Image.LANCZOS)
        r, col = divmod(i, per_row)
        x, y = col * cell, r * (cell + 26)
        d.text((x + 4, y + 1), labels[i], fill=(255, 255, 255), font=font)
        sheet.paste(c, (x + 4, y + 26))
    return sheet


def to_bytes(img: Image.Image, max_side: int = 1024) -> bytes:
    out = img.copy()
    if max(out.size) > max_side:
        s = max_side / float(max(out.size))
        out = out.resize((int(out.width * s), int(out.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=82)
    return buf.getvalue()
