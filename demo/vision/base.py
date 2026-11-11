"""Shared VLM contract: one prompt, one validated result shape, any provider."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


class VisionError(RuntimeError):
    pass


SYSTEM = (
    "You are a meticulous retail shelf auditor for a watch display case.\n"
    "Your job is to locate and count every watch visible in the photo. You do NOT "
    "identify brands, models, references or prices - that is out of scope and you "
    "must not attempt it.\n"
    "Rules for counting:\n"
    "- A watch counts if you can see its dial, its case, or a strap on a display stand.\n"
    "- Empty display stands, velvet cushions, dividers, boxes and tags are NOT watches.\n"
    "- One physical watch = exactly one entry. Never double-count a watch because both "
    "the dial and the strap are visible.\n"
    "- If an object is too cut off, blurred or reflected to be sure, leave it out and "
    "mention it in notes instead of guessing.\n"
    "- Ignore reflections in glass; only count real objects standing on the shelf.\n"
    "Coordinates: normalised to the image, x and y measured from the top-left corner, "
    "each value between 0.0 and 1.0. box = [x, y, width, height].\n"
    "desc: a short neutral visual description, max 8 words, e.g. 'dark dial, steel "
    "case, metal bracelet' or 'light dial, brown leather strap'. No brand names.\n"
    "Return JSON only, matching the requested schema exactly."
)

USER = (
    "Count every watch on this shelf. For each one return its bounding box and a short "
    "neutral visual description. Also report overall image quality. "
    "Set total equal to the number of entries in items."
)

_JSON_LOOSE = re.compile(r"\{.*\}", re.S)


@dataclass
class Detection:
    items: list[dict] = field(default_factory=list)
    total: int = 0
    quality: str = "good"
    notes: str = ""
    provider: str = ""
    model: str = ""
    latency_ms: int = 0

    def as_rows(self) -> list[dict]:
        return self.items


def sanitize_items(raw) -> list[dict]:
    """Coerce model output into validated item rows. Drops anything unusable."""
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        box = entry.get("box") or entry.get("bbox") or entry.get("bbox_2d")
        if box is None:
            continue
        if isinstance(box, dict):
            box = [box.get("x", 0), box.get("y", 0), box.get("w", box.get("width", 0)),
                   box.get("h", box.get("height", 0))]
        if isinstance(box, (list, tuple)):
            box = list(box)
        else:
            continue
        # Some models return 0-1000 pixel-space integers.
        if any(isinstance(v, (int, float)) and abs(v) > 2 for v in box[:4]):
            box = [v / 1000.0 for v in box[:4]]
        try:
            x, y, w, h = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        except (TypeError, ValueError, IndexError):
            continue
        if w <= 0 or h <= 0:
            continue
        x0 = min(max(x, 0.0), 1.0)
        y0 = min(max(y, 0.0), 1.0)
        x1 = min(max(x + w, 0.0), 1.0)
        y1 = min(max(y + h, 0.0), 1.0)
        if x1 - x0 < 0.004 or y1 - y0 < 0.004:
            continue
        desc = str(entry.get("desc") or entry.get("description") or "")
        desc = re.sub(r"\s+", " ", desc).strip()[:90]
        try:
            conf = float(entry.get("confidence", entry.get("conf", 0.9)))
        except (TypeError, ValueError):
            conf = 0.9
        out.append(
            {
                "box": [round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4)],
                "desc": desc,
                "conf": round(max(0.0, min(1.0, conf)), 2),
            }
        )
    return out


def _repair_truncated(text: str) -> str | None:
    """Close a JSON object that was cut off mid-output.

    A gateway that runs out of max_tokens stops in the middle of the items array.
    Everything up to the last complete element is still valid and useful, so keep
    it and drop the partial tail rather than throwing the whole answer away.
    Returns None when not even one complete element can be salvaged.
    """
    start = text.find("{")
    if start < 0:
        return None
    stack: list[str] = []
    in_str = esc = False
    cut = -1
    for i, ch in enumerate(text[start:], start):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if in_str:
            if ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None
            stack.pop()
            # back to { + [ means one whole array element just finished
            if stack[:2] == ["{", "["] and len(stack) == 2:
                cut = i + 1
    if cut < 0:
        return None
    return text[start:cut] + "]}"


def parse_json_object(text: str) -> dict:
    if not text:
        raise VisionError("empty model response")
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json_loads(text)
    except ValueError:
        pass
    m = _JSON_LOOSE.search(text)
    if m:
        try:
            return json_loads(m.group(0))
        except ValueError:
            pass
    # Last resort: the answer was probably truncated by the token limit. Salvage
    # the complete items and flag it so nobody trusts the resulting count.
    fixed = _repair_truncated(m.group(0) if m else text)
    if fixed:
        try:
            obj = json_loads(fixed)
            obj["_truncated"] = True
            return obj
        except ValueError:
            pass
    raise VisionError(f"model did not return valid JSON: {text[:200]}")


def json_loads(text: str) -> dict:
    import json

    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise VisionError("model JSON was not an object")
    return obj


def build_detection(parsed: dict, provider: str, model: str, latency_ms: int) -> Detection:
    # Set by parse_json_object when it had to close a cut-off answer.
    truncated = bool(parsed.pop("_truncated", False))
    items = sanitize_items(parsed.get("items"))
    quality = str(parsed.get("quality") or "good").lower().strip()
    if quality not in ("good", "glare", "blur", "too_far", "dark", "other"):
        quality = "other"
    try:
        total = int(parsed.get("total") or len(items))
    except (TypeError, ValueError):
        total = len(items)
    # Fail closed on a count/box mismatch: trust the boxes, flag it for review.
    if total != len(items):
        quality = quality if quality != "good" else "other"
    notes = str(parsed.get("notes") or "")
    if truncated:
        # A truncated list always under-counts, and `total` is usually the key
        # that got cut off, so the mismatch check above cannot catch it.
        quality = "other"
        notes = (notes + " | " if notes else "") + (
            f"output truncated at the token limit - only {len(items)} items recovered, "
            "raise AHADI_AI_MAX_TOKENS"
        )
    return Detection(
        items=items,
        total=len(items),
        quality=quality,
        notes=notes[:300],
        provider=provider,
        model=model,
        latency_ms=latency_ms,
    )
