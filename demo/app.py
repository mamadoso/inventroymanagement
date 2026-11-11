"""Ahadi shelf-count demo - FastAPI app.

Run:  ./run.sh     (or)  uvicorn app:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import io
import shutil
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import diff as diffmod
import imaging
import store
import vision
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from vision.base import VisionError

import config

config.ensure_dirs()
store.init()


def ensure_default_shelf() -> None:
    """A fresh install must have a working /capture?shelf=1 for the QR link."""
    if not store.rows("SELECT id FROM shelves LIMIT 1"):
        store.create_shelf("Shelf 1", "auto-created default")


ensure_default_shelf()

app = FastAPI(title="Ahadi Shelf Count Demo", version="0.1")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/files", StaticFiles(directory=str(config.IMAGE_DIR)), name="files")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

MAX_BYTES = config.MAX_UPLOAD_MB * 1024 * 1024


# ------------------------------------------------------------------ helpers


def _flash(request: Request) -> str:
    return request.query_params.get("msg", "")


def file_url(rel: str) -> str:
    return f"/files/{rel}" if rel else ""


def process_upload(
    shelf_id: int,
    view_key: str,
    data: bytes,
    source: str = "upload",
    taken_by: str = "",
) -> dict:
    """Store a photo, run detection, and diff it against that view's baseline."""
    shelf = store.get_shelf(shelf_id)
    if not shelf:
        raise HTTPException(404, "shelf not found")
    view = store.get_view(shelf_id, view_key)
    if not view:
        raise HTTPException(404, f"view '{view_key}' not found on shelf {shelf_id}")

    kind = "baseline" if not view.get("baseline_shot_id") else "capture"
    rel_dir = f"{shelf_id}/{view_key}"
    stem = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    rel = f"{rel_dir}/{stem}.jpg"
    dest = config.IMAGE_DIR / rel
    w, h = imaging.save_upload(data, dest)
    shot_id = store.save_shot(view["id"], kind, rel, w, h, source, taken_by)

    try:
        det = vision.detect(imaging.open_image(dest))
    except VisionError as exc:
        store.finish_shot(
            shot_id, status="failed", provider=config.active_provider(),
            model=config.AI_MODEL, latency_ms=0, total=0, quality="error",
            notes="", items=[], error=str(exc)[:700],
        )
        return {"ok": False, "shot_id": shot_id, "error": str(exc)[:700]}

    store.finish_shot(
        shot_id, status="done", provider=det.provider, model=det.model,
        latency_ms=det.latency_ms, total=det.total, quality=det.quality,
        notes=det.notes, items=det.items,
    )

    if kind == "baseline":
        store.set_baseline(shot_id)
        return {
            "ok": True, "kind": "baseline", "shot_id": shot_id,
            "total": det.total, "quality": det.quality, "latency_ms": det.latency_ms,
            "redirect": f"/shelves/{shelf_id}?msg=Baseline+set+for+{view_key}+({det.total}+watches)",
        }

    base_shot = store.get_shot(view["baseline_shot_id"])
    if not base_shot or base_shot["status"] != "done":
        return {"ok": False, "shot_id": shot_id,
                "error": "The baseline for this view has no usable detection."}

    base_img = imaging.open_image(config.IMAGE_DIR / base_shot["filename"])
    cap_img = imaging.open_image(dest)
    summary = diffmod.compare(base_img, cap_img, base_shot["items"], det.items)

    annot_cap_rel = f"{rel_dir}/{stem}-cap.jpg"
    annot_base_rel = ""
    imaging.annotate(cap_img, summary["cap_items"], config.IMAGE_DIR / annot_cap_rel)
    if summary["changes"]:
        annot_base_rel = f"{rel_dir}/{stem}-base.jpg"
        imaging.annotate(base_img, summary["base_items"], config.IMAGE_DIR / annot_base_rel)

    summary.pop("base_items", None)
    summary.pop("cap_items", None)
    diff_id = store.save_diff(
        shelf_id, view["id"], base_shot["id"], shot_id,
        summary["base_total"], summary["cap_total"], summary,
        annot_base_rel, annot_cap_rel,
    )
    return {
        "ok": True, "kind": "capture", "shot_id": shot_id, "diff_id": diff_id,
        "summary": summary, "total": det.total, "quality": det.quality,
        "latency_ms": det.latency_ms, "redirect": f"/diffs/{diff_id}",
    }


def shelf_summary(shelf: dict) -> dict:
    """Per-view verdicts plus a shelf-level headline."""
    order = {"idle": 0, "ok": 0, "warn": 1, "bad": 2}
    views = []
    for v in shelf["views"]:
        d = v.get("last_diff")
        if not d:
            state, text = "idle", "no capture yet"
        else:
            state, text = diffmod.verdict(d["summary"])
        views.append(
            {
                "key": v["key"],
                "label": v["label"],
                "state": state,
                "text": text,
                "has_baseline": bool(v.get("baseline")),
                "baseline_total": (v.get("baseline") or {}).get("total", 0),
                "last_capture": v.get("last_capture"),
                "diff_id": (d or {}).get("id"),
            }
        )
    states = [v["state"] for v in views]
    if not states or all(s == "idle" for s in states):
        worst = "idle"
    else:
        worst = max(states, key=lambda s: order.get(s, 0))
    return {"worst": worst, "views": views}


# -------------------------------------------------------------------- pages


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    shelves = store.list_shelves()
    for s in shelves:
        s["summary"] = shelf_summary(s)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "shelves": shelves,
            "msg": _flash(request),
            "provider": vision.provider_label(),
            "mock_mode": config.active_provider() == "mock",
            "recent": store.recent_diffs(8),
        },
    )


@app.post("/shelves")
def create_shelf(name: str = Form(...), note: str = Form("")):
    return RedirectResponse(
        f"/shelves/{store.create_shelf(name, note)}?msg=Shelf+created", status_code=303
    )


@app.get("/shelves/{sid}", response_class=HTMLResponse)
def shelf_page(request: Request, sid: int):
    shelf = store.get_shelf(sid)
    if not shelf:
        raise HTTPException(404, "shelf not found")
    shelf["summary"] = shelf_summary(shelf)
    for v in shelf["views"]:
        for k in ("baseline", "last_capture"):
            if v.get(k):
                v[k + "_url"] = file_url(v[k]["filename"])
        d = v.get("last_diff")
        if d:
            d["annot_base_url"] = file_url(d["annot_base"])
            d["annot_cap_url"] = file_url(d["annot_cap"])
    return templates.TemplateResponse(
        request,
        "shelf.html",
        {
            "request": request,
            "shelf": shelf,
            "msg": _flash(request),
            "provider": vision.provider_label(),
            "mock_mode": config.active_provider() == "mock",
            "capture_url": f"{config.public_url(request)}/capture?shelf={sid}",
        },
    )


@app.post("/shelves/{sid}/views")
def create_view(sid: int, key: str = Form(...), label: str = Form("")):
    try:
        store.add_view(sid, key, label)
    except ValueError:
        return RedirectResponse(f"/shelves/{sid}?msg=View+key+required", status_code=303)
    return RedirectResponse(f"/shelves/{sid}?msg=View+added", status_code=303)


@app.post("/shelves/{sid}/views/{key}/clear")
def clear_view_photos(sid: int, key: str):
    """Delete every photo for one angle. The next capture becomes the baseline."""
    shelf = store.get_shelf(sid)
    if not shelf:
        raise HTTPException(404, "shelf not found")
    view = store.get_view(sid, key)
    if not view:
        raise HTTPException(404, f"view '{key}' not found")
    n = store.clear_view(view["id"])
    shutil.rmtree(config.IMAGE_DIR / str(sid) / key, ignore_errors=True)
    return RedirectResponse(
        f"/shelves/{sid}?msg=Cleared+{n}+photo(s)+from+{key}+-+next+capture+sets+a+new+baseline",
        status_code=303,
    )


@app.post("/shelves/{sid}/clear")
def clear_shelf_photos(sid: int):
    """Delete every photo across all angles of one shelf."""
    shelf = store.get_shelf(sid)
    if not shelf:
        raise HTTPException(404, "shelf not found")
    n = 0
    for v in shelf["views"]:
        n += store.clear_view(v["id"])
        shutil.rmtree(config.IMAGE_DIR / str(sid) / v["key"], ignore_errors=True)
    return RedirectResponse(
        f"/shelves/{sid}?msg=Cleared+{n}+photo(s)+from+all+angles", status_code=303
    )


@app.post("/api/views/{sid}/{key}/clear")
def api_clear_view(sid: int, key: str):
    """JSON variant used by the phone capture screen."""
    view = store.get_view(sid, key)
    if not view:
        raise HTTPException(404, f"view '{key}' not found on shelf {sid}")
    n = store.clear_view(view["id"])
    shutil.rmtree(config.IMAGE_DIR / str(sid) / key, ignore_errors=True)
    return {"ok": True, "removed": n, "shelf": sid, "view": key}


@app.post("/shelves/{sid}/shots")
async def upload_form(
    sid: int,
    view: str = Form(...),
    file: UploadFile = File(...),
    source: str = Form("upload"),
    taken_by: str = Form(""),
):
    """Plain HTML form upload (desktop admin pages). Redirects to the report."""
    data = await file.read()
    if not data:
        return RedirectResponse(f"/shelves/{sid}?msg=No+file+selected", status_code=303)
    result = process_upload(sid, view, data, source, taken_by)
    if result.get("ok"):
        return RedirectResponse(result["redirect"], status_code=303)
    msg = quote((result.get("error") or "processing failed")[:180])
    return RedirectResponse(f"/shelves/{sid}?msg=Failed%3A+{msg}", status_code=303)


@app.get("/diffs/{did}", response_class=HTMLResponse)
def diff_page(request: Request, did: int):
    d = store.get_diff(did)
    if not d:
        raise HTTPException(404, "diff not found")
    d["annot_base_url"] = file_url(d["annot_base"])
    d["annot_cap_url"] = file_url(d["annot_cap"])
    d["baseline_url"] = file_url(d["baseline"]["filename"])
    d["capture_url"] = file_url(d["capture"]["filename"])
    d["state"], d["headline"] = diffmod.verdict(d["summary"])
    return templates.TemplateResponse(
        request,
        "diff.html",
        {
            "request": request,
            "d": d,
            "provider": vision.provider_label(),
            "mock_mode": config.active_provider() == "mock",
        },
    )


@app.get("/shots/{sid}", response_class=HTMLResponse)
def shot_page(request: Request, sid: int):
    shot = store.get_shot(sid)
    if not shot:
        raise HTTPException(404, "shot not found")
    shot["url"] = file_url(shot["filename"])
    return templates.TemplateResponse(
        request,
        "shot.html", {"request": request, "shot": shot, "provider": vision.provider_label()}
    )


@app.post("/shots/{sid}/baseline")
def make_baseline(sid: int):
    shot = store.get_shot(sid)
    if not shot:
        raise HTTPException(404, "shot not found")
    store.set_baseline(sid)
    return RedirectResponse(
        f"/shelves/{shot['view']['shelf_id']}?msg=Baseline+updated", status_code=303
    )


@app.get("/capture", response_class=HTMLResponse)
def capture_page(request: Request, shelf: int = 0, view: str = ""):
    """Capture screen. Falls back to the first shelf / first view so a stale or
    hand-typed link (and the printed QR) never dead-ends on a fresh install."""
    s = store.get_shelf(shelf) if shelf else None
    if not s:
        first = store.rows("SELECT id FROM shelves ORDER BY id LIMIT 1")
        if not first:
            raise HTTPException(404, "no shelves configured yet")
        s = store.get_shelf(first[0]["id"])
    valid = [v["key"] for v in s["views"]]
    if not valid:
        raise HTTPException(404, f"shelf {s['id']} has no views")
    if view not in valid:
        view = valid[0]
    return templates.TemplateResponse(
        request,
        "capture.html",
        {
            "request": request,
            "shelf": s,
            "view": view,
            "provider": vision.provider_label(),
            "mock_mode": config.active_provider() == "mock",
            "views": valid,
            "view_labels": {v["key"]: v["label"] for v in s["views"]},
        },
    )


# --------------------------------------------------------------------- API


@app.post("/api/capture")
async def api_capture(
    file: UploadFile = File(...),
    shelf: int = Form(...),
    view: str = Form("front"),
    taken_by: str = Form(""),
    source: str = Form("camera"),
):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"file larger than {config.MAX_UPLOAD_MB}MB")
    return JSONResponse(process_upload(shelf, view, data, source, taken_by))


@app.get("/api/shelves")
def api_shelves():
    out = []
    for s in store.list_shelves():
        s["summary"] = shelf_summary(s)
        out.append(
            {
                "id": s["id"],
                "name": s["name"],
                "views": [{"key": v["key"], "label": v["label"]} for v in s["views"]],
                "last_diff": s["last_diff"]["id"] if s.get("last_diff") else None,
            }
        )
    return {"shelves": out, "provider": config.active_provider(), "model": config.AI_MODEL}


@app.get("/api/summary/{sid}")
def api_summary(sid: int):
    shelf = store.get_shelf(sid)
    if not shelf:
        raise HTTPException(404, "shelf not found")
    return shelf_summary(shelf)


@app.get("/api/history/{sid}")
def api_history(sid: int):
    shelf = store.get_shelf(sid)
    if not shelf:
        raise HTTPException(404, "shelf not found")
    return [
        {
            "id": d["id"],
            "view": d["view_id"],
            "created_at": d["created_at"],
            "base_total": d["base_total"],
            "cap_total": d["cap_total"],
            "summary": d["summary"],
        }
        for d in shelf["diffs"]
    ]


@app.get("/api/shot/{sid}")
def api_shot(sid: int):
    shot = store.get_shot(sid)
    if not shot:
        raise HTTPException(404, "shot not found")
    shot["url"] = file_url(shot["filename"])
    return shot


# ------------------------------------------------------------------ health

_HEALTH: dict = {"ran": False, "result": None}


def health_check(force: bool = False) -> dict:
    if not _HEALTH["ran"] or force:
        if config.active_provider() == "mock":
            _HEALTH["result"] = {
                "ok": True,
                "provider": "mock",
                "detail": "offline heuristic mode - set AHADI_AI_API_KEY for real AI",
                "base_url": "-",
                "model": "heuristic-grid",
                "key_set": False,
            }
        else:
            from vision import openai_compat

            _HEALTH["result"] = openai_compat.check_connection()
        _HEALTH["ran"] = True
    return _HEALTH["result"]


@app.get("/health", response_class=HTMLResponse)
def health_page(request: Request):
    return templates.TemplateResponse(
        request,
        "health.html",
        {
            "request": request,
            "cfg": {
                "provider": config.active_provider(),
                "base_url": config.normalized_base_url(),
                "model": config.AI_MODEL,
                "key_set": bool(config.AI_API_KEY),
                "max_side": config.MAX_SIDE,
                "timeout": config.REQUEST_TIMEOUT,
                "data_dir": str(config.DATA_DIR),
                "match_min": config.MATCH_MIN_SCORE,
                "moved_min": config.MOVED_MIN_APPEAR,
            },
            "check": health_check(),
        },
    )


@app.post("/health/test")
def health_test():
    _HEALTH["ran"] = False
    return JSONResponse(health_check(force=True))


@app.get("/health/json")
def health_json():
    return JSONResponse(health_check())


@app.get("/qr/{sid}")
def qr(sid: int, request: Request):
    try:
        import segno
    except ImportError:
        raise HTTPException(404, "segno not installed - pip install segno")
    target = store.get_shelf(sid)
    if not target:
        first = store.rows("SELECT id FROM shelves ORDER BY id LIMIT 1")
        if not first:
            raise HTTPException(404, "no shelves configured yet")
        sid = first[0]["id"]
    buf = io.BytesIO()
    segno.make(f"{config.public_url(request)}/capture?shelf={sid}", error="m").save(
        buf, kind="png", scale=8, border=2
    )
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/reset")
def reset():
    """Wipe demo data - handy between manager demos."""
    for p in config.IMAGE_DIR.glob("*"):
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    for table in ("diffs", "shots", "views", "shelves"):
        store.run(f"DELETE FROM {table}")
    return RedirectResponse("/?msg=Demo+data+cleared", status_code=303)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)
