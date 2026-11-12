"""Tiny sqlite3 persistence layer. No ORM - this is a demo.

Append-only for shots and diffs: a past observation is never rewritten, which
is the property that makes the report defensible later.
"""
import json
import sqlite3
from datetime import datetime, timezone

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS shelves (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  note TEXT DEFAULT '',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS views (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  shelf_id INTEGER NOT NULL REFERENCES shelves(id),
  key TEXT NOT NULL,
  label TEXT NOT NULL,
  baseline_shot_id INTEGER,
  UNIQUE(shelf_id, key)
);

CREATE TABLE IF NOT EXISTS shots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  view_id INTEGER NOT NULL REFERENCES views(id),
  kind TEXT NOT NULL,              -- baseline | capture
  filename TEXT NOT NULL,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  source TEXT DEFAULT '',          -- camera | upload | filename
  taken_by TEXT DEFAULT '',
  created_at TEXT NOT NULL,
  status TEXT NOT NULL,            -- pending | done | failed
  error TEXT DEFAULT '',
  provider TEXT DEFAULT '',
  model TEXT DEFAULT '',
  latency_ms INTEGER DEFAULT 0,
  total INTEGER DEFAULT 0,
  quality TEXT DEFAULT '',
  notes TEXT DEFAULT '',
  items_json TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS diffs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  shelf_id INTEGER NOT NULL REFERENCES shelves(id),
  view_id INTEGER NOT NULL REFERENCES views(id),
  baseline_shot_id INTEGER NOT NULL REFERENCES shots(id),
  capture_shot_id INTEGER NOT NULL REFERENCES shots(id),
  base_total INTEGER DEFAULT 0,
  cap_total INTEGER DEFAULT 0,
  summary_json TEXT DEFAULT '{}',
  annot_base TEXT DEFAULT '',
  annot_cap TEXT DEFAULT '',
  created_at TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    config.ensure_dirs()
    conn = sqlite3.connect(str(config.DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init() -> None:
    conn = connect()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def rows(sql: str, args: tuple = ()) -> list[dict]:
    conn = connect()
    try:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]
    finally:
        conn.close()


def one(sql: str, args: tuple = ()) -> dict | None:
    out = rows(sql, args)
    return out[0] if out else None


def insert(sql: str, args: tuple) -> int:
    conn = connect()
    try:
        cur = conn.execute(sql, args)
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def run(sql: str, args: tuple = ()) -> None:
    conn = connect()
    try:
        conn.execute(sql, args)
        conn.commit()
    finally:
        conn.close()


# ---------- shelves / views ----------

def create_shelf(name: str, note: str = "") -> int:
    sid = insert(
        "INSERT INTO shelves(name, note, created_at) VALUES (?,?,?)",
        (name.strip() or "Shelf", note.strip(), now()),
    )
    for key, label in config.DEFAULT_VIEWS:
        insert(
            "INSERT INTO views(shelf_id, key, label) VALUES (?,?,?)",
            (sid, key, label),
        )
    return sid


def add_view(shelf_id: int, key: str, label: str) -> int:
    key = (key or "").strip().lower().replace(" ", "-")
    if not key:
        raise ValueError("view key required")
    return insert(
        "INSERT OR IGNORE INTO views(shelf_id, key, label) VALUES (?,?,?)",
        (shelf_id, key, label or key.title()),
    )


def list_shelves() -> list[dict]:
    out = []
    for s in rows("SELECT * FROM shelves ORDER BY id DESC"):
        s["views"] = rows(
            "SELECT * FROM views WHERE shelf_id=? ORDER BY id", (s["id"],)
        )
        shots = rows(
            "SELECT d.* FROM diffs d WHERE d.shelf_id=? ORDER BY d.id DESC LIMIT 1",
            (s["id"],),
        )
        s["last_diff"] = shots[0] if shots else None
        s["diff_count"] = len(
            rows("SELECT id FROM diffs WHERE shelf_id=?", (s["id"],))
        )
        for v in s["views"]:
            _decorate_view(v)
        out.append(s)
    return out


def get_shelf(shelf_id: int) -> dict | None:
    s = one("SELECT * FROM shelves WHERE id=?", (shelf_id,))
    if not s:
        return None
    s["views"] = rows("SELECT * FROM views WHERE shelf_id=? ORDER BY id", (shelf_id,))
    for v in s["views"]:
        _decorate_view(v)
    s["diffs"] = rows(
        "SELECT * FROM diffs WHERE shelf_id=? ORDER BY id DESC LIMIT 20", (shelf_id,)
    )
    for d in s["diffs"]:
        d["summary"] = json.loads(d.get("summary_json") or "{}")
    s["total_shots"] = one(
        """SELECT COUNT(*) AS n FROM shots
           WHERE view_id IN (SELECT id FROM views WHERE shelf_id=?)""",
        (shelf_id,),
    )["n"]
    return s


def get_view(shelf_id: int, view_key: str) -> dict | None:
    v = one(
        "SELECT * FROM views WHERE shelf_id=? AND key=?", (shelf_id, view_key)
    )
    if not v:
        return None
    _decorate_view(v)
    return v


def reap_stale_pending() -> int:
    """Mark shots left in `pending` by an interrupted request.

    A shot is written as pending before the AI call and updated after it, so a
    restart, a crash or a proxy timeout strands rows in pending forever. They
    look like work in progress on the shelf page when nothing is running.
    """
    rows_ = rows("SELECT id FROM shots WHERE status='pending'")
    if rows_:
        run(
            "UPDATE shots SET status='failed', error='interrupted before the AI call "
            "finished (restart, crash or proxy timeout)' WHERE status='pending'"
        )
    return len(rows_)


def clear_view(view_id: int) -> int:
    """Delete every shot and diff belonging to a view.

    Returns the number of shots removed. The view row itself stays, so the next
    photo uploaded becomes the new baseline automatically.
    """
    n = one("SELECT COUNT(*) AS n FROM shots WHERE view_id=?", (view_id,))["n"]
    run("UPDATE views SET baseline_shot_id=NULL WHERE id=?", (view_id,))
    run("DELETE FROM diffs WHERE view_id=?", (view_id,))
    run("DELETE FROM shots WHERE view_id=?", (view_id,))
    return n


def _decorate_view(v: dict) -> None:
    bid = v.get("baseline_shot_id")
    v["baseline"] = one("SELECT * FROM shots WHERE id=?", (bid,)) if bid else None
    if v["baseline"]:
        v["baseline"]["items"] = json.loads(v["baseline"]["items_json"] or "[]")
    last = one(
        "SELECT * FROM shots WHERE view_id=? AND kind='capture' ORDER BY id DESC LIMIT 1",
        (v["id"],),
    )
    v["last_capture"] = last
    if last:
        last["items"] = json.loads(last["items_json"] or "[]")
    d = one(
        "SELECT * FROM diffs WHERE view_id=? ORDER BY id DESC LIMIT 1", (v["id"],)
    )
    if d:
        d["summary"] = json.loads(d.get("summary_json") or "{}")
    v["last_diff"] = d
    v["shot_count"] = one(
        "SELECT COUNT(*) AS n FROM shots WHERE view_id=?", (v["id"],)
    )["n"]
    v["queued"] = one(
        "SELECT COUNT(*) AS n FROM shots WHERE view_id=? AND status='queued'",
        (v["id"],),
    )["n"]
    # A failed angle must not read as "nothing happened" on the combined report.
    v["last_failed"] = one(
        "SELECT * FROM shots WHERE view_id=? AND status='failed' ORDER BY id DESC LIMIT 1",
        (v["id"],),
    )


def get_shot(shot_id: int) -> dict | None:
    sh = one("SELECT * FROM shots WHERE id=?", (shot_id,))
    if sh:
        sh["items"] = json.loads(sh["items_json"] or "[]")
        sh["view"] = one("SELECT * FROM views WHERE id=?", (sh["view_id"],))
    return sh


def save_shot(
    view_id: int,
    kind: str,
    filename: str,
    width: int,
    height: int,
    source: str,
    taken_by: str,
    status: str = "pending",
) -> int:
    return insert(
        """INSERT INTO shots(view_id, kind, filename, width, height, source,
                             taken_by, created_at, status)
           VALUES (?,?,?,?,?,?,?,?, ?)""",
        (view_id, kind, filename, width, height, source, taken_by, now(), status),
    )


def queued_shots(shelf_id: int) -> list[dict]:
    """Photos staged for this shelf that have not been sent to the AI yet."""
    return rows(
        """SELECT s.*, v.key AS view_key, v.label AS view_label, v.baseline_shot_id
           FROM shots s JOIN views v ON v.id = s.view_id
           WHERE v.shelf_id=? AND s.status='queued'
           ORDER BY v.id, s.id""",
        (shelf_id,),
    )


def set_status(shot_id: int, status: str, error: str = "") -> None:
    run("UPDATE shots SET status=?, error=? WHERE id=?", (status, error, shot_id))


def discard_queued(shelf_id: int) -> int:
    """Throw away staged-but-unanalysed photos (a retake)."""
    gone = rows(
        """SELECT s.id, s.filename FROM shots s JOIN views v ON v.id = s.view_id
           WHERE v.shelf_id=? AND s.status='queued'""",
        (shelf_id,),
    )
    for s in gone:
        run("DELETE FROM shots WHERE id=?", (s["id"],))
        try:
            (config.IMAGE_DIR / s["filename"]).unlink(missing_ok=True)
        except OSError:
            pass
    return len(gone)


def finish_shot(
    shot_id: int,
    *,
    status: str,
    provider: str,
    model: str,
    latency_ms: int,
    total: int,
    quality: str,
    notes: str,
    items: list[dict],
    error: str = "",
) -> None:
    run(
        """UPDATE shots SET status=?, provider=?, model=?, latency_ms=?, total=?,
               quality=?, notes=?, items_json=?, error=? WHERE id=?""",
        (
            status,
            provider,
            model,
            latency_ms,
            total,
            quality,
            notes,
            json.dumps(items),
            error,
            shot_id,
        ),
    )


def set_baseline(shot_id: int) -> None:
    sh = get_shot(shot_id)
    if not sh:
        raise ValueError("unknown shot")
    run("UPDATE views SET baseline_shot_id=? WHERE id=?", (shot_id, sh["view_id"]))


def set_kind(shot_id: int, kind: str) -> None:
    """Re-label a shot once its real role is known at analysis time."""
    run("UPDATE shots SET kind=? WHERE id=?", (kind, shot_id))


def save_diff(
    shelf_id: int,
    view_id: int,
    baseline_id: int,
    capture_id: int,
    base_total: int,
    cap_total: int,
    summary: dict,
    annot_base: str,
    annot_cap: str,
) -> int:
    return insert(
        """INSERT INTO diffs(shelf_id, view_id, baseline_shot_id, capture_shot_id,
                             base_total, cap_total, summary_json, annot_base,
                             annot_cap, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            shelf_id,
            view_id,
            baseline_id,
            capture_id,
            base_total,
            cap_total,
            json.dumps(summary),
            annot_base,
            annot_cap,
            now(),
        ),
    )


def get_diff(diff_id: int) -> dict | None:
    d = one("SELECT * FROM diffs WHERE id=?", (diff_id,))
    if not d:
        return None
    d["summary"] = json.loads(d.get("summary_json") or "{}")
    d["baseline"] = get_shot(d["baseline_shot_id"])
    d["capture"] = get_shot(d["capture_shot_id"])
    d["view"] = one("SELECT * FROM views WHERE id=?", (d["view_id"],))
    d["shelf"] = one("SELECT * FROM shelves WHERE id=?", (d["shelf_id"],))
    return d


def recent_diffs(limit: int = 20) -> list[dict]:
    out = rows("SELECT * FROM diffs ORDER BY id DESC LIMIT ?", (limit,))
    for d in out:
        d["summary"] = json.loads(d.get("summary_json") or "{}")
        v = one("SELECT * FROM views WHERE id=?", (d["view_id"],))
        d["view"] = v
        d["shelf"] = one("SELECT * FROM shelves WHERE id=?", (d["shelf_id"],))
    return out
