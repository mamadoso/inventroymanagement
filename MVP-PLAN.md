# MVP Build Plan — Shelf Count & Change Detection

**Scope locked:** count + what-changed only. No model/SKU identification. Cloud API. 6 weeks.
**Companion doc:** `PLAN.md` (full 16-week vision, risk register, Phase 3 identity design)
**Date:** 2026-11-11

---

## 0. The one design decision that makes this MVP easy

Because identity is out of scope, **we do not need object detection at all.**

If each shelf has a fixed slot grid (dividers, tray pockets, or a marked layout), then after camera-pose rectification we already know *where* every slot is. The task collapses from "find and count N watches in a cluttered scene" (hard) to:

> "Here is crop #14. Is there a watch in it? Yes / No / Unclear."

That is a per-slot binary classification — dramatically more reliable, and it never merges two touching watches into one blob, which is the classic failure of free-running detectors.

**Two variants, ship A first:**

| Variant | When | How |
|---|---|---|
| **A — Grid-based (primary)** | Shelves with fixed dividers/pockets | Calibrate slot grid once → crop each slot → classify each crop |
| **B — Free detection (fallback)** | Shelves with loose/overlapping watches | VLM returns boxes on the rectified image, assign to nearest slot |

Variant A also makes the diff trivially correct, because slot identity is stable by construction.

---

## 1. In / out of scope

**In:** shelf registration, pose-controlled capture, quality gate, slot-grid calibration, per-slot presence + count, change detection vs last confirmed baseline, missing/new/moved/swapped report, human review queue for uncertain slots, immutable audit log, daily digest, accuracy dashboard.

**Out (explicitly, in writing):** brand identification, model/reference identification, serial numbers, price/valuation, theft attribution, ERP write-back. Read-only CSV/JSON export instead.

**Data we still capture anyway, for free:** every slot crop + its embedding, from day one. This is what makes Phase 3 identity possible later without re-running the whole collection. Cheap insurance, ~$0 extra.

---

## 2. Architecture (cloud-only, minimal moving parts)

```
Phone PWA (React + Vite, camera + capture guide)
        │  multipart: image + shelf_code + user + ts
        ▼
FastAPI on Fly.io / Railway  ──►  Postgres (Neon or Supabase) + pgvector
        │                              └── images/thumbs → Cloudflare R2
        ▼
worker queue (RQ or arq, same host, 1-2 dynos)
  1. quality gate      blur / exposure / glare / markers-visible
  2. rectify           ArUco → homography → shelf-plane image (OpenCV)
  3. crop slots        slot_boxes from calibration → N crops
  4. classify          ONE batched VLM call over the labelled crop sheet
  5. embed             CLIP/SigLIP embedding + pHash per crop (for change check)
  6. diff              vs last confirmed snapshot → change list
  7. persist           observations, snapshot, audit event, review items
        │
        ▼
Admin UI (React) — shelf map editor, review queue, history, accuracy dashboard
```

**No GPU server, no model training, no Triton.** Embeddings run on CPU (ONNX, ~40ms/crop) or via a small serverless GPU endpoint if the CPU budget is too tight. This is what "cloud API" buys us: the MVP is a systems-integration project, not an ML project.

### Choices

| Piece | Pick | Why |
|---|---|---|
| VLM | **Gemini 2.5 Flash** primary, GPT-4o secondary | Best cost/latency for structured visual QA; keep a provider abstraction from commit 1 |
| Embeddings | **SigLIP-base ONNX** (CPU) or Jina/Cohere embedding API | Robust "same physical object?" check under lighting change |
| Change check | pHash (dHash 64-bit) **+** cosine on embedding **+** SSIM on rectified crop | Three cheap signals, vote between them |
| Storage | Cloudflare **R2** | No egress fees — matters when you keep every capture forever |
| DB | Postgres 16 + **pgvector** | relational + vectors, one service |
| Queue | **arq** (Redis) | retries, idempotency, backpressure |
| PWA | React + Vite + `getUserMedia`, service worker offline queue | staff phones, no app store |
| Auth | Supabase Auth or Auth0, RBAC: staff / reviewer / admin | reviewer sign-off gates the baseline |

### Cost envelope

| Item | Est. |
|---|---|
| VLM: 50 shelves × 4 counts/day × 25 days = 5,000 shelf images | ~$15–60/mo |
| Embeddings (self-hosted CPU) | $0 |
| Fly.io/Railway API + worker | ~$20–40/mo |
| Neon/Supabase Postgres | $0–25/mo |
| R2 storage + requests | ~$5/mo |
| **Total run cost** | **< $150/mo** |
| Hardware: phone jig, printed ArUco markers, clip-on polariser, cheap tripod arm | < $150 one-time |

---

## 3. Capture rig — non-negotiable precondition

Diff accuracy is bounded by pose repeatability, not by the model. Budget 3 days of physical iteration here and it pays back for the whole project.

1. **Printed ArUco/AprilTag pair**, one at each end of the shelf rail, laminated, fixed with double-sided tape. Two coplanar markers on the shelf plane → stable homography.
2. **Phone position jig**: adjustable arm/bracket so the phone sits at the same distance and angle every time, or a floor-standing tripod at a marked spot with tape on the floor. Target: **±3cm, ±5°**.
3. **Clip-on polarising filter** on the phone lens if the shelf has glass/acrylic. Kills most glare.
4. **Capture guide overlay** in the PWA: ghost outline of the previous accepted frame + marker outlines + "hold steady" indicator. This is the single highest-leverage UX feature in the MVP.
5. **Fixed lighting**: shelf lights ON during every count; note time-of-day; reject frames with colour temperature far from the shelf's reference.

If the shop refuses the jig and markers, stop and renegotiate — the product will not hit its acceptance criteria.

---

## 4. Diff engine

Baseline = **last human-confirmed snapshot**, never "last photo". This prevents silent error drift.

```python
def diff(prev: Snapshot, cur: Snapshot) -> list[Change]:
    changes = []
    for slot in sorted(set(prev.slots) | set(cur.slots)):
        p, c = prev.slots.get(slot), cur.slots.get(slot)
        if p and p.present and not (c and c.present):
            changes.append(MISSING(slot, p))
        elif c and c.present and not (p and p.present):
            changes.append(NEW(slot, c))
        elif p.present and c.present and not same_object(p, c):
            changes.append(SWAPPED(slot, p, c))
        elif p.unclear or c.unclear:
            changes.append(UNCERTAIN(slot, p, c))
        else:
            changes.append(UNCHANGED(slot, c))
    return resolve_moves(changes)      # MISSING + NEW elsewhere → MOVED

def same_object(p, c) -> bool:
    if hamming(p.dhash, c.dhash) <= 6:      return True     # near-identical pixels
    if cosine(p.emb, c.emb) >= 0.86:        return True     # same object, new lighting
    return ssim(p.crop, c.crop) >= 0.55                      # last resort
```

Rules:
- **`resolve_moves` is mandatory.** Staff tidy shelves between counts; without it every tidy-up reports as theft. Match each MISSING against NEW slots on the same shelf by embedding similarity, emit `MOVED(from,to)` and remove both from the alarm set.
- **`UNCERTAIN` never counts as a change.** It goes to the review queue and blocks that slot from being auto-committed.
- **Count delta is reported separately from slot changes.** "24 → 22" plus "slot 7 empty, slot 12 changed" are different claims with different failure modes; keep them separate in the UI and in the metrics.
- **Thresholds live in config, per shelf.** A glossy glass shelf needs different SSIM than an open tray. Tune on the frozen eval set, never in production.
- Every decision is logged with the three signal values so a disputed report can be replayed.

---

## 5. Data model

```sql
users(id, name, role, created_at)

shelves(
  id, code UNIQUE, name, location,
  rows int, cols int,
  slot_boxes jsonb,              -- [{slot:'B3-07', x,y,w,h} in rectified coords]
  reference_image_key,           -- the accepted "good capture" for the overlay
  marker_pattern_id,
  thresholds jsonb,              -- per-shelf dhash/cosine/ssim overrides
  active bool, created_at
)

captures(
  id, shelf_id FK, image_key, rectified_key, thumb_key,
  taken_by FK, taken_at,
  quality jsonb,                 -- {blur, brightness, glare, markers}
  status enum(queued,processing,done,failed,rejected),
  error text, vlm_latency_ms, created_at
)

slot_observations(
  id, capture_id FK, slot_code,
  state enum(present, empty, unclear),
  sub_count int default 1,       -- >1 only in Variant B
  crop_key, dhash bit(64), embedding vector(768),
  confidence numeric, vlm_raw jsonb,
  UNIQUE(capture_id, slot_code)
)

snapshots(
  id, shelf_id FK, capture_id FK,
  total_count int, per_slot jsonb,
  is_baseline bool,
  confirmed_by FK, confirmed_at
)

changes(
  id, shelf_id FK, prev_snapshot_id FK, cur_snapshot_id FK,
  slot_code, kind enum(missing,new,moved,swapped,uncertain,unchanged),
  detail jsonb, created_at
)

review_items(id, capture_id FK, slot_code, reason, status, resolved_by, resolution, created_at)
audit_events(id, actor, action, entity, entity_id, payload jsonb, created_at)
```

`slot_observations` is append-only. Never update a past observation — corrections are new rows with an audit event. This is what makes the system defensible in a dispute.

---

## 6. API surface

```
POST   /api/v1/shelves                         register shelf
PUT    /api/v1/shelves/{code}/calibration      rows, cols, slot_boxes, reference image
GET    /api/v1/shelves/{code}/calibration      → + capture overlay geometry for the PWA
POST   /api/v1/captures                        multipart → 202 {capture_id}
GET    /api/v1/captures/{id}                   status, quality, result
GET    /api/v1/shelves/{code}/latest           current snapshot + last diff
GET    /api/v1/shelves/{code}/diff?from=&to=   arbitrary two snapshots
POST   /api/v1/captures/{id}/confirm           reviewer sign-off → new baseline
GET    /api/v1/review?status=open
POST   /api/v1/review/{id}/resolve             {resolution, actual_state}
GET    /api/v1/shelves/{code}/history?limit=
GET    /api/v1/metrics/accuracy?from=&to=      count match rate, false alarms
GET    /api/v1/export.csv?shelf=&from=&to=
```

Idempotency: `POST /captures` accepts an `Idempotency-Key` (client-generated UUID) so a flaky shop Wi-Fi retry never double-counts.

---

## 7. VLM prompt (Variant A — the batched crop sheet)

One call per shelf, not per slot. Draw each slot crop into a labelled contact sheet (`07`, `08`, `09`… burned into the corner) and send that single image. ~1 call, ~$0.003, ~2s.

```
SYSTEM
You are a retail shelf auditor. You will receive ONE contact sheet. Each panel is a
fixed slot of a watch display shelf, labelled with its slot number in the corner.
For every panel, decide whether a watch is present.
Rules:
- A watch is PRESENT if you can see a dial, a case, or a strap on a display stand.
- A panel showing only the empty stand, velvet, or divider is EMPTY.
- If glare, blur, or an object at the panel edge prevents a confident answer, use UNCLEAR.
- Never guess. UNCLEAR is a correct and useful answer.
- Do not identify brands, models, or prices. Do not describe the watch beyond the fields asked.
- Count panels exactly. Output one object per label you see, no extras, no omissions.
Return ONLY JSON:
{"panels":[{"slot":"07","state":"present","confidence":0.93},
           {"slot":"08","state":"empty","confidence":0.97},
           {"slot":"09","state":"unclear","confidence":0.55,
            "reason":"strong glare across upper half"}]}

USER
[contact sheet image]
Labels present: 01..24. Return all 24.
```

Guardrails:
- Validate against the schema; **any missing or extra slot → whole capture fails closed** and is re-queued once, then sent to review. Never let a silently dropped slot become "empty".
- Ask for `reason` on UNCLEAR — it becomes the staff instruction in the review queue.
- Log prompt version + model name with every observation. Accuracy claims must be reproducible.
- Keep a `Provider` interface with two implementations from day 1; VLM pricing and quality move fast.

Variant B (loose shelves) uses free-detection prompting on the rectified full image instead; build it only for shelves that fail Variant A.

---

## 8. Six-week schedule

### Week 1 — Rig + ground truth + walking skeleton
- Buy/print jig, markers, polariser. Install on **one** shelf.
- Capture **150 photos** of that shelf: 50 known-unchanged, 50 with a known watch removed, 50 with a known watch moved. **Physically count and record the truth.** This is the frozen eval set.
- FastAPI + Postgres + R2 + queue skeleton, `POST /captures` stores and returns.
- Quality gate (blur/exposure/glare/markers) server-side.
**Gate:** homography reprojection error < 8px across the 150 captures. If not, fix the rig before writing more code.

### Week 2 — Geometry + calibration UI
- ArUco detection → rectification → slot box generation from rows×cols.
- Admin calibration screen: upload reference photo, mark 4 corners, set rows/cols, nudge slot boxes, save.
- PWA: shelf code entry, capture with ghost overlay + marker outlines, offline queue.
**Gate:** a non-technical staff member can calibrate a shelf in < 10 minutes, unassisted.

### Week 3 — Counting
- Contact-sheet builder + VLM call + schema validation + retries + prompt versioning.
- `GET /shelves/{code}/latest` returns count and per-slot states.
- Run against the frozen eval set. Tune thresholds.
**Gate:** exact count match ≥ 92% on the 150 eval captures; zero silent slot drops; p95 capture→result < 15s.

### Week 4 — Diff + review + audit
- Diff engine incl. `resolve_moves`, baseline = last confirmed, `POST /confirm`, review queue, audit log, daily digest email/Telegram.
**Gate:** on the 50 "removed watch" captures, ≥ 95% report the correct slot as MISSING. On the 50 "unchanged" captures, ≤ 5% produce a false alarm. On the 50 "moved" captures, ≥ 85% classify as MOVED rather than missing+new.

### Week 5 — Multi-shelf + hardening
- Roll to 5–10 shelves, including the worst-lit and glass-fronted ones. Build Variant B for any shelf where the grid doesn't hold.
- Idempotency, retries, rate limits, provider fallback, cost metering per shelf, error UX ("tilt slightly to avoid glare"), role-based access.
- Accuracy dashboard: count match rate, false alarms, review-queue volume, VLM cost per count.
**Gate:** 50 shelves/day processed with < 2% failed captures.

### Week 6 — Live pilot + go/no-go
- Staff run real daily counts for 5–10 working days. Every report cross-checked against a physical count for the first week.
- Write the SOP (1 page, laminated next to the jig).
- Deliver pilot report with measured numbers against the gates above, and a Phase 3 recommendation.
**Go/no-go:** if count match < 90% or false alarms > 10% after 6 weeks, the problem is the rig or the shelf layout, not the model — fix that before adding identity.

---

## 9. Acceptance criteria (put these in the contract)

| Metric | Target |
|---|---|
| Exact count match vs physical count | ≥ 95% of shelf counts |
| Correct slot identified for a removed watch | ≥ 95% |
| False alarms on unchanged shelves | ≤ 5% of shelf counts |
| Moved watch classified as MOVED, not missing | ≥ 85% |
| Silent slot drops (missing VLM panel treated as empty) | **0** — fails closed |
| Capture → report latency, p95 | ≤ 15s |
| Staff time per shelf count | ≤ 60s (vs manual baseline) |
| Uncertain slots routed to review, never auto-committed | 100% |
| Full replayable audit trail per count | required |

---

## 10. Repo layout

```
ahadiInventoryManagementAI/
  PLAN.md  MVP-PLAN.md
  backend/
    app/main.py  settings.py
    app/api/{shelves,captures,review,metrics}.py
    app/domain/{calibration,quality,rectify,contact_sheet,diff,slots}.py
    app/inference/{vlm.py,providers/base.py,providers/gemini.py,providers/openai.py,embed.py}
    app/worker/tasks.py
    app/db/{models.py,migrations/}
    tests/{test_diff.py,test_rectify.py,golden/}
  pwa/            React + Vite, capture + overlay + offline queue
  admin/          React, calibration + review + dashboard
  eval/           frozen eval set, ground-truth CSVs, scoring scripts
  infra/          docker-compose, Fly/Railway config, R2 setup, marker PDFs
  docs/           SOP.md, runbook.md, accuracy-report.md
```

`eval/` and `tests/golden/` are first-class. Every threshold change gets a scoring run against the frozen set, in CI. No "it looked better on my machine".

---

## 11. Known limits to state to the client now

1. Two identical watches in different slots are interchangeable to this system. It counts and detects change; it cannot say *which* unit.
2. A watch moved to a **different shelf** shows as missing on one and new on the other unless we add cross-shelf matching (Phase 5).
3. A watch removed and replaced by a similar-looking one in the same slot may read as UNCHANGED if the SSIM/embedding thresholds are loose. Tighten per shelf, and keep periodic physical audits.
4. Occlusion at slot edges, watches stacked on top of each other, or items placed outside the grid will be missed. The SOP must forbid off-slot placement.
5. This is a control, not a security system. It does not prove who took a watch or when.
