# Watch Shelf Photo-Count & Change Detection — Delivery Plan

**Project:** Ahadi Inventory Management AI
**Goal:** Staff photograph a watch shelf → system counts watches, compares with the previous photo of that same shelf, and reports exactly what changed (missing / new / moved / swapped).
**Status:** Pre-build plan, v1
**Date:** 2026-11-11

> **Decisions locked 2026-11-11** → build plan is in `MVP-PLAN.md`
> 1. **Identity level:** count + what-changed only. Brand/model/serial identification is **out of scope** for the MVP.
> 2. **Catalog photos:** messy/partial → the lightbox shoot is **deferred**, not needed for MVP scope. Slot crops + embeddings are still logged from day 1 so Phase 3 costs no extra collection.
> 3. **Deployment:** cloud API (VLM + hosted inference). No on-prem GPU box for the MVP.
> 4. **Scope:** 6-week MVP, then a measured go/no-go on Phase 3 identity.
>
> Consequence worth noting: with identity out of scope, **object detection is no longer required** — a calibrated slot grid turns the problem into per-slot presence classification, which is far more reliable. See `MVP-PLAN.md` §0.

---

## 0. Direct answers to the two questions

### "Can we compare a shelf photo to the previous one and report changes?"
**Yes — this is a solved-class problem and it is the most reliable part of the project.**

The important design insight: **change detection does not require knowing the watch model.** If we can (a) locate each watch slot in the image and (b) describe the crop in that slot as a vector, then comparing two photos of the same shelf is just matching slot-to-slot and comparing crops. We can report:

> "Shelf B3: count 24 → 22. Slot 7 is now empty. Slot 12 now holds a different item. Everything else unchanged."

…without ever naming a single brand. That is a shippable MVP in weeks, not months.

### "Can AI tell the exact watch model?"
**Partly. Be honest with the client about the three tiers:**

| Tier | What it means | Realistic accuracy | Verdict |
|---|---|---|---|
| **T1 — Category/attributes** | "steel watch, black dial, rotating bezel, oyster bracelet, ~40mm" | 90%+ | Reliable, free from any VLM today |
| **T2 — Brand + collection** | "Rolex Submariner-type", "Omega Seamaster-type" | 75–92% with a catalog; brand text on dial via OCR is near-perfect when legible | Reliable enough to *rank candidates*, not to auto-commit |
| **T3 — Exact reference / serial** | "126610LN-0001, serial 8xxxxx" | **Not reliably possible from a front shelf photo** | Do not promise it |

Why T3 fails: reference numbers differ by movement, case-back engraving, certificate, or a 0.5mm bezel change — invisible from the front at shelf distance. Serial numbers are on the case-back, which faces the stand. Identical SKUs on the same tray are **visually indistinguishable from each other**, so "which of my five identical Datejusts is missing" cannot be answered by vision at all.

**The practical answer:** build **fine-grained image retrieval against the shop's own catalog** (embed each detected watch, nearest-neighbour search over their SKUs, return top-3 + confidence). Above a confidence threshold → auto-assign. Below → human review queue. For per-unit uniqueness (serial-level), the correct tool is **RFID/NFC tags or a barcode on the display card**, not a camera.

> **Strong recommendation:** plan for a **hybrid** system. Vision does counting, placement, change detection, and visual verification. RFID/barcode does unique-unit identity. Vision-only identity is the single biggest scope risk in this project.

---

## 1. What actually determines success (before any code)

| Factor | Why it dominates | Mitigation |
|---|---|---|
| **Camera repeatability** | A handheld photo at a different angle makes slot-matching noisy | Printed **ArUco/AprilTag marker** at each shelf corner + a cheap 3D-printed phone jig. Target: same pose ±3cm, ±5° |
| **Glass/ acrylic glare** | Reflections destroy both detection and retrieval | Polarizing filter on the lens, off-axis lighting, capture SOP ("tilt 15°"), reject glare-heavy frames automatically |
| **Lighting consistency** | Embeddings shift under different colour temperature | Capture with shelf lights ON, fixed white balance, plus a colour-checker patch in the frame |
| **Catalog photos** | Retrieval quality is capped by the quality of reference images | Collect 3–6 angles per SKU on a lightbox. This is a one-time staff task, ~5 min/SKU |
| **Watch density & orientation** | Overlapping/tilted watches merge into one blob | Cap slot size, use dividers/trays with fixed pockets, define max 1 watch per slot |
| **Resolution** | A 400px-wide crop cannot distinguish two similar bezels | ≥12MP phone, ≤1.5m distance, ≤20 watches per frame → ≥600px per watch |

**If the client will not accept a phone jig + markers + tray dividers, expect accuracy to drop by half.** Make this a contractual precondition.

---

## 2. Pipeline design

```
CAPTURE (PWA on phone/tablet)
  scan shelf QR  →  guided overlay  →  quality gate on-device
      blur (Laplacian var) ≥ 100
      mean brightness 60–190
      ≥ 3 ArUco markers detected
      no over-exposed blob > 4% of frame
  upload original + EXIF + shelf_id + user + timestamp
                    │
                    ▼
BACKEND (queue, idempotent per capture)
 1. PREPROCESS   homography from markers → rectified shelf plane image
                 colour patch → white-balance normalisation
 2. LAYOUT       shelf map (rows × cols of slots) from calibration store
                 auto-detect dividers to refine slot boxes
 3. DETECT       YOLO11/YOLOv8-seg → one instance per watch (+ mask area)
 4. ASSIGN       each detection → nearest slot centroid (in shelf-plane coords)
                 slot states: OCCUPIED / EMPTY / MULTI / OCCLUDED
 5. CROP+EMBED   per-slot crop → SigLIP/DINOv2 vector (768-d)
                 + attribute head: dial colour, case metal, bezel, strap, shape
                 + OCR on dial (brand text) → PaddleOCR/TrOCR
 6. IDENTIFY     ANN search in shop catalog index (pgvector) → top-5 SKUs
                 accept if score ≥ τ_auto and margin ≥ m
                 else T2 candidate list → human review
 7. DIFF         compare vs baseline snapshot for this shelf
                 MISSING / NEW / MOVED / SWAPPED / UNCHANGED / UNCERTAIN
 8. REPORT       push to staff + inventory system, write immutable audit log
                 low-confidence items → review queue
```

### Step 7 — the diff algorithm (the core IP)

Match on **slot identity**, never raw pixel coordinates:

```python
def diff(prev: ShelfState, cur: ShelfState):
    out = []
    for slot in union(prev.slots, cur.slots):
        p, c = prev.get(slot), cur.get(slot)
        if p and not c:                      out.append(MISSING(slot, p))
        elif c and not p:                    out.append(NEW(slot, c))
        elif p.sku and c.sku and p.sku == c.sku:
            out.append(UNCHANGED(slot, c) if cos(p.vec, c.vec) > 0.82
                        else SWAPPED(slot, p, c))       # same SKU, different unit
        elif p.sku and c.sku:                out.append(SWAPPED(slot, p, c))
        else:                                out.append(UNCERTAIN(slot, p, c))
    # second pass: re-id MISSING against NEW elsewhere on the shelf → MOVED
    return resolve_moves(out)
```

Notes:
- **`resolve_moves`** matters: staff tidy shelves between counts. Without it, every tidy-up looks like theft.
- **Cosine on the crop embedding** is the "is this literally the same physical object" check, which works even for SKUs we cannot name.
- **Baseline policy:** baseline = last *confirmed* count, not last photo. A confirmed count is one a human accepted. Prevents error drift.
- **Fallback with zero ML identity:** per-slot **SSIM/template diff** after rectification. Even with no catalog at all this reliably answers "which 3 slots changed". Use it as the safety net and as the MVP.

---

## 3. Tech stack (opinionated, boring, deployable)

| Layer | Choice | Rationale |
|---|---|---|
| Capture app | **PWA (React + Vite + PWA manifest)**, `getUserMedia`, on-device quality gate with OpenCV.js | No app-store cycle, works on staff phones, offline queue |
| API | **Python 3.12 + FastAPI + Pydantic** | CV ecosystem, typed contracts |
| Async work | **Redis + arq** (or Celery) | capture ≠ inference, need retries & backpressure |
| Storage | **Postgres 16 + pgvector**, images in **S3/MinIO** | one DB for relational + ANN at this scale |
| Detection | **YOLO11m / YOLOv8m**, exported to **ONNX** | 30–60ms GPU, cheap, private |
| Embeddings | **SigLIP-so400m** (or DINOv2 ViT-L) | best fine-grained retrieval, 768-d |
| Attributes | small multi-label head on the embedding | dial colour / metal / strap / bezel |
| OCR | **PaddleOCR** (dial text), **TrOCR** for printed refs on paperwork | brand confirmation |
| VLM fallback | **GPT-4o / Gemini 2.x vision**, structured JSON output | unknown items, attribute sanity, PoC bootstrap |
| Serving | **Triton** or FastAPI + ONNXRuntime, Docker Compose | single on-prem GPU box |
| Admin UI | **React + TanStack Query** — shelf map editor, review queue, audit log | humans are part of the system |
| Observability | Prometheus + Grafana, every prediction logged with model version | you cannot defend accuracy claims without this |

**On-prem vs cloud:** watches are high-value and the shop will not want images of stock leaving the building. Default to **on-prem GPU box** (RTX 4000 Ada / RTX 5090 class, ~$2.5–5k, or a used L4). Keep a cloud VLM path as an optional, switchable enrichment step.

---

## 4. Data plan (this is the real schedule driver)

| Dataset | Volume | Source | Cost |
|---|---|---|---|
| Shelf photos (detection) | **600–1,200 images**, all shelves, varied lighting/angles | staff, 2 weeks of daily counts | ~$400–800 annotation (CVAT + Roboflow, ~8s/image) |
| Watch crops | auto-generated from shelf boxes, ~15–30k | pipeline | free |
| Catalog reference images | **3–6 angles × every SKU** | shop lightbox shoot | biggest staff time sink — plan a day per ~150 SKUs |
| Hard negatives | near-identical pairs, mislabels | from review queue | free, accumulates |
| Eval set | **150 shelf photos with ground-truth counts + slot lists**, frozen | manual audit with physical count | non-negotiable, do it before training |

**Rule: freeze the eval set before you tune anything.** Otherwise every "accuracy" number is fiction.

---

## 5. Phases, milestones, acceptance criteria

### Phase 0 — Discovery & feasibility (1 week)
Deliverables: shelf inventory (how many shelves/trays, watches per shelf, layout photos), SKU count, lighting audit, network on shop floor, decision on RFID, signed-off acceptance criteria.
**Gate:** if the shop needs serial-level identity, RFID is in scope.

### Phase 1 — PoC: count + change detection, no model ID (3 weeks)
- 1 shelf, phone jig + ArUco, VLM-only detection (no training) + per-slot SSIM/embedding diff.
- Output: a report that says "count 24→22, slots 7 and 12 changed".
**Acceptance:** count exact-match ≥ 90% on 50 captures; false "missing" alarms ≤ 5% of shelves; capture→report ≤ 20s.
**Kill/pivot decision here.** If pose repeatability is bad, fix the jig before spending on models.

### Phase 2 — Production detection + slot calibration (4 weeks)
- Annotate, train YOLO11, build shelf-map editor, homography pipeline, diff engine, PWA, audit log.
**Acceptance:** detection mAP@0.5 ≥ 0.90; count exact-match ≥ 97%; ≤ 1 false alarm per 100 shelf counts; 5 concurrent shelves ≤ 15s each.

### Phase 3 — Identity: catalog retrieval + human review (4 weeks)
- Lightbox catalog shoot, embedding index, top-5 candidates, attribute head, dial OCR, review queue, confidence thresholds.
**Acceptance:** top-1 correct ≥ 80%, top-3 ≥ 92% on known SKUs; ≥ 70% of items auto-committed without review; **zero auto-commits below τ**.

### Phase 4 — Pilot → rollout (4 weeks)
- 3–5 shelves live for 4 weeks, daily reconciliation, staff SOP, exception workflow, ERP/inventory export, monitoring, load test.
**Acceptance:** staff time per shelf count drops ≥ 50% vs manual; ≥ 95% of shelves reconcile without escalation; documented SOP.

### Phase 5 — Optional
RFID unit-level identity, multi-camera fixed rigs (lights-out counting), theft/after-hours anomaly alerts, ERP two-way sync, analytics (sell-through per slot, planogram compliance).

**Total: ~16 weeks to a production pilot with 3–4 people.**

---

## 6. Team & cost envelope

| Role | Effort |
|---|---|
| CV/ML engineer | 1.0 FTE, weeks 1–16 |
| Backend engineer | 0.75 FTE, weeks 3–16 |
| Frontend/PWA | 0.5 FTE, weeks 4–16 |
| PM / data & annotation lead | 0.4 FTE, weeks 0–16 |
| Shop staff (catalog shoot, ground truth) | ~10–20 person-days |

Infra: one GPU box (~$3–5k) + phones/jigs/markers (<$500) + optional cloud VLM ($25–150/mo at 5k shelf images).

---

## 7. Risk register

| Risk | Impact | Likelihood | Response |
|---|---|---|---|
| Client expects exact reference/serial from a photo | Contract dispute | **High** | Tier table in §0, written acceptance criteria, T3 explicitly out of scope unless RFID |
| Handheld capture varies too much | Diff unreliable | **High** | Jig + markers + reject-and-reprompt UX; hard precondition |
| Many near-identical SKUs | Retrieval top-1 collapses | **Medium-High** | Attribute head + dial OCR + human review; measure per-SKU confusion matrix |
| Glare on display glass | Detection + embedding noise | **Medium** | Polariser, SOP, automatic glare rejection |
| Not enough catalog photos | Identity slips | **Medium** | Phase 3 gate on catalog completeness; partial rollout per shelf |
| Error drift from bad baselines | Silent wrong inventory | **Medium** | Baseline = last human-confirmed count; periodic full physical audit |
| Staff bypass the process | No data | **Medium** | Make it faster than manual counting; make it the tool they must use to close the shop |
| Images of stock leave the building | Trust/legal | **Low-Med** | On-prem default; cloud enrichment opt-in, anonymised crops only |

---

## 8. What we need from the shop to start

1. Photos of every shelf/tray (one each) + count of shelves and total SKUs.
2. Do they have **existing digital catalog photos per SKU**? (Yes → Phase 3 is cheap. No → schedule the lightbox shoot.)
3. Is **serial/unit-level** tracking required by their insurer or ERP? (Yes → RFID in scope.)
4. Existing inventory system to integrate with (Excel, Odoo, custom, none?).
5. Network on the shop floor (Wi-Fi quality, is a local server box acceptable?).
6. Who counts, how often, and how long it takes today (baseline for the ROI claim).
7. Lighting conditions and whether they can be made consistent.

---

## 9. Recommended MVP (if only 6 weeks are available)

**→ Now specified in detail in `MVP-PLAN.md`.** Summary:

**Shelf photo → count + "what changed" report, no model naming.**

- PWA capture with shelf QR, ArUco markers, phone jig, on-device quality gate.
- Homography rectification + slot grid calibration UI (one-time per shelf, 10 min).
- VLM structured detection (Gemini/GPT-4o vision, JSON boxes) — no training needed.
- Per-slot embedding + SSIM change detection.
- Diff report: count delta, empty slots, new slots, changed slots, moved items.
- Daily digest + immutable audit log.

This is genuinely useful on day one ("we have 22 not 24, and slot 7 is empty"), carries no identity risk, and creates the exact dataset needed for Phase 3.
