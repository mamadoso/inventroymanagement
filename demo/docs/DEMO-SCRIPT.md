# The 5-minute demo for the manager

Goal: get agreement that **AI can count a shelf and report changes**, and get the
next step (a real pilot) approved. Do not try to prove it identifies watch models —
that is out of scope and it is the claim that would sink you.

## Before the meeting (do not skip)

1. `./run.sh` on the Ubuntu box, note the LAN URL.
2. Open `/health` → must be green ✅.
3. Set up **one real shelf** with ~10–20 watches. Take the baseline photo.
4. Take a second photo with nothing changed → must say **"No change"**.
   If it does not, fix the camera position before showing anyone anything.
5. Take a third photo with **one watch removed** → must say **"1 missing"**.
6. Take a fourth with a watch **moved to another spot** → must say **"moved"**, not missing.
7. Rehearse once end-to-end on your own phone via the QR code.
8. Keep `AHADI_PROVIDER=mock` as a fallback if the venue has no internet.

## Script

**0:00 — Frame it.**
> "Counting the vitrines today is manual, slow and nobody trusts the numbers. This is a
> phone photo. Watch what happens."

**0:30 — Scan and shoot.** Open the capture page on your phone, project it, take the photo.
Result appears in a few seconds with a count.

> "Twenty-two watches. It did not read a barcode, there are no tags, nothing was
> pre-registered. It looked at the photo."

**1:30 — The money moment.** Walk over, quietly take one watch off the shelf. Photograph again.

> "Now it is twenty-one, and it is telling me *which position* is empty — number fourteen,
> the gold one with the dark dial. That is the whole inventory check, in five seconds."

**2:30 — Show it is not just counting.** Put the watch back in a *different* spot. Photograph.

> "It says moved, not missing. That matters, because staff tidy the shelves all day. A
> system that cries wolf on every tidy-up gets switched off in a week."

**3:30 — Be the one to say what it cannot do.** This is what makes the rest believable.

> "Two things it will not do. It will not tell you the reference number of a watch from a
> shelf photo — nobody can, reliably, and anyone who promises that is selling you a demo.
> And if you have five identical watches, it can tell you one is gone but not *which* one.
> For unit-level identity we would add tags. Counting and change detection is what the
> camera is genuinely good at."

**4:30 — Ask for the pilot.**

> "Give me one vitrine for four weeks. I will report the count accuracy against your own
> physical counts, and how many minutes per day this saves. If the numbers do not hold,
> we stop."

## Objections, prepared

| They say | You say |
|---|---|
| "Is it accurate?" | "On this shelf, right now, you watched it get three cases right. The pilot measures it against your own counts — that is what the four weeks are for." |
| "What about glass reflections?" | "Real problem. Polarising filter on the lens and a fixed phone position. That is a 20-dollar part, and it is in the plan." |
| "Our data leaving the shop?" | "Currently it goes to the AI provider. There is a local-model mode that keeps every photo inside the building — slower and slightly less accurate, but it is a config change, not a rebuild." |
| "We already have RFID." | "Then we use the camera as the verification layer: RFID says what should be there, the photo says what is actually there. Disagreement is the alert." |
| "Can staff really do this?" | "Scan, aim, press. Under a minute per shelf, no training." |
| "How much?" | "The demo runs on a small PC. Cloud AI costs are a few cents per shelf count — under $100/month for the whole shop. The cost is the pilot, not the licence." |

## What to have on screen while you talk

- `/shelves/1` — the three-angle board with green/red pills.
- A `/diffs/N` report page: two photos, boxes drawn, red dashed on the missing one.
  Managers believe the annotated picture before they believe any number.
- `/health` — so you can show the model and latency honestly if asked.

## Never do this

- Do not promise model/brand identification.
- Do not demo on a shelf you have not pre-validated that morning.
- Do not let the manager hold the phone and freestyle. You drive, they watch.
