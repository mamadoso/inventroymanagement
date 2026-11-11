# Ahadi — Watch Shelf Photo Count & Change Detection

Camera-based inventory checks for a watch shop: photograph a shelf, the system counts the
watches, compares the photo with the previous accepted photo of that shelf, and reports
what is **missing / new / moved / swapped**.

**Deliberately out of scope:** identifying the brand, model or reference of a watch.
The reasoning is in [`PLAN.md`](PLAN.md) §0 — in short, a front-facing shelf photo cannot
reliably give a reference number, and identical units cannot be told apart. Counting and
change detection is where the camera is genuinely strong.

## Documents

| File | What it is |
|---|---|
| [`PLAN.md`](PLAN.md) | Full programme: architecture, phases, risks, identity (Phase 3) design |
| [`MVP-PLAN.md`](MVP-PLAN.md) | Locked 6-week MVP build plan: count + change detection, cloud AI |
| [`demo/`](demo) | **Working demo** — runs on Ubuntu, works on Android and iPhone browsers |
| [`demo/docs/AI-PROVIDERS.md`](demo/docs/AI-PROVIDERS.md) | OpenAI-compatible AI config (OpenAI, OpenRouter, Groq, Gemini, local vLLM/Ollama) |
| [`demo/docs/DEPLOY-UBUNTU.md`](demo/docs/DEPLOY-UBUNTU.md) | Install on an Ubuntu box, systemd, firewall, phones |
| [`demo/docs/DEMO-SCRIPT.md`](demo/docs/DEMO-SCRIPT.md) | The 5-minute pitch for the manager, with prepared objections |

## Try the demo in three commands

```bash
cd demo
./setup.sh
python selftest.py     # offline self-check, no API key, no browser
./run.sh
```

Then open `http://<server-ip>:8080` on a laptop and `http://<server-ip>:8080/capture?shelf=1`
on a phone on the same Wi-Fi. No AI key is needed to start — it runs in offline heuristic
mode; add a key to `demo/.env` for real vision.

`selftest.py` draws synthetic shelves and asserts the three behaviours that matter:
count is right, a removed watch reads as exactly 1 MISSING, and a moved watch reads as
MOVED rather than missing-plus-new. Run it after any change.

## How a comparison works

```
phone browser ──POST photo──► FastAPI ──► vision model (OpenAI-compatible)
                                             │  normalised boxes + neutral description
                                             ▼
                                    Hungarian match vs baseline photo
                                    (spatial proximity + crop appearance)
                                             │
                              missing / new / moved / swapped / unchanged
                                             ▼
                              annotated photos + report + audit row
```

Each shelf has three **views** (front, left, right). A view is only ever compared against
the baseline of the *same* view — cross-angle matching is not attempted in the demo.
The first photo of a view becomes its baseline; a baseline is only ever replaced by an
explicit human "accept", never silently.

## Stack

FastAPI + Jinja2 + SQLite + Pillow/numpy/scipy, `httpx` for the AI call. No GPU, no
Docker, no Postgres, no message queue. `data/` holds the database, every photo and
`ai_calls.jsonl`.
