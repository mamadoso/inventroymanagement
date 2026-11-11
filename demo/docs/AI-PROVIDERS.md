# AI provider configuration (OpenAI-compatible)

Everything vision-related is **one** OpenAI-compatible `POST {base_url}/chat/completions`
call. There is no provider-specific SDK anywhere in the code. Switching providers is
three lines in `demo/.env` and a restart.

## The three settings

```ini
AHADI_AI_BASE_URL=https://api.openai.com/v1
AHADI_AI_API_KEY=sk-...
AHADI_AI_MODEL=gpt-4o
```

`AHADI_PROVIDER=openai` (default) uses them. `AHADI_PROVIDER=mock` runs the offline
heuristic instead — no key, no network, useful for rehearsing the UI.

The base URL is forgiving: `https://host/v1`, `https://host/v1/`, or a bare `host:port`
all work, and a full `.../chat/completions` URL is accepted as-is.

## Recipes

### OpenAI
```ini
AHADI_AI_BASE_URL=https://api.openai.com/v1
AHADI_AI_API_KEY=sk-...
AHADI_AI_MODEL=gpt-4o
```
`gpt-4o` is the best default for box localisation. `gpt-4o-mini` is cheaper and slightly
worse on small/dense watches.

### OpenRouter (many vendors, one key)
```ini
AHADI_AI_BASE_URL=https://openrouter.ai/api/v1
AHADI_AI_API_KEY=sk-or-...
AHADI_AI_MODEL=google/gemini-2.5-flash
```
Other good ones there: `openai/gpt-4o`, `anthropic/claude-3.5-sonnet`, `qwen/qwen2.5-vl-72b-instruct`.

### Google Gemini via its OpenAI-compatible endpoint
```ini
AHADI_AI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
AHADI_AI_API_KEY=AIza...
AHADI_AI_MODEL=gemini-2.5-flash
```
Cheap, fast, strong at bounding boxes. Recommended for the demo if you want low cost.

### Groq (very fast, free tier)
```ini
AHADI_AI_BASE_URL=https://api.groq.com/openai/v1
AHADI_AI_API_KEY=gsk_...
AHADI_AI_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
```

### Together / Fireworks
```ini
AHADI_AI_BASE_URL=https://api.together.xyz/v1
AHADI_AI_API_KEY=...
AHADI_AI_MODEL=Qwen/Qwen2.5-VL-32B-Instruct-Turbo
```

### Local model — nothing leaves the LAN
```ini
# Ollama
AHADI_AI_BASE_URL=http://127.0.0.1:11434/v1
AHADI_AI_API_KEY=ollama
AHADI_AI_MODEL=qwen2.5vl:7b
AHADI_AI_JSON_SCHEMA=0
```
```ini
# vLLM
AHADI_AI_BASE_URL=http://127.0.0.1:8000/v1
AHADI_AI_API_KEY=none
AHADI_AI_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
AHADI_AI_JSON_SCHEMA=0
```
```ini
# LM Studio
AHADI_AI_BASE_URL=http://127.0.0.1:1234/v1
AHADI_AI_API_KEY=lm-studio
AHADI_AI_MODEL=llava-v1.6-mistral-7b
AHADI_AI_JSON_SCHEMA=0
```
For local models, `AHADI_MAX_SIDE=1024` keeps them responsive on modest GPUs.

## What the model is asked to return

```json
{
  "items": [ { "box": [0.12, 0.44, 0.09, 0.13], "desc": "dark dial, steel bracelet", "confidence": 0.9 } ],
  "total": 18,
  "quality": "good",
  "notes": "two watches at the right edge are cut off"
}
```
`box` is `[x, y, width, height]` normalised to 0..1 from the top-left. The client also
tolerates `bbox`, `bbox_2d`, dict-shaped boxes, and 0–1000 integer coordinates (some
models emit those) — they are normalised automatically.

**The prompt explicitly forbids brand/model identification.** The demo answers "how many,
where, what changed" — not "which Rolex is this". That boundary is deliberate; see
`../PLAN.md` §0.

## Robustness behaviour

The client degrades instead of failing when a gateway is quirky, in this order:

1. `response_format: json_schema` (strict)
2. `response_format: json_object`
3. plain prompt, JSON parsed out of the text

`AHADI_AI_JSON_SCHEMA=0` skips straight to step 2. Transient 429/5xx are retried
`AHADI_AI_RETRIES` times with backoff. Every call is appended to `data/ai_calls.jsonl`
with the variant used, latency and token usage — check that file first when a provider
behaves oddly.

## Verifying before the meeting

Open **`/health`** in a browser. It shows the effective config, calls `GET /models` to
confirm the model id actually exists on your key, and has a **Re-test connection** button.
Do this *before* you walk in with the manager.

## Cost of a demo

One shelf photo ≈ 1,200–1,600 input tokens + ~150 output. At `gpt-4o` prices that is
roughly **$0.006 per count**; `gemini-2.5-flash` is several times cheaper. A hundred demo
counts is pocket change.
