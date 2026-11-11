"""Single OpenAI-compatible vision client.

Works with any endpoint that implements POST {base_url}/chat/completions with
`messages` content parts and image_url data URIs: OpenAI, OpenRouter, Groq,
Together, Fireworks, DeepSeek-style gateways, Gemini's OpenAI-compatible URL,
and local vLLM / Ollama / LM Studio servers.

Defensive by design - gateways differ in which optional fields they accept, so
unsupported fields are stripped and the request retried rather than failing the
demo. Every request/response is logged to data/ai_calls.jsonl for debugging.
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import httpx
from PIL import Image

import config
import imaging
from vision.base import (
    SYSTEM,
    USER,
    Detection,
    VisionError,
    build_detection,
    parse_json_object,
)

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items", "total", "quality"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["box", "desc"],
                "properties": {
                    "box": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "[x, y, width, height] normalised 0..1",
                    },
                    "desc": {"type": "string"},
                    "confidence": {"type": "number"},
                },
            },
        },
        "total": {"type": "integer"},
        "quality": {"type": "string", "enum": ["good", "glare", "blur", "too_far", "dark", "other"]},
        "notes": {"type": "string"},
    },
}


def _log(record: dict) -> None:
    try:
        config.ensure_dirs()
        path = config.DATA_DIR / "ai_calls.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _data_uri(img: Image.Image) -> str:
    blob = imaging.to_bytes(img, config.MAX_SIDE)
    return "data:image/jpeg;base64," + base64.b64encode(blob).decode()


def _messages(img: Image.Image) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER},
                {"type": "image_url", "image_url": {"url": _data_uri(img)}},
            ],
        },
    ]


def _payload(img: Image.Image, *, schema: bool, temperature: bool, token_key: str) -> dict:
    body: dict = {"model": config.AI_MODEL, "messages": _messages(img)}
    if temperature:
        body["temperature"] = config.AI_TEMPERATURE
    body[token_key] = config.AI_MAX_TOKENS
    if schema:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "shelf_audit", "strict": True, "schema": RESPONSE_SCHEMA},
        }
    else:
        body["response_format"] = {"type": "json_object"}
    return body


def _post(client: httpx.Client, body: dict) -> httpx.Response:
    headers = {"Content-Type": "application/json"}
    if config.AI_API_KEY:
        headers["Authorization"] = f"Bearer {config.AI_API_KEY}"
    return client.post(config.chat_completions_url(), json=body, headers=headers)


def _extract(resp_json: dict) -> str:
    try:
        choice = resp_json["choices"][0]
        msg = choice.get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):  # some gateways return content parts
            content = "".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        if not content and msg.get("reasoning_content"):
            content = msg["reasoning_content"]
        if not content:
            raise VisionError("model returned no content")
        return content
    except (KeyError, IndexError, TypeError) as exc:
        raise VisionError(f"unexpected response shape: {json.dumps(resp_json)[:300]}") from exc


def detect(img: Image.Image) -> Detection:
    """Run one shelf photo through the configured vision model."""
    if not config.AI_MODEL:
        raise VisionError("AHADI_AI_MODEL is not set")

    # Gateways differ in which optional fields they accept, so degrade instead of
    # failing: json_schema -> json_object -> plain, and max_tokens ->
    # max_completion_tokens if the endpoint rejects that field name.
    variants: list[list] = []
    if config.AI_SEND_SCHEMA:
        variants.append(["json_schema", True, True, "max_tokens"])
        variants.append(["json_object", False, True, "max_tokens"])
    else:
        variants.append(["json_object", False, True, "max_tokens"])
    variants.append(["plain", False, False, "max_tokens"])

    errors: list[str] = []
    attempts = max(1, config.AI_RETRIES + 1)
    started = time.perf_counter()
    tried_token_alt = False

    with httpx.Client(timeout=config.REQUEST_TIMEOUT, follow_redirects=True) as client:
        for attempt in range(attempts):
            i = 0
            while i < len(variants):
                label, schema, temp, token_key = variants[i]
                body = _payload(img, schema=schema, temperature=temp, token_key=token_key)
                try:
                    resp = _post(client, body)
                except httpx.HTTPError as exc:
                    errors.append(f"{label}: transport {type(exc).__name__}: {exc}")
                    i += 1
                    continue

                if resp.status_code in (400, 404, 422):
                    detail = resp.text[:400]
                    low = detail.lower()
                    if (
                        token_key == "max_tokens"
                        and not tried_token_alt
                        and ("max_token" in low or "max_completion" in low)
                    ):
                        tried_token_alt = True
                        variants[i] = [label, schema, temp, "max_completion_tokens"]
                        errors.append(f"{label}: max_tokens rejected, using max_completion_tokens")
                        continue  # retry this same variant with the other field name
                    errors.append(f"{label}: HTTP {resp.status_code} {detail}")
                    if label == "plain":
                        raise VisionError(
                            f"endpoint rejected the request ({resp.status_code}). "
                            f"Check AHADI_AI_MODEL / AHADI_AI_BASE_URL. Detail: {detail}"
                        )
                    i += 1
                    continue
                if resp.status_code in (401, 403):
                    raise VisionError(
                        f"HTTP {resp.status_code} - AHADI_AI_API_KEY is wrong, lacks "
                        f"access, or is missing for {config.normalized_base_url()}"
                    )
                if resp.status_code == 429:
                    errors.append("429 rate limited")
                    time.sleep(min(10.0, 2.0 * (attempt + 1)))
                    break
                if resp.status_code >= 400:
                    errors.append(f"{label}: HTTP {resp.status_code} {resp.text[:300]}")
                    break

                try:
                    rj = resp.json()
                    parsed = parse_json_object(_extract(rj))
                except (VisionError, ValueError) as exc:
                    errors.append(f"{label}: {exc}")
                    i += 1
                    continue

                det = build_detection(
                    parsed, "openai", config.AI_MODEL, int((time.perf_counter() - started) * 1000)
                )
                usage = rj.get("usage") or {}
                _log(
                    {
                        "ts": int(time.time()),
                        "ok": True,
                        "variant": label,
                        "attempt": attempt,
                        "model": config.AI_MODEL,
                        "base": config.normalized_base_url(),
                        "items": len(det.items),
                        "quality": det.quality,
                        "latency_ms": det.latency_ms,
                        "usage": usage,
                    }
                )
                return det
            if attempt < attempts - 1:
                time.sleep(1.0)

    _log({"ts": int(time.time()), "ok": False, "base": config.normalized_base_url(),
          "model": config.AI_MODEL, "errors": errors})
    raise VisionError(" | ".join(errors[-4:]) or "AI call failed")


def check_connection() -> dict:
    """Used by the /health 'Test AI' button. Never raises."""
    out = {
        "provider": config.active_provider(),
        "base_url": config.normalized_base_url(),
        "model": config.AI_MODEL,
        "key_set": bool(config.AI_API_KEY),
        "ok": False,
        "detail": "",
    }
    if out["provider"] == "mock":
        out["detail"] = "offline heuristic mode - set AHADI_AI_API_KEY for real AI"
        out["ok"] = True
        return out

    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=25.0, follow_redirects=True) as client:
            r = client.get(
                config.models_url(),
                headers={"Authorization": f"Bearer {config.AI_API_KEY}"} if config.AI_API_KEY else {},
            )
            if r.status_code < 400:
                names: list[str] = []
                try:
                    data = r.json().get("data") or []
                    names = [d.get("id", "") for d in data if isinstance(d, dict)]
                except ValueError:
                    pass
                out["models_seen"] = len(names)
                out["model_available"] = config.AI_MODEL in names if names else None
                out["detail"] = f"GET /models -> {r.status_code}"
                if names and out["model_available"] is False:
                    out["detail"] += f" - '{config.AI_MODEL}' NOT in list. Try one of: " + ", ".join(
                        n for n in names if "vision" in n or "vl" in n or "gpt-4" in n
                    )[:220]
                out["ok"] = True
            else:
                out["detail"] = f"GET /models -> HTTP {r.status_code}: {r.text[:200]}"
    except httpx.HTTPError as exc:
        out["detail"] = f"{type(exc).__name__}: {exc}"

    # Prove a vision round-trip on a tiny blank image even if /models is blocked.
    if not out["ok"]:
        try:
            det = detect(Image.new("RGB", (64, 48), (30, 30, 30)))
            out["ok"] = True
            out["detail"] = f"test detection returned {det.total} items"
        except VisionError as exc:
            out["detail"] = str(exc)[:400]

    out["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    return out
