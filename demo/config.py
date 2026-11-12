"""Runtime configuration for the Ahadi shelf-count demo.

All AI access goes through ONE OpenAI-compatible client, so the provider is
purely a matter of three env vars: AHADI_AI_BASE_URL / AHADI_AI_API_KEY /
AHADI_AI_MODEL. That covers OpenAI, OpenRouter, Groq, Together, Azure-style
gateways, Gemini's OpenAI endpoint, and local vLLM / Ollama / LM Studio.

Everything is env-driven so the same code runs on a laptop or an Ubuntu box.
See docs/DEPLOY-UBUNTU.md and docs/AI-PROVIDERS.md.
"""
import os
import re
import time
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no external dep). Existing env vars win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


BASE_DIR = Path(__file__).resolve().parent
_load_dotenv(BASE_DIR / ".env")

DATA_DIR = Path(os.environ.get("AHADI_DATA", BASE_DIR / "data"))
IMAGE_DIR = DATA_DIR / "images"
DB_PATH = Path(os.environ.get("AHADI_DB", DATA_DIR / "demo.sqlite3"))


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return default


# ---------------------------------------------------------------- AI provider
# openai = any OpenAI-compatible chat/completions endpoint. mock = offline heuristic.
AI_PROVIDER = _env("AHADI_PROVIDER", "AI_PROVIDER").lower() or "openai"
AI_BASE_URL = _env(
    "AHADI_AI_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE", "AHADI_OPENAI_BASE_URL"
) or "https://api.openai.com/v1"
AI_API_KEY = _env("AHADI_AI_API_KEY", "OPENAI_API_KEY")
AI_MODEL = _env("AHADI_AI_MODEL", "OPENAI_MODEL") or "gpt-4o"

# max_tokens for the detection call. Each watch costs roughly 100-140 tokens of
# JSON (box + description), so a 12-watch shelf already needs ~1600. Too low and
# the model's output is cut off mid-JSON, which is a hard parse failure, not a
# soft one. If the response is ever truncated we double this and retry, up to the
# cap below.
AI_MAX_TOKENS = int(_env("AHADI_AI_MAX_TOKENS", default="4000"))
AI_MAX_TOKENS_CAP = int(_env("AHADI_AI_MAX_TOKENS_CAP", default="12000"))
AI_TEMPERATURE = float(_env("AHADI_AI_TEMPERATURE", default="0"))
AI_SEND_SCHEMA = _env("AHADI_AI_JSON_SCHEMA", default="1") not in ("0", "false", "no")
AI_RETRIES = int(_env("AHADI_AI_RETRIES", default="2"))
# Wall-clock budget for one detection, covering every fallback variant and retry.
# Must stay below the nginx proxy_read_timeout (180s) or the browser gives up
# while the server is still working, which looks like a hang rather than an error.
AI_DEADLINE = float(_env("AHADI_AI_DEADLINE", default="120"))

# --------------------------------------------------------------- image / perf
# Images are downscaled to this max side before being sent to the model.
# Bigger = better detection of small watches, slower and more expensive.
MAX_SIDE = int(_env("AHADI_MAX_SIDE", default="1536"))
JPEG_QUALITY = int(_env("AHADI_JPEG_QUALITY", default="86"))
THUMB_SIDE = int(_env("AHADI_THUMB_SIDE", default="420"))
REQUEST_TIMEOUT = float(_env("AHADI_TIMEOUT", default="90"))
MAX_UPLOAD_MB = int(_env("AHADI_MAX_UPLOAD_MB", default="25"))

# ------------------------------------------------------------- matching tuning
MATCH_MIN_SCORE = float(_env("AHADI_MATCH_MIN", default="0.40"))
MOVED_MIN_APPEAR = float(_env("AHADI_MOVED_MIN", default="0.72"))
SAME_OBJECT_APPEAR = float(_env("AHADI_SAME_OBJECT_MIN", default="0.62"))
# Two detections must be at least this spatially close to be considered the same
# *place*. Below it they are left unmatched so the move pass can pair them -
# otherwise a moved watch matches its new position and reports as "unchanged".
SAME_PLACE_SPATIAL = float(_env("AHADI_SAME_PLACE_MIN", default="0.45"))

# ------------------------------------------------------------------------ web
PUBLIC_BASE = _env("AHADI_PUBLIC_BASE")
HOST = _env("AHADI_HOST", default="0.0.0.0")
PORT = int(_env("AHADI_PORT", default="8080"))

DEFAULT_VIEWS = [
    ("front", "Front"),
    ("left", "Left angle"),
    ("right", "Right angle"),
]


def _git_version() -> str:
    """Short commit of the checkout this process was started from.

    Two debugging rounds were wasted on 'the fix did not work' when the fix was
    correct but the running process predated it. Printing the version and start
    time on /health makes that impossible to confuse.
    """
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=3,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


VERSION = _git_version()
STARTED_AT = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def ensure_dirs() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def normalized_base_url() -> str:
    """Accept 'https://host/v1', 'https://host/v1/' or a bare host."""
    url = AI_BASE_URL.rstrip("/")
    if not url:
        url = "https://api.openai.com/v1"
    if not re.match(r"^https?://", url):
        url = "https://" + url
    return url


def chat_completions_url() -> str:
    url = normalized_base_url()
    if url.endswith("/chat/completions"):
        return url
    return url + "/chat/completions"


def models_url() -> str:
    url = normalized_base_url()
    if url.endswith("/chat/completions"):
        return url[: -len("/chat/completions")] + "/models"
    return url.rstrip("/") + "/models"


def active_provider() -> str:
    """Fall back to the offline heuristic when no key is configured."""
    if AI_PROVIDER in ("mock", "offline", "none"):
        return "mock"
    if AI_API_KEY or "localhost" in AI_BASE_URL or "127.0.0.1" in AI_BASE_URL:
        return "openai"
    return "mock"


def public_url(request=None) -> str:
    """Base URL used for the QR code / share link."""
    if PUBLIC_BASE:
        return PUBLIC_BASE.rstrip("/")
    if request is not None:
        host = request.headers.get("host") or request.url.netloc
        scheme = request.headers.get("x-forwarded-proto") or request.url.scheme
        return f"{scheme}://{host}"
    return f"http://localhost:{PORT}"
