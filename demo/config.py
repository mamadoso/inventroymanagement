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

# max_tokens for the detection call. Small watch grids need room for many boxes.
AI_MAX_TOKENS = int(_env("AHADI_AI_MAX_TOKENS", default="1800"))
AI_TEMPERATURE = float(_env("AHADI_AI_TEMPERATURE", default="0"))
AI_SEND_SCHEMA = _env("AHADI_AI_JSON_SCHEMA", default="1") not in ("0", "false", "no")
AI_RETRIES = int(_env("AHADI_AI_RETRIES", default="2"))

# --------------------------------------------------------------- image / perf
# Images are downscaled to this max side before being sent to the model.
# Bigger = better detection of small watches, slower and more expensive.
MAX_SIDE = int(_env("AHADI_MAX_SIDE", default="1536"))
JPEG_QUALITY = int(_env("AHADI_JPEG_QUALITY", default="86"))
THUMB_SIDE = int(_env("AHADI_THUMB_SIDE", default="420"))
REQUEST_TIMEOUT = float(_env("AHADI_TIMEOUT", default="120"))
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
