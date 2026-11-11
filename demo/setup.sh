#!/usr/bin/env bash
# One-time setup on Ubuntu/Debian. Creates .venv and installs dependencies.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python3 not found. Run: sudo apt-get install -y python3 python3-venv python3-pip" >&2
  exit 1
fi

if ! "$PY" -c "import ensurepip" >/dev/null 2>&1; then
  echo "python3-venv missing. Run: sudo apt-get install -y python3-venv" >&2
  exit 1
fi

echo "Using $("$PY" --version)"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel >/dev/null
pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created demo/.env from the template — add your AI key there."
fi

mkdir -p data/images
echo
echo "Setup complete."
echo "  1. edit demo/.env   (AHADI_AI_API_KEY / AHADI_AI_MODEL)"
echo "  2. ./run.sh"
echo "  3. open http://localhost:${AHADI_PORT:-8080}"
