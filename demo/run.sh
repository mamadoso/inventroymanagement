#!/usr/bin/env bash
# Start the demo. Plain HTTP by default; AHADI_SSL=1 enables a self-signed cert
# (needed only if you want the in-page live viewfinder on iOS over the LAN).
set -euo pipefail
cd "$(dirname "$0")"

[ -d .venv ] || ./setup.sh
# shellcheck disable=SC1091
source .venv/bin/activate

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1090
  . ./.env
  set +a
fi

HOST="${AHADI_HOST:-0.0.0.0}"
PORT="${AHADI_PORT:-8080}"

LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
[ -n "$LAN_IP" ] || LAN_IP="localhost"

echo "──────────────────────────────────────────────────────────"
echo "  Ahadi shelf-count demo"
echo "  desktop : http://${LAN_IP}:${PORT}"
echo "  phone   : http://${LAN_IP}:${PORT}/capture?shelf=1   (same Wi-Fi)"
echo "  AI      : ${AHADI_PROVIDER:-openai} / ${AHADI_AI_MODEL:-gpt-4o}"
echo "  health  : http://${LAN_IP}:${PORT}/health"
echo "──────────────────────────────────────────────────────────"

if [ "${AHADI_SSL:-0}" = "1" ]; then
  mkdir -p data
  if [ ! -f data/cert.pem ]; then
    echo "generating self-signed certificate (accept the warning once on the phone)…"
    openssl req -x509 -newkey rsa:2048 -keyout data/key.pem -out data/cert.pem \
      -days 825 -nodes -subj "/CN=ahadi-demo" \
      -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" >/dev/null 2>&1
  fi
  exec uvicorn app:app --host "$HOST" --port "$PORT" \
    --ssl-keyfile data/key.pem --ssl-certfile data/cert.pem
else
  exec uvicorn app:app --host "$HOST" --port "$PORT"
fi
