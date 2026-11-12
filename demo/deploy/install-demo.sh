#!/usr/bin/env bash
#
# Privileged install for the Ahadi shelf-count demo on planbmanager (45.86.4.108).
#
#   sudo ./install-demo.sh
#
# Optional overrides:
#   DEMO_USER=shop DEMO_PASS='something' EMAIL=you@example.com sudo -E ./install-demo.sh
#
# WHAT IT CHANGES (all additive, nothing under /opt/PlanBManager is touched):
#   /etc/nginx/.htpasswd-demo              new   basic-auth file
#   /etc/systemd/system/ahadi-demo.service new   runs as mamadoso, 127.0.0.1:8080
#   /etc/nginx/sites-available/zz-demo.linkwb.ir       new   vhost
#   /etc/nginx/sites-enabled/zz-demo.linkwb.ir         new   symlink
#   certbot adds TLS to the demo vhost only
#
# WHY "zz-" IN THE NAME: nginx has no explicit default_server on :80 and loads
# sites-enabled alphabetically. "planb" is currently first, so it is the default
# and answers unknown hosts with 404. A file named "demo" would sort BEFORE it
# and silently steal the default vhost. "zz-demo" sorts after, so planb keeps
# its behaviour.
#
# SAFETY: nginx config is validated with `nginx -t` before reload, and the new
# vhost is removed automatically if validation fails. nginx is reloaded, never
# restarted, so existing connections are not dropped. PlanB service state is
# captured before and compared after.
#
set -euo pipefail

APP_DIR="/home/mamadoso/ahadi-demo/demo"
VENV="$APP_DIR/.venv"
DATA_DIR="$APP_DIR/data"
DOMAIN="demo.linkwb.ir"
VHOST_NAME="zz-${DOMAIN}"
HTPASSWD="/etc/nginx/.htpasswd-demo"
UNIT="/etc/systemd/system/ahadi-demo.service"
SERVICE="ahadi-demo"

DEMO_USER="${DEMO_USER:-ahadi}"
EMAIL="${EMAIL:-}"

[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
[ -d "$APP_DIR" ] || { echo "missing $APP_DIR - run the unprivileged stages first"; exit 1; }
[ -x "$VENV/bin/uvicorn" ] || { echo "missing $VENV - run ./setup stage first"; exit 1; }

echo "=============================================================="
echo " PlanB state BEFORE (must be identical after)"
echo "=============================================================="
before_planb() {
  for s in planb-bot-listen planb-queue nginx mariadb php8.3-fpm; do
    printf '  %-22s %s\n' "$s" "$(systemctl is-active "$s" 2>&1)"
  done
  printf '  nginx master pid: %s\n' "$(pgrep -o nginx || echo none)"
}
before_planb

echo
echo "=============================================================="
echo " 1. stop the temporary foreground instance (frees 127.0.0.1:8080)"
echo "=============================================================="
pkill -u mamadoso -f "uvicorn app:app" 2>/dev/null && echo "  stopped" || echo "  nothing to stop"
sleep 1

echo
echo "=============================================================="
echo " 2. basic-auth credentials"
echo "=============================================================="
if [ -z "${DEMO_PASS:-}" ]; then
  DEMO_PASS="$(openssl rand -base64 12 | tr -d '/+=' | cut -c1-14)"
fi
umask 077
printf '%s:%s\n' "$DEMO_USER" "$(printf '%s' "$DEMO_PASS" | openssl passwd -apr1 -stdin)" > "$HTPASSWD"
chmod 640 "$HTPASSWD"; chown root:www-data "$HTPASSWD"
echo "  wrote $HTPASSWD"
echo
echo "  >>> LOGIN  user: $DEMO_USER"
echo "  >>> LOGIN  pass: $DEMO_PASS"
echo "  >>> save these now - they are not stored anywhere else"

echo
echo "=============================================================="
echo " 3. systemd unit (loopback only, memory + cpu capped)"
echo "=============================================================="
cat > "$UNIT" <<UNIT_EOF
[Unit]
Description=Ahadi shelf-count demo (watch shelf photo counting)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=mamadoso
Group=mamadoso
WorkingDirectory=$APP_DIR
Environment=AHADI_DATA=$DATA_DIR
EnvironmentFile=-$APP_DIR/.env
ExecStart=$VENV/bin/uvicorn app:app --host 127.0.0.1 --port 8080 --log-level warning
Restart=always
RestartSec=3

# Cannot starve PlanBManager on this 1.9 GiB box.
MemoryMax=700M
MemoryHigh=560M
CPUQuota=150%
TasksMax=128

NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=$DATA_DIR

[Install]
WantedBy=multi-user.target
UNIT_EOF
systemctl daemon-reload
systemctl enable --now "$SERVICE"
sleep 3
systemctl is-active "$SERVICE" | grep -q active && echo "  service active" || { echo "  service FAILED"; systemctl status "$SERVICE" --no-pager | tail -15; exit 1; }
curl -sf -m 10 http://127.0.0.1:8080/health/json >/dev/null && echo "  app answering on 127.0.0.1:8080"

echo
echo "=============================================================="
echo " 4. nginx vhost"
echo "=============================================================="
cat > "/etc/nginx/sites-available/${VHOST_NAME}" <<NGINX_EOF
# Ahadi shelf-count demo. Managed by install-demo.sh - edit there.
# Named zz-* on purpose: it must load AFTER planb so planb stays the
# default server for unknown Host headers.

server {
    listen 80;
    listen [::]:80;
    server_name ${DOMAIN};

    # phone photos are large; keep below the app's own limit
    client_max_body_size 30M;

    access_log /var/log/nginx/demo.access.log;
    error_log  /var/log/nginx/demo.error.log;

    auth_basic "Ahadi demo";
    auth_basic_user_file ${HTPASSWD};

    # let Let's Encrypt in without a password
    location /.well-known/acme-challenge/ {
        auth_basic off;
        root /var/www/html;
    }

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host              \$host;
        proxy_set_header X-Real-IP         \$remote_addr;
        proxy_set_header X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 5s;
        proxy_read_timeout    180s;
        proxy_send_timeout    180s;
    }
}
NGINX_EOF
ln -sfn "/etc/nginx/sites-available/${VHOST_NAME}" "/etc/nginx/sites-enabled/${VHOST_NAME}"
echo "  wrote + enabled ${VHOST_NAME}"

if ! nginx -t; then
  echo "  nginx config test FAILED - removing the new vhost, reloading nothing"
  rm -f "/etc/nginx/sites-enabled/${VHOST_NAME}" "/etc/nginx/sites-available/${VHOST_NAME}"
  exit 1
fi
systemctl reload nginx
echo "  nginx reloaded (not restarted)"

echo
echo "=============================================================="
echo " 5. verify over HTTP before touching TLS"
echo "=============================================================="
code=$(curl -s -o /dev/null -w '%{http_code}' -m 15 "http://${DOMAIN}/" || echo fail)
echo "  no auth  -> $code (expect 401)"
code=$(curl -s -o /dev/null -w '%{http_code}' -m 20 -u "${DEMO_USER}:${DEMO_PASS}" "http://${DOMAIN}/" || echo fail)
echo "  with auth-> $code (expect 200 or 301)"

echo
echo "=============================================================="
echo " 6. TLS via certbot (demo vhost only)"
echo "=============================================================="
if [ -z "$EMAIL" ]; then
  echo "  set EMAIL=you@example.com to receive expiry notices; continuing without"
  certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos \
          --register-unsafely-without-email --redirect || {
    echo "  certbot failed (DNS not propagated yet?); HTTP still works."; }
else
  certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$EMAIL" --redirect || {
    echo "  certbot failed (DNS not propagated yet?); HTTP still works."; }
fi

echo
echo "=============================================================="
echo " PlanB state AFTER (compare with BEFORE)"
echo "=============================================================="
before_planb
echo
echo "  demo service: $(systemctl is-active "$SERVICE")  restarts=$(systemctl show "$SERVICE" -p NRestarts --value)"
echo "  demo memory : $(systemctl show "$SERVICE" -p MemoryCurrent --value | awk '{printf "%.0f MB", $1/1048576}')"
echo
echo "=============================================================="
echo " DONE"
echo "=============================================================="
echo "  URL   : https://${DOMAIN}/   (http:// also works)"
echo "  login : ${DEMO_USER} / ${DEMO_PASS}"
echo "  capture: https://${DOMAIN}/capture"
echo "  health : https://${DOMAIN}/health"
echo
echo "  To enable real AI, edit $APP_DIR/.env and set:"
echo "    AHADI_PROVIDER=openai"
echo "    AHADI_AI_BASE_URL=...   AHADI_AI_API_KEY=***   AHADI_AI_MODEL=..."
echo "  then: sudo systemctl restart $SERVICE"
echo
echo "  Rollback:"
echo "    sudo rm -f /etc/nginx/sites-enabled/${VHOST_NAME} /etc/nginx/sites-available/${VHOST_NAME}"
echo "    sudo nginx -t && sudo systemctl reload nginx"
echo "    sudo systemctl disable --now $SERVICE"
echo "    sudo rm -f $UNIT && sudo systemctl daemon-reload"
