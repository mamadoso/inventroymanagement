# Deploying the demo on Ubuntu

Target: an Ubuntu 22.04/24.04 box on the shop LAN that the manager's and staff's phones
can reach. No GPU, no Docker, no Postgres.

## 1. System packages

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip openssl
```

## 2. Code

```bash
sudo mkdir -p /opt/ahadi && sudo chown "$USER" /opt/ahadi
cp -r demo /opt/ahadi/demo
cd /opt/ahadi/demo
chmod +x setup.sh run.sh
./setup.sh
```

## 3. Configure the AI

```bash
nano .env
```
```ini
AHADI_AI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
AHADI_AI_API_KEY=AIza...
AHADI_AI_MODEL=gemini-2.5-flash
AHADI_PROVIDER=openai
```
Full recipe list: [AI-PROVIDERS.md](AI-PROVIDERS.md).

## 4. Run the self-test, then the app

```bash
./setup.sh 2>/dev/null || true          # already done above
.venv/bin/python selftest.py            # offline: no key, no network needed
./run.sh
```
`selftest.py` exercises detection, the diff engine, annotation and the HTTP routes and
prints `PASS`. Do not skip it — it catches a broken install in five seconds.

Note the printed LAN URL, e.g. `http://192.168.1.50:8080`.

## 5. Open the firewall

```bash
sudo ufw allow from 192.168.1.0/24 to any port 8080 proto tcp
```
Restrict to the shop subnet — do not expose this to the internet.

## 6. Run as a service

```bash
sudo useradd -r -m -s /usr/sbin/nologin ahadi
sudo mkdir -p /var/lib/ahadi-demo && sudo chown ahadi:ahadi /var/lib/ahadi-demo
sudo cp systemd/ahadi-demo.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ahadi-demo
sudo systemctl status ahadi-demo --no-pager
journalctl -u ahadi-demo -f
```

## 7. Phones

Same Wi-Fi as the server. On a laptop, open `http://<server-ip>:8080/qr/1`, print it,
tape it next to the shelf. Staff scan → capture screen opens → take photo → result.

iPhone works over plain HTTP because the capture button hands off to the **native camera
app** (`<input capture>`), which needs no secure context. The in-page live viewfinder does
require HTTPS — either use the QR/native path, or:

```bash
AHADI_SSL=1 ./run.sh     # self-signed cert, accept the warning once
```
For a permanent install, put Caddy or nginx in front with a real certificate.

## 8. Verify

| Check | URL |
|---|---|
| app alive | `http://<ip>:8080/` |
| AI configured | `http://<ip>:8080/health` → green ✅ |
| machine-readable | `curl -s http://<ip>:8080/health/json` |
| API | `curl -s http://<ip>:8080/api/shelves` |

## Data location & reset

`AHADI_DATA` (default `demo/data/`) holds `demo.sqlite3`, all photos and `ai_calls.jsonl`.
Deleting the folder is a full reset; the UI also has a **Clear all data** button.
Back up by copying that one directory.

## Upgrading

```bash
cd /opt/ahadi/demo && git pull && ./setup.sh && sudo systemctl restart ahadi-demo
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Phone can't load the page | Different Wi-Fi/VLAN, or `ufw` rule missing. Test `curl http://<ip>:8080/health` from the phone's browser. |
| `/health` red, 401 | Wrong or missing `AHADI_AI_API_KEY`. |
| `/health` red, model not in list | Model id is wrong for that key. The health page lists vision-capable ids it can see. |
| 400 on every detection | Gateway rejects structured output → `AHADI_AI_JSON_SCHEMA=0`. |
| Detection returns 0 items | Image too small or shelf too far. Raise `AHADI_MAX_SIDE` to 2048 and move closer. |
| Everything reads as "new"/"missing" | Camera moved between shots. Use the same standing position; see `../../MVP-PLAN.md` §3 on the phone jig. |
| iOS camera button does nothing | Page opened over `file://` or a captive portal. Use the LAN IP URL. |
| Slow first request | Cold venv; warm it with one capture before the demo. |
| `sqlite3.OperationalError: locked` | Two processes on the same DB. Only one service instance. |

Logs worth reading: `journalctl -u ahadi-demo -f` and `data/ai_calls.jsonl`.
