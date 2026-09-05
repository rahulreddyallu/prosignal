#!/bin/bash
# AWS "User data": pasted into the launch wizard and run once, as root, on
# first boot. Everything the deployment needs happens here so that no SSH,
# no key file and no terminal are required to stand the instance up.
#
# Two placeholders must be edited before pasting:
#   __DOMAIN__  the hostname you pointed at this machine
#   __TOKEN__   the access token, 24+ characters
#   __ALERT__   where a failure should reach you. See PROSIGNAL_ALERT_CMD
#               below -- leave it as the default and failures stay in a log
#               file on a box nobody logs into.
#
# Progress is written to /var/log/prosignal-setup.log. If the site does not
# come up, that file says why.
set -euxo pipefail
exec > >(tee -a /var/log/prosignal-setup.log) 2>&1

DOMAIN="__DOMAIN__"
TOKEN="__TOKEN__"
# A command that receives the message as its single argument. A healthcheck.io
# ping URL, a Telegram send, `mail -s`. Left empty, forward_run.sh's alerting is
# inert -- which is what it was, because this file wrote the token and the
# allocator variables into /etc/prosignal.env and never this one. Every failure
# the nightly is careful to detect went to a log file and stopped there.
ALERT_CMD="__ALERT__"
APP_USER="ubuntu"
APP_DIR="/home/${APP_USER}/prosignal"

echo "=== prosignal setup $(date -u) ==="

# --- 1. swap first -------------------------------------------------------
# 1 GB with no swap means the kernel kills the analysis instead of slowing it
# down, and a killed analysis is a missing observation in an 18-month test.
if [ ! -f /swapfile ]; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
  echo 'vm.swappiness=10' >> /etc/sysctl.conf
  sysctl -p || true
fi

# --- 2. packages ---------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3-venv python3-pip git curl debian-keyring \
                   debian-archive-keyring apt-transport-https

timedatectl set-timezone Asia/Kolkata

# --- 3. the engine -------------------------------------------------------
sudo -u "${APP_USER}" git clone https://github.com/rahulreddyallu/prosignal.git "${APP_DIR}"
cd "${APP_DIR}"
sudo -u "${APP_USER}" python3 -m venv .venv
sudo -u "${APP_USER}" ./.venv/bin/pip install --upgrade pip
sudo -u "${APP_USER}" ./.venv/bin/pip install -r requirements.txt
# src-layout: the package is importable only after installing the project
# itself, not just its dependencies.
sudo -u "${APP_USER}" ./.venv/bin/pip install --no-deps -e .

# --- 4. environment ------------------------------------------------------
# Without the three allocator settings the analysis peaks at 542 MB instead of
# 409 MB, which on a 1 GB instance is the difference between running and being
# killed. Arrow and glibc both retain freed pages in their own arenas.
# PROSIGNAL_PUBLIC sits beside the token deliberately. The process binds
# 127.0.0.1 and Caddy carries the internet to it, so the app's own fail-closed
# check -- which reads the bind address -- cannot see that it is exposed.
# With this marker, deleting the token line stops the service instead of
# opening /admin/reset/everything to the world.
cat > /etc/prosignal.env <<EOF
PROSIGNAL_AUTH_TOKEN=${TOKEN}
PROSIGNAL_PUBLIC=1
ARROW_DEFAULT_MEMORY_POOL=system
MALLOC_ARENA_MAX=2
PYTHONMALLOC=malloc
PROSIGNAL_ALERT_CMD=${ALERT_CMD}
EOF

# A FAILURE ALARM CANNOT REPORT THAT THE JOB NEVER STARTED.
# It only fires from inside a run. The job stopping -- a dead instance, an
# unloaded cron, a full disk -- is silent by construction, and silence and
# success look identical. If ALERT_CMD is a healthcheck.io-style ping URL,
# point its schedule at 20:30 IST daily and it alarms on ABSENCE, which is the
# failure this deployment has actually had.
if [ -z "${ALERT_CMD}" ] || [ "${ALERT_CMD}" = "__ALERT__" ]; then
  echo "WARNING: PROSIGNAL_ALERT_CMD is not set. Failures will be written to"
  echo "         ${APP_DIR}/data/ledger/forward.log and nowhere else."
fi
chmod 600 /etc/prosignal.env

# --- 5. the API as a service --------------------------------------------
cat > /etc/systemd/system/prosignal.service <<EOF
[Unit]
Description=prosignal API
After=network-online.target

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${APP_DIR}
EnvironmentFile=/etc/prosignal.env
ExecStart=${APP_DIR}/.venv/bin/uvicorn prosignal.api:create_app --factory \
  --host 127.0.0.1 --port 8000 --workers 1
Restart=always
RestartSec=10
MemoryMax=850M

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now prosignal

# --- 6. HTTPS ------------------------------------------------------------
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
  | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
  > /etc/apt/sources.list.d/caddy-stable.list
apt-get update -y
apt-get install -y caddy

cat > /etc/caddy/Caddyfile <<EOF
${DOMAIN} {
    encode gzip
    reverse_proxy 127.0.0.1:8000
    header {
        Strict-Transport-Security "max-age=31536000"
        X-Content-Type-Options    "nosniff"
        X-Frame-Options           "DENY"
        Referrer-Policy           "no-referrer"
    }
}
EOF
systemctl reload caddy || systemctl restart caddy

# --- 7. the daily observation -------------------------------------------
# 20:30 IST: the close is 15:30, the bhavcopy lands around 18:00-19:00 and
# delivery later, so this leaves five hours of margin.
#
# STILL DAILY, AND THAT IS DELIBERATE. The book now OPENS positions only every
# 21st session, but the job that runs must keep running every session: the
# disaster floor is a price level checked against every bar, the rank band and
# the eligibility screen can release a position on any day, and outcomes resolve
# from prices that arrive daily. Moving the cron itself to a 21-session rhythm
# would mean a stop breached on day 3 is noticed on day 21.
#
# The entry schedule lives in the config, not here --
# `stage6_entry.admission.entry_cadence_sessions` -- counted in SESSIONS from a
# fixed anchor against the exchange calendar, so it is reproducible and immune
# to holidays. cron cannot count sessions; it does not know an NSE holiday from
# a Tuesday. Putting the schedule in the only place that can see the calendar is
# what keeps the live entry dates and the backtested ones the same dates.
cat > /etc/cron.d/prosignal <<EOF
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
30 20 * * 1-5 ${APP_USER} set -a; . /etc/prosignal.env; set +a; ${APP_DIR}/scripts/forward_run.sh
EOF
chmod 644 /etc/cron.d/prosignal

# --- 8. open the forward test -------------------------------------------
# Registered here rather than by hand, because this deployment is meant to run
# without a terminal. The clock starts on the deploy date and the criteria are
# hashed at the same moment, which is the whole point of a pre-registration.
sudo -u "${APP_USER}" bash -c "cd ${APP_DIR} && set -a && . /etc/prosignal.env && set +a && \
  ./.venv/bin/python -m prosignal.cli research forward --start" || \
  echo "forward test already registered, leaving it alone"

echo "=== setup complete $(date -u) ==="
echo "site: https://${DOMAIN}"
