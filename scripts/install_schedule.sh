#!/usr/bin/env bash
# Install (or remove) the daily observation as a launchd agent.
#
# WHY launchd AND NOT cron. This repo describes its schedule twice -- a
# `/etc/cron.d` line in cloud-init.sh for the Linux deployment, and nothing at
# all for macOS -- and on the machine that actually runs the engine there was no
# scheduler of any kind. On macOS cron is the wrong tool anyway: it needs Full
# Disk Access granted to /usr/sbin/cron by hand, and it does not run a job whose
# window passed while the lid was shut. launchd does both -- a
# StartCalendarInterval that comes due while the machine is asleep fires on
# wake -- which is what a laptop needs to not silently skip observations.
#
# The times are LOCAL. The system zone here is Asia/Kolkata, so 20:30 is 20:30
# IST: five hours after the 15:30 close, which clears the bhavcopy (18:00-19:00)
# and delivery. Weekdays only; NSE holidays are handled inside the run by
# `analyse run --skip-if-recorded`, because cron and launchd both think a
# holiday is a Tuesday.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.prosignal.daily"
PLIST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
HOUR="${PROSIGNAL_HOUR:-20}"
MINUTE="${PROSIGNAL_MINUTE:-30}"

if [ "${1:-}" = "--uninstall" ]; then
  launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
  rm -f "$PLIST"
  echo "removed ${LABEL}"
  exit 0
fi

if [ ! -x "${ROOT}/.venv/bin/python" ]; then
  echo "no interpreter at ${ROOT}/.venv/bin/python -- create the venv first" >&2
  exit 1
fi

mkdir -p "${HOME}/Library/LaunchAgents" "${ROOT}/logs"

# One <dict> per weekday. launchd has no "Mon-Fri" shorthand: a
# StartCalendarInterval dict without a Weekday key means EVERY day, so the
# weekday has to be enumerated or the job runs on Saturday and Sunday too --
# harmless (the ingest finds nothing and --skip-if-recorded declines) but it
# fills the log with runs that were never going to record anything.
entries=""
for wd in 1 2 3 4 5; do
  entries="${entries}
      <dict>
        <key>Weekday</key><integer>${wd}</integer>
        <key>Hour</key><integer>${HOUR}</integer>
        <key>Minute</key><integer>${MINUTE}</integer>
      </dict>"
done

cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>${LABEL}</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${ROOT}/scripts/forward_run.sh</string>
  </array>

  <key>WorkingDirectory</key><string>${ROOT}</string>

  <key>StartCalendarInterval</key>
  <array>${entries}
  </array>

  <!-- The run must NOT start at load. Installing the agent is not an
       observation, and a run triggered by an install would be dated by
       whenever somebody happened to run this script. -->
  <key>RunAtLoad</key><false/>

  <!-- The allocator settings the deployment uses. Without them the analysis
       peaks around 542 MB instead of 409 MB; on this host that is only
       pressure, on a 1 GB instance it is the difference between running and
       being killed. Kept identical to cloud-init.sh so the two hosts behave
       the same way. -->
  <key>EnvironmentVariables</key>
  <dict>
    <key>ARROW_DEFAULT_MEMORY_POOL</key><string>system</string>
    <key>MALLOC_ARENA_MAX</key><string>2</string>
    <key>PYTHONMALLOC</key><string>malloc</string>
    <key>PATH</key><string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>

  <!-- forward_run.sh keeps its own rotated log. These two catch what escapes
       it: an interpreter that will not start, a syntax error, an OOM kill. -->
  <key>StandardOutPath</key><string>${ROOT}/logs/schedule.out</string>
  <key>StandardErrorPath</key><string>${ROOT}/logs/schedule.err</string>
</dict>
</plist>
PLIST_EOF

plutil -lint "$PLIST" >/dev/null

launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"

echo "installed ${LABEL}"
echo "  runs   : ${HOUR}:$(printf '%02d' "${MINUTE}") local (Mon-Fri), $(date '+%Z')"
echo "  script : ${ROOT}/scripts/forward_run.sh"
echo "  log    : ${ROOT}/data/ledger/forward.log"
echo "  remove : ${ROOT}/scripts/install_schedule.sh --uninstall"
