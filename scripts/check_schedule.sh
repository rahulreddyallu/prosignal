#!/usr/bin/env bash
# Is the daily observation actually running? Run this ON the host that runs it.
#
# Every line is a fact with a threshold, because "the script exists and the log
# was written today" is not evidence that a schedule is firing -- a log can be
# four hand-runs in a two-day window and look exactly the same from a distance.
# Exits non-zero if anything below fails, so it can be a cron line of its own.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
LOG="${ROOT}/data/ledger/forward.log"
FAIL=0

ok()   { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; FAIL=1; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$*"; }

echo "prosignal daily loop -- $(hostname) -- $(date '+%Y-%m-%d %H:%M %Z')"
echo

# -- 1. is anything scheduled -------------------------------------------
echo "SCHEDULE"
if [ -f /etc/cron.d/prosignal ]; then
  ok "/etc/cron.d/prosignal: $(grep -v '^\(#\|SHELL\|PATH\)' /etc/cron.d/prosignal | head -1 | cut -c1-60)"
elif crontab -l 2>/dev/null | grep -q forward_run; then
  ok "user crontab: $(crontab -l | grep forward_run | head -1 | cut -c1-60)"
elif launchctl list 2>/dev/null | grep -q com.prosignal.daily; then
  ok "launchd agent com.prosignal.daily"
else
  bad "NOTHING schedules forward_run.sh on this host"
fi

# The scheduler's clock is not necessarily the exchange's. NSE publishes the
# bhavcopy in the evening IST; a job on a UTC box fires before the data exists.
TZNOW="$(date '+%Z %z')"
case "$TZNOW" in
  *IST*|*+0530*) ok "host clock is IST ($TZNOW)" ;;
  *) warn "host clock is $TZNOW, not IST -- check the cron hour is after the 18:00-19:00 IST bhavcopy" ;;
esac

# -- 2. is it firing ----------------------------------------------------
echo
echo "OBSERVATIONS"
if [ ! -f "$LOG" ]; then
  bad "no $LOG -- the job has never run here"
else
  STARTS=$(grep -c 'observation start' "$LOG" 2>/dev/null || echo 0)
  LASTRUN=$(grep 'observation start' "$LOG" | tail -1 | cut -c2-20)
  ok "$STARTS observations logged; last at ${LASTRUN:-never} UTC"
  # A schedule fires at one time of day. Four runs at four different hours are
  # somebody typing, not a cron.
  HOURS=$(grep 'observation start' "$LOG" | tail -10 | cut -c13-14 | sort -u | tr '\n' ' ')
  case "$(echo "$HOURS" | wc -w)" in
    0) ;;
    1) ok "last runs all fired at ${HOURS}h UTC -- consistent with a schedule" ;;
    *) warn "last runs fired at hours: ${HOURS}UTC -- a schedule fires at one" ;;
  esac
  AGE_H=$(( ( $(date +%s) - $(stat -c %Y "$LOG" 2>/dev/null || stat -f %m "$LOG") ) / 3600 ))
  if [ "$AGE_H" -gt 96 ]; then bad "log untouched for ${AGE_H}h -- the loop has stopped"
  elif [ "$AGE_H" -gt 30 ]; then warn "log untouched for ${AGE_H}h (a weekend or a holiday is fine)"
  else ok "log written ${AGE_H}h ago"; fi
fi

# -- 3. did it leave the store usable -----------------------------------
echo
echo "STORE"
if [ -x "$PY" ]; then
  "$PY" -m prosignal.cli research ready 2>/dev/null \
    | grep -Ei 'session|stale|fundamental|delivery' | head -6 | sed 's/^/  /'
  if "$PY" -m prosignal.cli data manifest --verify 2>/dev/null | grep -q VERIFIED; then
    ok "manifest describes the store"
  else
    bad "manifest DRIFTED -- no figure computed against it describes this data"
  fi
else
  bad "no interpreter at $PY"
fi

# -- 4. can a failure reach a person ------------------------------------
echo
echo "ALERTING"
if [ -n "${PROSIGNAL_ALERT_CMD:-}" ]; then
  ok "PROSIGNAL_ALERT_CMD is set"
elif grep -q '^PROSIGNAL_ALERT_CMD=.' /etc/prosignal.env 2>/dev/null; then
  ok "PROSIGNAL_ALERT_CMD is set in /etc/prosignal.env"
else
  bad "no PROSIGNAL_ALERT_CMD -- failures reach a log file and nothing else"
fi
warn "a failure alarm cannot report that the job never STARTED; pair it with a dead-man's switch"

echo
[ "$FAIL" -eq 0 ] && echo "all checks passed" || echo "SOMETHING IS WRONG -- see FAIL above"
exit "$FAIL"
