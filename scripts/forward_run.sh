#!/usr/bin/env bash
# One forward-test observation: refresh the store, then record a run.
#
# Runs unattended, so it is deliberately conservative. It does not retry, does
# not repair, and does not continue past a failed ingest -- a run scored on a
# store that half-updated is worse than a missing observation, because the
# missing one is visible in the session count and the bad one is not.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
LOG="$ROOT/data/ledger/forward.log"
mkdir -p "$(dirname "$LOG")"

# Rotate at 4 MB, keeping four generations. This file is appended on every run
# forever and was never rotated; the ledger beside it is the record worth
# keeping, and an unbounded log on a 1 GB instance eventually competes with it
# for the disk the observations need.
if [ -f "$LOG" ] && [ "$(wc -c <"$LOG")" -gt 4194304 ]; then
  rm -f "${LOG}.4"
  for n in 3 2 1; do
    [ -f "${LOG}.${n}" ] && mv "${LOG}.${n}" "${LOG}.$((n+1))"
  done
  mv "$LOG" "${LOG}.1"
fi

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$LOG"; }

# The only notification this system has ever had is a line in this file, which
# means silence and success look identical: a job that stops in week three is
# discovered in week eleven. PROSIGNAL_ALERT_CMD is whatever you want to be
# told with -- a curl to a healthcheck, a Telegram send, `mail`. It receives
# the message as its single argument and its own failure is ignored, because a
# broken notifier must not become a missing observation.
#
# Unset, this is inert and the behaviour is exactly what it was.
alert() {
  say "$*"
  if [ -n "${PROSIGNAL_ALERT_CMD:-}" ]; then
    "${PROSIGNAL_ALERT_CMD}" "prosignal: $*" >/dev/null 2>&1 || true
  fi
}

export ARROW_DEFAULT_MEMORY_POOL=system MALLOC_ARENA_MAX=2

say "--- observation start"

# The operator can decline an observation from the interface. cron still
# wakes -- editing /etc/cron.d needs root and the service does not have it --
# so the decline is a flag this script honours, and it is written down. A gap
# nobody can explain is indistinguishable from a gap that was hidden.
PAUSE="$ROOT/data/ledger/cron.paused"
if [ -f "$PAUSE" ]; then
  say "PAUSED by operator -- no observation recorded ($(cat "$PAUSE" 2>/dev/null | head -c 200))"
  exit 0
fi

if ! "$PY" -m prosignal.cli data ingest >>"$LOG" 2>&1; then
  alert "ingest FAILED -- no run recorded for today"
  exit 1
fi

# THE MANIFEST DESCRIBES THE STORE, AND THE INGEST JUST CHANGED IT.
#
# Nothing re-manifested. `data manifest --write` was reachable only by hand, so
# the first ingest after any manifest left the two disagreeing -- and stayed
# that way, because the next ingest changed the store again. The restart gate
# reads the manifest, so it failed every morning for a reason that was not the
# reason it exists to catch, and a failing gate nobody can act on is a gate
# everybody learns to ignore.
#
# Written here, between the ingest and the analysis, so the run is scored
# against a store the manifest actually describes. Not fatal: a stale manifest
# makes the reproducibility claim weaker, it does not make the observation
# wrong.
if ! "$PY" -m prosignal.cli data manifest --write >>"$LOG" 2>&1; then
  alert "manifest write FAILED -- the run will still be recorded, but no figure computed against the manifest describes this store"
fi

# --skip-if-recorded, because cron cannot see an NSE holiday. On one the ingest
# fetches nothing and exits 0, the staleness gate counts a single weekday and
# passes, and the analysis re-ranks the previous session -- writing a SECOND
# ledger row for a date that already has one. A person pressing SCAN wants that
# rerun; an unattended job does not.
if ! "$PY" -m prosignal.cli analyse run --watch 0 --skip-if-recorded >>"$LOG" 2>&1; then
  alert "analysis FAILED -- no run recorded for today"
  exit 1
fi
say "observation recorded"

# Progress, and the integrity checks that go with it. `research forward` exits
# non-zero when the window is broken -- a configuration changed after
# registration, coverage below the 60% the pre-registration requires, the file
# edited. That is worth being told about: every night it runs unnoticed on a
# void window is a night of observations that cannot be used as evidence.
if ! "$PY" -m prosignal.cli research forward >>"$LOG" 2>&1; then
  alert "FORWARD TEST INVALID -- see the lines above; the observations are still recorded but are not evidence until it is re-registered"
fi

# Resolve outcomes and warm the API's caches while nobody is waiting.
#
# Resolution runs on read, which is self-healing and correct, and it means
# the first person to open History after a new run pays for it. The machine
# is already awake at 20:35 and has nothing else to do, so it pays instead.
# Over the loopback, with the token this script already has in its
# environment -- it never leaves the box.
#
# Failures are ignored on purpose: the observation is recorded either way,
# and a cold cache is slow rather than wrong.
API="${PROSIGNAL_API:-http://127.0.0.1:8000}"

# IS THERE AN API AT ALL? On a host that runs the job but not the service --
# a laptop, a container that only ingests -- every curl below fails and the
# last one alerts "the screen may be empty" on a night when nothing is wrong.
# An alarm that fires on every healthy run trains the reader to ignore the one
# that matters.
if ! curl -fsS -m 10 -o /dev/null "${API}/health" 2>/dev/null; then
  say "no API at ${API} -- skipping cache warm and the screen check (the run IS recorded)"
  say "--- observation complete"
  exit 0
fi

warm() {
  local path="$1"
  if curl -fsS -m 600 -o /dev/null \
       ${PROSIGNAL_AUTH_TOKEN:+-H "x-api-key: ${PROSIGNAL_AUTH_TOKEN}"} \
       "${API}${path}"; then
    say "warmed ${path}"
  else
    say "could not warm ${path} (harmless: it will resolve on first open)"
  fi
}
# EVERY ENDPOINT THE SCREEN OPENS WITH, not just the slow one.
#
# The interface fetches /ready, /today and /performance on load, and the
# History tab adds /history/names and /history. Warming two of the five left
# the first visitor of the day paying for the other three -- and /today is the
# one that rebuilds the whole view model, so the screen that was meant to be
# ready before anyone looked at it still opened on a skeleton.
warm "/ready"
warm "/today"
warm "/performance"
warm "/history/names?limit=60"
warm "/history?limit=30"

# A last, cheap assertion that the screen the operator opens has something on
# it. This job never checked its own output: an analysis exits 0 having
# produced an empty slate, and the only way that was ever discovered was by
# opening the page the next morning and finding it blank.
TODAY_JSON=$(curl -fsS -m 300 \
  ${PROSIGNAL_AUTH_TOKEN:+-H "x-api-key: ${PROSIGNAL_AUTH_TOKEN}"} \
  "${API}/today" 2>/dev/null || true)
case "$TODAY_JSON" in
  "")            alert "could not read /today after the run -- the screen may be empty" ;;
  *'"picks":[]'*) alert "the run put NO names on the screen -- check the funnel before acting on an empty shortlist" ;;
  *'"picks"'*)   say "screen ready" ;;
  *)             alert "/today returned no view -- the screen has nothing to show" ;;
esac
