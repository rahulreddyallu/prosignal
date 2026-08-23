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

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$LOG"; }

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
  say "ingest FAILED -- no run recorded for today"
  exit 1
fi
if ! "$PY" -m prosignal.cli analyse run --watch 0 >>"$LOG" 2>&1; then
  say "analysis FAILED -- no run recorded for today"
  exit 1
fi
say "observation recorded"
"$PY" -m prosignal.cli research forward 2>&1 | tail -4 >> "$LOG"
