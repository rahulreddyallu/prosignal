#!/usr/bin/env bash
# Retire the tuned-away engine and open a clean one. Run ONCE, after a tuning
# pass, before the first live observation of the new configuration.
#
# WHY THIS IS A SCRIPT AND NOT A README. Five things have to happen in one
# order, and doing four of them leaves the engine in a state that looks fine and
# is not: a refitted model against a stale manifest, or a new epoch whose
# forward test still points at the old configuration. The order is the content.
#
#   1. ARCHIVE the fitted model, then remove the live cache. Archive first,
#      always -- the old coefficients are the evidence for what the previous
#      epoch did, and an audit that cannot see them cannot check it.
#   2. PURGE the HTTP cache, so nothing downstream is served a payload fetched
#      under the old configuration.
#   3. RE-MANIFEST the store, which is what makes the DATA gate describe the
#      store as it actually is.
#   4. OPEN a new epoch. A material change is a new out-of-sample question, and
#      this pass changed what orders the book, what closes a position, and how
#      often one opens.
#   5. RESTART the forward test against the new configuration. This is the step
#      that discharges finding R1, and it is deliberately last: restarting the
#      clock while the tree is still drifting would void the new window for
#      exactly the reason the old one is void.
#
# The next `analyse run` refits from scratch -- roughly 3,000 sessions -- which
# is slower than a cached run and is the point.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# The venv first, as `forward_run.sh` does. `python3` on the PATH is not
# necessarily the interpreter the engine is installed into, and this script
# opens an epoch -- a permanent record -- so it must not run against a
# half-installed environment that imports a different prosignal.
if [ -n "${PROSIGNAL_PY:-}" ]; then
  PY="$PROSIGNAL_PY"
elif [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="python3"
fi
STAMP="$(date -u +%Y%m%d_%H%M%S)"
CURATED="${PROSIGNAL_CURATED:-$ROOT/data/curated}"
LABEL="${1:-momentum-rank-v1}"
NOTE="${2:-Ranking moved from the fitted composite to mom_6_1_r; target and thesis-invalidation exits disarmed; stop widened to an 8 ATR disaster floor clipped at 35%; entries on a 21-session cadence; book size 6, exit band 18.}"

say() { echo "[$(date -u +%H:%M:%S)] $*"; }

# --- 1. archive, then purge, the fitted model ------------------------------
if [ -f "$CURATED/crosssec_model.json" ]; then
  mkdir -p "$CURATED/crosssec_model_versions"
  cp "$CURATED/crosssec_model.json" \
     "$CURATED/crosssec_model_versions/crosssec_model_pre_${LABEL}_${STAMP}.json"
  rm -f "$CURATED/crosssec_model.json"
  say "archived and removed the fitted model; the next run refits from scratch"
else
  say "no fitted model cached; nothing to archive"
fi

# --- 2. HTTP payload cache --------------------------------------------------
"$PY" -m prosignal.cli data purge-cache

# --- 3. the manifest must describe the store as it is NOW -------------------
"$PY" -m prosignal.cli data manifest --write
"$PY" -m prosignal.cli data manifest --verify

# --- 4. retire the old epoch, then open the new one -------------------------
# CLOSE FIRST. Two open epochs leave every ledger row ambiguous about which
# experiment it belongs to, and `--force` exists to let you do that on purpose,
# which this is not. SUPERSEDED rather than VOID: the previous epoch's
# observations are not invalid, they measured a different engine, and its
# results stay readable under their own identity.
OPEN_ID="$("$PY" - <<'PYEOF'
from prosignal.config.loader import load_config
from prosignal.validation import epoch
cfg = load_config()
cur = epoch.active(cfg.paths.ledger)
print(cur.epoch_id if cur else "")
PYEOF
)"
if [ -n "$OPEN_ID" ]; then
  say "closing epoch $OPEN_ID as SUPERSEDED"
  "$PY" -m prosignal.cli research epoch close "$OPEN_ID" --status SUPERSEDED \
    --reason "Superseded by $LABEL. $NOTE"
fi
"$PY" -m prosignal.cli research epoch open --label "$LABEL" --note "$NOTE"

# --- 5. and only then, a new forward window ---------------------------------
# `--restart` refuses unless every restart gate passes, which is why it is here
# rather than three steps earlier.
"$PY" -m prosignal.cli research forward --restart

say "done. The engine is on a new epoch with an open forward test."
say "The first observation will refit the model; expect it to take longer."
