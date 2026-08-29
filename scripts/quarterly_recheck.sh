#!/usr/bin/env bash
# Quarterly re-check of the shipped v2 scorer.
#
# Runs the SAME holdout discipline that earned the deploy: the frozen
# configuration is applied to data it has not been fitted on, once, and the
# result is recorded. Nothing here refits, retunes or changes the model -- if
# the re-check fails, the next configuration has to be selected on training
# data and tested on a window this one has not seen.
#
# Install (macOS/Linux), first day of each quarter at 19:00 IST:
#   0 19 1 1,4,7,10 * cd /path/to/prosignal && ./scripts/quarterly_recheck.sh
set -euo pipefail

cd "$(dirname "$0")/.."
STAMP="$(date +%Y-%m-%d)"
OUT="research/recheck"
mkdir -p "$OUT"

echo "== ingesting the quarter's data =="
python3 -m prosignal.cli data ingest || echo "ingest reported a problem; the "\
"re-check continues on the store as it stands and the coverage line below "\
"will show what it had"

echo "== re-check =="
python3 -m prosignal.cli research v2 \
    --years 4 --recheck --holdout-months 12 --json \
    | tee "$OUT/recheck-${STAMP}.txt"

echo
echo "Written to $OUT/recheck-${STAMP}.txt"
echo "A verdict of FAILS is not a signal to retune against this window --"
echo "it is a signal that the next candidate needs its own sealed holdout."
