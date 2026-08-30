#!/usr/bin/env bash
# Quarterly re-check of the shipped scorers.
#
# Runs the SAME holdout discipline that earned each deploy: the frozen
# configuration is applied to data it was not fitted on, once, and the result is
# recorded. Nothing here refits, retunes or changes a model -- if a re-check
# fails, the next configuration has to be selected on training data and tested
# on a window this one has not seen. Patching the failed configuration against
# the window that failed it is exactly what the discipline forbids.
#
# v3 is the shipped ranking (`ranking.source: v3_composite`) and is the one that
# decides. v2 is still re-checked because it is still in the tree and a
# divergence between them is itself information -- if v2 holds on a window where
# v3 fails, the two-level structure is what broke, not the factors.
#
# Install (macOS/Linux), first day of each quarter at 19:00 IST:
#   0 19 1 1,4,7,10 * cd /path/to/prosignal && ./scripts/quarterly_recheck.sh
#
# If you already run the daily pipeline on a server, this is the one thing that
# schedule does NOT do: the daily run scores today, it never asks whether the
# model still works. That question needs a window, not a day.
set -euo pipefail

cd "$(dirname "$0")/.."
STAMP="$(date +%Y-%m-%d)"
OUT="research/recheck"
mkdir -p "$OUT"

echo "== ingesting the quarter's data =="
python3 -m prosignal.cli data ingest || echo "ingest reported a problem; the "\
"re-check continues on the store as it stands and the coverage line below "\
"will show what it had"

echo
echo "== v3 -- the shipped composite: per-factor IC, per-theme IC, dominance =="
python3 -m prosignal.cli research v3 \
    --monitor --recheck --years 4 --holdout-months 12 --json \
    | tee "$OUT/recheck-v3-${STAMP}.txt"

echo
echo "== v2 -- the previous scorer, kept as a cross-check =="
python3 -m prosignal.cli research v2 \
    --years 4 --recheck --holdout-months 12 --json \
    | tee "$OUT/recheck-v2-${STAMP}.txt"

echo
echo "Written to $OUT/recheck-v3-${STAMP}.txt and $OUT/recheck-v2-${STAMP}.txt"
echo
echo "How to read it:"
echo "  TOO_EARLY  the window does not yet hold as much independent evidence as"
echo "             the deploy was judged on. The numbers printed are running"
echo "             totals, not a verdict. Read the theme health anyway."
echo "  HOLDS      the ranking still orders outcomes, outside its own null."
echo "  WEAK       positive but inside the null. Neither failure nor edge."
echo "  FAILS      the ranking did not order outcomes on this window."
echo
echo "A FAILS is not a signal to retune against this window -- it is a signal"
echo "that the next candidate needs a sealed window of its own. A THEME flagged"
echo "as inverted or dominating is worth acting on sooner than the headline:"
echo "it needs less evidence, and it names which fifth of the model moved."
