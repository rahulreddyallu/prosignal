"""K-1 status: what the shipped ranking is doing on recent data, honestly gated.

The definitive K-1 experiment -- the traded book on a window that never touched
the 378-cell selection surface -- can only be answered by data that post-dates
that surface, i.e. the live forward test as it accrues past sealed window A. The
repo already builds exactly this discipline into `validation.panel.recheck`,
which:

  * applies the FROZEN scorer (no fitting) to the most recent dates,
  * refuses a verdict until it holds as much independent evidence as the deploy
    did (MIN_INDEPENDENT_WINDOWS), returning TOO_EARLY otherwise,
  * and NAMES the overlap when its window still sits inside a sealed holdout,
    because a re-read of the deploy's own dates is not independent evidence.

This runner surfaces that status so K-1's progress is visible now, rather than
re-implementing a book simulator that would only re-measure holdout-overlapping
data. It reuses the cached panel from audit_2026_09.py if present.

Usage:
    python research/v3/experiments/recheck_status.py --holdout-months 18
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prosignal.config.loader import load_config           # noqa: E402
from prosignal.data.store import DataStore                 # noqa: E402
from prosignal.validation.panel import build_panel, recheck  # noqa: E402

OUT = Path(__file__).resolve().parent
PANEL_CACHE = OUT / "panel_2026_09.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-months", type=int, default=18,
                    help="how many recent months to grade")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if PANEL_CACHE.exists() and not args.rebuild:
        panel = pd.read_parquet(PANEL_CACHE)
        print(f"[panel] reusing {PANEL_CACHE.name}", flush=True)
    else:
        cfg = load_config()
        store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
        print("[panel] building recent window from the shipped scorer...", flush=True)
        start = (pd.Timestamp.today() - pd.DateOffset(months=args.holdout_months + 30)).date()
        panel = build_panel(store, start=start)
        if panel.empty:
            print("PANEL EMPTY", flush=True)
            sys.exit(2)

    rc = recheck(panel, holdout_months=args.holdout_months)
    d = rc.to_dict()
    (OUT / "recheck_status.json").write_text(json.dumps(d, indent=1, default=str))

    print("\n" + "=" * 68)
    print(f"K-1 RANKING STATUS  (last {args.holdout_months} months)")
    print("=" * 68)
    print(f"  verdict        : {rc.verdict}")
    print(f"  window         : {rc.holdout_start} .. {rc.as_of}  ({rc.holdout_dates} dates)")
    print(f"  rank IC        : {rc.ic:+.4f}  (t {rc.ic_t:+.2f})")
    print(f"  quintile spread: {rc.spread:+.4f}  (t {rc.spread_t:+.2f})")
    print(f"  null p (spread): {rc.null_p_spread}")
    if rc.theme_health:
        print("  theme health:")
        for h in rc.theme_health:
            flags = []
            if h.get("inverted"):
                flags.append("INVERTED")
            if h.get("dominating"):
                flags.append("DOMINATING")
            print(f"    {h.get('name'):10s} {' '.join(flags) or 'ok'}")
    print("\n  NOTE:")
    for line in _wrap(rc.note):
        print(f"    {line}")
    print("=" * 68)


def _wrap(s: str, width: int = 74):
    words, line, out = s.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line); line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    main()
