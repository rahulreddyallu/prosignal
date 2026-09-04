"""K-5: bound the residual survivorship inflation of the OOS IC.

The universe is survivorship-FREE by construction for the collection period:
`build_v3_panel` admits any name liquid on each date, including ones that later
stop printing (the config's liquidity_pit choice, not index_snapshot). So the
question is not "are dead names excluded" -- they are present until they die --
but "when a name dies WITHIN the label horizon, its 21-session forward return
becomes NaN and drops from the IC; how much does that silent drop inflate the
IC?"

This measures:
  1. the disappearance rate (names whose last print precedes the store end),
     to check it matches the ~4.3%/yr the README claims;
  2. how many panel rows have a name that stops printing within the 21-session
     label horizon (these are the rows dropped from IC);
  3. a STRESSED IC in which those rows are assigned a delisting return, to bound
     how much the silent drop inflates the reported composite IC.

What it cannot do: names that delisted BEFORE data collection are absent from the
store entirely and cannot be recovered from it. That residual is unquantifiable
here and is stated, not estimated.

Usage:
    python research/v3/experiments/survivorship_bound.py --stress -0.30
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prosignal.config.loader import load_config      # noqa: E402
from prosignal.data.store import DataStore            # noqa: E402
from prosignal.data.types import DATE, SYMBOL          # noqa: E402

OUT = Path(__file__).resolve().parent
PANEL_CACHE = OUT / "panel_2026_09.parquet"
PRICE_CACHE = OUT / "book_price_panels.pkl"
HORIZON = 21


def _close_panel(cfg, store):
    if PRICE_CACHE.exists():
        with open(PRICE_CACHE, "rb") as fh:
            return pickle.load(fh)["close"]
    px = store.read_prices(columns=[DATE, SYMBOL, "close"])
    px[DATE] = pd.to_datetime(px[DATE]).dt.normalize()
    return px.pivot_table(index=DATE, columns=SYMBOL, values="close",
                          aggfunc="last", observed=True).sort_index()


def _ic(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 60:
        return np.nan
    ra, rb = pd.Series(a[m]).rank().to_numpy(), pd.Series(b[m]).rank().to_numpy()
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _agg(ics):
    x = np.asarray([v for v in ics if np.isfinite(v)])
    if len(x) < 5:
        return float("nan"), float("nan"), len(x)
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stress", type=float, default=-0.30,
                    help="forward return assigned to a name that delists within "
                         "the horizon (bound; -0.30 = down 30%)")
    args = ap.parse_args()

    cfg = load_config()
    store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
    vpanel = pd.read_parquet(PANEL_CACHE)
    vpanel["date"] = pd.to_datetime(vpanel["date"])
    close = _close_panel(cfg, store)

    # last print per symbol
    last_print = {}
    for s in close.columns:
        col = close[s].dropna()
        if len(col):
            last_print[s] = col.index[-1]
    store_end = close.index[-1]
    sess_index = {d: i for i, d in enumerate(close.index)}

    # (1) disappearance rate
    names = list(last_print)
    disappeared = [s for s in names if last_print[s] < store_end - pd.Timedelta(days=30)]
    span_years = (close.index[-1] - close.index[0]).days / 365.25
    rate = len(disappeared) / max(len(names), 1) / max(span_years, 1e-9)
    print("=" * 68)
    print("K-5 SURVIVORSHIP BOUND")
    print("=" * 68)
    print(f"  names in store            {len(names)}")
    print(f"  stopped printing early    {len(disappeared)} "
          f"({len(disappeared)/max(len(names),1):.1%})")
    print(f"  disappearance rate/yr     {rate:.1%}   (README claims ~4.3%/yr)")

    # (2) panel rows whose name dies within the horizon
    def dies_within(sym, dt):
        lp = last_print.get(sym)
        if lp is None:
            return False
        i = sess_index.get(pd.Timestamp(dt))
        if i is None:
            return False
        return sess_index.get(lp, 10**9) < i + 1 + HORIZON

    vp = vpanel[["date", "symbol", "score", "y21"]].copy()
    vp["dies"] = [dies_within(s, d) for s, d in zip(vp["symbol"], vp["date"])]
    n_rows = len(vp)
    n_die = int(vp["dies"].sum())
    n_die_nan = int((vp["dies"] & vp["y21"].isna()).sum())
    print(f"  panel rows                {n_rows:,}")
    print(f"  rows whose name dies <=21 {n_die:,} ({n_die/max(n_rows,1):.2%})"
          f"; of which y21 is NaN: {n_die_nan:,}")

    # (3) reported IC vs stressed IC
    rep_ics, str_ics = [], []
    for d, g in vp.groupby("date", sort=True):
        sc = g["score"].to_numpy("float64")
        y = g["y21"].to_numpy("float64")
        rep_ics.append(_ic(sc, y))
        y2 = y.copy()
        stress_mask = g["dies"].to_numpy() & ~np.isfinite(y)
        y2[stress_mask] = args.stress
        str_ics.append(_ic(sc, y2))
    rep = _agg(rep_ics)
    stt = _agg(str_ics)
    print(f"\n  composite IC reported     {rep[0]:+.4f} (t {rep[1]:+.2f})")
    print(f"  composite IC stressed     {stt[0]:+.4f} (t {stt[1]:+.2f})"
          f"   [delisting rows = {args.stress:+.0%}]")
    infl = rep[0] - stt[0]
    print(f"  survivorship inflation    {infl:+.4f} "
          f"({infl/rep[0]*100:+.1f}% of reported)" if np.isfinite(rep[0]) and rep[0]
          else "")
    print("\n  UNQUANTIFIABLE RESIDUAL: names that delisted BEFORE data "
          "collection are absent from the store and cannot be bounded from it.")
    print("=" * 68)

    res = {
        "names": len(names), "disappeared": len(disappeared),
        "disappearance_rate_per_year": rate,
        "panel_rows": n_rows, "rows_die_within_horizon": n_die,
        "rows_die_nan_y21": n_die_nan,
        "ic_reported": rep[0], "ic_reported_t": rep[1],
        "ic_stressed": stt[0], "ic_stressed_t": stt[1],
        "stress_return": args.stress,
        "survivorship_inflation_ic": infl,
    }
    (OUT / "survivorship_bound.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"[results] -> {OUT/'survivorship_bound.json'}")


if __name__ == "__main__":
    main()
