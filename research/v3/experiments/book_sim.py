"""K-1/K-2/K-6 (full book): net-of-cost P&L of the v3 book, the repo's own way.

Runs the SHIPPED scorer's rankings through the repository's OWN book simulator
(`validation.portfolio_sim.simulate` / `phase_summary`) with the shipped cost
model, so the number is produced by the same machinery `research portfolio` uses
-- not a bespoke reimplementation. It answers three of the open audit questions
together:

  K-2  does the 6-name book beat the benchmark net of cost, and does it beat the
       best single theme's book?
  K-6  how does that net edge move as the impact coefficient is stressed?
  K-1  how do window A (2025-03..2026-08) and window B (2021-07..2022-12) differ
       -- the 7x gross-edge disagreement the cost analysis surfaced?

WHAT IS FAITHFUL AND WHAT IS NOT.
  * Rankings ARE the shipped v3 composite score, with the shipped absolute floor
    applied to ENTRIES (close > 200-DMA AND >= 3 themes positive) -- names that
    fail are pushed below the entry band, held names ride the exit band.
  * Sizing, stop, target, invalidation, cost and benchmark ARE the shipped
    stage-6/7 settings via `cli._portfolio_params`.
  * The book model is `simulate`'s COHORT model (enter, hold the horizon, exit on
    stop/target/invalidation/rank) -- the same one `research portfolio` uses. It
    is NOT a bit-reproduction of the old `work/v3` holdout book (a different,
    continuous-weekly model on a dead cloud path), so numbers here will not match
    HOLDOUT_V3_A/B exactly. Read the ranking of configs, not the third decimal.
  * Windows A and B OVERLAP the 378-cell book-selection surface, so a positive
    book here is not clean OOS evidence -- it is the in-sample book, priced. Only
    forward data clears that (see recheck_status.py).

Usage:
    python research/v3/experiments/book_sim.py            # full sweep
    python research/v3/experiments/book_sim.py --rebuild  # rebuild price panels
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prosignal.config.loader import load_config          # noqa: E402
from prosignal.data.store import DataStore               # noqa: E402
from prosignal.costs import CostModel                     # noqa: E402
from prosignal.cli import _portfolio_inputs, _portfolio_params  # noqa: E402
from prosignal.validation.portfolio_sim import phase_summary    # noqa: E402

OUT = Path(__file__).resolve().parent
PANEL_CACHE = OUT / "panel_2026_09.parquet"       # the v3 score panel (from audit_2026_09)
PRICE_CACHE = OUT / "book_price_panels.pkl"        # OHLC/ATR/MA/ADTV/benchmark

WINDOWS = {
    "A_2025_26": ("2025-03-06", "2026-08-17"),
    "B_2021_22": ("2021-07-01", "2022-12-27"),
    "full":      (None, None),
}
COEFFS = [0.10, 0.25, 0.50]
# (label, max_positions, entry_rank, exit_rank)
BOOKS = [
    ("live_6", 6, 6, 18),        # the shipped book
    ("holdout_10", 10, 20, 30),  # the sealed-holdout-style book, for contrast
]
THEME_SUB = {"momentum": "momentum_sub", "quality": "quality_sub",
             "ownership": "ownership_sub", "risk": "risk_sub",
             "reversal": "reversal_sub"}


def _price_panels(cfg, store, rebuild: bool):
    if PRICE_CACHE.exists() and not rebuild:
        with open(PRICE_CACHE, "rb") as fh:
            return pickle.load(fh)
    sessions = store.price_sessions()
    end = sessions[-1]
    print("[panels] building OHLC/ATR/MA/ADTV/benchmark over the full store "
          "(slow, one-off)...", flush=True)
    panels = _portfolio_inputs(cfg, store, sessions, None, end)
    with open(PRICE_CACHE, "wb") as fh:
        pickle.dump(panels, fh)
    print(f"[panels] cached -> {PRICE_CACHE.name}", flush=True)
    return panels


def _rankings(vpanel: pd.DataFrame, close: pd.DataFrame, score_col: str):
    """[(date, score series best-first)] with the entries-only absolute floor.

    Names failing close>200DMA or n_themes_positive<3 are pushed to -inf so they
    cannot enter; a held name is governed by the exit band, not this mask.
    """
    ma200 = close.rolling(200, min_periods=200).mean()
    out = []
    idx = set(close.index)
    for d, g in vpanel.groupby("date", sort=True):
        ts = pd.Timestamp(d)
        if ts not in idx:
            continue
        s = pd.Series(g[score_col].to_numpy("float64"), index=g["symbol"].to_numpy())
        s = s[s.notna()]
        if s.empty:
            continue
        npos = pd.Series(g["n_themes_positive"].to_numpy("float64"),
                         index=g["symbol"].to_numpy()).reindex(s.index)
        c = close.loc[ts].reindex(s.index)
        m = ma200.loc[ts].reindex(s.index)
        floor_ok = (npos >= 3) & (c > m)
        s = s.where(floor_ok, other=-np.inf)
        out.append((ts, s.sort_values(ascending=False)))
    return out


def _cost_fn(cfg, coeff):
    params_copy = cfg.params.model_copy(deep=True)
    node = params_copy.costs.impact_model.coefficient
    if hasattr(node, "value"):
        node.value = coeff
    else:
        params_copy.costs.impact_model.coefficient = coeff
    model = CostModel(SimpleNamespace(params=params_copy))

    def cost_bps(price, quantity, adtv):
        if price <= 0 or quantity <= 0:
            return 0.0
        return float(model.round_trip(price, int(max(quantity, 1)),
                                      adtv_inr=adtv if adtv > 0 else None
                                      ).total_bps_of_buy)
    return cost_bps


def _window_dates(rankings, lo, hi):
    if lo is None:
        return [d for d, _ in rankings]
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    return [d for d, _ in rankings if lo <= d <= hi]


def _run(rankings, panels, params, dates):
    m = phase_summary(rankings, panels, params, step_sessions=21,
                      dates_allowed=dates)
    if not m or not m.get("benchmarked", False):
        return None
    ppy = m["periods_per_year"]
    return {
        "excess_ann": m["mean_excess"] * ppy,
        "ir": m["information_ratio"],
        "sharpe": m["sharpe"],
        "maxdd": m["worst_schedule_drawdown"],
        "cost_drag_ann": m["mean_cost"] * ppy,
        "gross_excess_ann": (m["mean_excess"] + m["mean_cost"]) * ppy,
        "avg_names": m["avg_names"],
        "turnover_per_period": m["avg_charged"],
        "n_periods": m["n_periods"],
        "beta": m.get("beta_to_benchmark", float("nan")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
    if not PANEL_CACHE.exists():
        print("ERROR: run audit_2026_09.py first to build the v3 score panel.",
              file=sys.stderr)
        sys.exit(2)
    vpanel = pd.read_parquet(PANEL_CACHE)
    vpanel["date"] = pd.to_datetime(vpanel["date"])
    panels = _price_panels(cfg, store, args.rebuild)
    close = panels["close"]
    base = _portfolio_params(cfg)
    print(f"[book] horizon {base.horizon_sessions}, capital {base.capital:,.0f}", flush=True)

    results = {"windows": {}, "single_theme": {}}

    # --- main sweep: books x coeffs x windows on the composite -------------
    print("\n" + "=" * 92)
    print(f"{'window':10s} {'book':11s} {'coeff':>5s} {'excess/yr':>10s} "
          f"{'IR':>6s} {'cost/yr':>8s} {'gross/yr':>9s} {'names':>6s} {'maxDD':>7s}")
    print("=" * 92)
    rk = _rankings(vpanel, close, "score")
    for wname, (lo, hi) in WINDOWS.items():
        dates = _window_dates(rk, lo, hi)
        results["windows"][wname] = {}
        for bname, K, er, xr in BOOKS:
            for coeff in COEFFS:
                p = dataclasses.replace(base, max_positions=K, entry_rank=er,
                                        exit_rank=xr, cost_fn=_cost_fn(cfg, coeff))
                r = _run(rk, panels, p, dates)
                results["windows"][wname][f"{bname}@{coeff}"] = r
                if r:
                    print(f"{wname:10s} {bname:11s} {coeff:>5.2f} "
                          f"{r['excess_ann']:>+9.1%} {r['ir']:>+6.2f} "
                          f"{r['cost_drag_ann']:>7.1%} {r['gross_excess_ann']:>+8.1%} "
                          f"{r['avg_names']:>6.1f} {r['maxdd']:>+7.1%}")
                else:
                    print(f"{wname:10s} {bname:11s} {coeff:>5.2f}   (no book)")

    # --- K-2: composite 6-name vs each single theme's 6-name book ----------
    print("\n" + "=" * 92)
    print("K-2  6-name book: composite vs each single theme (coeff 0.10, full window)")
    print("=" * 92)
    dates_full = _window_dates(rk, None, None)
    comp = _run(rk, panels, dataclasses.replace(
        base, max_positions=6, entry_rank=6, exit_rank=18,
        cost_fn=_cost_fn(cfg, 0.10)), dates_full)
    results["single_theme"]["composite"] = comp
    if comp:
        print(f"  composite   excess/yr {comp['excess_ann']:+.1%}  IR {comp['ir']:+.2f}  "
              f"gross/yr {comp['gross_excess_ann']:+.1%}")
    for theme, col in THEME_SUB.items():
        if col not in vpanel.columns:
            continue
        rkt = _rankings(vpanel, close, col)
        rt = _run(rkt, panels, dataclasses.replace(
            base, max_positions=6, entry_rank=6, exit_rank=18,
            cost_fn=_cost_fn(cfg, 0.10)), dates_full)
        results["single_theme"][theme] = rt
        if rt:
            print(f"  {theme:11s} excess/yr {rt['excess_ann']:+.1%}  IR {rt['ir']:+.2f}  "
                  f"gross/yr {rt['gross_excess_ann']:+.1%}")

    (OUT / "book_sim.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"\n[results] -> {OUT/'book_sim.json'}")
    print("\nCAVEATS: repo's cohort simulator (not the old holdout book); windows "
          "A/B overlap the 378-cell book-selection surface so a positive book is "
          "in-sample, not clean OOS; read config rankings, not third decimals.")


if __name__ == "__main__":
    main()
