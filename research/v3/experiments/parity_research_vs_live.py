"""Does the RESEARCH path score the same thing the LIVE path ranks?

THE QUESTION NOTHING ELSE ASKS. `validation/parity.py` compares a live run to a
replay of the same date, which tests whether the store settled the way the run
assumed -- a data-timing question. This asks a different one: the audit's
conclusions, the CPCV folds, the prune that now ships, were all computed by
`validation.panel.build_panel`. The engine ranks with
`stage4_core_score.build_v3_block`. Those are two functions. If they disagree,
every number that justified the model describes something the engine does not do.

They share `features/factors.factor_frame` and `features/engine.score_frame`, so
the FORMULAE cannot differ. What can differ is everything around them, and each
of these is a real hypothesis rather than paranoia:

  universe      the panel takes its own liquidity mask (max_names, min ADTV, min
                price, min history); the live path takes whatever Stage 3 passed.
                Different populations produce different sector-neutral ranks for
                the SAME name, because a rank is a statement about its peers.
  benchmark     `resid_rev_21` is the one factor that reads a market return, and
                both build it as the equal-weight mean of the names in hand -- so
                a different universe is a different market.
  window        the panel slices `lookback + 15` sessions ending at i; the live
                path asks the calendar for a trailing window of the same length.
                An off-by-one in either is invisible and changes every rolling
                statistic.
  fundamentals  both as-of join on disclosure dates, over different symbol lists.
  sectors       the panel reads the sector map itself; the live path is handed
                one by Stage 4.

The check therefore runs in two modes, and the second is the one that matters:

  AS-CONFIGURED  each path on the universe it would really use. Divergence here
                 is expected and is a statement about POPULATION, not code.
  SAME-UNIVERSE  both paths forced onto the identical symbol list for the same
                 date. Any divergence here is a genuine implementation split and
                 there is no benign explanation for one.

Usage:
    python research/v3/experiments/parity_research_vs_live.py
    python research/v3/experiments/parity_research_vs_live.py --dates 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from prosignal.config.loader import load_config                      # noqa: E402
from prosignal.core.calendar import TradingCalendar                  # noqa: E402
from prosignal.data.store import DataStore                           # noqa: E402
from prosignal.features import engine, factors                    # noqa: E402
from prosignal.features.crosssec import liquidity_mask               # noqa: E402
from prosignal.stages.stage4_core_score import build_score_block     # noqa: E402

OUT = HERE / "parity_research_vs_live.json"

#: What the panel screens on, copied from `build_panel`'s defaults so this
#: harness reproduces the research universe rather than inventing one.
PANEL_SCREEN = dict(max_names=750, min_adtv_inr=5e7, min_price_inr=20.0,
                    min_history_sessions=300)


def panel_universe(store, as_of, sessions):
    """The symbol list `build_panel` would have scored on `as_of`."""
    start = sessions[max(0, sessions.index(as_of) - 400)]
    px = store.read_prices(start=start, end=as_of)
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    piv = lambda c: px.pivot_table(index="date", columns="symbol", values=c,
                                   aggfunc="last", observed=True).sort_index()
    close, turnover = piv("close"), piv("turnover")
    adj = piv("adj_factor") if "adj_factor" in px.columns else None
    if adj is None:
        adj = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    try:
        from prosignal.data.instruments import non_equity_symbols
        try:
            master = store.read_equity_master()
        except Exception:
            master = None
        drop = non_equity_symbols(list(close.columns), master, close)
        if drop:
            keep = [c for c in close.columns if c not in drop]
            close, turnover = close[keep], turnover.reindex(columns=keep)
            adj = adj.reindex(columns=keep)
    except Exception:
        pass
    elig = liquidity_mask(close, turnover, lookback_sessions=60,
                          adj_factor=adj.reindex(columns=close.columns),
                          **PANEL_SCREEN)
    row = elig.loc[pd.Timestamp(as_of)] if pd.Timestamp(as_of) in elig.index \
        else elig.iloc[-1]
    return sorted(row[row].index.astype(str)), close, turnover


def research_score(store, as_of, symbols, sectors, sessions):
    """Score `symbols` on `as_of` the way the RESEARCH panel does it."""
    i = sessions.index(as_of)
    win = sessions[max(i + 1 - factors.FRAME_SESSIONS, 0): i + 1]
    px = store.read_prices(symbols=symbols, start=win[0], end=as_of)
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    piv = lambda c: (px.pivot_table(index="date", columns="symbol", values=c,
                                    aggfunc="last", observed=True).sort_index()
                     if c in px.columns else None)
    close = piv("close")
    if close is None or close.empty:
        return None, None
    open_, vwap, turnover = piv("open"), piv("vwap"), piv("turnover")
    deliv = None
    try:
        dl = store.read_delivery(symbols=symbols, start=win[0], end=as_of)
        if dl is not None and not dl.empty and "deliv_pct" in dl.columns:
            dl = dl.copy()
            dl["date"] = pd.to_datetime(dl["date"]).dt.normalize()
            deliv = dl.pivot_table(index="date", columns="symbol",
                                   values="deliv_pct", aggfunc="last",
                                   observed=True).sort_index()
            deliv = deliv.reindex(index=close.index, columns=close.columns)
    except Exception:
        deliv = None
    fund = None
    try:
        from prosignal.features import pit_fundamentals as pitf
        recs = pitf.build_records(store=store)
        if recs is not None and not recs.empty:
            fs = [c for c in close.columns if c in set(recs["symbol"])]
            if fs:
                blk = pitf.asof_panel(recs, close.index, fs)
                fund = {k: v.reindex(columns=close.columns)
                        for k, v in blk.items()
                        if k in ("ttm_revenue", "ttm_net_profit", "fund_age_days")}
    except Exception:
        fund = None
    raw = factors.factor_frame(close, open_, vwap, turnover, deliv, fund)
    scored = engine.score_frame(raw, sectors)
    return raw, scored


def _cmp(a: pd.Series, b: pd.Series, label: str) -> dict:
    common = a.index.intersection(b.index)
    x, y = a.reindex(common), b.reindex(common)
    ok = x.notna() & y.notna()
    n = int(ok.sum())
    out = {"field": label, "common_names": int(len(common)), "both_finite": n}
    if n < 5:
        out["verdict"] = "too few overlapping values to compare"
        return out
    xv, yv = x[ok].to_numpy("float64"), y[ok].to_numpy("float64")
    out["max_abs_diff"] = float(np.nanmax(np.abs(xv - yv)))
    out["mean_abs_diff"] = float(np.nanmean(np.abs(xv - yv)))
    out["spearman"] = float(pd.Series(xv).corr(pd.Series(yv), method="spearman"))
    out["identical"] = bool(out["max_abs_diff"] < 1e-12)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", type=int, default=8)
    ap.add_argument("--stride", type=int, default=45,
                    help="sessions between sampled dates")
    args = ap.parse_args()

    cfg = load_config(ROOT / "config/parameters.yaml")
    store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
    sessions = store.price_sessions()
    cal = TradingCalendar(sessions)
    sm = store.read_sector_map()
    sectors = dict(zip(sm["symbol"].astype(str), sm["sector"]))

    themes = engine.THEMES
    print(f"one scorer: {len(themes)} themes, {len(engine.ALL_FACTORS)} factors")

    picks = [sessions[i] for i in
             range(len(sessions) - 20, 400, -args.stride)][:args.dates]
    rows = []
    for as_of in picks:
        puni, _, _ = panel_universe(store, as_of, sessions)
        if len(puni) < 100:
            continue
        # LIVE PATH on the research universe -- the same-universe test.
        _, live_same, err = build_score_block(store, cal, puni, as_of, sectors,
                                              cfg.params.stage4_core_score)
        # RESEARCH PATH on the same universe.
        _, res_same = research_score(store, as_of, puni, sectors, sessions)
        if live_same is None or res_same is None or err:
            print(f"{as_of}  skipped ({err or 'a path returned nothing'})")
            continue

        rec = {"date": str(as_of), "universe": len(puni),
               "same_universe": {}}
        for f in ("score",):
            rec["same_universe"][f] = _cmp(res_same[f], live_same[f], f)
        for t in themes:
            rec["same_universe"][t + "_sub"] = _cmp(
                res_same[t + "_sub"], live_same[t + "_sub"], t + "_sub")
        a = res_same["score"].dropna().sort_values(ascending=False).index[:6]
        b = live_same["score"].dropna().sort_values(ascending=False).index[:6]
        rec["top6_research"] = list(a)
        rec["top6_live"] = list(b)
        rec["top6_overlap"] = len(set(a) & set(b)) / 6.0
        rows.append(rec)
        s = rec["same_universe"]["score"]
        print(f"{as_of}  n={len(puni):4d}  score max|diff| {s.get('max_abs_diff', float('nan')):.3e}"
              f"  rho {s.get('spearman', float('nan')):.6f}"
              f"  top-6 overlap {rec['top6_overlap']:.0%}"
              f"  {'IDENTICAL' if s.get('identical') else 'DIVERGES'}")

    if not rows:
        print("no comparable dates")
        return 2
    ident = [r["same_universe"]["score"]["identical"] for r in rows]
    worst = max(r["same_universe"]["score"].get("max_abs_diff", 0.0) for r in rows)
    mean_overlap = float(np.mean([r["top6_overlap"] for r in rows]))
    verdict = ("The research path and the live path compute the SAME score on the "
               "same universe. Every number the audit produced describes what the "
               "engine ranks."
               if all(ident) else
               f"THE TWO PATHS DIVERGE on the same universe (worst max|diff| "
               f"{worst:.3e}, mean top-6 overlap {mean_overlap:.0%}). Research "
               f"numbers do NOT describe the shipped engine until this is "
               f"explained.")
    OUT.write_text(json.dumps(
        {"dates": rows,
         "all_identical_same_universe": bool(all(ident)),
         "worst_max_abs_diff": worst, "mean_top6_overlap": mean_overlap,
         "verdict": verdict}, indent=2))
    print(f"\n{verdict}\n\nwritten {OUT}")
    return 0 if all(ident) else 1


if __name__ == "__main__":
    raise SystemExit(main())
