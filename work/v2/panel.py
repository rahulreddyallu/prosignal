"""Assemble the research panel: one row per (signal date, symbol).

Execution model, and it is deliberately conservative: a signal computed on the
close of session t is FILLED AT THE VWAP OF SESSION t+1 and exited at the VWAP
of t+1+H. Nothing in a feature reads past t; nothing in a label starts before
t+1. That gap is the manual next-session execution the product actually asks of
its user.
"""
from __future__ import annotations
import os, gc, numpy as np, pandas as pd
import data as D_, universe as U, factors as FA

CACHE = "/home/claude/psr/cache"
MIN_LOOKBACK = 300
STEP = 5                      # weekly signal dates
HORIZONS = (5, 10, 21, 42, 63)


def _wide(m, key, dates, syms):
    return pd.DataFrame(m[key], index=pd.DatetimeIndex(dates), columns=syms)


def build(step: int = STEP, force: bool = False, max_names_pool: int = 800):
    out = f"{CACHE}/panel_step{step}.parquet"
    if os.path.exists(out) and not force:
        return pd.read_parquet(out)

    m = D_.build()
    dates = pd.DatetimeIndex(m["dates"]); syms = list(m["symbols"])
    W = {k: _wide(m, k, dates, syms) for k in
         ("open", "high", "low", "close", "vwap", "volume", "turnover",
          "trades", "deliv_pct", "adj_factor")}
    del m; gc.collect()

    # Ever-eligible pool -- keeps the working matrices small without changing
    # any per-date screen, which is still resolved below.
    pool_mask = U.eligible_mask(W["close"], W["turnover"], W["adj_factor"],
                                max_names=max_names_pool)
    keep = pool_mask.any(axis=0)
    keep = keep[keep].index
    for k in W:
        W[k] = W[k].loc[:, keep]
    print(f"pool symbols {len(keep)} of {len(syms)}")

    elig = U.eligible_mask(W["close"], W["turnover"], W["adj_factor"], max_names=750)
    adtv = W["turnover"].rolling(60, min_periods=1).median()
    adtv_rank = adtv.where(elig).rank(axis=1, ascending=False, method="first")

    # The market: equal-weight return of the ELIGIBLE names, as it stood.
    bench_px = W["close"].where(elig)
    bench_ret = (bench_px.mean(axis=1) / bench_px.mean(axis=1).shift(1) - 1.0)

    F = FA.build_factors(W["close"], W["high"], W["low"], W["open"], W["vwap"],
                         W["volume"], W["turnover"], W["trades"], W["deliv_pct"],
                         bench_ret)
    print(f"factors built: {len(F)}")

    # fill price: next session's VWAP, then open, then close
    fill = W["vwap"].where(W["vwap"] > 0)
    fill = fill.fillna(W["open"]).fillna(W["close"])
    fill_np = fill.to_numpy("float32")
    low_np = W["low"].to_numpy("float32")
    high_np = W["high"].to_numpy("float32")
    elig_np = elig.to_numpy()
    T = len(dates)

    sig_idx = [i for i in range(MIN_LOOKBACK, T - min(HORIZONS) - 2, step)]
    print(f"signal dates {len(sig_idx)}: {dates[sig_idx[0]].date()} -> {dates[sig_idx[-1]].date()}")

    names = list(F.keys())
    Fnp = {k: F[k].to_numpy("float32") for k in names}
    del F; gc.collect()
    sector = pd.read_parquet(f"{D_.D}/sector_map.parquet").set_index("symbol")["sector"]
    atr = (Fnp["atr_pct_14"])
    rows = []
    cols_syms = np.array(W["close"].columns)
    for i in sig_idx:
        sel = elig_np[i]
        if sel.sum() < 60:
            continue
        idx = np.where(sel)[0]
        d = {"date": dates[i], "symbol": cols_syms[idx]}
        for k in names:
            d[k] = Fnp[k][i, idx]
        entry = fill_np[i + 1, idx]
        d["entry_px"] = entry
        d["adtv"] = adtv.to_numpy("float32")[i, idx]
        d["adtv_rank"] = adtv_rank.to_numpy("float32")[i, idx]
        d["close"] = W["close"].to_numpy("float32")[i, idx]
        d["atr_pct"] = atr[i, idx]
        # Path within the coming 5 sessions, so a stop or target can be simulated
        # at the same granularity the book rebalances on.
        seg_lo = low_np[i + 1: i + 6, idx]
        seg_hi = high_np[i + 1: i + 6, idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            d["mae5"] = np.nanmin(seg_lo, axis=0) / np.where(entry > 0, entry, np.nan) - 1.0
            d["mfe5"] = np.nanmax(seg_hi, axis=0) / np.where(entry > 0, entry, np.nan) - 1.0
        for h in HORIZONS:
            j = i + 1 + h
            if j >= T:
                d[f"y{h}"] = np.full(len(idx), np.nan, "float32")
                d[f"b{h}"] = np.full(len(idx), np.nan, "float32")
                continue
            ex = fill_np[j, idx]
            d[f"y{h}"] = ex / np.where(entry > 0, entry, np.nan) - 1.0
            bp = fill_np[j][sel]
            ep = fill_np[i + 1][sel]
            good = np.isfinite(bp) & np.isfinite(ep) & (ep > 0)
            bmean = float(np.mean(bp[good] / ep[good] - 1.0)) if good.sum() > 20 else np.nan
            d[f"b{h}"] = np.full(len(idx), bmean, "float32")
        rows.append(pd.DataFrame(d))
    panel = pd.concat(rows, ignore_index=True)
    panel["sector"] = panel["symbol"].map(sector).fillna("UNCLASSIFIED")
    for c in panel.columns:
        if panel[c].dtype == "float64":
            panel[c] = panel[c].astype("float32")
    panel.to_parquet(out, index=False)
    print("panel", panel.shape, "->", out)
    return panel


if __name__ == "__main__":
    p = build(force=True)
    print(p.shape)
    print(p[["date"]].agg(["min", "max"]))
    print("rows/date median", int(p.groupby("date").size().median()))
    print("label coverage:", {h: float(p[f'y{h}'].notna().mean()) for h in HORIZONS})
