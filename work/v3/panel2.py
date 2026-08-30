"""Themed research panel: one row per (signal date, symbol).

Adds the point-in-time fundamental block to the price/volume/delivery panel and
tags every factor with its theme. Execution is unchanged and deliberately
conservative: a signal on the close of t is filled at the VWAP of t+1.
"""
from __future__ import annotations
import os, gc, numpy as np, pandas as pd
import data as D_, universe as U, factors as FA, pit_fund as PF, themes as TH

CACHE = "/home/claude/psr/cache"
MIN_LOOKBACK = 300
STEP = 5
HORIZONS = (5, 10, 21, 42, 63)


def build(step: int = STEP, force: bool = False, max_names_pool: int = 800,
          max_age_days: float = PF.MAX_AGE_DAYS):
    out = f"{CACHE}/panel2_step{step}.parquet"
    if os.path.exists(out) and not force:
        return pd.read_parquet(out)

    m = D_.build()
    dates = pd.DatetimeIndex(m["dates"]); syms = list(m["symbols"])
    W = {k: pd.DataFrame(m[k], index=dates, columns=syms) for k in
         ("open", "high", "low", "close", "vwap", "volume", "turnover",
          "trades", "deliv_pct", "adj_factor")}
    del m; gc.collect()

    pool_mask = U.eligible_mask(W["close"], W["turnover"], W["adj_factor"],
                                max_names=max_names_pool)
    keep = pool_mask.any(axis=0)
    keep = keep[keep].index
    for k in W:
        W[k] = W[k].loc[:, keep]
    print(f"pool symbols {len(keep)} of {len(syms)}", flush=True)

    elig = U.eligible_mask(W["close"], W["turnover"], W["adj_factor"], max_names=750)
    adtv = W["turnover"].rolling(60, min_periods=1).median()
    adtv_rank = adtv.where(elig).rank(axis=1, ascending=False, method="first")
    bench_px = W["close"].where(elig)
    bench_ret = bench_px.mean(axis=1) / bench_px.mean(axis=1).shift(1) - 1.0

    F = FA.build_factors(W["close"], W["high"], W["low"], W["open"], W["vwap"],
                         W["volume"], W["turnover"], W["trades"], W["deliv_pct"],
                         bench_ret)
    print(f"price/volume factors: {len(F)}", flush=True)

    # ---- point-in-time fundamentals, on their own 200-name frame ----------
    recs = PF.build_records()
    fsyms = [s for s in recs["symbol"].unique() if s in set(W["close"].columns)]
    Fp = PF.asof_panel(recs, dates, fsyms)
    sub_close = W["close"][fsyms]
    sub_adj = W["adj_factor"][fsyms]
    FF = TH.fundamental_factors(Fp, sub_close, sub_adj, max_age_days=max_age_days)
    del Fp; gc.collect()
    print(f"fundamental factors: {len([k for k in FF if not k.startswith('_')])} "
          f"on {len(fsyms)} symbols", flush=True)

    fill = W["vwap"].where(W["vwap"] > 0).fillna(W["open"]).fillna(W["close"])
    fill_np = fill.to_numpy("float32")
    low_np, high_np = W["low"].to_numpy("float32"), W["high"].to_numpy("float32")
    elig_np = elig.to_numpy()
    T = len(dates)
    sig_idx = [i for i in range(MIN_LOOKBACK, T - min(HORIZONS) - 2, step)]
    print(f"signal dates {len(sig_idx)}: {dates[sig_idx[0]].date()} -> "
          f"{dates[sig_idx[-1]].date()}", flush=True)

    pnames = list(F.keys())
    Fnp = {k: F[k].to_numpy("float32") for k in pnames}
    del F; gc.collect()
    fnames = [k for k in FF if not k.startswith("_")] + ["_mcap", "_fund_age_days"]
    FFnp = {k: FF[k].to_numpy("float32") for k in fnames}
    fcol = {s: j for j, s in enumerate(fsyms)}
    del FF; gc.collect()

    sector = pd.read_parquet(f"{D_.D}/sector_map.parquet").set_index("symbol")["sector"]
    cols_syms = np.array(W["close"].columns)
    atr = Fnp["atr_pct_14"]
    close_np = W["close"].to_numpy("float32")
    adtv_np = adtv.to_numpy("float32"); rank_np = adtv_rank.to_numpy("float32")
    rows = []
    for i in sig_idx:
        sel = elig_np[i]
        if sel.sum() < 60:
            continue
        idx = np.where(sel)[0]
        names = cols_syms[idx]
        d = {"date": dates[i], "symbol": names}
        for k in pnames:
            d[k] = Fnp[k][i, idx]
        fidx = np.array([fcol.get(s, -1) for s in names])
        has = fidx >= 0
        for k in fnames:
            col = np.full(len(idx), np.nan, "float32")
            if has.any():
                col[has] = FFnp[k][i, fidx[has]]
            d[k.lstrip("_") if k.startswith("_") else k] = col
        entry = fill_np[i + 1, idx]
        d["entry_px"] = entry
        d["adtv"] = adtv_np[i, idx]
        d["adtv_rank"] = rank_np[i, idx]
        d["close"] = close_np[i, idx]
        d["atr_pct"] = atr[i, idx]
        seg_lo = low_np[i + 1: i + 6, idx]; seg_hi = high_np[i + 1: i + 6, idx]
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
            bp, ep = fill_np[j][sel], fill_np[i + 1][sel]
            good = np.isfinite(bp) & np.isfinite(ep) & (ep > 0)
            bm = float(np.mean(bp[good] / ep[good] - 1.0)) if good.sum() > 20 else np.nan
            d[f"b{h}"] = np.full(len(idx), bm, "float32")
        rows.append(pd.DataFrame(d))
    panel = pd.concat(rows, ignore_index=True)
    panel["sector"] = panel["symbol"].map(sector).fillna("UNCLASSIFIED")
    for c in panel.columns:
        if panel[c].dtype == "float64":
            panel[c] = panel[c].astype("float32")
    panel.to_parquet(out, index=False)
    print("panel", panel.shape, "->", out, flush=True)
    return panel


if __name__ == "__main__":
    p = build(force=True)
    print(p.shape, p.date.min(), p.date.max())
    fcols = [c for c in TH.FACTOR_THEME if c in p.columns]
    print(f"factors present: {len(fcols)} of {len(TH.FACTOR_THEME)}")
    missing = [c for c in TH.FACTOR_THEME if c not in p.columns]
    print("missing:", missing)
    print("\ncoverage by theme (share of rows with a value):")
    for t, fs in TH.THEMES.items():
        have = [f for f in fs if f in p.columns]
        if have:
            print(f"  {t:12s} {p[have].notna().mean().mean():6.1%}  ({len(have)} factors)")
