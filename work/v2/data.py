"""Panel assembly for the v2 signal-engine search.

Reads the curated store, applies corporate-action adjustment exactly as
`prosignal.data.corporate_actions.apply_adjustments` does (cumulative product of
every ratio whose ex-date is strictly after t), and caches wide matrices.

Nothing here reads a value dated after the row it belongs to.
"""
from __future__ import annotations
import os, numpy as np, pandas as pd

D = "/mnt/user-data/uploads/Pro Stock Signal BOT/data/curated"
CACHE = "/home/claude/psr/cache"
os.makedirs(CACHE, exist_ok=True)

PRICE_YEARS = list(range(2017, 2027))
EQUITY_SERIES = {"EQ", "BE"}


def _load_prices() -> pd.DataFrame:
    frames = []
    for y in PRICE_YEARS:
        p = f"{D}/prices/year={y}.parquet"
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p, columns=["date", "symbol", "series", "open", "high",
                                         "low", "close", "vwap", "volume", "turnover",
                                         "trades", "deliv_qty", "deliv_pct"])
        frames.append(df)
    px = pd.concat(frames, ignore_index=True)
    px = px[px["series"].isin(EQUITY_SERIES)].drop(columns=["series"])
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    px = px.drop_duplicates(subset=["date", "symbol"], keep="last")
    return px


def _load_delivery() -> pd.DataFrame:
    frames = []
    for y in range(2019, 2027):
        f = f"{D}/delivery/year={y}.parquet"
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f, columns=["date", "symbol", "series", "deliv_qty", "deliv_pct"])
        frames.append(d[d["series"].isin(EQUITY_SERIES)].drop(columns=["series"]))
    d = pd.concat(frames, ignore_index=True)
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    return d.drop_duplicates(subset=["date", "symbol"], keep="last")


def _adjust(px: pd.DataFrame) -> pd.DataFrame:
    acts = pd.read_parquet(f"{D}/corporate_actions.parquet")
    acts["ex_date"] = pd.to_datetime(acts["ex_date"]).dt.normalize()
    acts = acts.dropna(subset=["ex_date", "ratio"])
    acts = acts[(acts["ratio"] > 0) & (acts["ratio"] != 1.0)]
    px = px.sort_values(["symbol", "date"]).reset_index(drop=True)
    px["adj_factor"] = 1.0
    by = {s: g for s, g in acts.groupby("symbol", observed=True)}
    fac_all = np.ones(len(px), dtype="float64")
    starts = px.groupby("symbol", sort=False, observed=True).indices
    for sym, idx in starts.items():
        a = by.get(sym)
        if a is None or a.empty:
            continue
        d = px["date"].to_numpy()[idx]
        f = np.ones(len(idx))
        for r, ex in zip(a["ratio"].to_numpy(), a["ex_date"].to_numpy()):
            f = np.where(d < ex, f * float(r), f)
        fac_all[idx] = f
    px["adj_factor"] = fac_all
    for c in ("open", "high", "low", "close", "vwap"):
        px[c] = px[c].to_numpy() * fac_all
    with np.errstate(divide="ignore", invalid="ignore"):
        px["volume"] = np.where(fac_all > 0, px["volume"].to_numpy() / fac_all, np.nan)
        px["deliv_qty"] = np.where(fac_all > 0, px["deliv_qty"].to_numpy() / fac_all, np.nan)
    return px


def build(force: bool = False) -> dict:
    out = f"{CACHE}/panel.npz"
    if os.path.exists(out) and not force:
        z = np.load(out, allow_pickle=True)
        return {k: z[k] for k in z.files}
    px = _load_prices().drop(columns=["deliv_qty", "deliv_pct"])
    dl = _load_delivery()
    px = px.merge(dl, on=["date", "symbol"], how="left")
    px = _adjust(px)
    dates = np.array(sorted(px["date"].unique()))
    syms = np.array(sorted(px["symbol"].unique()))
    di = pd.Series(np.arange(len(dates)), index=pd.DatetimeIndex(dates))
    si = pd.Series(np.arange(len(syms)), index=syms)
    r = di.reindex(px["date"]).to_numpy()
    c = si.reindex(px["symbol"]).to_numpy()
    mats = {}
    for col in ("open", "high", "low", "close", "vwap", "volume", "turnover",
                "trades", "deliv_qty", "deliv_pct", "adj_factor"):
        M = np.full((len(dates), len(syms)), np.nan, dtype="float32")
        M[r, c] = px[col].to_numpy(dtype="float32")
        mats[col] = M
    mats["dates"] = dates
    mats["symbols"] = syms
    np.savez_compressed(out, **mats)
    return mats


if __name__ == "__main__":
    m = build(force=True)
    print("dates", len(m["dates"]), m["dates"][0], m["dates"][-1])
    print("symbols", len(m["symbols"]))
    cl = m["close"]
    print("close shape", cl.shape, "non-nan frac", float(np.isfinite(cl).mean()))
    print("names with a print per date: median",
          int(np.median(np.isfinite(cl).sum(1))))
