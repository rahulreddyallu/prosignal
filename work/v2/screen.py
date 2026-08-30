"""Factor screen. TRAIN ONLY.

The null is a PLACEBO ALIGNMENT: the factor cross-section from date t scored
against the label cross-section from date t+k for |k| large enough that no true
relation can survive (>= 60 signal steps, roughly 1.2 years). Those placebo IC
series carry the same cross-sectional structure and the same overlap-induced
autocorrelation as the real one and none of the signal, so their t-statistics
are the honest critical value -- an analytic N(0,1) is not, because overlapping
63-session labels inflate a naive t by roughly sqrt(H/step).

Everything is one matmul: with per-date rank vectors laid out as a (dates x
symbols) matrix, the whole cross-date IC surface is a matrix product. The real
series is its diagonal; the null is its far off-diagonals.
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import core

MIN_LAG = 60          # signal steps between factor date and label date in the null
MIN_OVERLAP = 30      # symbols in common before an IC is computed


def _mats(df, col, dates, syms):
    """(D x N) matrix of within-date pct-ranks centred on zero, NaN elsewhere."""
    v = df[col].to_numpy("float64")
    r = df.groupby("date", observed=True)[col].rank(pct=True).to_numpy("float64")
    r = (r - 0.5) * 2.0
    M = np.full((len(dates), len(syms)), np.nan)
    M[df["_di"].to_numpy(), df["_si"].to_numpy()] = r
    return M


def ic_surface(Z: np.ndarray, Y: np.ndarray):
    """Full cross-date rank-IC matrix IC[t, s] = corr(Z[t], Y[s]) on the
    symbols the two dates share."""
    Mz = np.isfinite(Z); My = np.isfinite(Y)
    Z0 = np.where(Mz, Z, 0.0); Y0 = np.where(My, Y, 0.0)
    Mzf = Mz.astype("float64"); Myf = My.astype("float64")
    n = Mzf @ Myf.T
    sz = Z0 @ Myf.T
    sy = Mzf @ Y0.T
    szz = (Z0 * Z0) @ Myf.T
    syy = Mzf @ (Y0 * Y0).T
    szy = Z0 @ Y0.T
    with np.errstate(invalid="ignore", divide="ignore"):
        num = szy - sz * sy / n
        den = np.sqrt(np.maximum(szz - sz * sz / n, 0) * np.maximum(syy - sy * sy / n, 0))
        ic = num / den
    ic[n < MIN_OVERLAP] = np.nan
    return ic


def t_of(v):
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return np.nan
    return float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))


def run(horizons=(5, 10, 21, 42, 63), panel=None, tag="train"):
    tr = core.load_train() if panel is None else panel
    tr = tr.sort_values(["date", "symbol"]).reset_index(drop=True)
    meta = {"date", "symbol", "sector", "entry_px", "adtv", "adtv_rank", "close", "atr_pct"}
    fcols = [c for c in tr.columns if c not in meta
             and not (c[0] in "yb" and c[1:].isdigit())]
    R = core.sector_neutral_ranks(tr, fcols)
    tr = pd.concat([tr, R], axis=1)
    dates = np.array(sorted(tr["date"].unique()))
    syms = np.array(sorted(tr["symbol"].unique()))
    tr["_di"] = pd.Series(np.arange(len(dates)), index=pd.DatetimeIndex(dates)).reindex(tr["date"]).to_numpy()
    tr["_si"] = pd.Series(np.arange(len(syms)), index=syms).reindex(tr["symbol"]).to_numpy()
    D = len(dates)
    print(f"{len(fcols)} factors | {D} dates | {len(syms)} symbols | {len(tr)} rows")

    lag = np.abs(np.subtract.outer(np.arange(D), np.arange(D)))
    far = lag >= MIN_LAG
    rows = []
    for h in horizons:
        Ymat = _mats(tr, f"y{h}", dates, syms)
        for f in fcols:
            Zmat = _mats(tr, f + "_r", dates, syms)
            S = ic_surface(Zmat, Ymat)
            real = np.diag(S)
            t = t_of(real)
            # null: every far off-diagonal band is one placebo experiment
            null_ts = []
            for k in range(MIN_LAG, D - 20):
                for sgn in (1, -1):
                    d = np.diagonal(S, offset=sgn * k)
                    if np.isfinite(d).sum() >= 20:
                        null_ts.append(t_of(d))
            null_ts = np.array([x for x in null_ts if np.isfinite(x)])
            crit = float(np.percentile(np.abs(null_ts), 95)) if len(null_ts) > 20 else np.nan
            rows.append({"factor": f, "h": h, "ic": float(np.nanmean(real)),
                         "t": t, "null_t95": crit,
                         "keep": bool(np.isfinite(t) and np.isfinite(crit) and abs(t) > crit),
                         "n_dates": int(np.isfinite(real).sum()),
                         "n_null": len(null_ts)})
        print(f"  h={h} done")
    out = pd.DataFrame(rows)
    out.to_csv(f"/home/claude/psr/cache/screen_{tag}.csv", index=False)
    return out


if __name__ == "__main__":
    df = run()
    for h in sorted(df.h.unique()):
        s = df[df.h == h].copy()
        s["at"] = s.t.abs()
        s = s.sort_values("at", ascending=False)
        print(f"\n===== h={h}  KEPT {int(s.keep.sum())}/{len(s)}  null|t|95 median {s.null_t95.median():.2f}")
        print(s.head(22)[["factor", "ic", "t", "null_t95", "keep"]].to_string(index=False))
