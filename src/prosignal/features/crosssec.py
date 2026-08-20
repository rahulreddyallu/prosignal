"""Cross-sectional feature panel and forward-return labels.

Every feature at date t uses only closes at or before t, and the label is the
forward return measured strictly after t. The two are built from disjoint slices
of the same price matrix so an off-by-one cannot quietly connect them.

Features are the ones with the strongest cross-sectional replication in equity
literature, restricted to what this store can actually serve: price and volume.
Fundamentals are excluded because the store's newest filing is 525 days old,
which is the same reason Stage 4 now drops them.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["FEATURES", "build_panel", "cross_sectional_rank"]

#: name -> (lookback sessions needed, description)
FEATURES: Dict[str, Tuple[int, str]] = {
    "mom_12_1":      (294, "12-1 momentum: return over 273 sessions ending 21 back (Jegadeesh & Titman 1993)"),
    "mom_6_1":       (147, "6-1 momentum"),
    "mom_3_1":       (84,  "3-1 momentum"),
    "reversal_1m":   (22,  "last 21-session return; short-horizon reversal (Jegadeesh 1990)"),
    "vol_60":        (61,  "realised volatility, 60 sessions, annualised"),
    "downside_vol":  (61,  "downside deviation of daily returns, 60 sessions"),
    "beta_120":      (121, "OLS beta against the equal-weight universe, 120 sessions"),
    "idio_vol":      (121, "residual volatility from that beta regression"),
    "amihud":        (61,  "Amihud (2002) illiquidity: mean(|ret| / turnover)"),
    "turnover_ratio":(61,  "mean turnover over 60 sessions, log"),
    "rel_strength":  (126, "stock return minus universe return, 126 sessions"),
    "dist_200dma":   (201, "close / 200-session mean - 1"),
    "trend_r2":      (121, "R-squared of an OLS fit on log close, 120 sessions"),
    "max_dd_120":    (121, "maximum drawdown over 120 sessions"),
    "prox_52w":      (253, "close / 252-session high - 1 (George & Hwang 2004)"),
    "max5_21":       (22,  "mean of the 5 largest daily returns in 21 sessions; lottery demand (Bali, Cakici & Whitelaw 2011)"),
    "resid_mom":     (253, "momentum of market-residual returns, 252 to 21 back (Blitz, Huij & Martens 2011)"),
    "deliv_pct":     (61,  "mean delivered fraction of traded volume, 60 sessions"),
    "deliv_trend":   (127, "delivered fraction, 21-session mean less 126-session mean"),
}

#: Factors that rank neutral rather than dropping the row when their input is
#: missing. Delivery is published per session in sec_bhavdata_full and covers
#: about 82% of the panel; requiring it would discard a fifth of the universe
#: over a feed gap. This matches how Stage 4 treats an unavailable factor and
#: how _attach_fundamentals treats a name with no usable filing.
NEUTRAL_WHEN_MISSING = frozenset({"deliv_pct", "deliv_trend"})

MIN_LOOKBACK = max(v[0] for v in FEATURES.values())


def _ols_beta_resid(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """Beta of y on x, and the residual standard deviation."""
    if len(y) < 20 or np.allclose(x.std(), 0):
        return np.nan, np.nan
    xc, yc = x - x.mean(), y - y.mean()
    denom = float((xc * xc).sum())
    if denom <= 0:
        return np.nan, np.nan
    beta = float((xc * yc).sum() / denom)
    resid = yc - beta * xc
    return beta, float(resid.std(ddof=1))


def _trend_r2(logp: np.ndarray) -> float:
    n = len(logp)
    if n < 20:
        return np.nan
    t = np.arange(n, dtype="float64")
    tc, yc = t - t.mean(), logp - logp.mean()
    denom = float((tc * tc).sum())
    if denom <= 0:
        return np.nan
    slope = float((tc * yc).sum() / denom)
    fitted = slope * tc
    ss_tot = float((yc * yc).sum())
    if ss_tot <= 0:
        return np.nan
    return float(1.0 - ((yc - fitted) ** 2).sum() / ss_tot)


def _features_at(
    close: pd.DataFrame,
    turnover: pd.DataFrame,
    i: int,
    bench_ret: np.ndarray,
    delivery: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Features for every symbol using rows 0..i inclusive. Never touches i+1."""
    hist = close.iloc[: i + 1]
    tno = turnover.iloc[: i + 1]
    ret = hist.pct_change()
    out: Dict[str, pd.Series] = {}
    last = hist.iloc[-1]

    def past(k: int) -> pd.Series:
        return hist.iloc[-1 - k] if len(hist) > k else pd.Series(index=hist.columns, dtype="float64")

    out["mom_12_1"] = past(21) / past(294) - 1.0
    out["mom_6_1"] = past(21) / past(147) - 1.0
    out["mom_3_1"] = past(21) / past(84) - 1.0
    out["reversal_1m"] = last / past(21) - 1.0

    r60 = ret.tail(60)
    out["vol_60"] = r60.std(ddof=1) * np.sqrt(252)
    out["downside_vol"] = r60.where(r60 < 0).std(ddof=1) * np.sqrt(252)

    out["prox_52w"] = last / hist.tail(252).max() - 1.0
    out["max5_21"] = ret.tail(21).apply(lambda s: s.nlargest(5).mean(), axis=0)

    # Residual momentum: strip the market component, then accumulate. Blitz,
    # Huij & Martens (2011) find the residual carries the momentum premium with
    # far less of the beta exposure that drives momentum crashes.
    win = ret.tail(252)
    out["resid_mom"] = pd.Series(np.nan, index=hist.columns, dtype="float64")
    if len(win) >= 60 and len(bench_ret) >= len(win):
        b = np.asarray(bench_ret[-len(win):], dtype="float64")
        bc = b - np.nanmean(b)
        bvar = float(np.nanmean(bc * bc))
        # A market with no dispersion leaves beta undefined; the factor stays NaN
        # rather than vanishing, so the column is always present for the model.
        if bvar > 1e-12:
            beta_m = win.mul(bc, axis=0).mean() / bvar
            resid = win.sub(np.outer(b, beta_m.to_numpy()), fill_value=np.nan)
            resid.columns = win.columns
            out["resid_mom"] = resid.iloc[:-21].sum(axis=0)

    r120 = ret.tail(120)
    bench = bench_ret[-len(r120):] if len(bench_ret) >= len(r120) else bench_ret
    betas, idios = {}, {}
    for s in r120.columns:
        y = r120[s].to_numpy(dtype="float64")
        mask = np.isfinite(y) & np.isfinite(bench)
        if mask.sum() < 40:
            betas[s], idios[s] = np.nan, np.nan
            continue
        b, rsd = _ols_beta_resid(y[mask], bench[mask])
        betas[s], idios[s] = b, (rsd * np.sqrt(252) if np.isfinite(rsd) else np.nan)
    out["beta_120"] = pd.Series(betas)
    out["idio_vol"] = pd.Series(idios)

    t60 = tno.tail(60)
    absret = ret.tail(60).abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        illiq = (absret / t60.replace(0, np.nan)).mean() * 1e9
    out["amihud"] = illiq
    out["turnover_ratio"] = np.log1p(t60.mean())

    out["rel_strength"] = (last / past(126) - 1.0) - float(
        np.nanmean((last / past(126) - 1.0).to_numpy(dtype="float64"))
    )
    out["dist_200dma"] = last / hist.tail(200).mean() - 1.0

    logp = np.log(hist.tail(120))
    out["trend_r2"] = pd.Series(
        {s: _trend_r2(logp[s].dropna().to_numpy(dtype="float64")) for s in logp.columns}
    )
    win = hist.tail(120)
    out["max_dd_120"] = (win / win.cummax() - 1.0).min()

    # Delivered fraction of traded volume. NSE settles intraday positions
    # without delivery, so a high ratio means buyers took the stock rather than
    # churning it -- conviction that has no clean analogue in most markets.
    if delivery is not None and not delivery.empty:
        dl = delivery.reindex(index=hist.index, columns=hist.columns).iloc[: i + 1]
        out["deliv_pct"] = dl.tail(60).mean()
        out["deliv_trend"] = dl.tail(21).mean() - dl.tail(126).mean()
    else:
        out["deliv_pct"] = pd.Series(np.nan, index=hist.columns, dtype="float64")
        out["deliv_trend"] = pd.Series(np.nan, index=hist.columns, dtype="float64")

    frame = pd.DataFrame(out)
    return frame.replace([np.inf, -np.inf], np.nan)


def cross_sectional_rank(s: pd.Series) -> pd.Series:
    """Rank to [-1, 1] within a date.

    Rank rather than z-score: a single midcap doubling on an order announcement
    wrecks a cross-sectional mean and standard deviation, and the resulting
    z-scores collapse every other name toward zero. Fitted per date, so no
    scaling parameter is ever carried across time.
    """
    r = s.rank(pct=True, na_option="keep")
    return (r - 0.5) * 2.0


def build_panel(
    close: pd.DataFrame,
    turnover: pd.DataFrame,
    horizon: int,
    step: int = 21,
    min_names: int = 40,
    delivery: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Assemble the panel. One row per (date, symbol).

    ``label`` is the forward return from i to i+horizon, so it uses only prices
    strictly after the feature date.
    """
    dates = list(close.index)
    bench_full = close.mean(axis=1).pct_change().to_numpy(dtype="float64")
    rows: List[pd.DataFrame] = []
    for i in range(MIN_LOOKBACK, len(dates) - horizon, step):
        feats = _features_at(close, turnover, i, bench_full[: i + 1], delivery=delivery)
        fwd = close.iloc[i + horizon] / close.iloc[i] - 1.0
        feats = feats.assign(label=fwd)
        feats = feats[np.isfinite(feats["label"]) & feats["label"].abs().lt(1.0)]
        required = [c for c in FEATURES if c not in NEUTRAL_WHEN_MISSING]
        feats = feats.dropna(subset=required, thresh=int(len(required) * 0.7))
        if len(feats) < min_names:
            continue
        feats["date"] = dates[i]
        feats["symbol"] = feats.index
        rows.append(feats.reset_index(drop=True))
    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows, ignore_index=True)
    del rows
    # float32 across the feature block. Ranks and returns carry nowhere near
    # seven significant figures, and the panel is the largest object the signal
    # path allocates.
    for c in panel.columns:
        if panel[c].dtype == "float64" and c != "label":
            panel[c] = panel[c].astype("float32")
    panel["label_rank"] = panel.groupby("date")["label"].transform(cross_sectional_rank)
    for f in FEATURES:
        r = panel.groupby("date")[f].transform(cross_sectional_rank)
        # A neutral rank is 0.0 by construction, so a name with no delivery
        # print contributes nothing to the score instead of being discarded.
        panel[f + "_r"] = r.fillna(0.0) if f in NEUTRAL_WHEN_MISSING else r
    return panel
