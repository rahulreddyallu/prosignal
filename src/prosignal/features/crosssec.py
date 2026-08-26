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

from .labels import BarrierSpec, average_uniqueness, triple_barrier

__all__ = ["FEATURES", "build_panel", "cross_sectional_rank",
           "liquidity_mask", "sector_neutral_rank", "MIN_SECTOR_NAMES",
           "BarrierSpec"]

#: name -> (lookback sessions needed, description)
#:
#: Seven factors were removed after a marginal-IC test. Each was regressed on
#: the others in its correlation block and the residual scored against forward
#: returns: a factor whose residual carries no information is a restatement of
#: its neighbours, however well it scores alone.
#:
#:   momentum    mom_12_1 (marginal t -0.64) and mom_3_1 (-0.27) are spanned by
#:               mom_6_1 (+1.62) and resid_mom (+1.43)
#:   volatility  vol_60 (-0.96) and idio_vol (-0.92) are spanned by
#:               downside_vol (-2.89)
#:   trend       dist_200dma (-0.34), rel_strength (+0.06) and trend_r2 (+0.48)
#:               are spanned by prox_52w (+2.70)
#:
#: dist_200dma is the clearest case: raw IC t 2.86, marginal t -0.34. It is a
#: good factor and it says nothing prox_52w has not already said.
#:
#: Holdout after the trim, 36 periods: IC +0.0512 -> +0.0530 (t 2.24 -> 2.43)
#: with seven fewer factors. Standardised unexpected earnings was built and
#: measured as a replacement and rejected: selection IC +0.0479 (t 2.60),
#: holdout -0.0003 (t -0.02).
FEATURES: Dict[str, Tuple[int, str]] = {
    "mom_6_1":       (147, "6-1 momentum"),
    # Residual, not raw. Blitz, Huij, Lansdorp & Martens (2013): conventional
    # short-term reversal carries dynamic exposure to the market and size
    # factors, and building it on residual returns avoids them and earns
    # roughly twice the risk-adjusted return. Standardised by the residual's own
    # trailing dispersion so a volatile name does not dominate by construction.
    "resid_reversal": (253, "21-session residual return over its trailing residual sd (Blitz, Huij, Lansdorp & Martens 2013)"),
    "downside_vol":  (61,  "downside deviation of daily returns, 60 sessions"),
    "beta_120":      (121, "OLS beta against the equal-weight universe, 120 sessions"),
    "amihud":        (61,  "Amihud (2002) illiquidity: mean(|ret| / turnover)"),
    "turnover_ratio":(61,  "mean turnover over 60 sessions, log"),
    "max_dd_120":    (121, "maximum drawdown over 120 sessions"),
    "prox_52w":      (253, "close / 252-session high - 1 (George & Hwang 2004)"),
    "max5_21":       (22,  "mean of the 5 largest daily returns in 21 sessions; lottery demand (Bali, Cakici & Whitelaw 2011)"),
    "resid_mom":     (253, "momentum of market-residual returns, 252 to 21 back (Blitz, Huij & Martens 2011)"),
    "idio_vol":      (253, "annualised std of market-residual returns, 126 sessions (Ang, Hodrick, Xing & Zhang 2006)"),
    "idio_skew":     (253, "skewness of market-residual returns, 126 sessions; lottery demand"),
    "deliv_pct":     (61,  "mean delivered fraction of traded volume, 60 sessions"),
    "deliv_trend":   (127, "delivered fraction, 21-session mean less 126-session mean"),
}

#: Factors that rank neutral rather than dropping the row when their input is
#: missing. Delivery is published per session in sec_bhavdata_full and covers
#: about 82% of the panel; requiring it would discard a fifth of the universe
#: over a feed gap. This matches how Stage 4 treats an unavailable factor and
#: how _attach_fundamentals treats a name with no usable filing.
NEUTRAL_WHEN_MISSING = frozenset({"deliv_pct", "deliv_trend"})

#: Window for the idiosyncratic moments. Long enough for a third moment to
#: mean anything -- skewness on 60 points is mostly noise -- and short enough to
#: describe the name as it is now.
RESID_WINDOW = 126

#: Window the RESIDUAL REVERSAL is standardised over. Blitz, Huij, Lansdorp &
#: Martens (2013) standardise by the trailing 36-month residual standard
#: deviation, which is 756 sessions. Requiring all of it would exclude any name
#: with under three years of history against a universe floor of 300 sessions,
#: so what is available is used down to RESID_WINDOW and no further -- a
#: dispersion estimate on less than six months is not one.
REVERSAL_STD_WINDOW = 756

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
    # fill_method=None, matching every other module. pandas pads by default, which
    # invents a 0% return on a session the stock did not trade and then books the
    # whole multi-session move on the next real one. 35% of names carry an interior
    # gap, so the pad understates downside_vol and max5_21, makes amihud read more
    # liquid than the name is, and estimates beta against a shifted series.
    # universe.forbid_forward_fill_across_sessions declares this must not happen.
    ret = hist.pct_change(fill_method=None)
    out: Dict[str, pd.Series] = {}
    last = hist.iloc[-1]

    def past(k: int) -> pd.Series:
        return hist.iloc[-1 - k] if len(hist) > k else pd.Series(index=hist.columns, dtype="float64")

    out["mom_6_1"] = past(21) / past(147) - 1.0

    r60 = ret.tail(60)
    out["downside_vol"] = r60.where(r60 < 0).std(ddof=1) * np.sqrt(252)

    out["prox_52w"] = last / hist.tail(252).max() - 1.0
    out["max5_21"] = ret.tail(21).apply(lambda s: s.nlargest(5).mean(), axis=0)

    # Residual momentum: strip the market component, then accumulate. Blitz,
    # Huij & Martens (2011) find the residual carries the momentum premium with
    # far less of the beta exposure that drives momentum crashes.
    # Long enough for the reversal's 36-month standardisation where the history
    # exists. resid_mom still reads only its own 252 sessions off the tail.
    win = ret.tail(max(252, REVERSAL_STD_WINDOW))
    for name in ("resid_mom", "idio_vol", "idio_skew", "resid_reversal"):
        out[name] = pd.Series(np.nan, index=hist.columns, dtype="float64")
    # An all-NaN benchmark slice is a real state, not a defect: under a
    # point-in-time universe the earliest dates have no eligible names, so the
    # equal-weight market is undefined there. nanmean warns and returns NaN on
    # such a slice, so the emptiness is tested before it is averaged.
    if (len(win) >= 60 and len(bench_ret) >= len(win)
            and np.isfinite(np.asarray(bench_ret[-len(win):], dtype="float64")).any()):
        b = np.asarray(bench_ret[-len(win):], dtype="float64")
        bc = b - np.nanmean(b)
        bvar = float(np.nanmean(bc * bc))
        # A market with no dispersion leaves beta undefined; the factor stays NaN
        # rather than vanishing, so the column is always present for the model.
        if bvar > 1e-12:
            beta_m = win.mul(bc, axis=0).mean() / bvar
            resid = win.sub(np.outer(b, beta_m.to_numpy()), fill_value=np.nan)
            resid.columns = win.columns
            # 252 to 21 sessions back, off the tail, whatever the window holds.
            mom_win = resid.tail(252)
            out["resid_mom"] = mom_win.iloc[:-21].sum(axis=0)

            # -- the rest of the residual block ---------------------------
            # One regression, four factors. Everything below is a moment of the
            # SAME residual series, so computing it here costs nothing and
            # guarantees the four cannot drift apart in definition.
            r_tail = resid.tail(RESID_WINDOW)
            sd = r_tail.std(ddof=1)

            # Idiosyncratic volatility. Ang, Hodrick, Xing & Zhang (2006), and
            # in India one of the lottery-demand measures that is NOT subsumed
            # by the others.
            out["idio_vol"] = sd * np.sqrt(252)

            # Idiosyncratic skewness. Positive skew is what a lottery buyer is
            # paying for, and it is priced negatively.
            demeaned = r_tail.sub(r_tail.mean())
            m3 = (demeaned ** 3).mean()
            out["idio_skew"] = m3 / (sd.replace(0.0, np.nan) ** 3)

            # Residual reversal, standardised by its own trailing dispersion.
            # Blitz, Huij, Lansdorp & Martens (2013): conventional short-term
            # reversal carries dynamic exposure to the market and size factors,
            # and reversal built on RESIDUAL returns avoids them and earns
            # roughly twice the risk-adjusted return. The sign is the raw
            # factor's: a name that has run up over the last month is expected
            # to give some back.
            # Standardised by the trailing 36-month residual dispersion where
            # the history is there, and by whatever is there when it is not.
            long_sd = resid.tail(REVERSAL_STD_WINDOW).std(ddof=1)
            recent = resid.tail(21).sum(axis=0)
            out["resid_reversal"] = recent / long_sd.replace(0.0, np.nan)

    r120 = ret.tail(120)
    # Lengths must match before they are masked together. When the benchmark is
    # shorter than the window, both are cut to the overlap rather than left to
    # broadcast -- an unequal `&` raises, and a silent pad would align a stock's
    # returns against the wrong sessions.
    bench = np.asarray(bench_ret[-len(r120):], dtype="float64")
    if len(bench) < len(r120):
        r120 = r120.tail(len(bench))
    betas = {}
    for s in r120.columns:
        y = r120[s].to_numpy(dtype="float64")
        mask = np.isfinite(y) & np.isfinite(bench)
        if mask.sum() < 40:
            betas[s] = np.nan
            continue
        b, _ = _ols_beta_resid(y[mask], bench[mask])
        betas[s] = b
    out["beta_120"] = pd.Series(betas)

    t60 = tno.tail(60)
    absret = ret.tail(60).abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        illiq = (absret / t60.replace(0, np.nan)).mean() * 1e9
    out["amihud"] = illiq
    out["turnover_ratio"] = np.log1p(t60.mean())

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


def liquidity_mask(
    close: pd.DataFrame,
    turnover: pd.DataFrame,
    *,
    min_adtv_inr: float,
    lookback_sessions: int,
    max_names: int,
    min_history_sessions: int,
    min_price_inr: float,
) -> pd.DataFrame:
    """Which names the liquidity screen would have admitted on each date.

    A boolean frame shaped like ``close``. Mirrors
    ``UniverseResolver.resolve_liquidity_pit`` -- median turnover over the
    trailing window, a price floor, a listing-history floor, then the top
    ``max_names`` by turnover -- computed across every date at once instead of
    one date at a time.

    WHY THIS EXISTS. The panel was built from ONE universe: the names eligible
    on the most recent session, projected backwards over every training date.
    Measured against the screen resolved properly per date, that set is wrong in
    both directions and by a lot:

        as of 2024-08-12   750 eligible, 203 of them (27%) absent from today's
        as of 2021-07-19   523 eligible, 148 of them (28%) absent from today's

    A name eligible in 2024 that has since fallen out contributed no training
    row -- it was excluded for what happened afterwards. A name eligible today
    that was not eligible in 2024 contributed 2024 rows it could never have been
    traded on. That is look-ahead selection on both sides, in the training set
    and in the validation built from it.

    History counts SESSIONS SINCE THE FIRST PRINT, not prints, which is what
    ``UniverseResolver._listed_at_least`` means by listed history. Counting
    prints instead penalises a name for a suspension it has since come back
    from, and the two disagreed on 10-15% of the universe.
    """
    # min_periods=1 matches the resolver, which takes the median of whatever
    # rows the window holds and sets no minimum count. Requiring a full window
    # here would apply a stricter screen to training than to the decision.
    adtv = turnover.rolling(int(lookback_sessions), min_periods=1).median()
    listed = close.notna().cummax().cumsum()
    ok = (
        (adtv >= float(min_adtv_inr))
        & (close >= float(min_price_inr))
        & (listed >= int(min_history_sessions))
    )
    # The cap is a ranking, so it has to be applied per date rather than
    # globally. `first` breaks ties deterministically.
    rank = adtv.where(ok).rank(axis=1, ascending=False, method="first")
    return (ok & (rank <= int(max_names))).fillna(False)


#: Below this a sector is too thin to rank within: a bucket of three names
#: produces ranks of -1, 0 and +1 regardless of what the values were. Names in
#: such a sector fall back to the whole-universe rank.
MIN_SECTOR_NAMES = 12


def sector_neutral_rank(
    values: pd.Series, sectors: Optional[pd.Series] = None
) -> pd.Series:
    """Cross-sectional rank taken WITHIN sector where the sector is big enough.

    Ranking a value ratio across the whole market compares a bank's book-to-price
    with an IT services firm's, and the difference between those is an accounting
    convention rather than a signal. Every factor ranked that way picks up an
    unintended sector bet: the model ends up long whichever sector happens to
    screen cheap or fast, which is not what any of these factors claim to
    measure.

    A sector under ``MIN_SECTOR_NAMES`` is not ranked within -- three names give
    ranks of -1, 0 and +1 whatever the values were -- and falls back to the
    universe rank. A name with no sector does the same, which is common here:
    the point-in-time universe reaches past any index constituent file, so
    sectors are genuinely absent for part of it.
    """
    universe = cross_sectional_rank(values)
    if sectors is None:
        return universe
    sectors = sectors.reindex(values.index)
    out = universe.copy()
    for name, idx in values.groupby(sectors, observed=True).groups.items():
        if name is None or str(name) in ("", "Unknown"):
            continue
        if len(idx) < MIN_SECTOR_NAMES:
            continue
        out.loc[idx] = cross_sectional_rank(values.loc[idx])
    return out

def build_panel(
    close: pd.DataFrame,
    turnover: pd.DataFrame,
    horizon: int,
    step: int = 21,
    min_names: int = 40,
    delivery: Optional[pd.DataFrame] = None,
    eligible: Optional[pd.DataFrame] = None,
    sectors: Optional[Dict[str, str]] = None,
    barriers: Optional["BarrierSpec"] = None,
    high: Optional[pd.DataFrame] = None,
    low: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Assemble the panel. One row per (date, symbol).

    ``label`` is the forward return from i to i+horizon, so it uses only prices
    strictly after the feature date.

    ``eligible`` is a boolean frame shaped like ``close`` saying which names the
    universe screen would have admitted on each date -- see
    :func:`liquidity_mask`. Without it the panel is drawn from whatever columns
    ``close`` happens to carry on every date, which in practice meant today's
    universe projected backwards: names excluded for what happened after the
    date, and names included on dates they could not have been traded on.

    The benchmark is built from the eligible names too. An equal-weight mean
    over today's survivors is not the market as it stood.
    """
    dates = list(close.index)
    if eligible is not None:
        eligible = eligible.reindex(index=close.index,
                                    columns=close.columns).fillna(False)
        bench_src = close.where(eligible)
    else:
        bench_src = close
    bench_full = bench_src.mean(axis=1).pct_change(fill_method=None).to_numpy(dtype="float64")
    rows: List[pd.DataFrame] = []
    for i in range(MIN_LOOKBACK, len(dates) - horizon, step):
        feats = _features_at(close, turnover, i, bench_full[: i + 1], delivery=delivery)
        if barriers is not None:
            # The label is the trade the engine would actually have taken:
            # whichever of the profit, stop or time barrier is touched first.
            # The horizon return is blind to the path and books a name that
            # fell 20% and recovered by day 63 as a winner the engine was
            # stopped out of in week two.
            bars = triple_barrier(close, i, barriers, high=high, low=low)
            feats = feats.assign(label=bars["ret"].reindex(feats.index),
                                 barrier_side=bars["side"].reindex(feats.index),
                                 held=bars["held"].reindex(feats.index),
                                 t0=float(i),
                                 t1=bars["t1"].reindex(feats.index))
        else:
            fwd = close.iloc[i + horizon] / close.iloc[i] - 1.0
            feats = feats.assign(label=fwd, barrier_side=np.nan,
                                 held=float(horizon), t0=float(i),
                                 t1=float(i + horizon))
        if eligible is not None:
            # Point-in-time: only the names the screen admitted on THIS date.
            feats = feats[eligible.iloc[i].reindex(feats.index).fillna(False).to_numpy()]
        feats = feats[np.isfinite(feats["label"]) & feats["label"].abs().lt(1.0)]
        required = [c for c in FEATURES if c not in NEUTRAL_WHEN_MISSING]
        feats = feats.dropna(subset=required, thresh=int(len(required) * 0.7))
        if len(feats) < min_names:
            continue
        feats["date"] = dates[i]
        feats["symbol"] = feats.index
        if sectors is not None:
            feats["sector"] = feats.index.map(sectors)
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
    # How much of its own span each label holds alone. Consecutive rows share
    # most of their outcome window, so an unweighted fit counts one market
    # shock once per overlapping row.
    if {"t0", "t1"} <= set(panel.columns):
        # WITHIN each symbol. Overlap is a label sharing its outcome window
        # with the SAME name's other labels. Thirty names on one date are
        # thirty correlated observations, not a thirtieth of one -- pooling
        # them into the concurrency count returned a uniqueness of 0.014 and
        # would have thrown away almost the whole panel.
        panel["uniqueness"] = np.nan
        for _, g in panel.groupby("symbol", sort=False, observed=True):
            panel.loc[g.index, "uniqueness"] = average_uniqueness(
                g["t0"].to_numpy("float64"), g["t1"].to_numpy("float64"),
                len(dates))
    panel["label_rank"] = panel.groupby("date")["label"].transform(cross_sectional_rank)
    # Ranked WITHIN sector where the sector is big enough. Ranking across the
    # whole market compares a bank's leverage with an IT firm's, and every
    # factor then carries an unintended sector bet on top of what it measures.
    has_sector = "sector" in panel.columns and panel["sector"].notna().any()
    for f in FEATURES:
        if has_sector:
            # Built group by group into one flat Series. `groupby().apply()`
            # returns a DataFrame when the groups line up, which then cannot be
            # assigned to a single column.
            r = pd.Series(np.nan, index=panel.index, dtype="float64")
            for _, g in panel.groupby("date", sort=False, observed=True):
                r.loc[g.index] = sector_neutral_rank(g[f], g["sector"]).to_numpy()
        else:
            r = panel.groupby("date")[f].transform(cross_sectional_rank)
        # A neutral rank is 0.0 by construction, so a name with no delivery
        # print contributes nothing to the score instead of being discarded.
        panel[f + "_r"] = r.fillna(0.0) if f in NEUTRAL_WHEN_MISSING else r
    return panel
