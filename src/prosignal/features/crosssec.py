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

from .exits import ExitRules, atr_panel, ma_panel
from .labels import (BarrierSpec, average_uniqueness, engine_barrier,
                     triple_barrier)

__all__ = ["FEATURES", "build_panel", "features_for_date",
           "cross_sectional_rank",
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
    "prox_52w":      (274, "close 21 back / 252-session high ending 21 back - 1 (George & Hwang 2004, with a reversal-avoiding skip)"),
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

    # SKIPPING THE LAST 21 SESSIONS, which George & Hwang (2004) do NOT do --
    # they measure nearness on the current price. The deviation is deliberate
    # and the reason is `resid_reversal` sitting in the same model: using
    # today's close puts the last month inside prox_52w, which is exactly the
    # window the reversal theme prices with the OPPOSITE sign, so momentum and
    # reversal partially cancel through this factor. Measured within date the
    # two correlate +0.378 as shipped and -0.029 with the skip. mom_6_1 and
    # resid_mom already skip the same window for the same reason (Jegadeesh
    # 1990; Lehmann 1990).
    #
    # hist.iloc[-273:-21] is the 252 sessions ending one month back. This needs
    # 273 rows, so FEATURES["prox_52w"] is 274 rather than 253 -- see there.
    out["prox_52w"] = past(21) / hist.iloc[-273:-21].max() - 1.0
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
            # THE REGRESSION HAS AN INTERCEPT. It did not: the residual was
            # `r - beta*b` with `b` in raw form, so its mean over the window is
            # the name's own alpha and every subsequent sum accumulates 231
            # copies of it. What was called "residual momentum" was therefore
            # raw momentum with a beta-times-market term removed -- the name's
            # drift, which is the component the construction exists to strip,
            # survived in full.
            #
            # With an intercept, eps = (r - rbar) - beta*(b - bbar), which is
            # the textbook OLS residual and is mean-zero over the estimation
            # window by construction. The estimation window (756 sessions) is
            # deliberately longer than the momentum window (231), exactly as in
            # the cited paper, so the sub-window sum is not forced to zero.
            #
            # A CONSEQUENCE WORTH STATING. With an intercept, a name whose
            # residual drifts at a constant rate across the whole estimation
            # window has that drift absorbed into alpha and scores zero. The
            # factor prices RECENT residual out-performance against the name's
            # own long-run norm, which is what the paper intends and what the
            # no-intercept version could not express.
            resid = (win.sub(win.mean(axis=0), axis=1)
                        .sub(np.outer(bc, beta_m.to_numpy())))
            resid.columns = win.columns
            # 252 to 21 sessions back, off the tail, whatever the window holds.
            mom_win = resid.tail(252)
            formation = mom_win.iloc[:-21]
            # STANDARDISED BY THE RESIDUAL'S OWN DISPERSION over the same
            # formation window. Blitz, Huij & Martens (2011) divide the
            # cumulative residual by the standard deviation of the residual
            # returns over that period; omitting it left the factor scaling
            # with volatility, which pulled it into the lottery block it is
            # supposed to be orthogonal to.
            f_sd = formation.std(ddof=1)
            # A name that is EXACTLY a linear function of the market has no
            # residual to accumulate and no dispersion to divide by: the
            # standardised factor is 0/0 and its honest value is undefined, not
            # a large number produced by float noise. The test is dimensionless
            # -- residual dispersion as a fraction of the name's own -- so it
            # catches numerical degeneracy without imposing a scale. Real names
            # sit around R^2 = 0.3; only an exact multiple of the index reaches
            # this branch.
            raw_sd = win.tail(252).iloc[:-21].std(ddof=1)
            degenerate = ~(f_sd > 1e-10 * raw_sd.abs())
            out["resid_mom"] = (formation.sum(axis=0)
                                / f_sd.mask(degenerate | (f_sd == 0.0)))

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


#: Distinguishes "the caller has no adjustment factors" from "the caller forgot
#: to pass them". The price floor is a look-ahead trap when it is applied to a
#: back-adjusted series, and a silent default is how it stayed one.
NO_ADJUSTMENT = object()


def liquidity_mask(
    close: pd.DataFrame,
    turnover: pd.DataFrame,
    *,
    min_adtv_inr: float,
    lookback_sessions: int,
    max_names: int,
    min_history_sessions: int,
    min_price_inr: float,
    adj_factor: object = NO_ADJUSTMENT,
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

    # THE PRICE FLOOR IS A TRADEABILITY TEST AND MUST READ THE QUOTED PRICE.
    # `close` is back-adjusted on every store read, so a name that traded at
    # Rs 200 in 2019 and later split 1:20 reads Rs 10 on that 2019 date and
    # fails a floor it comfortably cleared at the time. Membership in a past
    # universe then depends on a corporate action nobody had announced yet.
    #
    # Measured against the raw bhavcopy close over 3,848,322 (date, symbol)
    # cells: 58,411 excluded that the raw series admits, across 165 symbols, and
    # ZERO in the other direction. 4,905 of them clear every other floor, so the
    # universe genuinely grows -- ADANIPOWER, BEL, CANBK, ASHOKLEY among them.
    # The bias runs one way and removes names that later split.
    #
    # `adj_factor` multiplies PRE-ex-date prices (0.5 for a 1:1 bonus), so the
    # quoted price is the adjusted one divided by it.
    price = close
    if adj_factor is not NO_ADJUSTMENT and adj_factor is not None:
        fac = adj_factor.reindex(index=close.index, columns=close.columns)
        price = close.divide(fac.where(fac > 0))
    ok = (
        (adtv >= float(min_adtv_inr))
        & (price >= float(min_price_inr))
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

    EVERY NAME IS RANKED WITHIN A GROUP. A sector under ``MIN_SECTOR_NAMES`` is
    not ranked within -- three names give ranks of -1, 0 and +1 whatever the
    values were -- and neither is a name with no sector at all, which is common
    here: the point-in-time universe reaches past any index constituent file, so
    sectors are genuinely absent for part of it. Those names form ONE RESIDUAL
    GROUP, `UNCLASSIFIED`, and are ranked within that.

    WHY, AND WHAT THIS REPLACES. The fallback used to be the UNIVERSE rank, so
    a single column carried two different quantities: "where you sit among your
    peers" for a sectored name and "where you sit in the whole market" for the
    rest. Measured on the shipped panel, 58.0% of rows carried a sector label
    and a median 46.0% of names per date were ranked within one -- so roughly
    half of every cross-section was on the other scale. A within-sector rank of
    +0.9 in a fourteen-name sector can belong to a name that is unremarkable
    market-wide; a universe rank of +0.9 cannot. Both were averaged into the
    same family aggregate and handed to the same regression, whose design
    column was therefore not one variable. On the last panel date 49% of the
    top `mom_f` decile came from the 45% of names that were NOT within-sector
    ranked.

    A residual group is heterogeneous, and that is a real cost. It is a smaller
    one than mixing two normalisations: the column now means one thing
    everywhere, and the heterogeneity is visible in the group's name rather
    than hidden in a fallback branch. The universe rank survives only where the
    residual group is itself too small to rank within, which on this universe
    means it barely exists.

    Not done: dropping unsectored names, which discards 42% of the universe and
    selects on which names a vendor file happens to cover; nor lowering
    ``MIN_SECTOR_NAMES``, which buys coverage with noise dressed as neutrality.
    """
    universe = cross_sectional_rank(values)
    if sectors is None:
        return universe
    sectors = sectors.reindex(values.index)
    out = universe.copy()
    residual = []
    for name, idx in values.groupby(sectors, observed=True).groups.items():
        if name is None or str(name) in ("", "Unknown") or len(idx) < MIN_SECTOR_NAMES:
            residual.extend(list(idx))
            continue
        out.loc[idx] = cross_sectional_rank(values.loc[idx])
    # Names the grouping dropped entirely -- a NaN sector is not a group key.
    missing = values.index.difference(
        pd.Index([i for name, idx in values.groupby(sectors, observed=True).groups.items()
                  for i in idx]))
    residual.extend(list(missing))
    if residual:
        resid = pd.Index(residual).unique()
        if len(resid) >= MIN_SECTOR_NAMES:
            out.loc[resid] = cross_sectional_rank(values.loc[resid])
        # else: too few to rank within, and they keep the universe rank. This
        # is the one surviving mixed case and it is bounded by 11 names.
    return out


def sector_rank_coverage(sectors: pd.Series) -> Dict[str, float]:
    """How much of a cross-section is ranked within a REAL sector.

    The 58% figure that opened this finding was only discoverable by
    instrumenting the code from outside. A property that decides what a whole
    feature column means should be readable from the inside.
    """
    s = sectors.dropna()
    n = int(len(sectors))
    if n == 0:
        return {"n": 0, "within_sector": 0.0, "unclassified": 0.0, "n_sectors": 0}
    counts = s[~s.astype(str).isin(("", "Unknown"))].value_counts()
    real = counts[counts >= MIN_SECTOR_NAMES]
    within = int(real.sum())
    return {
        "n": n,
        "within_sector": within / n,
        "unclassified": (n - within) / n,
        "n_sectors": int(len(real)),
    }

#: How much history the LIVE feature row reads. `resid_reversal` standardises by
#: the trailing residual dispersion over REVERSAL_STD_WINDOW where it exists, so
#: a live row built off MIN_LOOKBACK sessions computes a different statistic from
#: the training rows it is scored against. Ranking is cross-sectional and absorbs
#: most of that, but "most" is not "all" and the fix costs one wider read.
LIVE_HISTORY_SESSIONS = max(MIN_LOOKBACK, REVERSAL_STD_WINDOW) + 1


def features_for_date(
    close: pd.DataFrame,
    turnover: pd.DataFrame,
    delivery: Optional[pd.DataFrame] = None,
    sectors: Optional[Dict[str, str]] = None,
    min_names: int = 40,
    eligible: Optional[pd.Series] = None,
    admissible: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """The feature row for the LAST session in ``close``. One date, no label.

    WHY THIS EXISTS. Both live scoring paths used to reach the decision date
    through `build_panel`:

        live_hist = hist.tail(MIN_LOOKBACK + 5)              # 279 rows
        live      = build_panel(live_hist, horizon=1, step=21)

    `build_panel` is a TRAINING panel builder. Its loop is

        for i in range(MIN_LOOKBACK, len(dates) - horizon, step)

    and the `- horizon` bound exists to guarantee every row has a label. On a
    279-row frame with horizon=1 and step=21 the only reachable i is 274, which
    is FOUR ROWS BEFORE THE END -- and at any horizon >= 1 the bound makes the
    last row structurally unreachable. Verified live: a run as_of 2026-08-25
    scored features dated 2026-08-19.

    Measured over 88 panel dates, the cost of that staleness was small
    statistically (rank IC +0.0751 -> +0.0730, top-decile excess +0.98% ->
    +0.83%, both inside the noise band) and large operationally: only 64% of the
    top eight names agreed with the top eight the same model produces on the
    decision date. Roughly three names on every card were four sessions old,
    priced at today's close with today's ATR stop attached.

    A decision row wants no label, no stride and no forward bound, so it does not
    go through a builder that has all three. The ranking columns are built by the
    SAME `_features_at` and the SAME `sector_neutral_rank` the panel uses, so
    training and inference cannot drift apart in definition.

    ``eligible`` is the screen for THIS date, as a boolean Series over symbols.
    ``admissible`` narrows the ROWS further -- to the names Stage 6 can open --
    without narrowing the BENCHMARK. The two are deliberately separate: in
    `build_panel` the equal-weight market is `close.where(eligible)`, computed
    before the admissibility mask, so folding both into one argument here would
    measure beta and residual momentum against a different market live than in
    training. That mismatch was introduced while fixing the population and caught
    by re-reading the fix against the builder it has to match.
    """
    if close.empty or len(close.index) == 0:
        return pd.DataFrame()
    i = len(close.index) - 1
    if i < MIN_LOOKBACK:
        return pd.DataFrame()

    # THE MARKET, from the eligible universe only -- exactly as `build_panel`
    # forms it. Never narrowed by `admissible`.
    #
    # Broadcast to a frame rather than passing the Series to `where(axis=1)`:
    # that overload raises on a column-indexed condition, which made the
    # eligible-benchmark path a latent crash.
    if eligible is None:
        bench_src = close
    else:
        col_ok = eligible.reindex(close.columns).fillna(False).to_numpy(dtype=bool)
        bench_src = close.where(pd.DataFrame(
            np.broadcast_to(col_ok, close.shape),
            index=close.index, columns=close.columns))
    bench = bench_src.mean(axis=1).pct_change(fill_method=None).to_numpy("float64")

    feats = _features_at(close, turnover, i, bench[: i + 1], delivery=delivery)
    if eligible is not None:
        feats = feats[eligible.reindex(feats.index).fillna(False).to_numpy()]
    if admissible is not None:
        feats = feats[admissible.reindex(feats.index).fillna(False).to_numpy()]
    # The SAME completeness rule the panel applies. No label filter: a name is
    # not excluded from today's ranking because a past session did not print.
    required = [c for c in FEATURES if c not in NEUTRAL_WHEN_MISSING]
    feats = feats.dropna(subset=required, thresh=int(len(required) * 0.7))
    if len(feats) < min_names:
        return pd.DataFrame()

    feats = feats.copy()
    feats["date"] = close.index[i]
    feats["symbol"] = feats.index
    if sectors is not None:
        feats["sector"] = feats.index.map(sectors)
    has_sector = "sector" in feats.columns and feats["sector"].notna().any()
    for f in FEATURES:
        r = (sector_neutral_rank(feats[f], feats["sector"]) if has_sector
             else cross_sectional_rank(feats[f]))
        feats[f + "_r"] = r.fillna(0.0) if f in NEUTRAL_WHEN_MISSING else r
    return feats.reset_index(drop=True)


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
    exit_rules: Optional["ExitRules"] = None,
    high: Optional[pd.DataFrame] = None,
    low: Optional[pd.DataFrame] = None,
    open_: Optional[pd.DataFrame] = None,
    admissible: Optional[pd.DataFrame] = None,
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
    # ATR and the invalidation moving average are computed ONCE for the whole
    # panel rather than per date. They are the engine's own, from the same
    # `true_range` Stage 7 uses.
    atr = ma = None
    if exit_rules is not None and high is not None and low is not None:
        atr = atr_panel(high, low, close, exit_rules.atr_period_sessions,
                        exit_rules.atr_method)
        ma = ma_panel(close, exit_rules.invalidation_ma_sessions)
    rows: List[pd.DataFrame] = []
    for i in range(MIN_LOOKBACK, len(dates) - horizon, step):
        feats = _features_at(close, turnover, i, bench_full[: i + 1], delivery=delivery)
        if exit_rules is not None:
            # The label is the trade the engine would actually have taken, in
            # the ENGINE'S OWN geometry: its ATR stop, its 3R target, its
            # thesis-invalidation exit. Sigma barriers described a trade with a
            # 1.33:1 reward-to-risk profile that this engine never takes.
            bars = engine_barrier(close, i, exit_rules, high=high, low=low,
                                  open_=open_, atr=atr, ma=ma)
            feats = feats.assign(label=bars["ret"].reindex(feats.index),
                                 barrier_side=bars["side"].reindex(feats.index),
                                 held=bars["held"].reindex(feats.index),
                                 t0=float(i),
                                 t1=bars["t1"].reindex(feats.index))
        elif barriers is not None:
            # Sigma barriers. Research only -- kept so the shipped geometry can
            # be measured against the one it replaced.
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
        if admissible is not None:
            # THE POPULATION THE ENGINE CAN ACTUALLY BUY FROM. Stage 6 refuses a
            # name already below its own invalidation level -- it satisfies its
            # first exit condition at the moment it is opened. While the label
            # was a triple barrier, `resolve_exits` applied the same predicate
            # and training and admission agreed. Turning the barrier off removed
            # it from training and left the live gate standing, so the model
            # began ranking a population 23.3% of which it could never buy, and
            # 1.55 of its top eight were refused on 72% of dates.
            #
            # Ranks are taken AFTER this mask, so a rank means the same thing in
            # training and at the decision.
            feats = feats[admissible.iloc[i].reindex(feats.index).fillna(False).to_numpy()]
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
