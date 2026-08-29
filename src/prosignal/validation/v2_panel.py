"""Research panel for the v2 scorer, and the quarterly re-check.

The panel is one row per (signal date, symbol), built with the SAME
`features.v2.factor_frame` the live path calls -- there is no second
implementation to drift from. The universe is resolved per date by the same
liquidity screen Stage 3 applies, so a name appears on a date only if it could
have been traded on it.

THE RE-CHECK IS THE SAME DISCIPLINE AS THE ORIGINAL DEPLOY. The most recent
`holdout_months` of signal dates are sealed, everything before them is the
comparison window, and the shipped configuration is applied to the sealed part
ONCE. Nothing is fitted here -- the signs and weights are frozen in
`features/v2.py` -- so the re-check answers one question: is the ranking still
doing on new data what it did on the holdout that earned the deploy.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..features import v2
from ..features.crosssec import liquidity_mask

__all__ = ["build_v2_panel", "quintile_spread", "rank_ic", "recheck",
           "V2Recheck", "SIGNAL_STRIDE_SESSIONS", "LABEL_HORIZON_SESSIONS"]

#: Signal dates are one trading week apart, matching the shipped refresh.
SIGNAL_STRIDE_SESSIONS = 5
#: The label the scorer was selected and measured against.
LABEL_HORIZON_SESSIONS = 42


def _pivot(df: pd.DataFrame, col: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty or col not in df.columns:
        return None
    return df.pivot_table(index="date", columns="symbol", values=col,
                          aggfunc="last", observed=True).sort_index()


def build_v2_panel(store, *, start: Optional[dt.date] = None,
                   end: Optional[dt.date] = None,
                   stride: int = SIGNAL_STRIDE_SESSIONS,
                   horizons: Sequence[int] = (21, LABEL_HORIZON_SESSIONS),
                   max_names: int = 750, min_adtv_inr: float = 5e7,
                   min_price_inr: float = 20.0,
                   min_history_sessions: int = 300) -> pd.DataFrame:
    """One row per (signal date, symbol), with factor ranks, score and labels.

    THE LABEL IS THE TRADE, NOT THE TAPE. Entry is the VWAP of the session
    AFTER the signal and exit is the VWAP `h` sessions later, because that is
    the manual next-session execution the product asks of its user. A
    close-to-close label would report an edge nobody could have taken.
    """
    px = store.read_prices(start=start, end=end)
    if px is None or px.empty:
        return pd.DataFrame()
    px = px.copy()
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    close = _pivot(px, "close")
    open_ = _pivot(px, "open")
    turnover = _pivot(px, "turnover")
    vwap = _pivot(px, "vwap")
    if close is None or turnover is None:
        return pd.DataFrame()
    fill = (vwap.where(vwap > 0) if vwap is not None else None)
    fill = close if fill is None else fill.fillna(open_ if open_ is not None else close).fillna(close)

    deliv = None
    try:
        dl = store.read_delivery(start=start, end=end)
        if dl is not None and not dl.empty:
            dl = dl.copy()
            dl["date"] = pd.to_datetime(dl["date"]).dt.normalize()
            deliv = _pivot(dl, "deliv_pct")
            if deliv is not None:
                deliv = deliv.reindex(index=close.index, columns=close.columns)
    except Exception:
        deliv = None

    try:
        smap = store.read_sector_map()
        sectors = (dict(zip(smap["symbol"], smap["sector"]))
                   if smap is not None and not smap.empty else {})
    except Exception:
        sectors = {}

    adj = _pivot(px, "adj_factor")
    if adj is None:
        adj = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    elig = liquidity_mask(close, turnover, min_adtv_inr=min_adtv_inr,
                          lookback_sessions=60, max_names=max_names,
                          min_history_sessions=min_history_sessions,
                          min_price_inr=min_price_inr, adj_factor=adj)
    adtv = turnover.rolling(60, min_periods=1).median()
    adtv_rank = adtv.where(elig).rank(axis=1, ascending=False, method="first")

    dates = list(close.index)
    T = len(dates)
    lo = v2.MIN_LOOKBACK_SESSIONS
    hi = T - min(horizons) - 2
    rows: List[pd.DataFrame] = []
    for i in range(lo, max(hi, lo), stride):
        sel = elig.iloc[i]
        syms = list(sel[sel].index)
        if len(syms) < 60:
            continue
        win = slice(max(i - lo - 20, 0), i + 1)
        raw = v2.factor_frame(close.iloc[win][syms], turnover.iloc[win][syms],
                              open_.iloc[win][syms] if open_ is not None else None,
                              deliv.iloc[win][syms] if deliv is not None else None)
        if raw.empty:
            continue
        scored = v2.score_frame(raw, sectors)
        blk = pd.concat([raw, scored], axis=1)
        blk["date"] = dates[i]
        blk["symbol"] = blk.index
        blk["sector"] = [sectors.get(s, "UNCLASSIFIED") for s in blk.index]
        blk["adtv"] = adtv.iloc[i].reindex(blk.index).to_numpy()
        blk["adtv_rank"] = adtv_rank.iloc[i].reindex(blk.index).to_numpy()
        entry = fill.iloc[i + 1].reindex(blk.index).to_numpy("float64")
        blk["entry_px"] = entry
        for h in horizons:
            j = i + 1 + h
            if j >= T:
                blk[f"y{h}"] = np.nan
                continue
            ex = fill.iloc[j].reindex(blk.index).to_numpy("float64")
            with np.errstate(invalid="ignore", divide="ignore"):
                blk[f"y{h}"] = ex / np.where(entry > 0, entry, np.nan) - 1.0
        rows.append(blk.reset_index(drop=True))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def rank_ic(panel: pd.DataFrame, label: str, score: str = "score") -> Tuple[float, float, int]:
    ics = []
    for _, g in panel.groupby("date", sort=True):
        a = g[score].to_numpy("float64"); b = g[label].to_numpy("float64")
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 60:
            continue
        ra = pd.Series(a[m]).rank().to_numpy(); rb = pd.Series(b[m]).rank().to_numpy()
        if ra.std() < 1e-12 or rb.std() < 1e-12:
            continue
        ics.append(float(np.corrcoef(ra, rb)[0, 1]))
    ics = np.asarray(ics)
    if len(ics) < 5:
        return float("nan"), float("nan"), len(ics)
    return (float(ics.mean()),
            float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))), len(ics))


def quintile_spread(panel: pd.DataFrame, label: str, q: int = 5,
                    score: str = "score") -> Tuple[float, float, int]:
    """Top-fifth minus bottom-fifth realised return, per period.

    THE STATISTIC WITH POWER, and the reason it is the headline rather than the
    book's annual excess: a permuted-label test run before the original deploy
    put a ten-name book's five-year excess almost entirely inside its own null,
    while this spread sat six standard deviations outside it. Judge the scorer
    on the number that can tell signal from noise.
    """
    sp = []
    for _, g in panel.groupby("date", sort=True):
        g = g.dropna(subset=[score, label])
        if len(g) < 100:
            continue
        k = max(len(g) // q, 5)
        o = g.sort_values(score, ascending=False)
        sp.append(float(o[label].head(k).mean() - o[label].tail(k).mean()))
    sp = np.asarray(sp)
    if len(sp) < 5:
        return float("nan"), float("nan"), len(sp)
    return (float(sp.mean()),
            float(sp.mean() / (sp.std(ddof=1) / np.sqrt(len(sp)))), len(sp))


@dataclass
class V2Recheck:
    """One quarterly verdict. Nothing here changes the model."""
    as_of: dt.date
    holdout_start: Optional[dt.date]
    holdout_dates: int
    ic: float
    ic_t: float
    spread: float
    spread_t: float
    null_p_spread: float
    reference: Dict[str, float]
    verdict: str
    note: str
    factor_health: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        d = {k: v for k, v in self.__dict__.items()}
        d["as_of"] = self.as_of.isoformat()
        d["holdout_start"] = self.holdout_start.isoformat() if self.holdout_start else None
        return d


#: Independent label windows the re-check needs before it issues a verdict.
#: The deploy was judged on 8.6 of them; a quarterly window holds about 1.5.
MIN_INDEPENDENT_WINDOWS = 8.0

#: What the deploy earned, on 2025-03-06 to 2026-08-17, evaluated once.
DEPLOY_REFERENCE = {"ic": 0.0451, "ic_t": 2.59, "spread": 0.0165, "spread_t": 2.56,
                    "book_excess_ann": 0.0241, "book_maxdd": -0.140}


def recheck(panel: pd.DataFrame, *, holdout_months: int = 3,
            label: str = f"y{LABEL_HORIZON_SESSIONS}",
            n_null: int = 200, seed: int = 0) -> V2Recheck:
    """Apply the frozen scorer to the most recent `holdout_months` of dates.

    The shuffled-score null permutes the score WITHIN each cross-section, which
    breaks the score/outcome link and preserves everything else, and is the
    reference the original holdout was judged against.
    """
    from .. import v2_monitor as vm

    if panel is None or panel.empty:
        return V2Recheck(dt.date.today(), None, 0, float("nan"), float("nan"),
                         float("nan"), float("nan"), float("nan"),
                         DEPLOY_REFERENCE, "NO_DATA",
                         "the panel is empty; nothing to re-check")
    dates = np.array(sorted(pd.to_datetime(panel["date"]).unique()))
    end = pd.Timestamp(dates[-1])
    cut = end - pd.DateOffset(months=int(holdout_months))
    ho = panel[pd.to_datetime(panel["date"]) >= cut]
    n_dates = int(pd.to_datetime(ho["date"]).nunique()) if len(ho) else 0
    # HOW MANY OF THOSE DATES ARE INDEPENDENT. Signal dates are five sessions
    # apart and the label spans forty-two, so consecutive dates share five
    # sixths of their outcome window. The count that matters is
    # dates x stride / horizon, and it is the count the ORIGINAL deploy was
    # judged on -- 72 dates, 8.6 independent windows. A quarterly window holds
    # about 1.5 of them.
    #
    # So the re-check RUNS quarterly and issues a VERDICT when it has as much
    # evidence as the deploy had, and says so plainly in between. A quarterly
    # pass/fail on 1.5 independent observations would be a coin flip with a
    # decimal point on it, and worse, it would be one the engine invited
    # somebody to act on.
    indep = n_dates * SIGNAL_STRIDE_SESSIONS / float(LABEL_HORIZON_SESSIONS)
    if indep < MIN_INDEPENDENT_WINDOWS:
        ic0, ic0_t, _ = (rank_ic(ho, label) if n_dates >= 5
                         else (float("nan"), float("nan"), 0))
        sp0, sp0_t, _ = (quintile_spread(ho, label) if n_dates >= 5
                         else (float("nan"), float("nan"), 0))
        need = int(np.ceil(MIN_INDEPENDENT_WINDOWS * LABEL_HORIZON_SESSIONS
                           / SIGNAL_STRIDE_SESSIONS))
        return V2Recheck(
            end.date(), cut.date(), n_dates, ic0, ic0_t, sp0, sp0_t, float("nan"),
            DEPLOY_REFERENCE, "TOO_EARLY",
            f"{n_dates} scored dates is {indep:.1f} independent "
            f"{LABEL_HORIZON_SESSIONS}-session windows, against the "
            f"{MIN_INDEPENDENT_WINDOWS:.0f} the deploy itself was judged on. "
            f"Running numbers are shown above and are NOT a verdict; about "
            f"{need} dates are needed. Widen --holdout-months to accumulate, "
            f"and read the factor health below, which needs less evidence to "
            f"be worth acting on.")
    ic, ic_t, _ = rank_ic(ho, label)
    sp, sp_t, _ = quintile_spread(ho, label)

    rng = np.random.default_rng(seed)
    d = ho.copy()
    idx = d.groupby("date", sort=False).indices
    base = d["score"].to_numpy("float64").copy()
    nulls = []
    for _ in range(int(n_null)):
        s2 = base.copy()
        for ii in idx.values():
            s2[ii] = rng.permutation(s2[ii])
        d["score"] = s2
        nulls.append(quintile_spread(d, label)[0])
    nulls = np.asarray([x for x in nulls if np.isfinite(x)])
    p = float((nulls >= sp).mean()) if len(nulls) and np.isfinite(sp) else float("nan")

    ic_frame = vm.rolling_factor_ic(ho, label)
    health = [h.to_dict() for h in vm.review_factors(ic_frame, window=len(ic_frame),
                                                    min_periods=min(20, n_dates))]
    inverted = [h["factor"] for h in health if h["inverted"]]

    if not np.isfinite(sp):
        verdict, note = "NO_VERDICT", "the spread could not be computed"
    elif p <= 0.05 and sp > 0:
        verdict = "HOLDS"
        note = (f"quintile spread +{sp:.2%} per {LABEL_HORIZON_SESSIONS} sessions "
                f"(t {sp_t:.2f}), outside a {len(nulls)}-draw shuffled null at "
                f"p={p:.3f}. The deploy earned +{DEPLOY_REFERENCE['spread']:.2%} "
                f"(t {DEPLOY_REFERENCE['spread_t']:.2f}).")
    elif sp > 0:
        verdict = "WEAK"
        note = (f"spread is positive (+{sp:.2%}) but inside the shuffled null "
                f"(p={p:.3f}). Not evidence of failure and not evidence of edge; "
                f"the window is short.")
    else:
        verdict = "FAILS"
        note = (f"quintile spread is NEGATIVE ({sp:.2%}, t {sp_t:.2f}) on this "
                f"window. The ranking did not order outcomes. This is the "
                f"condition the re-check exists to surface -- it does not "
                f"change the model, and the next configuration has to be "
                f"selected on training data and tested on a window this one "
                f"has not seen.")
    if inverted:
        note += f" Factors flagged as inverted: {', '.join(inverted)}."
    return V2Recheck(end.date(), cut.date(), n_dates, ic, ic_t, sp, sp_t, p,
                     DEPLOY_REFERENCE, verdict, note, health)
