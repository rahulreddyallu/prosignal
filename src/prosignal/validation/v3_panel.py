"""Research panel for the v3 thematic scorer, and the quarterly re-check.

One row per (signal date, symbol), built with the SAME `features.v3_factors`
and `features.v3` the live path calls -- there is no second implementation to
drift from. The universe is resolved per date by the liquidity screen Stage 3
applies, minus the non-equity instruments NSE publishes in the same EQ-series
bhavcopy, so a name appears on a date only if it could have been traded on it.

THE RE-CHECK IS THE SAME DISCIPLINE AS THE DEPLOY. The most recent
`holdout_months` of signal dates are sealed, everything before is comparison,
and the frozen configuration is applied to the sealed part ONCE. Nothing is
fitted here -- signs, horizons and theme weights live in `features/v3.py` and
are not touched -- so the re-check answers one question: is the ranking still
doing on new data what it did on the holdouts that earned the deploy.

WHAT IT ADDS OVER THE v2 RE-CHECK. The composite can hold up in aggregate while
one theme has inverted, and the composite's own IC will not say which; so the
verdict carries per-theme IC and per-theme influence share alongside the
headline, and a theme that has inverted or is running more of the ranking than
it was given is named in the note whatever the headline says.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..features import v3, v3_factors
from ..features.crosssec import liquidity_mask
from .metrics import quintile_spread, rank_ic

__all__ = ["build_v3_panel", "recheck", "V3Recheck",
           "SIGNAL_STRIDE_SESSIONS", "LABEL_HORIZON_SESSIONS",
           "MIN_INDEPENDENT_WINDOWS", "DEPLOY_REFERENCE"]

#: Signal dates one trading week apart, matching the shipped refresh.
SIGNAL_STRIDE_SESSIONS = 5
#: The label the COMPOSITE was measured against on both sealed windows. Themes
#: are oriented at their own horizons (42/21/21/10/10); this is the horizon the
#: blended score was judged on, and changing it would make the re-check answer
#: a different question from the one the deploy answered.
LABEL_HORIZON_SESSIONS = 21


def _pivot(df: pd.DataFrame, col: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty or col not in df.columns:
        return None
    return df.pivot_table(index="date", columns="symbol", values=col,
                          aggfunc="last", observed=True).sort_index()


def build_v3_panel(store, *, start: Optional[dt.date] = None,
                   end: Optional[dt.date] = None,
                   stride: int = SIGNAL_STRIDE_SESSIONS,
                   horizons: Sequence[int] = (LABEL_HORIZON_SESSIONS, 42),
                   max_names: int = 750, min_adtv_inr: float = 5e7,
                   min_price_inr: float = 20.0,
                   min_history_sessions: int = 300,
                   drop_non_equity: bool = True) -> pd.DataFrame:
    """Factor ranks, theme sub-scores, contributions, composite and labels.

    THE LABEL IS THE TRADE, NOT THE TAPE. Entry is the VWAP of the session AFTER
    the signal and exit the VWAP `h` sessions later, because that is the manual
    next-session execution the product asks of its user.
    """
    px = store.read_prices(start=start, end=end)
    if px is None or px.empty:
        return pd.DataFrame()
    px = px.copy()
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    close, open_ = _pivot(px, "close"), _pivot(px, "open")
    turnover, vwap = _pivot(px, "turnover"), _pivot(px, "vwap")
    if close is None or turnover is None:
        return pd.DataFrame()

    # NON-EQUITY INSTRUMENTS OUT, BEFORE ANYTHING IS RANKED. The first
    # evaluation of holdout window A did not do this and ETFs, gold funds and
    # liquid funds took 26.25% of its top-ten slots.
    if drop_non_equity:
        try:
            from ..data.instruments import non_equity_symbols
            master = None
            try:
                master = store.read_equity_master()
            except Exception:
                master = None
            drop = non_equity_symbols(list(close.columns), master, close)
            if drop:
                keep = [c for c in close.columns if c not in drop]
                close = close[keep]
                turnover = turnover.reindex(columns=keep)
                open_ = open_.reindex(columns=keep) if open_ is not None else None
                vwap = vwap.reindex(columns=keep) if vwap is not None else None
        except Exception:
            # The exclusion is a correctness fix, not a nicety, but a panel that
            # refuses to build teaches nothing. It proceeds and the ETF check in
            # the re-check note below is what would catch the consequence.
            pass

    fill = (vwap.where(vwap > 0) if vwap is not None else None)
    fill = close if fill is None else fill.fillna(
        open_ if open_ is not None else close).fillna(close)

    # DELIVERY FROM ITS OWN TABLE, never `prices.deliv_pct` -- that column is a
    # write-time placeholder and reading it would silently neutralise the whole
    # ownership theme, 19% of the composite, without erroring.
    deliv = None
    try:
        dl = store.read_delivery(start=start, end=end)
        if dl is not None and not dl.empty and "deliv_pct" in dl.columns:
            dl = dl.copy()
            dl["date"] = pd.to_datetime(dl["date"]).dt.normalize()
            deliv = _pivot(dl, "deliv_pct")
            if deliv is not None:
                deliv = deliv.reindex(index=close.index, columns=close.columns)
    except Exception:
        deliv = None

    fund_recs = None
    try:
        from ..features import pit_fundamentals as pitf
        fund_recs = pitf.build_records(store=store)
        if fund_recs is not None and fund_recs.empty:
            fund_recs = None
    except Exception:
        fund_recs = None

    try:
        smap = store.read_sector_map()
        sectors = (dict(zip(smap["symbol"], smap["sector"]))
                   if smap is not None and not smap.empty else {})
    except Exception:
        sectors = {}

    adj = _pivot(px, "adj_factor")
    if adj is None:
        adj = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    adj = adj.reindex(columns=close.columns)
    elig = liquidity_mask(close, turnover, min_adtv_inr=min_adtv_inr,
                          lookback_sessions=60, max_names=max_names,
                          min_history_sessions=min_history_sessions,
                          min_price_inr=min_price_inr, adj_factor=adj)
    adtv = turnover.rolling(60, min_periods=1).median()
    adtv_rank = adtv.where(elig).rank(axis=1, ascending=False, method="first")

    bench = close.mean(axis=1)
    bench_ret = bench / bench.shift(1) - 1.0

    dates = list(close.index)
    T = len(dates)
    lo = v3_factors.LOOKBACK_SESSIONS
    hi = T - min(horizons) - 2
    rows: List[pd.DataFrame] = []
    for i in range(lo, max(hi, lo), stride):
        sel = elig.iloc[i]
        syms = list(sel[sel].index)
        if len(syms) < 60:
            continue
        win = slice(max(i - lo - 15, 0), i + 1)
        fund = None
        if fund_recs is not None:
            try:
                from ..features import pit_fundamentals as pitf
                fs = [s for s in syms if s in set(fund_recs["symbol"])]
                if fs:
                    blk = pitf.asof_panel(fund_recs, close.index[win], fs)
                    fund = {k: v.reindex(columns=syms) for k, v in blk.items()
                            if k in ("ttm_revenue", "ttm_net_profit",
                                     "fund_age_days")}
            except Exception:
                fund = None
        raw = v3_factors.factor_frame(
            close.iloc[win][syms],
            open_.iloc[win][syms] if open_ is not None else None,
            vwap.iloc[win][syms] if vwap is not None else None,
            turnover.iloc[win][syms],
            deliv.iloc[win][syms] if deliv is not None else None,
            bench_ret.iloc[win], fund)
        if raw.empty:
            continue
        scored = v3.score_frame(raw, sectors)
        if scored.empty:
            continue
        blk = pd.concat([raw, scored], axis=1)
        blk = blk.loc[:, ~blk.columns.duplicated()]
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


@dataclass
class V3Recheck:
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
    theme_health: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        d = dict(self.__dict__)
        d["as_of"] = self.as_of.isoformat()
        d["holdout_start"] = (self.holdout_start.isoformat()
                              if self.holdout_start else None)
        return d


#: Independent label windows the re-check needs before it issues a verdict. The
#: deploy was judged on 17.1 of them across the two sealed windows; a quarterly
#: window holds about 3.
MIN_INDEPENDENT_WINDOWS = 8.0

#: THE SEALED WINDOWS, so the re-check can tell the reader when it is reading
#: them back. A re-check whose window overlaps a sealed holdout is not
#: independent evidence of anything: it is the deploy's own number arriving a
#: second time wearing a verdict. That is not a defect in the re-check -- run
#: the day after a deploy it CANNOT avoid the overlap -- but it has to be said
#: out loud, or the first quarterly run reads as confirmation.
SEALED_WINDOWS = {
    "A": (dt.date(2025, 3, 6), dt.date(2026, 8, 17)),
    "B": (dt.date(2021, 7, 1), dt.date(2022, 12, 27)),
}


def sealed_overlap(start: dt.date, end: dt.date) -> Dict[str, float]:
    """Share of [start, end] that falls inside each sealed holdout window."""
    span = max((end - start).days, 1)
    out = {}
    for name, (a, b) in SEALED_WINDOWS.items():
        lo, hi = max(start, a), min(end, b)
        if hi > lo:
            out[name] = min((hi - lo).days / span, 1.0)
    return out


#: WHAT THE DEPLOY EARNED. Window A equity-only, 2025-03-06 to 2026-08-17,
#: evaluated once on the frozen configuration. The pre-fix run of the same
#: window read ic 0.0586 / spread 0.0112 and is recorded in `parameters.yaml`.
DEPLOY_REFERENCE = {
    "ic": 0.0493, "ic_t": 3.69, "spread": 0.0107, "spread_t": 2.89,
    "topk": 0.0038, "topk_t": 0.81,
    "book_excess_ann": -0.0283, "book_maxdd": -0.2392,
    "cost_drag_ann": 0.0966,
    "ic_window_b": 0.0357, "spread_window_b": 0.0086,
}


def recheck(panel: pd.DataFrame, *, holdout_months: int = 3,
            label: str = f"y{LABEL_HORIZON_SESSIONS}",
            n_null: int = 200, seed: int = 0) -> V3Recheck:
    """Apply the frozen v3 scorer to the most recent `holdout_months` of dates.

    The shuffled-score null permutes the composite WITHIN each cross-section,
    breaking the score/outcome link and preserving everything else. It is the
    reference both sealed windows were judged against (p 0.02 and 0.01).
    """
    from .. import v3_monitor as vm

    if panel is None or panel.empty:
        return V3Recheck(dt.date.today(), None, 0, float("nan"), float("nan"),
                         float("nan"), float("nan"), float("nan"),
                         DEPLOY_REFERENCE, "NO_DATA",
                         "the panel is empty; nothing to re-check")
    score = "score" if "score" in panel.columns else "v3_score"
    if score not in panel.columns or label not in panel.columns:
        return V3Recheck(dt.date.today(), None, 0, float("nan"), float("nan"),
                         float("nan"), float("nan"), float("nan"),
                         DEPLOY_REFERENCE, "NO_DATA",
                         f"the panel carries no {score!r}/{label!r} column")

    dates = np.array(sorted(pd.to_datetime(panel["date"]).unique()))
    end = pd.Timestamp(dates[-1])
    cut = end - pd.DateOffset(months=int(holdout_months))
    ho = panel[pd.to_datetime(panel["date"]) >= cut]
    n_dates = int(pd.to_datetime(ho["date"]).nunique()) if len(ho) else 0

    # HOW MANY OF THOSE DATES ARE INDEPENDENT. Signal dates are five sessions
    # apart and the label spans twenty-one, so consecutive dates share four
    # fifths of their outcome window. dates x stride / horizon is the count
    # that matters, and it is the count the deploy itself was judged on.
    #
    # So the re-check RUNS quarterly and issues a VERDICT once it holds as much
    # evidence as the deploy did. A quarterly pass/fail on three independent
    # observations is a coin flip with a decimal point on it -- and worse, one
    # the engine would be inviting somebody to act on.
    indep = n_dates * SIGNAL_STRIDE_SESSIONS / float(LABEL_HORIZON_SESSIONS)

    # THE PER-THEME READ IS COMPUTED EITHER WAY. It needs less evidence than the
    # headline to be worth looking at, and an inverted theme is actionable
    # before a verdict on the composite is.
    theme_health: List[Dict[str, object]] = []
    factor_health: List[Dict[str, object]] = []
    if n_dates >= 5:
        tic = vm.rolling_theme_ic(ho, label)
        theme_health = [h.to_dict() for h in vm.review_themes(
            tic, ho, window=len(tic), min_periods=min(20, n_dates))]
        fic = vm.rolling_factor_ic(ho, label)
        factor_health = [h.to_dict() for h in vm.review_factors(
            fic, window=len(fic), min_periods=min(20, n_dates))]
    inverted_f = [h["name"] for h in factor_health if h.get("inverted")]
    inverted_t = [h["name"] for h in theme_health if h.get("inverted")]
    dominating = [h["name"] for h in theme_health if h.get("dominating")]

    ov = sealed_overlap(cut.date(), end.date())
    overlap_note = ""
    if ov:
        worst = max(ov.values())
        which = ", ".join(f"{k} ({v:.0%})" for k, v in sorted(ov.items()))
        overlap_note = (
            f"THIS WINDOW OVERLAPS SEALED HOLDOUT {which}. The result below is "
            f"substantially a re-reading of the dates the deploy was already "
            f"judged on, not independent evidence: agreement with the deploy "
            f"reference is expected and confirms nothing. "
            + ("Wait for the overlap to fall away before treating a verdict as "
               "news." if worst > 0.5 else
               "The non-overlapping part is small; read it as directional."))

    def _tail(note: str) -> str:
        bits = [note]
        if overlap_note:
            bits.append(overlap_note)
        if inverted_t:
            bits.append(f"THEMES flagged as inverted: {', '.join(inverted_t)}. A "
                        f"theme inverting is a bigger event than a factor doing "
                        f"it -- it is a fifth of the composite pointing the "
                        f"wrong way.")
        if dominating:
            bits.append(f"THEMES running more of the ranking than they were "
                        f"given: {', '.join(dominating)}. The cap constrains "
                        f"the coefficient, not the influence.")
        if inverted_f:
            bits.append(f"Factors flagged as inverted: {', '.join(inverted_f)}.")
        return " ".join(bits)

    if indep < MIN_INDEPENDENT_WINDOWS:
        ic0, ic0_t, _ = (rank_ic(ho, label, score) if n_dates >= 5
                         else (float("nan"), float("nan"), 0))
        sp0, sp0_t, _ = (quintile_spread(ho, label, score=score) if n_dates >= 5
                         else (float("nan"), float("nan"), 0))
        need = int(np.ceil(MIN_INDEPENDENT_WINDOWS * LABEL_HORIZON_SESSIONS
                           / SIGNAL_STRIDE_SESSIONS))
        return V3Recheck(
            end.date(), cut.date(), n_dates, ic0, ic0_t, sp0, sp0_t,
            float("nan"), DEPLOY_REFERENCE, "TOO_EARLY",
            _tail(f"{n_dates} scored dates is {indep:.1f} independent "
                  f"{LABEL_HORIZON_SESSIONS}-session windows, against the "
                  f"{MIN_INDEPENDENT_WINDOWS:.0f} the deploy was judged on. The "
                  f"numbers above are running totals and are NOT a verdict; "
                  f"about {need} dates are needed. Widen --holdout-months to "
                  f"accumulate, and read the theme health below, which needs "
                  f"less evidence to be worth acting on."),
            factor_health, theme_health)

    ic, ic_t, _ = rank_ic(ho, label, score)
    sp, sp_t, _ = quintile_spread(ho, label, score=score)

    rng = np.random.default_rng(seed)
    d = ho.copy()
    idx = d.groupby("date", sort=False).indices
    base = d[score].to_numpy("float64").copy()
    nulls = []
    for _ in range(int(n_null)):
        s2 = base.copy()
        for ii in idx.values():
            s2[ii] = rng.permutation(s2[ii])
        d[score] = s2
        nulls.append(quintile_spread(d, label, score=score)[0])
    nulls = np.asarray([x for x in nulls if np.isfinite(x)])
    p = (float((nulls >= sp).mean()) if len(nulls) and np.isfinite(sp)
         else float("nan"))

    if not np.isfinite(sp):
        verdict, note = "NO_VERDICT", "the spread could not be computed"
    elif p <= 0.05 and sp > 0:
        verdict = "HOLDS"
        note = (f"quintile spread +{sp:.2%} per {LABEL_HORIZON_SESSIONS} "
                f"sessions (t {sp_t:.2f}), outside a {len(nulls)}-draw shuffled "
                f"null at p={p:.3f}. The deploy earned "
                f"+{DEPLOY_REFERENCE['spread']:.2%} "
                f"(t {DEPLOY_REFERENCE['spread_t']:.2f}) on window A and "
                f"+{DEPLOY_REFERENCE['spread_window_b']:.2%} on window B.")
    elif sp > 0:
        verdict = "WEAK"
        note = (f"spread is positive (+{sp:.2%}) but inside the shuffled null "
                f"(p={p:.3f}). Not evidence of failure and not evidence of "
                f"edge; the window is short.")
    else:
        verdict = "FAILS"
        note = (f"quintile spread is NEGATIVE ({sp:.2%}, t {sp_t:.2f}) on this "
                f"window. The ranking did not order outcomes. This is the "
                f"condition the re-check exists to surface -- it does not "
                f"change the model, and the next configuration has to be "
                f"selected on training data and tested on a window this one "
                f"has not seen. Patching the failed configuration against this "
                f"window is exactly what the holdout discipline forbids.")
    return V3Recheck(end.date(), cut.date(), n_dates, ic, ic_t, sp, sp_t, p,
                     DEPLOY_REFERENCE, verdict, _tail(note),
                     factor_health, theme_health)
