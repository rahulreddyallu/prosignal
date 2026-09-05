"""ProSignal v3 -- the two-level thematic composite. THE SHIPPED SCORER.

    factors  ->  one sub-score per THEME  ->  a capped blend of themes

WHY TWO LEVELS. A flat weighted sum of twenty-two factors is whichever theme
brought the most factors, wearing a disguise. Momentum brought ten and quality
brought two; summed flat, the composite is a momentum bet with decoration. So
each theme is combined on its own first, every sub-score is re-ranked within the
date so the themes are commensurable, and only then are they blended.

THREE CONSTRAINTS ON THE BLEND, each measured rather than asserted:

  cap 0.40      no theme may exceed 40% of the composite. Uncapped, momentum
                and quality took 74% between them.
  floor 0.06    no theme that cleared its screen may fall below 6%. On
                validation the floor alone moved max drawdown from -38.5% to
                -34.9% at no cost in excess.
  coverage      a theme is also capped at the SHARE OF NAMES IT CAN SPEAK
                ABOUT. Weights renormalise over the themes a name has, so
                carrying `quality` at 40% while only 19% of names have
                fundamentals ranks those two populations by different models.
                Fitted without this constraint, quality took the cap.

EACH THEME IS ORIENTED AT THE HORIZON IT WORKS AT. Reversal is a two-week
effect and momentum is a six-month one; oriented against a single 42-session
label the reversal sub-score came out ANTI-predictive at t -3.96. Sub-scores are
ranks, so they blend regardless of which horizon oriented them.

WHAT THE SEALED HOLDOUTS SAID. Two windows, one evaluation each, no re-tuning:

                              A 2025-03..2026-08   B 2021-07..2022-12
    rank IC (t), h=21            +0.049 (3.69)        +0.036 (3.83)
    quintile spread (t)          +1.07% (2.89)        +0.86% (3.05)
    top-ten excess (t)           +0.38% (0.81)        +1.37% (2.50)
    themes with positive IC          5 of 5              3 of 3
    ten-name book, net excess        -2.8%/yr            +2.0%/yr
    modelled cost drag                9.7%/yr           13.7%/yr
    max drawdown                     -23.9%             -16.4%

For window B the ENTIRE pipeline -- screen, stability, admission, weights -- was
re-run on data ending 2021-02-17 and evaluated once on the eighteen months that
followed. The ranking is what generalised. The concentrated book is not: it
earned roughly 15.7% gross on window B and paid 13.7% of it away in costs. See
CHANGELOG.md; the book's cost curve is in CHANGELOG.md and turnover
is reported on every run rather than left to be discovered.

THOSE ARE THE SECOND EVALUATION OF WINDOW A, and the first stands on the record
beside them. The first run scored a universe that still contained ETFs, gold and
liquid funds -- NSE publishes them in the same EQ-series bhavcopy as equities --
and they took 26.25% of its top-ten slots. It read +0.059 (3.66) / +1.12%
(2.65) / -7.2% book. The UNIVERSE was defective, not the configuration, so the
fix was to exclude non-equity instruments and re-run BOTH windows with every
parameter untouched; nothing was tuned after either number was seen. Window A
has therefore been evaluated three times counting the pre-seal dry run, and its
t-statistics should be read with that multiplicity charged against them. Window
B -- once before the defect was found, once after, both positive -- is the
cleaner read of the two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["Theme", "THEMES", "FACTOR_THEME", "ALL_FACTORS", "MIN_THEMES",
           "MIN_LOOKBACK_SESSIONS", "sector_neutral_rank", "theme_subscore",
           "score_frame", "attribution", "absolute_floor", "cap_weights",
           "BOOK", "BOOK_NOTE", "HOLDOUT_BOOK", "RESEARCH_BOOK",
           "LIVE_BOOK", "EXCLUDED_THEMES"]


@dataclass(frozen=True)
class Theme:
    weight: float
    horizon: int
    coverage: float
    factors: Tuple[Tuple[str, int], ...]

    @property
    def names(self) -> List[str]:
        return [n for n, _ in self.factors]

    @property
    def signs(self) -> Dict[str, float]:
        return {n: float(s) for n, s in self.factors}


#: The frozen configuration. Weights are post-cap, post-floor, post-coverage-cap,
#: fitted on 2018-11-27 to 2024-10-25 and not refitted since.
THEMES: Dict[str, Theme] = {
    "momentum": Theme(
        weight=0.40, horizon=42, coverage=0.9988,
        factors=(("intraday_mom_126", 1), ("mom_12_6", 1), ("mom_2_0", 1),
                 ("mom_3_1", 1), ("mom_accel", -1), ("mom_consist_126", 1),
                 ("prox_52w", 1), ("prox_52w_now", 1), ("voladj_mom_12_1", 1),
                 ("voladj_mom_6_1", 1)),
    ),
    "quality": Theme(
        weight=0.18991, horizon=21, coverage=0.1899,
        factors=(("margin_stability", -1), ("net_margin", -1)),
    ),
    "ownership": Theme(
        weight=0.18939, horizon=10, coverage=0.8985,
        factors=(("deliv_chg_5", 1), ("deliv_pct_60", 1), ("deliv_z_21", 1)),
    ),
    "risk": Theme(
        weight=0.11088, horizon=21, coverage=0.9983,
        factors=(("downside_vol_60", -1), ("ret_kurt_126", -1), ("ulcer_120", -1)),
    ),
    "reversal": Theme(
        weight=0.10982, horizon=10, coverage=0.9993,
        factors=(("max5_21", -1), ("price_vs_vwap_20", -1), ("resid_rev_21", -1),
                 ("rev_1w", -1)),
    ),
}

FACTOR_THEME: Dict[str, str] = {f: t for t, th in THEMES.items() for f in th.names}
ALL_FACTORS: Tuple[str, ...] = tuple(FACTOR_THEME)

#: Themes with NO validated factor, kept here because their absence is a finding
#: and not an oversight. Each was built in full and measured.
EXCLUDED_THEMES = {
    "value": "0 of 8 factors clear the placebo screen at any horizon. Built PIT-"
             "correct: earnings, sales, book, tangible book, EV/EBITDA, EV/sales, "
             "FCF and OCF yields. Balance-sheet data begins 2023 and the median "
             "training date has ZERO names with a book value.",
    "liquidity": "0 of 9 clear. `volume_shock_5` clears at h=42 and its sign "
                 "flips between the halves of its own life.",
    "seasonality": "0 of 2 clear. The placebo |t| threshold is 9.9 -- the "
                   "statistic is so persistent that a year-shifted alignment "
                   "reproduces it -- against a real t of -1.2.",
}

#: A name needs this many themes before it is scored at all.
MIN_THEMES = 3
#: Longest window any shipped factor reads.
MIN_LOOKBACK_SESSIONS = 274
#: Ranks neutral rather than dropping the row when the input is missing.
NEUTRAL_WHEN_MISSING = frozenset()
MIN_SECTOR_NAMES = 12


def sector_neutral_rank(values: pd.Series,
                        sectors: Optional[pd.Series] = None) -> pd.Series:
    """Rank to [-1, 1] WITHIN sector where the bucket is big enough, and within
    one residual group otherwise, so the column means one thing everywhere."""
    def _rank(s: pd.Series) -> pd.Series:
        return (s.rank(pct=True, na_option="keep") - 0.5) * 2.0

    if sectors is None:
        return _rank(values)
    sec = sectors.reindex(values.index).astype("object")
    sec = sec.where(sec.notna() & ~sec.astype(str).isin(("", "Unknown", "nan")),
                    "__RESID__")
    counts = sec.value_counts()
    small = set(counts[counts < MIN_SECTOR_NAMES].index)
    key = sec.where(~sec.isin(small), "__RESID__")
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    for _, idx in key.groupby(key, observed=True).groups.items():
        if len(idx) >= 2:
            out.loc[idx] = _rank(values.loc[idx])
    left = out.isna() & values.notna()
    if left.any():
        out.loc[left] = _rank(values.loc[left])
    return out


def theme_subscore(ranks: pd.DataFrame, theme: Theme,
                   min_factors: int = 1) -> pd.Series:
    """One theme's sub-score: sign-oriented mean of its factor ranks, re-ranked.

    The re-rank is what makes themes commensurable. Without it a theme whose
    factors happen to be more dispersed dominates the blend for a reason that
    has nothing to do with information.
    """
    cols = [c for c in theme.names if c in ranks.columns]
    if not cols:
        return pd.Series(np.nan, index=ranks.index, dtype="float64")
    M = ranks[cols].to_numpy("float64")
    S = np.array([theme.signs[c] for c in cols])
    ok = np.isfinite(M)
    n = ok.sum(axis=1)
    num = np.nansum(np.where(ok, M * S, 0.0), axis=1)
    raw = np.where(n >= min_factors, num / np.maximum(n, 1), np.nan)
    s = pd.Series(raw, index=ranks.index)
    return (s.rank(pct=True) - 0.5) * 2.0


def cap_weights(raw: Dict[str, float], cap: float = 0.40, floor: float = 0.06,
                coverage: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Normalise, floor, then cap -- per theme, at `cap` and at its coverage."""
    w = {k: max(float(v), 0.0) for k, v in raw.items()}
    tot = sum(w.values())
    if tot <= 0:
        return {k: 1.0 / max(len(w), 1) for k in w}
    w = {k: v / tot for k, v in w.items()}
    if floor > 0 and w:
        n = len(w)
        floor = min(float(floor), 1.0 / n)
        slack = 1.0 - floor * n
        w = {k: floor + slack * v for k, v in w.items()}
    caps = {k: max(min(float(cap), float((coverage or {}).get(k, 1.0))), 1e-6)
            for k in w}
    if sum(caps.values()) < 1.0:
        t = sum(caps.values())
        return {k: v / t for k, v in caps.items()}
    for _ in range(80):
        over = {k for k, v in w.items() if v > caps[k] + 1e-12}
        if not over:
            break
        excess = sum(w[k] - caps[k] for k in over)
        for k in over:
            w[k] = caps[k]
        free = [k for k in w if k not in over]
        pool = sum(w[k] for k in free)
        if not free or pool <= 0:
            break
        for k in free:
            w[k] += excess * w[k] / pool
    return w


def score_frame(raw: pd.DataFrame, sectors: Optional[Dict[str, str]] = None,
                min_themes: int = MIN_THEMES,
                themes: Optional[Dict[str, Theme]] = None) -> pd.DataFrame:
    """Rank, combine within theme, blend. One row per symbol.

    Weights renormalise over the themes a name actually has, and `n_themes`
    records how many that was -- a name scored on three of five is not the same
    measurement as one scored on five and the card says so.

    `themes` exists so a DERIVED specification can reuse this blend instead of
    copying it. It defaults to the frozen `THEMES` and the default path is
    byte-identical to what the sealed holdouts measured; `features/v4.py` passes
    its own pruned table. The alternative was a second implementation of the
    two-level blend, and a scorer that drifts from the one that earned the
    out-of-sample numbers is the failure this module exists to prevent.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()
    themes = THEMES if themes is None else themes
    factors = tuple(f for th in themes.values() for f in th.names)
    sec = pd.Series(sectors).reindex(raw.index) if sectors else None
    cols = [c for c in factors if c in raw.columns]
    ranks = pd.DataFrame({c: sector_neutral_rank(raw[c], sec) for c in cols},
                         index=raw.index)
    out = pd.DataFrame(index=raw.index)
    for c in factors:
        out[c + "_r"] = ranks[c] if c in ranks else np.nan
    subs = {}
    for tname, th in themes.items():
        s = theme_subscore(ranks, th)
        subs[tname] = s
        out[tname + "_sub"] = s
    M = np.column_stack([subs[t].to_numpy("float64") for t in themes])
    W = np.array([themes[t].weight for t in themes])
    ok = np.isfinite(M)
    num = np.nansum(np.where(ok, M * W, 0.0), axis=1)
    den = np.where(ok, W, 0.0).sum(axis=1)
    cnt = ok.sum(axis=1)
    out["n_themes"] = cnt
    out["n_themes_positive"] = ((M > 0) & ok).sum(axis=1)
    out["score"] = np.where((den > 0) & (cnt >= min_themes),
                            num / np.maximum(den, 1e-12), np.nan)
    # per-theme contribution, which sums to the score by construction
    for i, tname in enumerate(themes):
        contrib = np.where(ok[:, i] & np.isfinite(out["score"].to_numpy()),
                           M[:, i] * W[i] / np.maximum(den, 1e-12), np.nan)
        out[tname + "_contrib"] = contrib
    out["score_rank"] = out["score"].rank(ascending=False, method="first")
    return out


#: The shipped book. NOT holdout-tested at this setting -- see BOOK_NOTE.
#: THREE BOOKS EXIST AND ONLY ONE OF THEM TRADES. Keeping them apart matters
#: more than any of their contents, because a number measured on one and quoted
#: about another is how a backtest becomes a claim it never made.
#:
#: 1. HOLDOUT_BOOK -- the book the sealed windows actually evaluated.
#: 2. RESEARCH_BOOK -- the lower-turnover replacement chosen on TRAINING data
#:    after the holdout showed costs were eating the book. Never traded.
#: 3. The LIVE book -- what production trades. It is NOT either of these and it
#:    does not live here: it is `capital.max_open_positions`,
#:    `stage6_entry.admission.{entry_rank,exit_rank,entry_cadence_sessions}` in
#:    `config/parameters.yaml`, which is the only place that can change it.
#:    Mirrored below for reading only; `tests/test_v3_score.py` fails if this
#:    copy drifts from the config, because a stale mirror is worse than none.
HOLDOUT_BOOK = {"slots": 10, "entry_rank": 20, "exit_rank": 30,
                "rebalance_every_signal_dates": 1,
                "signal_date_stride_sessions": 5,
                "floor_applies_to": "whole_population"}

RESEARCH_BOOK = {"slots": 12, "entry_rank": 24, "exit_rank": 48,
                 "rebalance_every_signal_dates": 2,
                 "signal_date_stride_sessions": 5,
                 "max_per_sector": 3, "weighting": "equal",
                 "universe_max_names": 750, "floor_applies_to": "entries_only"}

#: Read-only mirror of the LIVE book. The config is the source of truth.
LIVE_BOOK = {"slots": 6, "entry_rank": 6, "exit_rank": 18,
             "entry_cadence_sessions": 21}

#: Kept as the name older code imported. It is the RESEARCH book -- which is
#: not what trades -- so anything reading it for the live configuration is
#: reading the wrong thing.
BOOK = RESEARCH_BOOK

BOOK_NOTE = (
    "THE COMPOSITE CARRIES TWO SEALED-HOLDOUT EVALUATIONS. NO BOOK DOES. "
    "The book the windows evaluated -- 10 slots, exit 30, weekly, the floor "
    "filtering the whole population -- lost to the benchmark by 2.8% a year on "
    "window A and beat it by 2.0% on window B, and on both the reason was "
    "transaction costs of 9.7% and 13.7% a year. "
    "AND THAT IS NOT THE BOOK THAT TRADES. Production runs SIX positions on a "
    "21-session cadence with a 3x exit band (18), which is both slower and far "
    "more concentrated than anything either window measured. Slower cuts the "
    "cost drag that sank the tested book, and turnover needs no labels to "
    "verify. More concentrated cuts the other way, and it leans on the "
    "statistic that generalised LEAST: top-ten excess on window A was +0.38% "
    "at t 0.81, indistinguishable from zero, while the quintile spread held at "
    "t 2.89. Ordering within the top few names is the part of this model the "
    "holdouts did not support, and a six-name book is a bet on exactly that. "
    "Read the shortlist as drawn from an evidenced ranking; the concentration "
    "is an operator's risk choice, not a validated one. Both windows are spent, "
    "so no book can be settled here -- the quarterly re-check is what will do "
    "it, once its window stops overlapping window A.")


def absolute_floor(scored: pd.DataFrame, dist_200dma: pd.Series,
                   min_positive_themes: int = 3) -> pd.Series:
    """The quality floor, and it is ABSOLUTE on purpose.

    A floor on a cross-sectional RANK cannot fire -- somebody is top of the list
    every day however weak the day is. Measured on 235 validation dates, "three
    themes above the median" alone never left fewer than 87 names. Adding the
    trend condition took the count to 11 at the COVID trough and to 8 in the 2022
    drawdown -- below a ten-name book on one date, which is the point.

    IT GATES ENTRIES, NOT HOLDINGS. Filtering the whole population ejected a name
    the moment it slipped below its 200-session average even while it was still
    ranked third, and those forced exits were most of the book's turnover:
    measured on the training window, cost drag falls from 8.8% a year to 5.6%
    when the same floor is applied to purchases only. Requiring quality to BUY
    and exiting on rank or stop is both the ordinary discipline and the cheaper
    one.
    """
    trend = dist_200dma.reindex(scored.index) > 0
    broad = scored["n_themes_positive"] >= int(min_positive_themes)
    return (trend & broad).fillna(False)


def attribution(raw: pd.DataFrame, scored: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """FACTOR / THEME / Z / WEIGHT / CONTRIB for one name.

    ``Z`` is the sector-neutral cross-sectional rank in [-1, 1] -- the quantity
    the score is built from. The theme rows carry the theme's own sub-score and
    the weight it was blended at; the factor rows carry the ranks that theme was
    built from. Both are shown because "momentum +0.42" does not say WHICH
    momentum moved, and a reader cannot check the theme against its parts.
    """
    if symbol not in scored.index:
        return pd.DataFrame(columns=["FACTOR", "THEME", "VALUE", "Z", "WEIGHT",
                                     "CONTRIB", "LEVEL"])
    rows = []
    for tname, th in THEMES.items():
        sub = scored.at[symbol, tname + "_sub"]
        con = scored.at[symbol, tname + "_contrib"]
        rows.append({"FACTOR": tname, "THEME": tname, "VALUE": np.nan,
                     "Z": float(sub) if pd.notna(sub) else np.nan,
                     "WEIGHT": th.weight,
                     "CONTRIB": float(con) if pd.notna(con) else np.nan,
                     "LEVEL": "theme"})
        for fname in th.names:
            z = scored.at[symbol, fname + "_r"] if fname + "_r" in scored else np.nan
            rows.append({
                "FACTOR": fname, "THEME": tname,
                "VALUE": (float(raw.at[symbol, fname])
                          if fname in raw.columns and symbol in raw.index
                          and pd.notna(raw.at[symbol, fname]) else np.nan),
                "Z": float(z) if pd.notna(z) else np.nan,
                "WEIGHT": th.signs[fname] / max(len(th.names), 1),
                "CONTRIB": np.nan, "LEVEL": "factor"})
    out = pd.DataFrame(rows)
    order = out[out.LEVEL == "theme"].sort_values("CONTRIB", ascending=False,
                                                  na_position="last")["THEME"].tolist()
    out["_o"] = out["THEME"].map({t: i for i, t in enumerate(order)})
    out = out.sort_values(["_o", "LEVEL"], ascending=[True, True]).drop(columns=["_o"])
    return out.reset_index(drop=True)
