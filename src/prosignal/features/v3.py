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

AND THE CAP IS APPLIED TWICE, WHICH IT WAS NOT. The three constraints above
describe how the weights were CHOSEN. Applying them once, at fit time, produces
a weight vector correct for a name carrying all five themes and for no other
name -- and 91% of the live universe carries four, because fundamentals reach
8.8% of it. The blend then divided by the sum of the weights a name actually
had, which removes the cap rather than re-imposing it: momentum's 0.40 became
0.40/0.81009 = 49.38%. Measured on 2026-09-03, momentum ran at 48.55% mean
effective weight and 63.59% of the realised cross-sectional spread against a cap
of 40%, which is precisely the "momentum bet with decoration" the two-level
structure exists to prevent. `score_frame` now re-caps per name over the themes
it has; see `_weights_for_pattern`.

THE HOLDOUT NUMBERS BELOW DESCRIBE THE UNCAPPED BLEND. They were measured
before that fix and they are not re-run here: both windows are spent, and
re-evaluating either to make a repaired model look validated is the exact
laundering the seal exists to stop. The repair moves the ranking -- Spearman
0.977, but one of six book names and two of the top twenty change -- so the
right reading of the table is "this is what the ranking did when momentum was
carrying 49%", not "this is what ships". What ships has no sealed evaluation.

EACH THEME IS ORIENTED AT THE HORIZON IT WORKS AT. Reversal is a two-week
effect and momentum is a six-month one; oriented against a single 42-session
label the reversal sub-score came out ANTI-predictive at t -3.96. Sub-scores are
ranks, so they blend regardless of which horizon oriented them.

WHAT THE SEALED HOLDOUTS SAID, ON THE UNCAPPED BLEND. Two windows, one
evaluation each, no re-tuning:

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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["Theme", "THEMES", "FACTOR_THEME", "ALL_FACTORS", "MIN_THEMES",
           "MIN_LOOKBACK_SESSIONS", "sector_neutral_rank", "theme_subscore",
           "score_frame", "attribution", "absolute_floor", "cap_weights",
           "BOOK", "BOOK_NOTE", "HOLDOUT_BOOK", "RESEARCH_BOOK",
           "score_dispersion", "TYPICAL_DISPERSION", "residual_bucket_size",
           "LIVE_BOOK", "EXCLUDED_THEMES"]


@dataclass(frozen=True)
class Theme:
    weight: float
    horizon: int
    coverage: float
    factors: Tuple[Tuple[str, int], ...]
    #: WHAT THIS THEME ACTUALLY BUYS, in the reader's words rather than the
    #: search's shorthand. The dict KEYS are frozen -- they name the `_sub`,
    #: `_contrib` and `_w` columns, the tests pin them, and every recorded run
    #: carries them -- so the honest name has to live beside the key rather than
    #: replace it. The interface kept its own copy of this table and the run
    #: notes kept none, so the screen said "Low-margin tilt" while the scoring
    #: note written into the same run's record said "quality 19%".
    label: str = ""

    @property
    def names(self) -> List[str]:
        return [n for n, _ in self.factors]

    @property
    def signs(self) -> Dict[str, float]:
        return {n: float(s) for n, s in self.factors}


#: The frozen configuration. Weights are post-cap, post-floor, post-coverage-cap,
#: fitted on 2018-11-27 to 2024-10-25 and not refitted since.
#:
#: THE SIGNS ARE MEASUREMENTS, NOT INTENTIONS, and two of them read backwards
#: against their own theme name. `research/V3_SEARCH.md` records the screen:
#: `margin_stability` came in at IC -0.0351, t -5.52, holding its sign across
#: both halves of its life (-2.83 / -5.30), so it ships at -1. Because
#: `v3_factors` computes it as MINUS the standard deviation of the TTM margin,
#: that -1 makes the theme prefer UNSTABLE margins; `net_margin` at -1 makes it
#: prefer LOW ones. Both are what the search found and neither is a typo -- the
#: same treatment `mom_accel` gets inside momentum, which V3_SEARCH.md pins with
#: a test precisely "so nobody 'corrects' it".
#:
#: What that means is that the key `quality` is the wrong word for it. The name
#: is kept because it is load-bearing everywhere; `label` carries the truth.
THEMES: Dict[str, Theme] = {
    "momentum": Theme(
        weight=0.40, horizon=42, coverage=0.9988,
        label="Momentum",
        factors=(("intraday_mom_126", 1), ("mom_12_6", 1), ("mom_2_0", 1),
                 ("mom_3_1", 1), ("mom_accel", -1), ("mom_consist_126", 1),
                 ("prox_52w", 1), ("prox_52w_now", 1), ("voladj_mom_12_1", 1),
                 ("voladj_mom_6_1", 1)),
    ),
    "quality": Theme(
        weight=0.18991, horizon=21, coverage=0.1899,
        label="Low-margin tilt",
        factors=(("margin_stability", -1), ("net_margin", -1)),
    ),
    "ownership": Theme(
        weight=0.18939, horizon=10, coverage=0.8985,
        label="Delivery strength",
        factors=(("deliv_chg_5", 1), ("deliv_pct_60", 1), ("deliv_z_21", 1)),
    ),
    "risk": Theme(
        weight=0.11088, horizon=21, coverage=0.9983,
        label="Downside risk",
        factors=(("downside_vol_60", -1), ("ret_kurt_126", -1), ("ulcer_120", -1)),
    ),
    "reversal": Theme(
        weight=0.10982, horizon=10, coverage=0.9993,
        label="Short-horizon reversal",
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


def residual_bucket_size(index, sectors: Optional[Dict[str, str]]) -> Dict[str, int]:
    """How many names `sector_neutral_rank` ranks inside `__RESID__`.

    "Sector-neutral" is a claim the engine makes on every card, and it is only
    true for the names in a bucket big enough to rank within. Everything else --
    a missing or `Unknown` sector, and every sector with fewer than
    MIN_SECTOR_NAMES members -- is pooled into one residual group and ranked
    against the others there.

    Measured on the live cross-section of 386 eligible names, `__RESID__` held
    150 of them (38.9%): 79 genuinely unclassified, plus 71 drawn from THIRTEEN
    real sectors folded in for being too small -- Power, Realty, Telecom,
    Textiles, Metals & Mining and eight more. A Power stock is neutralised
    against Realty. Any sector tilt inside that bucket is not neutralised at
    all, and nothing said so.
    """
    if not sectors:
        return {"resid": len(index), "unknown": len(index), "folded": 0,
                "buckets": 0}
    sec = pd.Series({k: sectors.get(k) for k in index})
    unknown = sec.isna() | sec.astype(str).isin(("", "Unknown", "nan"))
    named = sec.where(~unknown, "__RESID__")
    counts = named.value_counts()
    small = {k for k, n in counts.items()
             if k != "__RESID__" and n < MIN_SECTOR_NAMES}
    key = named.where(~named.isin(small), "__RESID__")
    return {
        "resid": int((key == "__RESID__").sum()),
        "unknown": int(unknown.sum()),
        "folded": int(sum(counts[k] for k in small)),
        "buckets": int(key.nunique()),
        "folded_sectors": sorted(small),
    }


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


def _weights_for_pattern(available: Tuple[bool, ...],
                         cap: float = 0.40) -> np.ndarray:
    """The blend weights for one AVAILABILITY PATTERN, re-capped.

    THE CAP HAS TO BE RE-APPLIED HERE, and this is the whole of the fix.

    `THEMES[t].weight` is already post-cap, post-floor and post-coverage-cap:
    the search normalised each theme's validated top-decile excess, floored at
    6%, capped at 40%, and capped again at the share of names the theme can
    speak about. Quality's 0.18991 IS its 0.1899 coverage cap.

    Those weights are correct for a name that has all five themes and for
    nothing else. The blend then divided by the sum of the weights a name
    actually had -- which is the right shape and the wrong arithmetic, because
    dividing by 0.81009 does not re-impose the constraint the weights were
    chosen under, it removes it. Momentum's 0.40 became 0.40/0.81009 = 49.38%
    on every name without fundamentals, which is 91% of the live universe.
    Measured on 2026-09-03 over 386 eligible names: momentum ran at 48.55% mean
    effective weight and 63.59% of the realised cross-sectional spread, against
    a cap of 40%. `cap_weights` existed, was correct, and was never called at
    scoring time.

    THE CAP IS RE-APPLIED AND THE FLOOR IS NOT, which is not an oversight.

    The floor is a fit-time constraint: it exists so a theme that cleared its
    screen cannot be handed a weight so small it may as well have been cut, and
    `THEMES[t].weight` already carries it. Re-applying it here re-floors an
    already-floored vector and compresses every weight toward equal --
    `cap_weights` blends `floor + (1 - floor*n) * w`, so momentum's 0.40 comes
    back as 0.06 + 0.70*0.40 = 0.34 EVEN FOR A NAME THAT HAS ALL FIVE THEMES.
    That is a different model from the one that was fitted, applied to the
    names the fit was correct for. Caught by
    `test_a_full_coverage_name_keeps_the_fitted_weights`.

    The cap is different: it is a constraint on the RESULT, and dropping a
    theme renormalises the survivors upward, which is exactly the direction
    that can breach it. Nothing here can push a weight DOWN below the floor --
    renormalisation only raises, and the cap's redistribution only adds to the
    themes that were not capped -- so the floor has no work left to do.

    Re-capping over the available themes returns momentum to 40.00% (from
    49.38%) on a four-theme name and lifts ownership to 27.71%, risk to 16.23%
    and reversal to 16.07%. On the live cross-section the ranking moves:
    Spearman 0.977 to the uncapped order, but one of six book names and two of
    the top twenty change, so this is a change to what gets bought and not a
    presentational one.

    Keyed on the PATTERN rather than on the name: the weights depend only on
    WHICH themes are present, so there are at most 2**len(THEMES) of them and
    the iterative cap runs once per distinct pattern instead of once per symbol.
    """
    names = [t for t, keep in zip(THEMES, available) if keep]
    out = np.zeros(len(THEMES), dtype="float64")
    if not names:
        return out
    w = cap_weights({t: THEMES[t].weight for t in names}, cap=cap, floor=0.0)
    for i, t in enumerate(THEMES):
        out[i] = float(w.get(t, 0.0))
    return out


def score_frame(raw: pd.DataFrame, sectors: Optional[Dict[str, str]] = None,
                min_themes: int = MIN_THEMES) -> pd.DataFrame:
    """Rank, combine within theme, blend. One row per symbol.

    Weights are re-capped over the themes a name actually has -- see
    `_weights_for_pattern` -- and `n_themes` records how many that was, because
    a name scored on three of five is not the same measurement as one scored on
    five and the card says so.

    Emits `<theme>_w` beside `<theme>_sub` and `<theme>_contrib`: the weight
    that produced THIS name's contribution. Without it the presentation layer
    had only `Theme.weight` to show, so every card displayed a weight that did
    not multiply its own z into its own contribution.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()
    sec = pd.Series(sectors).reindex(raw.index) if sectors else None
    cols = [c for c in ALL_FACTORS if c in raw.columns]
    ranks = pd.DataFrame({c: sector_neutral_rank(raw[c], sec) for c in cols},
                         index=raw.index)
    out = pd.DataFrame(index=raw.index)
    for c in ALL_FACTORS:
        out[c + "_r"] = ranks[c] if c in ranks else np.nan
    subs = {}
    for tname, th in THEMES.items():
        s = theme_subscore(ranks, th)
        subs[tname] = s
        out[tname + "_sub"] = s
    M = np.column_stack([subs[t].to_numpy("float64") for t in THEMES])
    ok = np.isfinite(M)
    cnt = ok.sum(axis=1)

    cache: Dict[Tuple[bool, ...], np.ndarray] = {}
    Weff = np.zeros_like(M)
    for r in range(M.shape[0]):
        pattern = tuple(bool(x) for x in ok[r])
        vec = cache.get(pattern)
        if vec is None:
            vec = cache[pattern] = _weights_for_pattern(pattern)
        Weff[r] = vec

    out["n_themes"] = cnt
    out["n_themes_positive"] = ((M > 0) & ok).sum(axis=1)
    scored = cnt >= min_themes
    # The weights already sum to 1 over the themes a name has, so the blend is
    # a plain dot product. There is no second renormalisation, which is exactly
    # what went wrong before.
    total = np.nansum(np.where(ok, M * Weff, 0.0), axis=1)
    out["score"] = np.where(scored & (Weff.sum(axis=1) > 0), total, np.nan)
    finite = np.isfinite(out["score"].to_numpy())
    for i, tname in enumerate(THEMES):
        # The weight THIS name was blended at, and the contribution it produced.
        # `w * sub == contrib` holds row by row, which is the property the card
        # invites the reader to check and could not previously satisfy.
        out[tname + "_w"] = np.where(ok[:, i] & finite, Weff[:, i], np.nan)
        out[tname + "_contrib"] = np.where(ok[:, i] & finite,
                                           M[:, i] * Weff[:, i], np.nan)
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


#: What the composite's cross-sectional spread normally is, measured on the
#: TRAINING window (2018-11-27 .. 2024-10-25) over 61 sampled dates:
#:
#:     min 0.5010   p05 0.5361   median 0.5732   p75 0.6020   max 0.7358
#:
#: Frozen here beside the weights, for the same reason they are: a number that
#: can be edited in a config file can drift away from the thing that was
#: measured. Neither sealed window was touched to produce it.
TYPICAL_DISPERSION = 0.5732


def score_dispersion(scores: "pd.Series") -> Optional[float]:
    """Top-decile mean minus median: how far the composite separated the
    universe today.

    The quantity `stage8_final_signal.scarcity.min_dispersion_ratio` gates on,
    and the reason that gate has been inert. It read `prediction_dispersion`,
    which only the deleted fitted model ever populated, so a `is not None` guard
    skipped it on every v3 run.

    READ THE LIMITS BEFORE TRUSTING IT. A blend of cross-sectional RANKS has a
    spread bounded by construction -- each sub-score is uniform on [-1, 1] every
    single day -- so what varies is only how much the themes AGREE. Measured on
    61 training dates the whole range is 0.5010 to 0.7358 and the worst day is
    0.874x the median; simulated, a rank blend runs 0.54 when its themes are
    independent and 0.88 when they move together, with a day-to-day sd near
    0.027 inside either regime.

    So at the shipped ratio of 0.50 this CANNOT fire, and that is not a defect
    to fix by lowering the bar -- it is what the control is for. The config says
    so in its own words: "meant to catch a day the fit has degenerated, not to
    tune signal count". It detects a broken scorer, not a bad market. What
    refuses to open a book on a bad market is the BOOK-LEVEL CASH RULE, which
    counts names above their long moving average and can go to zero for the
    whole market at once.
    """
    s = pd.Series(scores).dropna()
    if len(s) < 30:
        return None
    k = max(int(len(s) * 0.10), 3)
    return float(s.nlargest(k).mean() - s.median())


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
        # The weight THIS name was blended at, falling back to the fit-time one
        # only for a frame written before `_w` existed. Serving `th.weight`
        # unconditionally is what put a WEIGHT beside a CONTRIB it does not
        # produce -- the same defect the card carried, and this helper is
        # currently called by nothing, so it would have surfaced the day
        # somebody revived it.
        wcol = tname + "_w"
        w = scored.at[symbol, wcol] if wcol in scored.columns else np.nan
        rows.append({"FACTOR": tname, "THEME": tname, "VALUE": np.nan,
                     "Z": float(sub) if pd.notna(sub) else np.nan,
                     "WEIGHT": float(w) if pd.notna(w) else th.weight,
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
