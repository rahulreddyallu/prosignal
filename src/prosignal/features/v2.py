"""ProSignal v2 cross-sectional score.

THE SHIPPED SCORER. Ten factors, each a survivor of a placebo-alignment screen
run on 2018-11 to 2024-10 (see `research/V2_SEARCH.md`), each carried at equal
weight with the sign the training window measured. The only fitted parameter is
that sign, and it was identical in all eight walk-forward folds.

WHY THE DEFINITIONS LIVE HERE RATHER THAN IN `crosssec.FEATURES`. Several of
these share a NAME with a factor in that module and are not the same
construction -- `crosssec.mom_6_1` spans 126 sessions ending 21 back, this one
spans 105. A scorer that earned a sealed-holdout number has to compute what it
was measured computing, so it carries its own definitions and its own tests
rather than borrowing a neighbour's.

NOTHING HERE READS PAST `t`. Every window ends at or before the decision row,
and the caller fills at the next session's VWAP, which is what the holdout
measured.

NONE OF THE TEN NEEDS A MARKET RESIDUAL. That is a property worth stating: the
live path needs only this name's own close, open, turnover and delivered
fraction, so a score cannot be moved by a change in how the benchmark universe
is assembled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "V2Factor", "V2_FACTORS", "V2_FACTOR_NAMES", "MIN_LOOKBACK_SESSIONS",
    "factor_frame", "sector_neutral_rank", "score_frame", "attribution",
]


@dataclass(frozen=True)
class V2Factor:
    name: str
    sign: int
    weight: float
    lookback: int
    family: str
    note: str


#: Sign is the direction the TRAINING window measured, not a prior. Weight is
#: 1/n for every factor: the search found equal weighting matched or beat ridge,
#: lasso, elastic net, PCR and gradient boosting on out-of-sample top-decile
#: behaviour, and it has no coefficient to go stale.
V2_FACTORS: Tuple[V2Factor, ...] = (
    V2Factor("ret_kurt_126", -1, 0.1, 127, "risk",
             "kurtosis of daily returns over 126 sessions; fat-tailed names "
             "underperform (negative)"),
    V2Factor("voladj_mom_12_1", +1, 0.1, 254, "momentum",
             "231-session return ending 21 back, divided by 252-session return sd"),
    V2Factor("mom_consist_126", +1, 0.1, 148, "momentum",
             "share of positive sessions over 126, ending 21 back "
             "(Da, Gurun & Warachka 2014, information discreteness)"),
    V2Factor("intraday_mom_126", +1, 0.1, 148, "momentum",
             "sum of close-over-open returns over 126 sessions ending 21 back "
             "(Lou, Polk & Skouras 2019)"),
    V2Factor("prox_52w", +1, 0.1, 274, "momentum",
             "close 21 back over the 252-session high ending 21 back "
             "(George & Hwang 2004, with a reversal-avoiding skip)"),
    V2Factor("voladj_mom_6_1", +1, 0.1, 148, "momentum",
             "105-session return ending 21 back, divided by 126-session return sd"),
    V2Factor("deliv_z_21", +1, 0.1, 254, "delivery",
             "21-session mean delivered fraction less its 252-session mean, over "
             "the 252-session sd; NSE-specific ownership-conviction signal"),
    V2Factor("prox_52w_now", +1, 0.1, 253, "momentum",
             "close over the trailing 252-session high, no skip"),
    V2Factor("mom_3_1", +1, 0.1, 64, "momentum",
             "42-session return ending 21 back"),
    V2Factor("volume_shock_5", +1, 0.1, 61, "liquidity",
             "log 20-session mean turnover less log 60-session mean turnover"),
)

V2_FACTOR_NAMES: Tuple[str, ...] = tuple(f.name for f in V2_FACTORS)
MIN_LOOKBACK_SESSIONS = max(f.lookback for f in V2_FACTORS)

#: Ranks neutral (0.0) rather than dropping the row when its input is missing.
#: Delivery covers about 90% of the panel and is absent before 2019-07; a name
#: with no delivery print contributes nothing through this factor instead of
#: being excluded from the ranking entirely.
NEUTRAL_WHEN_MISSING = frozenset({"deliv_z_21"})

#: Below this a sector is too thin to rank within -- three names produce ranks
#: of -1, 0 and +1 whatever the values were. Matches `crosssec.MIN_SECTOR_NAMES`.
MIN_SECTOR_NAMES = 12


def _tail(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.iloc[-n:] if len(df) >= n else df


def _mask(values: pd.Series, window: pd.DataFrame, min_obs: int) -> pd.Series:
    """Blank a column that has fewer than ``min_obs`` real observations.

    THIS IS NOT COSMETIC. The search measured every factor through
    ``rolling(w, min_periods=m)``, which counts NON-NaN observations PER COLUMN.
    A live path that instead checks the length of the slice gives a value to a
    name with thirty prints in a 252-session window and the training panel gave
    it nothing -- so the two rank different populations, which is the exact
    train/inference divergence `crosssec.features_for_date` was written to
    prevent. Measured before this was added: three of the ten factors
    disagreed with the researched definition on 0.2-0.5% of names per date.
    """
    n = window.notna().sum(axis=0)
    return values.where(n >= int(min_obs))


def factor_frame(
    close: pd.DataFrame,
    turnover: pd.DataFrame,
    open_: Optional[pd.DataFrame] = None,
    deliv_pct: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Raw factor values for the LAST row of ``close``. One date, no label.

    ``close`` and ``open_`` must be corporate-action adjusted, as the store
    serves them. Rows are dates ascending, columns are symbols. Minimum
    observation counts mirror the ``min_periods`` the search measured each
    factor with, per column.
    """
    if close.empty or len(close.index) < 2:
        return pd.DataFrame(columns=list(V2_FACTOR_NAMES))
    hist = close
    n = len(hist)
    ret = hist.pct_change(fill_method=None)
    out: Dict[str, pd.Series] = {}
    nan = pd.Series(np.nan, index=hist.columns, dtype="float64")

    def back(k: int) -> pd.Series:
        return hist.iloc[-1 - k] if n > k else nan

    # -- momentum ---------------------------------------------------------
    out["mom_3_1"] = back(21) / back(63) - 1.0
    mom_6_1 = back(21) / back(126) - 1.0
    mom_12_1 = back(21) / back(252) - 1.0

    r126, r252 = _tail(ret, 126), _tail(ret, 252)
    sd126 = _mask(r126.std(ddof=1), r126, 75)
    sd252 = _mask(r252.std(ddof=1), r252, 151)
    out["voladj_mom_6_1"] = mom_6_1 / sd126.replace(0.0, np.nan)
    out["voladj_mom_12_1"] = mom_12_1 / sd252.replace(0.0, np.nan)

    # The positive-session share and intraday momentum are both measured over
    # the 126 sessions ENDING 21 BACK -- the same formation window the price
    # momenta use, so the block cannot pick up the reversal month by the back
    # door. With the last row at i = n-1, rows i-146..i-21 inclusive is
    # iloc[n-147 : n-21]. An off-by-one here shifts the whole formation window
    # by a session and was caught by the parity test against the search code.
    form = ret.iloc[max(n - 147, 0): n - 21] if n > 147 else ret.iloc[0:0]
    if len(form):
        pos = (form > 0).astype("float64").where(form.notna())
        out["mom_consist_126"] = _mask(pos.mean(), form, 75)
    else:
        out["mom_consist_126"] = nan

    if open_ is not None and not open_.empty and n > 147:
        op = open_.reindex(index=hist.index, columns=hist.columns)
        intraday = hist / op.where(op > 0) - 1.0
        seg = intraday.iloc[max(n - 147, 0): n - 21]
        out["intraday_mom_126"] = _mask(seg.sum(min_count=1), seg, 75)
    else:
        out["intraday_mom_126"] = nan

    if n > 273:
        skip = hist.iloc[max(n - 273, 0): n - 21]
        out["prox_52w"] = _mask(back(21) / skip.max().replace(0.0, np.nan) - 1.0,
                                skip, 200)
    else:
        out["prox_52w"] = nan
    hi252 = _tail(hist, 252)
    out["prox_52w_now"] = _mask(hist.iloc[-1] / hi252.max().replace(0.0, np.nan) - 1.0,
                                hi252, 200)

    # -- risk -------------------------------------------------------------
    out["ret_kurt_126"] = _mask(r126.kurt(), r126, 90)

    # -- liquidity --------------------------------------------------------
    t20, t60 = _tail(turnover, 20), _tail(turnover, 60)
    out["volume_shock_5"] = _mask(
        _mask(np.log1p(t20.mean()) - np.log1p(t60.mean()), t20, 12), t60, 36)

    # -- delivery ---------------------------------------------------------
    if deliv_pct is not None and not deliv_pct.empty:
        dl = deliv_pct.reindex(index=hist.index, columns=hist.columns)
        d21, d252 = _tail(dl, 21), _tail(dl, 252)
        sd = _mask(d252.std(ddof=1), d252, 150)
        z = (_mask(d21.mean(), d21, 12) - _mask(d252.mean(), d252, 150)) \
            / sd.replace(0.0, np.nan)
        out["deliv_z_21"] = z
    else:
        out["deliv_z_21"] = nan

    frame = pd.DataFrame({f.name: out[f.name] for f in V2_FACTORS})
    return frame.replace([np.inf, -np.inf], np.nan)


def sector_neutral_rank(values: pd.Series,
                        sectors: Optional[pd.Series] = None) -> pd.Series:
    """Cross-sectional rank to [-1, 1], taken WITHIN sector where the bucket is
    big enough and within ONE residual group otherwise.

    Every name is ranked inside some group, so the column means one thing
    everywhere. Mixing within-sector ranks with whole-universe ranks in a single
    column is the defect `crosssec.sector_neutral_rank` documents at length; the
    same rule is applied here.
    """
    def _rank(s: pd.Series) -> pd.Series:
        r = s.rank(pct=True, na_option="keep")
        return (r - 0.5) * 2.0

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
    unranked = out.isna() & values.notna()
    if unranked.any():
        out.loc[unranked] = _rank(values.loc[unranked])
    return out


def score_frame(raw: pd.DataFrame,
                sectors: Optional[Dict[str, str]] = None,
                min_factors: int = 7) -> pd.DataFrame:
    """Rank, orient, weight and sum. Returns one row per symbol with the
    per-factor rank, weight and contribution alongside the composite.

    ``min_factors`` is the number of NON-NEUTRAL factors a name must have to be
    scored at all. A name ranked on four of ten is not comparable with one
    ranked on ten, and median-filling the gap ranks it by a number nobody
    computed for it.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()
    sec = pd.Series(sectors).reindex(raw.index) if sectors else None
    cols = {}
    contrib = {}
    for f in V2_FACTORS:
        r = sector_neutral_rank(raw[f.name], sec)
        if f.name in NEUTRAL_WHEN_MISSING:
            r = r.fillna(0.0)
        cols[f.name + "_r"] = r
        contrib[f.name + "_contrib"] = r * f.sign * f.weight
    out = pd.DataFrame(cols, index=raw.index)
    out = pd.concat([out, pd.DataFrame(contrib, index=raw.index)], axis=1)
    required = [f.name for f in V2_FACTORS if f.name not in NEUTRAL_WHEN_MISSING]
    n_present = raw[required].notna().sum(axis=1) + len(NEUTRAL_WHEN_MISSING)
    out["n_factors"] = n_present
    # Weights renormalise over the factors this name actually has, so a missing
    # input dilutes rather than silently biases the composite toward zero.
    wsum = pd.Series(0.0, index=raw.index)
    total = pd.Series(0.0, index=raw.index)
    for f in V2_FACTORS:
        r = out[f.name + "_r"]
        ok = r.notna()
        wsum = wsum.add(ok.astype(float) * f.weight, fill_value=0.0)
        total = total.add((r.fillna(0.0)) * f.sign * f.weight, fill_value=0.0)
    out["score"] = (total / wsum.replace(0.0, np.nan)).where(n_present >= min_factors)
    out["score_rank"] = out["score"].rank(ascending=False, method="first")
    return out


def attribution(raw: pd.DataFrame, scored: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """The per-stock FACTOR / Z / WEIGHT / CONTRIB table for one name.

    ``Z`` is the sector-neutral cross-sectional rank in [-1, 1] -- the quantity
    the score is actually built from. It is labelled Z because that is the
    column the card has always carried; it is a rank, and calling it a z-score
    without saying so is how a reader ends up thinking it is a standard
    deviation count.
    """
    if symbol not in scored.index:
        return pd.DataFrame(columns=["FACTOR", "VALUE", "Z", "WEIGHT", "CONTRIB", "FAMILY"])
    rows = []
    for f in V2_FACTORS:
        z = scored.at[symbol, f.name + "_r"]
        rows.append({
            "FACTOR": f.name,
            "VALUE": (float(raw.at[symbol, f.name])
                      if symbol in raw.index and pd.notna(raw.at[symbol, f.name]) else np.nan),
            "Z": float(z) if pd.notna(z) else np.nan,
            "WEIGHT": f.sign * f.weight,
            "CONTRIB": float(scored.at[symbol, f.name + "_contrib"]),
            "FAMILY": f.family,
        })
    out = pd.DataFrame(rows)
    return out.sort_values("CONTRIB", key=abs, ascending=False).reset_index(drop=True)
