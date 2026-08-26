"""Standalone factor diagnostics: IC, ICIR, decay and breakeven turnover.

Three questions that have to be answered per factor BEFORE any blending, and
were not being answered at all.

IC AND ICIR. Rank IC, so a single outlier cannot carry the correlation. The mean
across dates is the edge; the mean divided by its standard deviation is how
reliable that edge is date to date, which is the number that survives blending.
A factor with a good IC and an ICIR near zero is a factor that worked in a few
periods and did nothing in the rest.

DECAY. The holding period is a MEASUREMENT, not a choice. IC computed at several
horizons says how long the information lasts, and the horizon to run is the one
maximising cumulative IC net of what it costs to turn the book over that often --
not the one with the highest raw IC, which is almost always the shortest.

There is a tension worth stating rather than averaging away: signals decay at
different speeds. Blending a fast signal with a slow one into one score means no
single holding period is right for the result.

BREAKEVEN TURNOVER. T* = alpha_gross / cost. A factor whose implied turnover
exceeds it loses money however good the gross IC looks. STT alone is 0.1% each
way on delivery -- 20 bps round trip before spread, brokerage, stamp duty,
exchange fees and impact -- so a book turned six times a year pays roughly 120
bps against a gross premium that is plausibly 200-400.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["FactorIC", "rank_ic", "factor_ic", "ic_decay", "breakeven_turnover"]


def rank_ic(pred: Sequence[float], actual: Sequence[float]) -> float:
    """Spearman correlation between a forecast and what happened.

    Ranks rather than levels: one midcap doubling on an order announcement would
    otherwise carry the whole correlation.
    """
    p = pd.Series(pred, dtype="float64")
    a = pd.Series(actual, dtype="float64").reindex(p.index)
    ok = p.notna() & a.notna()
    if ok.sum() < 8:
        return float("nan")
    pr, ar = p[ok].rank(), a[ok].rank()
    if pr.std(ddof=0) == 0 or ar.std(ddof=0) == 0:
        return float("nan")
    return float(np.corrcoef(pr, ar)[0, 1])


@dataclass(frozen=True)
class FactorIC:
    factor: str
    n_dates: int
    ic_mean: float
    ic_std: float
    icir: float
    t_stat: float
    hit_rate: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "factor": self.factor, "n_dates": self.n_dates,
            "ic_mean": round(self.ic_mean, 5), "ic_std": round(self.ic_std, 5),
            "icir": round(self.icir, 4), "t_stat": round(self.t_stat, 3),
            "hit_rate": round(self.hit_rate, 4),
        }


def factor_ic(panel: pd.DataFrame, factor: str,
              label: str = "label") -> Optional[FactorIC]:
    """Per-date rank IC for one factor, summarised.

    ``t_stat`` is ICIR * sqrt(n_dates) and is NOT overlap-corrected -- the panel
    is sampled every `step` sessions against a longer label, so neighbouring
    dates share most of their outcome window. Treat it as an upper bound and use
    the project's overlap-corrected machinery for anything reportable.
    """
    if factor not in panel.columns or label not in panel.columns:
        return None
    per_date: List[float] = []
    for _, g in panel.groupby("date", sort=True, observed=True):
        value = rank_ic(g[factor], g[label])
        if np.isfinite(value):
            per_date.append(value)
    if len(per_date) < 4:
        return None
    ic = np.asarray(per_date, dtype="float64")
    mean, std = float(ic.mean()), float(ic.std(ddof=1))
    icir = mean / std if std > 0 else float("nan")
    return FactorIC(
        factor=factor, n_dates=len(ic), ic_mean=mean, ic_std=std, icir=icir,
        t_stat=icir * np.sqrt(len(ic)) if np.isfinite(icir) else float("nan"),
        hit_rate=float((ic > 0).mean()),
    )


def ic_decay(
    close: pd.DataFrame,
    factor_values: pd.DataFrame,
    horizons: Sequence[int] = (5, 10, 21, 42, 63, 126),
) -> pd.DataFrame:
    """Mean rank IC at several horizons, for one factor.

    ``factor_values`` is (date x symbol) and must be aligned to ``close``. The
    label at horizon h is the forward return from the factor date to h sessions
    later, so nothing at or before the factor date enters it.
    """
    dates = list(close.index)
    rows: List[Dict[str, object]] = []
    for h in horizons:
        per_date: List[float] = []
        for i, when in enumerate(dates):
            if i + h >= len(dates) or when not in factor_values.index:
                continue
            fwd = close.iloc[i + h] / close.iloc[i] - 1.0
            value = rank_ic(factor_values.loc[when], fwd)
            if np.isfinite(value):
                per_date.append(value)
        if len(per_date) < 4:
            continue
        ic = np.asarray(per_date, dtype="float64")
        mean = float(ic.mean())
        std = float(ic.std(ddof=1))
        rows.append({
            "horizon": h, "n_dates": len(ic), "ic_mean": mean,
            "icir": mean / std if std > 0 else np.nan,
            # Held h sessions, a position is turned over 252/h times a year.
            "turnovers_per_year": 252.0 / h,
        })
    return pd.DataFrame(rows)


def breakeven_turnover(alpha_gross_annual: float, cost_bps_round_trip: float) -> float:
    """T* = alpha_gross / cost. Round trips a year the edge can pay for.

    Above it the factor loses money however good the gross IC looks. Returns
    infinity for a costless round trip and 0.0 for a non-positive edge, both of
    which are statements rather than errors.
    """
    if cost_bps_round_trip <= 0:
        return float("inf")
    if alpha_gross_annual <= 0:
        return 0.0
    return float(alpha_gross_annual / (cost_bps_round_trip / 10_000.0))


def net_of_cost(ic_table: pd.DataFrame, cost_bps_round_trip: float,
                alpha_per_ic: float = 0.10) -> pd.DataFrame:
    """Rank horizons by edge NET of what turning the book over that often costs.

    ``alpha_per_ic`` converts a rank IC into an annual gross return. It is a
    rule of thumb, not a measurement -- stated as an argument so the assumption
    is visible and can be replaced with a fitted number.

    The horizon to run is the one that maximises this, which is rarely the
    horizon with the highest raw IC.
    """
    if ic_table.empty:
        return ic_table
    out = ic_table.copy()
    out["gross_annual"] = out["ic_mean"] * alpha_per_ic
    out["cost_annual"] = out["turnovers_per_year"] * (cost_bps_round_trip / 10_000.0)
    out["net_annual"] = out["gross_annual"] - out["cost_annual"]
    return out.sort_values("net_annual", ascending=False)
