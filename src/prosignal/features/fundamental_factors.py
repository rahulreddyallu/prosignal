"""Fundamental factor layer.

Separated from the price factors because the two have different failure modes.
A price factor is wrong when the arithmetic is wrong; a fundamental factor is
wrong when it was computed from a number the market did not have yet, and that
failure is invisible in the output. Everything here therefore carries two
dates: the period the figure describes, and the date it could first have been
acted on.

Availability. Statement feeds carry period end, not filing date. SEBI LODR
Regulation 33 requires quarterly results within 45 days of quarter end and
audited annual results within 60. Using the deadline rather than the typical
lag is deliberate: companies file early, not late, so the deadline is the
conservative choice and never grants the model information ahead of the market.
Where a true filing date exists it is used instead and the lag is ignored.

Families. Factors are grouped because they are not independent. Five ways of
saying "this company is cheap" are one piece of information, and scoring them
as five confirmations is how a model talks itself into a position. Family
membership is declared here so the weighting can act on it.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "FUNDAMENTAL_FACTORS", "FAMILIES", "FactorSpec",
    "build_fundamental_panel", "available_as_of", "winsorise", "sector_neutralise",
]

#: SEBI LODR Regulation 33 filing deadlines, in calendar days after period end.
QUARTERLY_LAG_DAYS = 45
ANNUAL_LAG_DAYS = 60


@dataclass(frozen=True)
class FactorSpec:
    """One fundamental factor: how to build it and what it means."""

    name: str
    family: str
    #: Higher raw value is a better prospect. False flips the sign at ranking.
    higher_is_better: bool
    rationale: str
    #: Statement fields the factor needs; absent fields leave it NaN.
    requires: Sequence[str] = field(default_factory=tuple)
    enabled: bool = True


FUNDAMENTAL_FACTORS: List[FactorSpec] = [
    # -- value: price paid per unit of what the business produces -------------
    FactorSpec("earnings_yield", "value", True,
               "TTM net income over market cap. Basu (1977); the value premium's "
               "most direct expression.",
               ("Net Income",)),
    FactorSpec("book_to_price", "value", True,
               "Common equity over market cap. Fama & French (1992).",
               ("Common Stock Equity",)),
    FactorSpec("ebitda_to_ev", "value", True,
               "TTM EBITDA over enterprise value. Capital-structure neutral, so "
               "it does not penalise a levered firm the way earnings yield does.",
               ("EBITDA", "Total Debt")),
    FactorSpec("fcf_yield", "value", True,
               "Free cash flow over market cap. Harder to manage than earnings.",
               ("Free Cash Flow",)),
    FactorSpec("sales_to_price", "value", True,
               "TTM revenue over market cap. Survives a loss-making quarter, "
               "where earnings yield goes undefined.",
               ("Total Revenue",)),

    # -- quality: whether the earnings are real and repeatable ----------------
    FactorSpec("roe", "quality", True,
               "TTM net income over common equity.",
               ("Net Income", "Common Stock Equity")),
    FactorSpec("roce", "quality", True,
               "TTM EBIT over capital employed. Comparable across leverage.",
               ("EBIT", "Common Stock Equity", "Total Debt")),
    FactorSpec("gross_margin", "quality", True,
               "Gross profit over revenue. Novy-Marx (2013) finds gross "
               "profitability the cleanest quality signal.",
               ("Gross Profit", "Total Revenue")),
    FactorSpec("ebit_margin", "quality", True,
               "EBIT over revenue.",
               ("EBIT", "Total Revenue")),
    FactorSpec("net_margin", "quality", True,
               "Net income over revenue.",
               ("Net Income", "Total Revenue")),
    FactorSpec("interest_coverage", "quality", True,
               "EBIT over interest expense. Distress proxy; Altman (1968).",
               ("EBIT", "Interest Expense")),
    FactorSpec("accruals", "quality", False,
               "Net income less operating cash flow, over assets. Sloan (1996): "
               "earnings the cash flow does not support tend to reverse. "
               "Lower is better, so the sign is flipped.",
               ("Net Income", "Operating Cash Flow", "Total Assets")),
    FactorSpec("fcf_conversion", "quality", True,
               "Free cash flow over net income.",
               ("Free Cash Flow", "Net Income")),

    # -- growth: direction of the business ------------------------------------
    FactorSpec("revenue_growth", "growth", True,
               "TTM revenue against the prior year.",
               ("Total Revenue",)),
    FactorSpec("earnings_growth", "growth", True,
               "TTM net income against the prior year.",
               ("Net Income",)),
    FactorSpec("ebitda_growth", "growth", True,
               "TTM EBITDA against the prior year.",
               ("EBITDA",)),
    FactorSpec("margin_expansion", "growth", True,
               "Change in net margin year on year. Separates growth that drops "
               "to the bottom line from growth bought with discounting.",
               ("Net Income", "Total Revenue")),

    # -- leverage: what happens when the cycle turns --------------------------
    FactorSpec("debt_to_equity", "leverage", False,
               "Total debt over common equity. Lower is better.",
               ("Total Debt", "Common Stock Equity")),
    FactorSpec("net_debt_to_ebitda", "leverage", False,
               "Net debt over TTM EBITDA. Lower is better.",
               ("Total Debt", "EBITDA")),

    # -- fundamental momentum -------------------------------------------------
    FactorSpec("earnings_acceleration", "fundamental_momentum", True,
               "Change in the earnings growth rate. Fundamental momentum is "
               "distinct from price momentum and decays more slowly.",
               ("Net Income",)),
]

FAMILIES: Dict[str, List[str]] = {}
for _spec in FUNDAMENTAL_FACTORS:
    FAMILIES.setdefault(_spec.family, []).append(_spec.name)


def available_as_of(
    period_end: pd.Series,
    kind: object = "annual",
    filing_date: Optional[pd.Series] = None,
) -> pd.Series:
    """When each figure could first have been acted on.

    A true filing date wins wherever one exists, because it is the fact the
    deadline only approximates. Otherwise the deadline is used rather than the
    typical lag: companies file early, not late, so the deadline errs toward
    showing the model less than the market had, never more.
    """
    ends = pd.to_datetime(period_end)
    kinds = kind if isinstance(kind, pd.Series) else pd.Series(kind, index=ends.index)
    lag = np.where(kinds.to_numpy() == "annual", ANNUAL_LAG_DAYS, QUARTERLY_LAG_DAYS)
    deadline = ends + pd.to_timedelta(lag, unit="D")
    if filing_date is None:
        return deadline
    filed = pd.to_datetime(filing_date)
    return filed.where(filed.notna(), deadline)


def winsorise(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip to cross-sectional quantiles.

    Fundamental ratios have unbounded tails -- one company with equity near zero
    produces an ROE of several thousand percent, and an unclipped z-score then
    collapses every other name toward zero.
    """
    v = s.dropna()
    if len(v) < 10:
        return s
    lo, hi = v.quantile(lower), v.quantile(upper)
    return s.clip(lo, hi)


def sector_neutralise(s: pd.Series, sectors: pd.Series) -> pd.Series:
    """Subtract the sector median.

    A bank's leverage and a software firm's are not comparable, and an
    un-neutralised leverage factor is largely a bet on sector composition.
    Names with no sector are left as they are rather than pooled into one
    bucket, which would create a fictitious peer group.
    """
    known = sectors.reindex(s.index)
    out = s.copy()
    mask = known.notna() & (known != "Unknown")
    if not mask.any():
        return out
    med = s[mask].groupby(known[mask]).transform("median")
    out.loc[mask] = s[mask] - med
    return out


def _ttm(frame: pd.DataFrame, field_name: str, as_of: pd.Timestamp,
         min_periods: int = 1) -> pd.Series:
    """Trailing figure per symbol using only rows available at ``as_of``."""
    usable = frame[frame["available_on"] <= as_of]
    if usable.empty or field_name not in usable.columns:
        return pd.Series(dtype="float64")
    usable = usable.dropna(subset=[field_name])
    if usable.empty:
        return pd.Series(dtype="float64")
    latest = usable.sort_values("period_end").groupby("symbol").tail(1)
    return latest.set_index("symbol")[field_name]


def _prior(frame: pd.DataFrame, field_name: str, as_of: pd.Timestamp) -> pd.Series:
    """The figure one period before the latest available one."""
    usable = frame[frame["available_on"] <= as_of]
    if usable.empty or field_name not in usable.columns:
        return pd.Series(dtype="float64")
    usable = usable.dropna(subset=[field_name]).sort_values("period_end")
    prior = usable.groupby("symbol").tail(2).groupby("symbol").head(1)
    return prior.set_index("symbol")[field_name]


def _safe_div(a: pd.Series, b: pd.Series, floor: float = 1e-6) -> pd.Series:
    a, b = a.align(b, join="inner")
    denom = b.where(b.abs() > floor)
    return a / denom


def build_fundamental_panel(
    statements: pd.DataFrame,
    market_cap: pd.Series,
    as_of: dt.date,
    enabled: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Raw fundamental factor values for one decision date.

    ``statements`` must carry symbol, period_end, kind and the statement fields;
    ``available_on`` is derived here if absent. Nothing dated after ``as_of`` is
    read, so the frame can safely hold the full history.
    """
    if statements is None or statements.empty:
        return pd.DataFrame()
    f = statements.copy()
    if "available_on" not in f.columns:
        f["available_on"] = available_as_of(f["period_end"], f.get("kind", "annual"))
    ts = pd.Timestamp(as_of)
    f = f[f["available_on"] <= ts]
    if f.empty:
        return pd.DataFrame()

    ann = f[f.get("kind", "annual") == "annual"]
    if ann.empty:
        ann = f

    want = set(enabled) if enabled is not None else {s.name for s in FUNDAMENTAL_FACTORS if s.enabled}
    g = lambda name: _ttm(ann, name, ts)
    p = lambda name: _prior(ann, name, ts)
    mc = market_cap.dropna()
    out: Dict[str, pd.Series] = {}

    ni, rev, ebitda, ebit = g("Net Income"), g("Total Revenue"), g("EBITDA"), g("EBIT")
    eq, debt, assets = g("Common Stock Equity"), g("Total Debt"), g("Total Assets")
    gp, ie = g("Gross Profit"), g("Interest Expense")
    ocf, fcf = g("Operating Cash Flow"), g("Free Cash Flow")

    if "earnings_yield" in want:   out["earnings_yield"] = _safe_div(ni, mc)
    if "book_to_price" in want:    out["book_to_price"] = _safe_div(eq, mc)
    if "sales_to_price" in want:   out["sales_to_price"] = _safe_div(rev, mc)
    if "fcf_yield" in want:        out["fcf_yield"] = _safe_div(fcf, mc)
    if "ebitda_to_ev" in want:
        ev = mc.add(debt.reindex(mc.index).fillna(0.0), fill_value=0.0)
        out["ebitda_to_ev"] = _safe_div(ebitda, ev)

    if "roe" in want:               out["roe"] = _safe_div(ni, eq)
    if "roce" in want:
        cap = eq.add(debt.reindex(eq.index).fillna(0.0), fill_value=0.0)
        out["roce"] = _safe_div(ebit, cap)
    if "gross_margin" in want:      out["gross_margin"] = _safe_div(gp, rev)
    if "ebit_margin" in want:       out["ebit_margin"] = _safe_div(ebit, rev)
    if "net_margin" in want:        out["net_margin"] = _safe_div(ni, rev)
    if "interest_coverage" in want: out["interest_coverage"] = _safe_div(ebit, ie.abs())
    if "accruals" in want:          out["accruals"] = _safe_div(ni.sub(ocf, fill_value=np.nan), assets)
    if "fcf_conversion" in want:    out["fcf_conversion"] = _safe_div(fcf, ni)

    if "revenue_growth" in want:    out["revenue_growth"] = _safe_div(rev, p("Total Revenue").abs()) - 1.0
    if "earnings_growth" in want:   out["earnings_growth"] = _safe_div(ni, p("Net Income").abs()) - 1.0
    if "ebitda_growth" in want:     out["ebitda_growth"] = _safe_div(ebitda, p("EBITDA").abs()) - 1.0
    if "margin_expansion" in want:
        out["margin_expansion"] = _safe_div(ni, rev) - _safe_div(p("Net Income"), p("Total Revenue"))

    if "debt_to_equity" in want:      out["debt_to_equity"] = _safe_div(debt, eq)
    if "net_debt_to_ebitda" in want:  out["net_debt_to_ebitda"] = _safe_div(debt, ebitda)

    if "earnings_acceleration" in want:
        g1 = _safe_div(ni, p("Net Income").abs()) - 1.0
        out["earnings_acceleration"] = g1 - g1.median()

    if not out:
        return pd.DataFrame()
    panel = pd.DataFrame(out)
    return panel.replace([np.inf, -np.inf], np.nan)
