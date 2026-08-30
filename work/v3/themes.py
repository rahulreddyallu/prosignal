"""Themed factor library.

Eight themes. REVERSAL IS SEPARATE FROM MOMENTUM on purpose: the two carry
opposite signs over the same price series, so folding them into one sub-score
makes them cancel and reports the cancellation as "no signal in the theme".

Fundamental factors are built from `pit_fund`, which lags every number to when
it was actually disclosed. Market capitalisation uses the QUOTED price and the
share count reported at that filing:

    mcap_t = adj_close_t * shares_at_filing / adj_factor_at_filing

which is invariant across splits and bonuses -- using the adjusted price with a
raw share count halves the market cap of every name that has ever split, and
turns a corporate action into a value signal.
"""
from __future__ import annotations
import numpy as np, pandas as pd

EPS = 1e-12


def _roll(df, w, how, mp=None):
    mp = mp if mp is not None else max(int(w * 0.6), 2)
    return getattr(df.rolling(w, min_periods=mp), how)()


def _safe(num, den, floor=None):
    d = den.replace(0.0, np.nan)
    if floor is not None:
        d = d.where(d.abs() > floor)
    return (num / d).replace([np.inf, -np.inf], np.nan)


def fundamental_factors(F: dict, close: pd.DataFrame, adj_factor: pd.DataFrame,
                        max_age_days: float = 420.0) -> dict:
    """Value and quality, from the point-in-time fundamental panel.

    ``F`` is the output of `pit_fund.asof_panel`; every frame is already
    as-of-joined and carries `fund_age_days`.
    """
    out = {}
    cols = close.columns
    g = lambda k: (F[k].reindex(index=close.index, columns=cols).astype("float64")
                   if k in F else pd.DataFrame(np.nan, index=close.index, columns=cols))
    age = g("fund_age_days")
    fresh = age <= float(max_age_days)

    shares = g("shares")
    # The share count is the one reported at the filing; the price is adjusted.
    # Dividing the count by the adjustment factor AT THE FILING puts both on the
    # same basis, so a split between the filing and today cannot move mcap.
    adj_at_filing = adj_factor.where(shares.notna()).ffill()
    mcap = close * _safe(shares, adj_at_filing)
    mcap = mcap.where(mcap > 0)

    eq, ta, td, cash = g("equity"), g("total_assets"), g("total_debt"), g("cash")
    ev = mcap + td.fillna(0.0) - cash.fillna(0.0)
    ev = ev.where(ev > 0)

    # ---- value ----------------------------------------------------------
    out["earnings_yield"] = _safe(g("ttm_net_profit"), mcap)
    out["sales_to_price"] = _safe(g("ttm_revenue"), mcap)
    out["book_to_price"] = _safe(eq, mcap)
    out["tangible_bp"] = _safe(g("tangible_book"), mcap)
    out["ebitda_to_ev"] = _safe(g("ttm_ebitda"), ev)
    out["sales_to_ev"] = _safe(g("ttm_revenue"), ev)
    out["fcf_yield"] = _safe(g("ttm_fcf"), mcap)
    out["ocf_to_price"] = _safe(g("ttm_ocf"), mcap)

    # ---- quality --------------------------------------------------------
    out["roe"] = _safe(g("ttm_net_profit"), eq)
    out["roa"] = _safe(g("ttm_net_profit"), ta)
    out["roce"] = _safe(g("ttm_ebit"), g("invested_capital"))
    out["gross_profitability"] = _safe(g("ttm_gross_profit"), ta)
    out["net_margin"] = _safe(g("ttm_net_profit"), g("ttm_revenue"))
    # Sloan (1996): the part of earnings that is not cash reverses. Negative is
    # the good side, and the sign is left to the screen rather than assumed.
    out["accruals"] = _safe(g("ttm_net_profit") - g("ttm_ocf"), ta)
    out["debt_to_equity"] = _safe(td, eq)
    out["interest_coverage"] = _safe(g("ttm_ebit"), g("ttm_interest").abs())
    out["cash_to_assets"] = _safe(cash, ta)
    out["current_ratio"] = _safe(g("current_assets"), g("current_liabilities"))
    # Year-on-year change, off the AS-OF panel, so the comparison is between two
    # things that were both public at their own times.
    out["asset_growth"] = _safe(ta, ta.shift(252)) - 1.0
    out["net_issuance"] = _safe(shares, shares.shift(252)) - 1.0
    out["profit_growth"] = _safe(g("ttm_net_profit"), g("ttm_net_profit").shift(252)) - 1.0
    out["revenue_growth"] = _safe(g("ttm_revenue"), g("ttm_revenue").shift(252)) - 1.0
    # Earnings stability: dispersion of the TTM margin over two years. A steady
    # margin is a quality signal; the level is priced by `net_margin`.
    out["margin_stability"] = -_roll(out["net_margin"], 504, "std", mp=250)

    # THE STALENESS GATE. Applied to the FIELD, not to the row: a name whose
    # newest disclosure is 18 months old is not described by it, and a value
    # ratio built on it is a ratio of today's price to a number from another
    # year -- which reads as "cheap" for exactly the companies that stopped
    # reporting.
    for k in list(out):
        out[k] = out[k].where(fresh).replace([np.inf, -np.inf], np.nan).astype("float32")
    out["_mcap"] = mcap.astype("float32")
    out["_fund_age_days"] = age.astype("float32")
    return out


#: Theme membership. Every factor the search may use belongs to exactly one.
THEMES = {
    "momentum": [
        "mom_2_0", "mom_3_1", "mom_6_1", "mom_12_1", "mom_12_6", "mom_9_1",
        "mom_accel", "voladj_mom_6_1", "voladj_mom_12_1", "mom_consist_126",
        "fip_6", "intraday_mom_126", "overnight_mom_126", "prox_52w",
        "prox_52w_now", "dist_low_52w", "resid_mom_252_21", "resid_mom_126_21",
        "dist_50dma", "dist_200dma", "ma_50_200", "trend_slope_120",
        "trend_r2_120",
    ],
    "reversal": [
        "mom_1_0", "rev_1w", "rev_2w", "rev_1m_scaled", "max5_21", "min5_21",
        "resid_rev_21", "intraday_mom_21", "overnight_mom_21",
        "price_vs_vwap_20",
        # RSI SITS HERE ON THE EVIDENCE, not on the convention. Measured on the
        # within-date sector-neutral ranks it correlates +0.87 with
        # price_vs_vwap_20, +0.76 with rev_2w and +0.75 with rev_1m_scaled, and
        # under +0.1 with every price-momentum factor. Filed under momentum --
        # where a technical-indicator taxonomy puts it -- it would have carried
        # a reversal signal into the momentum sub-score with the wrong sign.
        "rsi_14",
    ],
    "value": [
        "earnings_yield", "sales_to_price", "book_to_price", "tangible_bp",
        "ebitda_to_ev", "sales_to_ev", "fcf_yield", "ocf_to_price",
    ],
    "quality": [
        "roe", "roa", "roce", "gross_profitability", "net_margin", "accruals",
        "debt_to_equity", "interest_coverage", "cash_to_assets",
        "current_ratio", "asset_growth", "net_issuance", "profit_growth",
        "revenue_growth", "margin_stability",
    ],
    "risk": [
        "vol_20", "vol_60", "vol_252", "vol_ratio_20_120", "downside_vol_60",
        "idio_vol_126", "idio_skew_126", "ret_skew_126", "ret_kurt_126",
        "beta_126", "beta_252", "beta_down", "max_dd_120", "max_dd_252",
        "ulcer_120", "ulcer_252", "parkinson_60", "garman_klass_60",
        "atr_pct_14",
    ],
    "liquidity": [
        "log_adtv_60", "turnover_trend", "amihud_60", "amihud_trend",
        "zero_ret_60", "avg_trade_size", "volume_shock_5", "vol_of_turnover",
        "cs_spread_60",
    ],
    "ownership": [
        "deliv_pct_60", "deliv_trend", "deliv_z_21", "deliv_chg_5",
        "deliv_x_turnover", "deliv_value_trend",
    ],
    "seasonality": ["seasonal_same_month", "seasonal_rel_month"],
}

FACTOR_THEME = {f: t for t, fs in THEMES.items() for f in fs}
