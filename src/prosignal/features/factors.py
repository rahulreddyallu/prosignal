"""The fifteen shipped factors, computed exactly as the search measured them.

Every window here matches the v3 search code (removed 2026-09-03; the windows are frozen here) to
machine precision -- `tests/test_v3_score.py` checks it on real data across five
dates. A scorer that earned a sealed-holdout number has to compute what it was
measured computing, and the failure mode is silent: an off-by-one in a skip
window or a min-periods rule applied to the slice instead of the column produces
numbers that look entirely reasonable and are a different factor.

Nothing reads a session after the decision row.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .engine import ALL_FACTORS

__all__ = ["factor_frame", "LOOKBACK_SESSIONS"]

#: Sessions of history the block needs. `prox_52w` reads 273 of them.
LOOKBACK_SESSIONS = 300

#: HOW MANY SESSIONS A CALLER MUST HAND `factor_frame`, and there is exactly one
#: right answer because two callers used two.
#:
#: `validation.panel` sliced `[i - LOOKBACK - 15 : i + 1]`, which is 316 rows
#: inclusive. `stage4_core_score.build_v3_block` asked the calendar for a
#: trailing window of `LOOKBACK + 15`, which is 315. One extra leading bar shifts
#: the start of every rolling window, so the two produced scores that agreed to
#: Spearman 0.99997 and were not equal -- close enough that six sampled dates
#: still ranked an identical top six, and different enough that the model
#: selected on the research path was not quite the model the engine ran.
#:
#: 316 is the researched convention: it is what `build_panel` used for both
#: sealed holdouts and every experiment since, so production moves to it rather
#: than the other way round. Both callers now read this constant, and
#: `tests/test_research_live_parity.py` fails if either stops.
FRAME_SESSIONS = LOOKBACK_SESSIONS + 16


def _roll(df, w, how, mp=None):
    mp = mp if mp is not None else max(int(w * 0.6), 2)
    return getattr(df.rolling(w, min_periods=mp), how)()


def factor_frame(close: pd.DataFrame, open_: Optional[pd.DataFrame] = None,
                 vwap: Optional[pd.DataFrame] = None,
                 turnover: Optional[pd.DataFrame] = None,
                 deliv_pct: Optional[pd.DataFrame] = None,
                 fundamentals: Optional[Dict[str, pd.DataFrame]] = None,
                 last_row_only: bool = True):
    """The factor block. Rows are dates ascending, columns are symbols.

    EVERY FACTOR HERE IS AN OWN-SERIES STATISTIC. `bench_ret` used to be a
    parameter because `resid_rev_21` needed a market return; that factor was
    removed on 2026-09-05 and the parameter went with it. Nothing in this block
    now depends on how the benchmark is assembled, which removes the one way a
    change in the universe could silently move a factor value.
    """
    F: Dict[str, pd.DataFrame] = {}
    ret = close / close.shift(1) - 1.0

    # ---- momentum -------------------------------------------------------
    mom_12_1 = close.shift(21) / close.shift(252) - 1.0
    F["mom_12_6"] = close.shift(126) / close.shift(252) - 1.0
    F["voladj_mom_12_1"] = mom_12_1 / _roll(ret, 252, "std").replace(0, np.nan)
    pos = (ret > 0).astype("float64").where(ret.notna())
    F["mom_consist_126"] = _roll(pos, 126, "mean").shift(21)
    if open_ is not None:
        intraday = close / open_.where(open_ > 0) - 1.0
        F["intraday_mom_126"] = _roll(intraday, 126, "sum").shift(21)
    else:
        F["intraday_mom_126"] = pd.DataFrame(np.nan, index=close.index,
                                             columns=close.columns)
    hi252 = _roll(close, 252, "max", mp=200)
    F["prox_52w"] = close.shift(21) / hi252.shift(21).replace(0, np.nan) - 1.0
    F["prox_52w_now"] = close / hi252.replace(0, np.nan) - 1.0

    # ---- reversal -------------------------------------------------------
    F["rev_1w"] = close / close.shift(5) - 1.0
    if vwap is not None:
        F["price_vs_vwap_20"] = close / _roll(vwap, 20, "mean").replace(0, np.nan) - 1.0
    else:
        F["price_vs_vwap_20"] = pd.DataFrame(np.nan, index=close.index,
                                             columns=close.columns)
    arr = ret.to_numpy("float64")
    T, N = arr.shape
    out_max = np.full((T, N), np.nan)
    if T >= 21:
        win = np.lib.stride_tricks.sliding_window_view(arr, 21, axis=0)
        with np.errstate(invalid="ignore"):
            srt = np.sort(win, axis=2)
            cnt = np.isfinite(win).sum(2)
            top5 = np.nanmean(srt[:, :, -5:], axis=2)
        out_max[20:] = np.where(cnt >= 15, top5, np.nan)
        del win, srt
    F["max5_21"] = pd.DataFrame(out_max, index=close.index, columns=close.columns)

    # ---- risk ------------------------------------------------------------
    dsr = ret.where(ret < 0)
    F["downside_vol_60"] = dsr.rolling(60, min_periods=20).std()
    F["ret_kurt_126"] = _roll(ret, 126, "kurt", mp=90)

    # ---- ownership: the delivered fraction --------------------------------
    if deliv_pct is not None:
        dp = deliv_pct.reindex(index=close.index, columns=close.columns)
        F["deliv_pct_60"] = _roll(dp, 60, "mean")
        sd = _roll(dp, 252, "std", mp=150)
        F["deliv_z_21"] = (_roll(dp, 21, "mean")
                           - _roll(dp, 252, "mean", mp=150)) / sd.replace(0, np.nan)
    else:
        for k in ("deliv_pct_60", "deliv_z_21"):
            F[k] = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    # ---- quality, from point-in-time fundamentals -------------------------
    if fundamentals:
        g = lambda k: (fundamentals[k].reindex(index=close.index, columns=close.columns)
                       .astype("float64") if k in fundamentals
                       else pd.DataFrame(np.nan, index=close.index, columns=close.columns))
        rev = g("ttm_revenue")
        nm = (g("ttm_net_profit") / rev.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        age = g("fund_age_days")
        from .pit_fundamentals import MAX_AGE_DAYS
        nm = nm.where(age <= MAX_AGE_DAYS)
        F["net_margin"] = nm
        F["margin_stability"] = -_roll(nm, 504, "std", mp=250)
    else:
        F["net_margin"] = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        F["margin_stability"] = pd.DataFrame(np.nan, index=close.index,
                                             columns=close.columns)

    for k in list(F):
        F[k] = F[k].replace([np.inf, -np.inf], np.nan)
    missing = [f for f in ALL_FACTORS if f not in F]
    if missing:
        raise KeyError(f"the shipped factor set names {missing}, which this "
                       f"module does not compute")
    if not last_row_only:
        return {k: F[k] for k in ALL_FACTORS}
    return pd.DataFrame({k: F[k].iloc[-1] for k in ALL_FACTORS})
