"""The twenty-two shipped factors, computed exactly as the search measured them.

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

from .v3 import ALL_FACTORS

__all__ = ["factor_frame", "LOOKBACK_SESSIONS"]

#: Sessions of history the block needs.
#:
#: NOT set by the longest WINDOW -- `prox_52w` reads 273 -- but by the longest
#: CHAIN. `resid_rev_21` is six rolling stages deep: a 21-session sum of a
#: residual, over a 126-session idiosyncratic vol, over a beta from a
#: 126-session covariance, over a 126-session demeaned benchmark. That reaches
#: roughly 375 sessions behind the decision row, and every stage carries a
#: `min_periods` relaxation, so a short window does not produce NaN -- it
#: produces a DIFFERENT NUMBER, silently.
#:
#: Measured against a 1,200-session reference on live data: at 300 (+15, the 315
#: Stage 4 actually read) `resid_rev_21` was wrong by up to 4.5e-2 on the last
#: row, and up to 0.507 on another date. At 400 and beyond it is exact to 2e-14.
#: Every other factor in the block was already bit-stable at 315. 420 is 400
#: with a quarter of slack.
LOOKBACK_SESSIONS = 420

#: Own sessions a SYMBOL needs before `resid_rev_21` means anything. The frame
#: has to be deep enough (LOOKBACK_SESSIONS) and so does the column: a name
#: listed 320 sessions ago has no 375-session chain however much history the
#: reader loaded. Measured convergence is at 400; below it the value is wrong
#: rather than absent, because every stage relaxes on `min_periods`.
RESID_REV_MIN_SESSIONS = 400


def _roll(df, w, how, mp=None):
    mp = mp if mp is not None else max(int(w * 0.6), 2)
    return getattr(df.rolling(w, min_periods=mp), how)()


def factor_frame(close: pd.DataFrame, open_: Optional[pd.DataFrame] = None,
                 vwap: Optional[pd.DataFrame] = None,
                 turnover: Optional[pd.DataFrame] = None,
                 deliv_pct: Optional[pd.DataFrame] = None,
                 bench_ret: Optional[pd.Series] = None,
                 fundamentals: Optional[Dict[str, pd.DataFrame]] = None,
                 last_row_only: bool = True):
    """The factor block. Rows are dates ascending, columns are symbols.

    ``bench_ret`` is the equal-weight return of the ELIGIBLE universe -- the
    market as it stood, not as today's survivors describe it. Only
    `resid_rev_21` uses it; the other twenty-one are own-series statistics, so a
    change in how the benchmark is assembled cannot move them.
    """
    F: Dict[str, pd.DataFrame] = {}
    ret = close / close.shift(1) - 1.0

    # ---- momentum -------------------------------------------------------
    F["mom_2_0"] = close / close.shift(42) - 1.0
    F["mom_3_1"] = close.shift(21) / close.shift(63) - 1.0
    mom_6_1 = close.shift(21) / close.shift(126) - 1.0
    mom_12_1 = close.shift(21) / close.shift(252) - 1.0
    F["mom_12_6"] = close.shift(126) / close.shift(252) - 1.0
    F["mom_accel"] = F["mom_3_1"] - mom_6_1
    F["voladj_mom_6_1"] = mom_6_1 / _roll(ret, 126, "std").replace(0, np.nan)
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

    # ---- risk, and the market residual the reversal factor needs ---------
    dsr = ret.where(ret < 0)
    F["downside_vol_60"] = dsr.rolling(60, min_periods=20).std()
    F["ret_kurt_126"] = _roll(ret, 126, "kurt", mp=90)
    cm = close.rolling(120, min_periods=72).max()
    dd = close / cm.replace(0, np.nan) - 1.0
    F["ulcer_120"] = np.sqrt((dd ** 2).rolling(120, min_periods=72).mean())

    if bench_ret is not None:
        b = pd.Series(bench_ret).reindex(close.index).astype("float64")
        bc = b - b.rolling(126, min_periods=90).mean()
        bvar = (bc * bc).rolling(126, min_periods=90).mean()
        beta = ret.mul(bc, axis=0).rolling(126, min_periods=90).mean() \
                  .div(bvar.replace(0, np.nan), axis=0)
        resid = ret.sub(_roll(ret, 126, "mean", mp=90)).sub(beta.mul(bc, axis=0))
        idio = _roll(resid, 126, "std", mp=90)
        rr = _roll(resid, 21, "sum", mp=15) / idio.replace(0, np.nan)
        # PER NAME, NOT ONLY PER FRAME. Loading enough sessions fixes the read;
        # it does not help a SYMBOL that has not traded for that long. Every
        # stage of the chain above relaxes on `min_periods`, so a name with 320
        # sessions of its own gets a number rather than a NaN -- the same defect
        # LOOKBACK_SESSIONS closes for the frame, one column at a time. On the
        # live universe 64 of 750 names sit between 300 (the eligibility floor)
        # and the depth this needs.
        #
        # NaN is the honest answer and it is cheap: `theme_subscore` averages
        # the factors a name HAS, so reversal falls back to its other three and
        # the name keeps a sub-score instead of losing the theme.
        seen = ret.notna().cumsum()
        F["resid_rev_21"] = rr.where(seen >= RESID_REV_MIN_SESSIONS)
    else:
        F["resid_rev_21"] = pd.DataFrame(np.nan, index=close.index,
                                         columns=close.columns)

    # ---- ownership: the delivered fraction --------------------------------
    if deliv_pct is not None:
        dp = deliv_pct.reindex(index=close.index, columns=close.columns)
        F["deliv_pct_60"] = _roll(dp, 60, "mean")
        F["deliv_chg_5"] = _roll(dp, 5, "mean", mp=3) - _roll(dp, 60, "mean")
        sd = _roll(dp, 252, "std", mp=150)
        F["deliv_z_21"] = (_roll(dp, 21, "mean")
                           - _roll(dp, 252, "mean", mp=150)) / sd.replace(0, np.nan)
    else:
        for k in ("deliv_pct_60", "deliv_chg_5", "deliv_z_21"):
            F[k] = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    # ---- quality, from point-in-time fundamentals -------------------------
    if fundamentals:
        g = lambda k: (fundamentals[k].reindex(index=close.index, columns=close.columns)
                       .astype("float64") if k in fundamentals
                       else pd.DataFrame(np.nan, index=close.index, columns=close.columns))
        rev = g("ttm_revenue")
        nm = (g("ttm_net_profit") / rev.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        age = g("fund_age_days")
        # ONE staleness limit, not two. `stage4_core_score.
        # max_fundamental_age_days` (450) governs the legacy family block and
        # this governs the shipped one; they disagreed by a month and nothing
        # reconciled them. MAX_AGE_DAYS is the stricter and is the one the v3
        # quality theme was measured under, so it wins -- the config key is now
        # documented as applying to the family block alone.
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
