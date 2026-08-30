"""Wide candidate factor library.

Every factor is a (T x N) matrix whose value at row t uses only data at or
before t. Nothing reads t+1. Factors are computed once on the full grid and
sliced at signal dates.

Families: momentum, reversal, volatility/risk, liquidity/size, delivery
(the NSE-specific ownership-conviction proxy), trend, seasonality, and a
fundamental block built from filing-dated statements.
"""
from __future__ import annotations
import numpy as np, pandas as pd

EPS = 1e-12


def _sh(df: pd.DataFrame, k: int) -> pd.DataFrame:
    return df.shift(k)


def _roll(df, w, how, mp=None):
    mp = mp if mp is not None else max(int(w * 0.6), 2)
    r = df.rolling(w, min_periods=mp)
    return getattr(r, how)()


def build_factors(close, high, low, open_, vwap, volume, turnover, trades,
                  deliv_pct, bench_ret) -> dict:
    """bench_ret: Series of equal-weight eligible-universe returns, index=dates."""
    F = {}
    ret = close / close.shift(1) - 1.0
    logc = np.log(close.where(close > 0))
    b = bench_ret.reindex(close.index)

    # ---------- momentum ------------------------------------------------
    def px_mom(a, bk):
        return close.shift(a) / close.shift(bk) - 1.0
    F["mom_1_0"]   = px_mom(0, 21)
    F["mom_2_0"]   = px_mom(0, 42)
    F["mom_3_1"]   = px_mom(21, 63)
    F["mom_6_1"]   = px_mom(21, 126)
    F["mom_12_1"]  = px_mom(21, 252)
    F["mom_12_6"]  = px_mom(126, 252)
    F["mom_9_1"]   = px_mom(21, 189)
    F["mom_accel"] = F["mom_3_1"] - F["mom_6_1"]

    vol60 = _roll(ret, 60, "std")
    vol20 = _roll(ret, 20, "std")
    vol126 = _roll(ret, 126, "std")
    vol252 = _roll(ret, 252, "std")
    F["vol_20"], F["vol_60"], F["vol_252"] = vol20, vol60, vol252
    F["vol_ratio_20_120"] = vol20 / _roll(ret, 120, "std").replace(0, np.nan)
    F["voladj_mom_6_1"] = F["mom_6_1"] / vol126.replace(0, np.nan)
    F["voladj_mom_12_1"] = F["mom_12_1"] / vol252.replace(0, np.nan)

    pos = (ret > 0).astype("float32").where(ret.notna())
    F["mom_consist_126"] = _roll(pos, 126, "mean").shift(21)
    # Frog-in-the-pan (Da, Gurun & Warachka): continuous information is priced
    # more slowly. ID = sign(PRET) * (%neg - %pos).
    pct_pos = _roll(pos, 126, "mean").shift(21)
    F["fip_6"] = np.sign(F["mom_6_1"]) * ((1.0 - pct_pos) - pct_pos)

    # Intraday vs overnight decomposition (Lou, Polk & Skouras 2019).
    intraday = (close / open_.where(open_ > 0) - 1.0)
    overnight = (open_ / close.shift(1) - 1.0)
    F["intraday_mom_126"] = _roll(intraday, 126, "sum").shift(21)
    F["overnight_mom_126"] = _roll(overnight, 126, "sum").shift(21)
    F["intraday_mom_21"] = _roll(intraday, 21, "sum")
    F["overnight_mom_21"] = _roll(overnight, 21, "sum")

    # 52-week nearness (George & Hwang), with and without the reversal skip.
    hi252 = _roll(close, 252, "max", mp=200)
    lo252 = _roll(close, 252, "min", mp=200)
    F["prox_52w"] = close.shift(21) / hi252.shift(21).replace(0, np.nan) - 1.0
    F["prox_52w_now"] = close / hi252.replace(0, np.nan) - 1.0
    F["dist_low_52w"] = close / lo252.replace(0, np.nan) - 1.0

    # ---------- reversal ------------------------------------------------
    F["rev_1w"] = close / close.shift(5) - 1.0
    F["rev_2w"] = close / close.shift(10) - 1.0
    F["rev_1m_scaled"] = F["mom_1_0"] / vol60.replace(0, np.nan)
    r21 = ret.rolling(21, min_periods=15)
    F["max5_21"] = r21.max()            # cheap proxy kept alongside the true one
    F["min5_21"] = r21.min()
    # true MAX(5): mean of the five largest daily returns in 21 sessions
    arr = ret.to_numpy("float32")
    T, N = arr.shape
    out_max = np.full((T, N), np.nan, "float32")
    out_min = np.full((T, N), np.nan, "float32")
    win = np.lib.stride_tricks.sliding_window_view(arr, 21, axis=0)  # (T-20,N,21)
    with np.errstate(invalid="ignore"):
        srt = np.sort(win, axis=2)
        cnt = np.isfinite(win).sum(2)
        np.seterr(all="ignore"); top5 = np.nanmean(srt[:, :, -5:], axis=2)
        bot5 = np.nanmean(srt[:, :, :5], axis=2)
    ok = cnt >= 15
    out_max[20:] = np.where(ok, top5, np.nan)
    out_min[20:] = np.where(ok, bot5, np.nan)
    F["max5_21"] = pd.DataFrame(out_max, index=close.index, columns=close.columns)
    F["min5_21"] = pd.DataFrame(out_min, index=close.index, columns=close.columns)
    del win, srt, arr, out_max, out_min

    # ---------- market-residual block ------------------------------------
    bv = b.to_numpy("float64")
    bmean126 = pd.Series(bv, index=b.index).rolling(126, min_periods=90).mean()
    bc = pd.Series(bv, index=b.index) - bmean126
    bvar126 = (bc * bc).rolling(126, min_periods=90).mean()
    cov126 = ret.mul(bc, axis=0).rolling(126, min_periods=90).mean()
    beta126 = cov126.div(bvar126.replace(0, np.nan), axis=0)
    F["beta_126"] = beta126
    bmean252 = pd.Series(bv, index=b.index).rolling(252, min_periods=180).mean()
    bc2 = pd.Series(bv, index=b.index) - bmean252
    bvar252 = (bc2 * bc2).rolling(252, min_periods=180).mean()
    F["beta_252"] = ret.mul(bc2, axis=0).rolling(252, min_periods=180).mean().div(
        bvar252.replace(0, np.nan), axis=0)
    # downside beta: co-movement on down-market days only
    dn = bc.where(bc < 0)
    F["beta_down"] = ret.mul(dn, axis=0).rolling(126, min_periods=40).mean().div(
        (dn * dn).rolling(126, min_periods=40).mean().replace(0, np.nan), axis=0)

    rmean126 = _roll(ret, 126, "mean", mp=90)
    resid = ret.sub(rmean126).sub(beta126.mul(bc, axis=0))
    F["idio_vol_126"] = _roll(resid, 126, "std", mp=90)
    m3 = _roll(resid ** 3, 126, "mean", mp=90)
    F["idio_skew_126"] = m3 / (F["idio_vol_126"].replace(0, np.nan) ** 3)
    F["ret_skew_126"] = _roll(ret, 126, "skew", mp=90)
    F["ret_kurt_126"] = _roll(ret, 126, "kurt", mp=90)
    rs = _roll(resid, 252, "sum", mp=180) - _roll(resid, 21, "sum", mp=15)
    F["resid_mom_252_21"] = rs / F["idio_vol_126"].replace(0, np.nan)
    F["resid_mom_126_21"] = (_roll(resid, 126, "sum", mp=90)
                             - _roll(resid, 21, "sum", mp=15)) / F["idio_vol_126"].replace(0, np.nan)
    F["resid_rev_21"] = _roll(resid, 21, "sum", mp=15) / F["idio_vol_126"].replace(0, np.nan)

    # ---------- risk / drawdown ------------------------------------------
    dsr = ret.where(ret < 0)
    F["downside_vol_60"] = dsr.rolling(60, min_periods=20).std()
    for w in (120, 252):
        cm = close.rolling(w, min_periods=int(w * 0.6)).max()
        dd = close / cm.replace(0, np.nan) - 1.0
        F[f"max_dd_{w}"] = dd.rolling(w, min_periods=int(w * 0.6)).min()
        F[f"ulcer_{w}"] = np.sqrt((dd ** 2).rolling(w, min_periods=int(w * 0.6)).mean())
    hl = np.log(high / low.replace(0, np.nan))
    F["parkinson_60"] = np.sqrt((hl ** 2).rolling(60, min_periods=30).mean() / (4 * np.log(2)))
    co = np.log(close / open_.replace(0, np.nan))
    F["garman_klass_60"] = np.sqrt((0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2)
                                   .rolling(60, min_periods=30).mean().clip(lower=0))
    tr = pd.concat([(high - low),
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()]).groupby(level=0).max()
    tr = tr.reindex(close.index)
    F["atr_pct_14"] = tr.rolling(14, min_periods=10).mean() / close.replace(0, np.nan)

    # ---------- liquidity / size ------------------------------------------
    adtv60 = _roll(turnover, 60, "mean")
    adtv20 = _roll(turnover, 20, "mean")
    adtv120 = _roll(turnover, 120, "mean")
    F["log_adtv_60"] = np.log1p(adtv60)
    F["turnover_trend"] = np.log1p(adtv20) - np.log1p(adtv120)
    F["amihud_60"] = _roll(ret.abs() / turnover.replace(0, np.nan), 60, "mean") * 1e9
    F["amihud_trend"] = (_roll(ret.abs() / turnover.replace(0, np.nan), 21, "mean")
                         - _roll(ret.abs() / turnover.replace(0, np.nan), 126, "mean", mp=80)) * 1e9
    zero = (ret.abs() < 1e-9).astype("float32").where(ret.notna())
    F["zero_ret_60"] = _roll(zero, 60, "mean")
    F["avg_trade_size"] = np.log1p(_roll(turnover / trades.replace(0, np.nan), 60, "mean"))
    F["volume_shock_5"] = np.log1p(adtv20.shift(0)) - np.log1p(adtv60)
    F["vol_of_turnover"] = _roll(np.log1p(turnover), 60, "std")
    # Corwin-Schultz high-low spread estimator (2012), 2-day version, smoothed.
    hh = pd.concat([high, high.shift(1)]).groupby(level=0).max().reindex(close.index)
    ll = pd.concat([low, low.shift(1)]).groupby(level=0).min().reindex(close.index)
    beta_cs = (np.log(high / low.replace(0, np.nan)) ** 2
               + np.log(high.shift(1) / low.shift(1).replace(0, np.nan)) ** 2)
    gamma_cs = np.log(hh / ll.replace(0, np.nan)) ** 2
    k = 3 - 2 * np.sqrt(2.0)
    alpha_cs = (np.sqrt(2 * beta_cs) - np.sqrt(beta_cs)) / k - np.sqrt(gamma_cs / k)
    spread = 2 * (np.exp(alpha_cs) - 1) / (1 + np.exp(alpha_cs))
    F["cs_spread_60"] = _roll(spread.clip(lower=0), 60, "mean")

    # ---------- delivery (India-specific ownership conviction) -------------
    dp = deliv_pct
    F["deliv_pct_60"] = _roll(dp, 60, "mean")
    F["deliv_trend"] = _roll(dp, 21, "mean") - _roll(dp, 126, "mean", mp=80)
    dsd = _roll(dp, 252, "std", mp=150)
    F["deliv_z_21"] = (_roll(dp, 21, "mean") - _roll(dp, 252, "mean", mp=150)) / dsd.replace(0, np.nan)
    F["deliv_chg_5"] = _roll(dp, 5, "mean", mp=3) - _roll(dp, 60, "mean")
    F["deliv_x_turnover"] = F["deliv_pct_60"] * np.log1p(adtv60)
    # rupee value that actually settled, relative to the name's own norm
    dval = dp * turnover
    F["deliv_value_trend"] = np.log1p(_roll(dval, 21, "mean")) - np.log1p(_roll(dval, 126, "mean", mp=80))

    # ---------- trend / technical -----------------------------------------
    ma20 = _roll(close, 20, "mean"); ma50 = _roll(close, 50, "mean")
    ma200 = _roll(close, 200, "mean", mp=150)
    F["dist_50dma"] = close / ma50.replace(0, np.nan) - 1.0
    F["dist_200dma"] = close / ma200.replace(0, np.nan) - 1.0
    F["ma_50_200"] = ma50 / ma200.replace(0, np.nan) - 1.0
    F["price_vs_vwap_20"] = close / _roll(vwap, 20, "mean").replace(0, np.nan) - 1.0
    up = ret.clip(lower=0); dnr = (-ret).clip(lower=0)
    rs14 = _roll(up, 14, "mean", mp=10) / _roll(dnr, 14, "mean", mp=10).replace(0, np.nan)
    F["rsi_14"] = 100 - 100 / (1 + rs14)
    # trend R^2 over 120 sessions: regression of log price on time
    w = 120
    t = np.arange(w, dtype="float64"); t = t - t.mean()
    sst = float((t * t).sum())
    lp = logc
    m_lp = _roll(lp, w, "mean", mp=80)
    cov_t = lp.mul(0).copy()
    # sum(t_c * y) via convolution
    lparr = lp.to_numpy("float64")
    kern = t[::-1]
    num = np.full(lparr.shape, np.nan)
    valid = np.isfinite(lparr)
    lp0 = np.where(valid, lparr, 0.0)
    from numpy.lib.stride_tricks import sliding_window_view as swv
    sw = swv(lp0, w, axis=0)
    cnt = swv(valid.astype("float64"), w, axis=0).sum(2)
    num[w - 1:] = np.where(cnt >= 80, (sw * t).sum(2), np.nan)
    slope = num / sst
    var_y = _roll(lp, w, "var", mp=80) * (w - 1)
    F["trend_r2_120"] = ((slope ** 2) * sst) / var_y.replace(0, np.nan)
    F["trend_slope_120"] = slope * 252
    del sw, lp0, num

    # ---------- seasonality (Heston & Sadka, India calendar) ---------------
    mon = pd.Series(close.index.month, index=close.index)
    r21f = close / close.shift(21) - 1.0
    same_month = pd.DataFrame(np.nan, index=close.index, columns=close.columns, dtype="float32")
    for m in range(1, 13):
        rows = mon[mon == m].index
        if len(rows) < 40:
            continue
        sub = r21f.loc[rows]
        # mean of PRIOR same-month 21-day returns only -- expanding, shifted
        prior = sub.shift(1).expanding(min_periods=2).mean()
        same_month.loc[rows] = prior.to_numpy("float32")
    F["seasonal_same_month"] = same_month

    for k in list(F):
        v = F[k]
        if not isinstance(v, pd.DataFrame):
            v = pd.DataFrame(v, index=close.index, columns=close.columns)
        F[k] = v.replace([np.inf, -np.inf], np.nan).astype("float32")
    return F


FAMILY = {
    "momentum": ["mom_1_0", "mom_2_0", "mom_3_1", "mom_6_1", "mom_12_1", "mom_12_6",
                 "mom_9_1", "mom_accel", "voladj_mom_6_1", "voladj_mom_12_1",
                 "mom_consist_126", "fip_6", "intraday_mom_126", "overnight_mom_126",
                 "prox_52w", "prox_52w_now", "dist_low_52w",
                 "resid_mom_252_21", "resid_mom_126_21"],
    "reversal": ["rev_1w", "rev_2w", "rev_1m_scaled", "max5_21", "min5_21",
                 "resid_rev_21", "intraday_mom_21", "overnight_mom_21"],
    "risk":     ["vol_20", "vol_60", "vol_252", "vol_ratio_20_120", "downside_vol_60",
                 "idio_vol_126", "idio_skew_126", "ret_skew_126", "ret_kurt_126",
                 "beta_126", "beta_252", "beta_down", "max_dd_120", "max_dd_252",
                 "ulcer_120", "ulcer_252", "parkinson_60", "garman_klass_60", "atr_pct_14"],
    "liquidity": ["log_adtv_60", "turnover_trend", "amihud_60", "amihud_trend",
                  "zero_ret_60", "avg_trade_size", "volume_shock_5", "vol_of_turnover",
                  "cs_spread_60"],
    "delivery": ["deliv_pct_60", "deliv_trend", "deliv_z_21", "deliv_chg_5",
                 "deliv_x_turnover", "deliv_value_trend"],
    "trend":    ["dist_50dma", "dist_200dma", "ma_50_200", "price_vs_vwap_20",
                 "rsi_14", "trend_r2_120", "trend_slope_120"],
    "seasonal": ["seasonal_same_month"],
}
