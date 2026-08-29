"""Market-regime features, point-in-time.

Every series at date t is built from sessions <= t. These drive the entry gate:
when the gate is shut the book opens nothing new and holds cash, which is the
NO TRADE state the product is required to be able to reach.
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import data as D_, universe as U


def build() -> pd.DataFrame:
    m = D_.build()
    dates = pd.DatetimeIndex(m["dates"]); syms = list(m["symbols"])
    close = pd.DataFrame(m["close"], index=dates, columns=syms)
    turn = pd.DataFrame(m["turnover"], index=dates, columns=syms)
    adjf = pd.DataFrame(m["adj_factor"], index=dates, columns=syms)
    elig = U.eligible_mask(close, turn, adjf, max_names=750)
    # CHAINED equal-weight return index, not a mean of prices: the eligible set
    # changes every session, so a price mean jumps whenever a name enters or
    # leaves and the "index" then records composition as if it were a return.
    px = close.where(elig)
    r = (px / px.shift(1) - 1.0).mean(axis=1)
    r = r.fillna(0.0)
    mkt = (1.0 + r).cumprod()

    out = pd.DataFrame(index=dates)
    out["mkt"] = mkt
    out["ma50"] = mkt.rolling(50, min_periods=30).mean()
    out["ma200"] = mkt.rolling(200, min_periods=150).mean()
    out["above_ma200"] = (mkt > out["ma200"]).astype(float)
    out["above_ma50"] = (mkt > out["ma50"]).astype(float)
    out["vol21"] = r.rolling(21, min_periods=15).std() * np.sqrt(252)
    out["vol252_med"] = out["vol21"].rolling(252, min_periods=150).median()
    out["vol_ratio"] = out["vol21"] / out["vol252_med"].replace(0, np.nan)
    out["dd"] = mkt / mkt.cummax() - 1.0
    above50 = (close > close.rolling(50, min_periods=30).mean()).where(elig)
    out["breadth"] = above50.sum(axis=1) / elig.sum(axis=1).replace(0, np.nan)
    out["breadth_ma21"] = out["breadth"].rolling(21, min_periods=10).mean()
    out["mkt_mom_126"] = mkt / mkt.shift(126) - 1.0
    # 12-month absolute momentum, the classic time-series filter
    out["mkt_mom_252"] = mkt / mkt.shift(252) - 1.0
    return out


GATES = {
    "always":       lambda g: True,
    "ma200":        lambda g: g["above_ma200"] > 0.5,
    "ma50":         lambda g: g["above_ma50"] > 0.5,
    "tsmom12":      lambda g: g["mkt_mom_252"] > 0,
    "vol_calm":     lambda g: g["vol_ratio"] < 1.5,
    "ma200_vol":    lambda g: (g["above_ma200"] > 0.5) & (g["vol_ratio"] < 1.5),
    "breadth":      lambda g: g["breadth_ma21"] > 0.40,
    "ma200_breadth": lambda g: (g["above_ma200"] > 0.5) | (g["breadth_ma21"] > 0.55),
    "dd_shallow":   lambda g: g["dd"] > -0.12,
}


def gate_series(reg: pd.DataFrame, name: str, signal_dates) -> dict:
    s = GATES[name](reg)
    if isinstance(s, bool):
        return {pd.Timestamp(d): True for d in signal_dates}
    s = s.fillna(True)
    s = s.reindex(pd.DatetimeIndex(signal_dates), method="ffill").fillna(True)
    return {pd.Timestamp(d): bool(v) for d, v in s.items()}


if __name__ == "__main__":
    reg = build()
    reg.to_parquet("/home/claude/psr/cache/regime.parquet")
    print(reg.tail(3).round(3).to_string())
    for g in GATES:
        s = GATES[g](reg)
        if isinstance(s, bool):
            print(f"{g:16s} always open")
        else:
            print(f"{g:16s} open {float(s.fillna(True).mean()):.1%} of sessions")
