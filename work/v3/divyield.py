"""Dividend yield: the value theme's one buildable factor. TRAIN ONLY.

WHY THIS AND NOT THE EIGHT VALUE FACTORS ALREADY BUILT. `sales_to_price`,
`book_to_price`, `fcf_yield`, `ebitda_to_ev` and the rest need a balance sheet,
and this store's balance-sheet history begins in 2023 -- the median training
date has ZERO names with a book value, so none of them could be screened at all.
A dividend needs only a price and a payment record, and the payment record here
runs dense from 2017.

IT IS STILL A MINORITY FACTOR: 181 of the 750-name live universe have any
dividend history, so a `value` theme built on it would be capped at ~24%
coverage, in the same way `quality` is capped at 19%.

SPLIT INVARIANCE, the same problem market cap had. Prices in this panel are
adjusted (`adj_close = raw_close * adj_factor`), so a rupee dividend paid before
a 1:10 split is not comparable to the price after it. Each payment is therefore
converted into the adjusted currency of its own ex-date before being summed:

    yield_t = sum over ex in (t-365, t] of ( amount_ex * adj_factor_ex )
              / adj_close_t

POINT-IN-TIME. The ex-date is used, not the announcement date. A dividend is
announced BEFORE it goes ex, so keying on the ex-date is strictly conservative:
the number is certainly public by then. Keying on the announcement would be
better and this store does not carry one.
"""
from __future__ import annotations
import re
import numpy as np, pandas as pd

D = "/mnt/user-data/uploads/Pro Stock Signal BOT/data/curated"
_AMT = re.compile(r"dividend\s+([0-9]*\.?[0-9]+)", re.I)


def dividend_events() -> pd.DataFrame:
    ca = pd.read_parquet(f"{D}/corporate_actions.parquet")
    d = ca[ca["action_type"] == "dividend"].copy()
    d["amount"] = d["raw_details"].astype(str).str.extract(_AMT, expand=False).astype(float)
    d = d.dropna(subset=["amount"])
    d = d[d["amount"] > 0]
    d["ex_date"] = pd.to_datetime(d["ex_date"]).dt.normalize()
    return d[["symbol", "ex_date", "amount"]].sort_values(["symbol", "ex_date"])


def build(panel: pd.DataFrame, px: pd.DataFrame, window_days: int = 365) -> pd.Series:
    """Trailing-`window_days` dividend yield for every (date, symbol) in `panel`.

    `px` must carry date, symbol, close (adjusted) and adj_factor.
    """
    ev = dividend_events()
    fac = px[["date", "symbol", "adj_factor"]].copy()
    fac["date"] = pd.to_datetime(fac["date"]).dt.normalize()
    # the adjustment factor in force on the ex-date, so the payment is expressed
    # in the same currency as the adjusted price it will be divided by
    ev = ev.merge(fac.rename(columns={"date": "ex_date"}), on=["symbol", "ex_date"],
                  how="left")
    # a payment on a non-trading date takes the next session's factor
    miss = ev["adj_factor"].isna()
    if miss.any():
        f2 = fac.sort_values(["symbol", "date"])
        ev = ev.sort_values(["symbol", "ex_date"])
        filled = pd.merge_asof(ev[miss].sort_values("ex_date"),
                               f2.sort_values("date").rename(columns={"date": "ex_date"}),
                               on="ex_date", by="symbol", direction="forward",
                               suffixes=("", "_f"))
        ev.loc[miss, "adj_factor"] = filled["adj_factor_f"].to_numpy()
    ev["adj_amount"] = ev["amount"] * ev["adj_factor"].fillna(1.0)

    out = pd.Series(np.nan, index=panel.index, dtype="float64")
    pn = panel[["date", "symbol", "close"]].copy()
    pn["date"] = pd.to_datetime(pn["date"]).dt.normalize()
    by = {s: g for s, g in ev.groupby("symbol", sort=False)}
    w = pd.Timedelta(days=int(window_days))
    for sym, g in pn.groupby("symbol", sort=False):
        e = by.get(sym)
        if e is None or e.empty:
            continue                       # no history -> NaN, never zero
        ed = e["ex_date"].to_numpy("datetime64[ns]")
        amt = e["adj_amount"].to_numpy("float64")
        cum = np.concatenate([[0.0], np.cumsum(amt)])
        dts = g["date"].to_numpy("datetime64[ns]")
        hi = np.searchsorted(ed, dts, side="right")
        lo = np.searchsorted(ed, dts - w.to_timedelta64(), side="right")
        paid = cum[hi] - cum[lo]
        cl = g["close"].to_numpy("float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            out.loc[g.index] = np.where(cl > 0, paid / cl, np.nan)
    return out
