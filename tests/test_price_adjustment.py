"""Splits are share-count changes, not returns.

Left unadjusted a 1:10 split reads as a -90% session. Measured on the live
store, 80 of 100 split/bonus events showed a drop beyond 30%, affecting 72 of
the 200 index names, and every momentum or volatility figure computed across
one of those dates was describing an accounting event.
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from prosignal.data.store import DataStore
from prosignal.data.universe import UniverseResolver


@pytest.fixture
def sessions(live_cfg):
    return DataStore(live_cfg.paths.curated, live_cfg.paths.snapshots).price_sessions()


def _panel(cfg, adjust):
    store = DataStore(cfg.paths.curated, cfg.paths.snapshots, adjust_prices=adjust)
    s = store.price_sessions()
    syms = sorted(set(UniverseResolver(store, cfg.params).resolve("NIFTY 200", s[-1]).symbols))
    px = store.read_prices(symbols=syms, start=s[0], end=s[-1],
                           columns=["date", "symbol", "close"])
    px["date"] = pd.to_datetime(px["date"])
    wide = px.pivot_table(index="date", columns="symbol", values="close",
                          aggfunc="last", observed=True)
    return wide.sort_index().pct_change(fill_method=None).stack().dropna()


def test_adjustment_removes_the_phantom_split_crashes(live_cfg):
    raw = _panel(live_cfg, adjust=False)
    adj = _panel(live_cfg, adjust=True)
    extreme_raw = int((raw.abs() > 0.30).sum())
    extreme_adj = int((adj.abs() > 0.30).sum())
    # Measured on the live store: 112 -> 41.
    assert extreme_adj < extreme_raw * 0.6, "adjustment should remove most artifacts"

    # Residual is expected and is a limit of the ratio feed, not of the
    # adjustment. yfinance records a single ratio per ex-date, so a compound
    # event lands short: BAJFINANCE's June-2025 1:2 split plus 4:1 bonus is
    # stored as 0.5 and still prints near -80% after adjustment. The test
    # pins the improvement rather than pretending the feed is complete.
    assert float(adj.min()) > float(raw.min()), "worst print must improve"


def test_adjusted_prices_are_the_default(live_cfg):
    store = DataStore(live_cfg.paths.curated, live_cfg.paths.snapshots)
    assert store.adjust_prices is True


def test_adjustment_leaves_the_requested_columns_alone(live_cfg, sessions):
    store = DataStore(live_cfg.paths.curated, live_cfg.paths.snapshots)
    want = ["date", "symbol", "close"]
    out = store.read_prices(symbols=None, start=sessions[-40], end=sessions[-1],
                            columns=want)
    assert list(out.columns) == want, "adjustment must not leak helper columns"


def test_turnover_is_not_rescaled(live_cfg, sessions):
    """Rupee turnover is invariant across a share-count change, so it must not
    be adjusted even though price and volume are."""
    a = DataStore(live_cfg.paths.curated, live_cfg.paths.snapshots, adjust_prices=False)
    b = DataStore(live_cfg.paths.curated, live_cfg.paths.snapshots, adjust_prices=True)
    kw = dict(symbols=["TATAINVEST"], start=sessions[-400], end=sessions[-1],
              columns=["date", "symbol", "turnover"])
    left, right = a.read_prices(**kw), b.read_prices(**kw)
    if left.empty or right.empty:
        pytest.skip("symbol absent from this store")
    np.testing.assert_allclose(
        left["turnover"].to_numpy(), right["turnover"].to_numpy(), rtol=1e-9
    )


def test_the_label_is_close_to_close_but_entry_is_the_next_open(live_cfg, sessions):
    """The model is trained on close[t] -> close[t+H]; the engine enters at
    open[t+1]. The overnight gap between them is not capturable.

    Measured on the holdout it accounts for 2.6% of the excess return
    (+0.126%/period against +4.888%), so the label is left close-to-close.
    This test exists so the assumption is checked rather than assumed: if the
    gap ever becomes material, the label construction has to change.
    """
    import numpy as np
    import pandas as pd

    store = DataStore(live_cfg.paths.curated, live_cfg.paths.snapshots)
    syms = sorted(set(UniverseResolver(store, live_cfg.params)
                      .resolve("NIFTY 200", sessions[-1]).symbols))
    px = store.read_prices(symbols=syms, start=sessions[-260], end=sessions[-1],
                           columns=["date", "symbol", "open", "close"])
    px["date"] = pd.to_datetime(px["date"])
    close = px.pivot_table(index="date", columns="symbol", values="close",
                           aggfunc="last", observed=True)
    open_ = px.pivot_table(index="date", columns="symbol", values="open",
                           aggfunc="last", observed=True)
    gap = (open_.shift(-1) / close - 1.0).stack().dropna()

    # A typical overnight gap is small. If the median absolute gap ever exceeds
    # 1%, close-to-close labels stop being a fair proxy for the tradeable return.
    assert float(gap.abs().median()) < 0.01
