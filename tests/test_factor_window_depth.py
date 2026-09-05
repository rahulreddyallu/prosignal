"""A factor must not change value because the reader loaded a longer window.

Two properties, and only one of them was ever tested.

NO LOOKAHEAD -- the value at T is unchanged when sessions AFTER T are added.
`v3_factors` passes this on all 22 factors and always did; the module docstring
claims it and an independent probe on live data confirms it.

WINDOW INVARIANCE -- the value at T is unchanged when sessions BEFORE the
window start are added. `resid_rev_21` failed this. It is six rolling stages
deep (a 21-session sum of a residual, over a 126-session idiosyncratic vol, over
a beta from a 126-session covariance, over a 126-session demeaned benchmark),
which reaches roughly 375 sessions behind the decision row, and every stage
carries a `min_periods` relaxation -- so at the 315 sessions Stage 4 read it did
not return NaN, it returned a DIFFERENT NUMBER. Measured against a
1,200-session reference on live data: wrong by up to 4.5e-2 on one date and
0.507 on another, across 118 of 120 names, while every other factor was
bit-stable at 315.

`LOOKBACK_SESSIONS` is therefore set by the longest CHAIN, not the longest
window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prosignal.features import v3, v3_factors


def _panel(n_days: int, n_sym: int = 12, seed: int = 5):
    """A synthetic OHLCV panel long enough to exercise the deepest chain."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    syms = [f"S{i:02d}" for i in range(n_sym)]
    steps = rng.normal(0.0005, 0.018, (n_days, n_sym))
    close = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=syms)
    open_ = close.shift(1).fillna(close.iloc[0])
    vwap = (close + open_) / 2.0
    turnover = pd.DataFrame(rng.uniform(1e7, 5e8, (n_days, n_sym)), index=idx, columns=syms)
    deliv = pd.DataFrame(rng.uniform(20, 90, (n_days, n_sym)), index=idx, columns=syms)
    bench = close.mean(axis=1)
    return close, open_, vwap, turnover, deliv, bench / bench.shift(1) - 1.0


def _last_row(depth: int, total: int = 1300):
    """The factor block at the final date, computed over the trailing `depth`
    sessions of the same longer panel."""
    close, open_, vwap, turnover, deliv, bench = _panel(total)
    sl = slice(-depth, None)
    return v3_factors.factor_frame(
        close.iloc[sl], open_.iloc[sl], vwap.iloc[sl], turnover.iloc[sl],
        deliv.iloc[sl], bench.iloc[sl])


def test_the_declared_lookback_is_deep_enough_for_every_factor():
    """The gate. Everything the block computes must have converged by
    `LOOKBACK_SESSIONS`, measured against a much longer window."""
    shipped = _last_row(v3_factors.LOOKBACK_SESSIONS)
    reference = _last_row(1200)
    worst = {}
    for f in v3.ALL_FACTORS:
        a, b = shipped[f], reference[f]
        m = a.notna() & b.notna()
        if not m.any():
            continue
        scale = max(float(b[m].abs().max()), 1.0)
        worst[f] = float((a[m] - b[m]).abs().max()) / scale
    bad = {f: d for f, d in worst.items() if d > 1e-9}
    assert not bad, (
        f"these factors have not converged at LOOKBACK_SESSIONS="
        f"{v3_factors.LOOKBACK_SESSIONS}: "
        + ", ".join(f"{f} off by {d:.2e}" for f, d in sorted(bad.items()))
    )


def test_a_short_window_yields_nothing_rather_than_something_wrong():
    """The property that replaces the defect, and it is stronger than "the
    constant is bigger now".

    At 300 sessions the chain cannot be evaluated. What it used to return was a
    number -- every stage relaxes on `min_periods`, so the shortfall showed up
    as a plausible value rather than as an absence. It now returns NaN, and
    `theme_subscore` averages the factors a name HAS, so reversal falls back to
    its other three instead of carrying a fabricated one.
    """
    short = _last_row(300)
    assert short["resid_rev_21"].isna().all(), (
        "a window too short for the residual chain must produce NaN, not a "
        "number computed off a truncated one"
    )
    assert short[["rev_1w", "max5_21", "price_vs_vwap_20"]].notna().any().all(), (
        "the rest of the reversal theme must survive, or the guard has taken "
        "the whole theme with it"
    )


def test_a_name_with_too_little_of_its_own_history_gets_no_residual_reversal():
    """Depth is a property of the COLUMN as well as the frame. A name listed
    320 sessions ago has no 375-session chain however much history the reader
    loaded -- 64 of the 750 live universe names sit in that gap, above the
    300-session eligibility floor and below what this factor needs."""
    close, open_, vwap, turnover, deliv, bench = _panel(1300)
    short_name = close.columns[0]
    close = close.copy()
    close.iloc[:-320, close.columns.get_loc(short_name)] = np.nan
    block = v3_factors.factor_frame(close, open_, vwap, turnover, deliv, bench)
    assert pd.isna(block.loc[short_name, "resid_rev_21"])
    assert block["resid_rev_21"].notna().sum() >= len(close.columns) - 1, (
        "only the short name should lose the factor"
    )


def test_lookback_covers_the_residual_reversal_chain():
    """The arithmetic behind the constant, so it is not a magic number: the
    deepest chain is bench-demean(126) -> beta(126) -> idio(126) -> sum(21)."""
    assert v3_factors.LOOKBACK_SESSIONS >= 126 * 3 - 2 + 21, (
        "LOOKBACK_SESSIONS is shorter than the residual-reversal chain reaches"
    )


def test_no_factor_reads_a_session_after_the_decision_row():
    """Unchanged and re-pinned beside its neighbour: the two properties are
    easy to confuse and only one of them was ever broken."""
    close, open_, vwap, turnover, deliv, bench = _panel(700)
    cut = 650
    truncated = v3_factors.factor_frame(
        close.iloc[:cut], open_.iloc[:cut], vwap.iloc[:cut], turnover.iloc[:cut],
        deliv.iloc[:cut], bench.iloc[:cut])
    full = v3_factors.factor_frame(
        close, open_, vwap, turnover, deliv, bench, last_row_only=False)
    at_t = pd.DataFrame({f: full[f].iloc[cut - 1] for f in v3.ALL_FACTORS})
    for f in v3.ALL_FACTORS:
        a, b = truncated[f], at_t[f]
        m = a.notna() & b.notna()
        if not m.any():
            continue
        np.testing.assert_allclose(
            a[m].to_numpy(), b[m].to_numpy(), rtol=1e-9, atol=1e-12,
            err_msg=f"{f} changed at T when sessions after T were added")
