"""The training panel must be the universe as it stood, not today's survivors.

The panel was built from ONE universe: the names the liquidity screen admits on
the most recent session, projected backwards over every training date. Measured
against the screen resolved properly per date:

    as of 2024-08-12   750 eligible, 203 of them (27%) absent from today's set
    as of 2021-07-19   523 eligible, 148 of them (28%) absent from today's set

A name eligible in 2024 that has since fallen out contributed no training row --
it was excluded for what happened afterwards. A name eligible today that was not
eligible in 2024 contributed 2024 rows it could never have been traded on. Both
directions are look-ahead selection, and they were in the fit AND in the
validation built from it.

Refitting on the corrected panel moves the model materially, in the shape
survivorship predicts:

    amihud (illiquidity)  +0.00737 -> -0.00917   sign flip
    beta_120              +0.00298 -> -0.00274   sign flip
    mom_6_1               +0.01389 -> +0.00288   -79%

Illiquidity and beta look rewarded when only the survivors are kept, because the
risky illiquid names that did not make it are missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.features.crosssec import MIN_LOOKBACK, build_panel, liquidity_mask


def _panel_frames(n_sessions=400, n_names=60, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_sessions)
    cols = [f"S{i:03d}" for i in range(n_names)]
    close = pd.DataFrame(
        100.0 * np.cumprod(1 + rng.normal(0.0004, 0.012, (n_sessions, n_names)), axis=0),
        index=idx, columns=cols,
    )
    turnover = pd.DataFrame(1e8, index=idx, columns=cols)
    return close, turnover


# ------------------------------------------------------------- the mask
def test_the_mask_admits_only_names_over_the_turnover_floor():
    close, turnover = _panel_frames()
    turnover.iloc[:, :20] = 1e6                       # far below the floor
    mask = liquidity_mask(close, turnover, min_adtv_inr=5e7, lookback_sessions=60,
                          max_names=750, min_history_sessions=100, min_price_inr=20.0)
    last = mask.iloc[-1]
    assert not last[close.columns[:20]].any()
    assert last[close.columns[20:]].all()


def test_the_mask_admits_only_names_over_the_price_floor():
    close, turnover = _panel_frames()
    close.iloc[:, :10] = 5.0
    mask = liquidity_mask(close, turnover, min_adtv_inr=5e7, lookback_sessions=60,
                          max_names=750, min_history_sessions=100, min_price_inr=20.0)
    assert not mask.iloc[-1][close.columns[:10]].any()


def test_the_mask_requires_listed_history():
    close, turnover = _panel_frames()
    close.iloc[:-50, :5] = np.nan                     # listed only 50 sessions ago
    mask = liquidity_mask(close, turnover, min_adtv_inr=5e7, lookback_sessions=60,
                          max_names=750, min_history_sessions=100, min_price_inr=20.0)
    assert not mask.iloc[-1][close.columns[:5]].any()


def test_history_counts_sessions_since_listing_not_prints():
    """A name that was suspended and came back has the listed history it has.
    Counting prints instead penalises it for the suspension, and that disagreed
    with the production screen on 10-15% of the universe."""
    close, turnover = _panel_frames()
    close.iloc[100:250, 0] = np.nan                   # a long halt, then resumes
    mask = liquidity_mask(close, turnover, min_adtv_inr=5e7, lookback_sessions=60,
                          max_names=750, min_history_sessions=300, min_price_inr=20.0)
    assert bool(mask.iloc[-1][close.columns[0]]), (
        "sessions since the first print exceed 300 even with the halt"
    )


def test_the_cap_is_applied_per_date_by_turnover():
    close, turnover = _panel_frames(n_names=60)
    turnover = turnover * np.arange(1, 61)            # S059 the most liquid
    mask = liquidity_mask(close, turnover, min_adtv_inr=1.0, lookback_sessions=60,
                          max_names=10, min_history_sessions=100, min_price_inr=1.0)
    kept = set(mask.columns[mask.iloc[-1].to_numpy()])
    assert len(kept) == 10
    assert kept == set(close.columns[-10:]), "the cap must keep the most liquid"


def test_the_mask_never_looks_forward():
    """Truncating the input must not change any date the truncated frame still
    holds. A mask that used future turnover would move."""
    close, turnover = _panel_frames()
    full = liquidity_mask(close, turnover, min_adtv_inr=5e7, lookback_sessions=60,
                          max_names=40, min_history_sessions=100, min_price_inr=20.0)
    cut = liquidity_mask(close.iloc[:300], turnover.iloc[:300], min_adtv_inr=5e7,
                         lookback_sessions=60, max_names=40,
                         min_history_sessions=100, min_price_inr=20.0)
    pd.testing.assert_frame_equal(full.iloc[:300], cut)


# ------------------------------------------------------------- the panel
def test_the_panel_drops_a_name_on_dates_it_was_not_eligible():
    close, turnover = _panel_frames(n_sessions=420)
    # S000 only becomes liquid in the last 60 sessions.
    turnover.iloc[:-60, 0] = 1e6
    mask = liquidity_mask(close, turnover, min_adtv_inr=5e7, lookback_sessions=60,
                          max_names=750, min_history_sessions=100, min_price_inr=20.0)
    panel = build_panel(close, turnover, horizon=21, step=21, min_names=5,
                        eligible=mask)
    early = panel[panel["date"] < close.index[-80]]
    assert not early.empty
    assert "S000" not in set(early["symbol"]), (
        "a name contributed rows on dates it could not have been traded on"
    )


def test_without_a_mask_the_panel_keeps_every_column():
    """The old behaviour, still reachable, so the change is opt-in per caller."""
    close, turnover = _panel_frames(n_sessions=420)
    turnover.iloc[:-60, 0] = 1e6
    panel = build_panel(close, turnover, horizon=21, step=21, min_names=5)
    assert "S000" in set(panel["symbol"])


def test_the_benchmark_is_built_from_eligible_names_only():
    """An equal-weight mean over today's survivors is not the market as it
    stood, and beta and residual momentum are both measured against it."""
    close, turnover = _panel_frames(n_sessions=420)
    # Half the names are never eligible and are given a wildly different path.
    turnover.iloc[:, 30:] = 1e6
    close.iloc[:, 30:] *= np.linspace(1.0, 6.0, len(close))[:, None]
    mask = liquidity_mask(close, turnover, min_adtv_inr=5e7, lookback_sessions=60,
                          max_names=750, min_history_sessions=100, min_price_inr=20.0)
    with_mask = build_panel(close, turnover, horizon=21, step=21, min_names=5,
                            eligible=mask)
    without = build_panel(close, turnover, horizon=21, step=21, min_names=5)
    a = with_mask[with_mask["symbol"] == "S000"].set_index("date")["beta_120_r"]
    b = without[without["symbol"] == "S000"].set_index("date")["beta_120_r"]
    common = a.index.intersection(b.index)
    assert len(common) > 2
    assert not np.allclose(a.loc[common], b.loc[common]), (
        "the ineligible names still moved the benchmark"
    )


def test_an_all_nan_benchmark_does_not_warn_or_crash():
    """Under a point-in-time universe the earliest dates have no eligible names,
    so the equal-weight market is undefined there. That is a real state, and it
    must degrade to NaN quietly rather than emit `Mean of empty slice` on every
    refit."""
    import warnings

    from prosignal.features.crosssec import _features_at

    close, turnover = _panel_frames()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = _features_at(close, turnover, len(close) - 1,
                           np.full(len(close), np.nan))
    assert out["resid_mom"].isna().all()
    assert out["beta_120"].isna().all()


def test_a_short_benchmark_is_cut_to_the_overlap_not_broadcast():
    """An unequal `&` raises; a silent pad would align a stock's returns against
    the wrong sessions."""
    from prosignal.features.crosssec import _features_at

    close, turnover = _panel_frames()
    out = _features_at(close, turnover, len(close) - 1, np.zeros(80))
    assert "beta_120" in out.columns


# ------------------------------------------------- the callers must not undo it
def test_the_production_fit_reads_the_store_unrestricted_on_a_refit():
    """The defect was in the CALLER, not the panel builder. Restricting the read
    to today's universe makes the mask a no-op: names that were eligible then
    and are not now have no column to be admitted from."""
    import inspect

    from prosignal.stages import stage4_core_score as s4

    src = inspect.getsource(s4._cross_sectional_model)
    assert "refitting = cached is None" in src
    assert "symbols=None if refitting else list(symbols)" in src, (
        "a refit must span every name that was ever eligible"
    )
    assert "liquidity_mask(" in src and "eligible=eligible" in src
    assert "score_symbols=list(symbols)" in src, (
        "today's ranking still covers today's eligible universe"
    )


def test_both_research_entry_points_build_a_point_in_time_panel():
    """The holdout claim and the portfolio CPCV both came through these. Either
    one left restricted would keep publishing a survivorship-biased number."""
    import inspect

    from prosignal import cli

    for fn in (cli.cmd_research_cpcv, cli.cmd_research_portfolio):
        src = inspect.getsource(fn)
        assert "liquidity_mask(" in src, f"{fn.__name__} builds no per-date mask"
        assert "eligible=eligible" in src, f"{fn.__name__} does not apply it"
        assert "symbols=symbols" not in src, (
            f"{fn.__name__} still restricts the panel read to today's universe"
        )
