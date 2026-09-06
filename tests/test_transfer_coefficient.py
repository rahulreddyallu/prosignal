"""A measured IC that does not appear in the book has two possible causes.

Grinold: IR = TC x IC x sqrt(breadth). The transfer coefficient is the
correlation between the positions a book takes and the positions the signal
implies. This engine measures the ORDERING well and builds a six-name long-only
book off the very top of it, and nothing could tell whether a signal that fails
to show up in that book is a broken signal or a book that cannot hold the
signal's opinion. Those call for opposite responses.

Measured out of sample -- after the fit window closed, h=63, 87 dates, median
750 names, GROSS of cost:

    shape                        kind         gross excess       t  transfer  names
    D10-D1                       LONG_SHORT       +2.6729%   +1.08    +0.762    147
    top-half minus bottom-half   LONG_SHORT       +1.8120%   +1.54    +0.822    735
    D6-D8                        LONG_ONLY        +1.1934%   +2.42    +0.296    221
    top-half                     LONG_ONLY        +0.9052%   +1.54    +0.822    368
    D10                          LONG_ONLY        +0.3242%   +0.26    +0.553     74
    top6  (SHIPPED)              LONG_ONLY        -1.5196%   -0.67    +0.206      6

Three things follow. The shipped shape is the worst of the six and its
out-of-sample excess is NEGATIVE. The decile it is drawn from is
indistinguishable from zero on its own. And D6-D8 -- the band where the decile
profile peaks out of sample -- is the only long-only shape that clears t = 2,
at a transfer coefficient of 0.296 against the book's 0.206.

So the long-only constraint is NOT what binds. A tradeable long-only shape
carries the signal; the six-name concentration does not. The long-short shapes
earn more gross and clear no significance bar either, and India has no retail
borrow market to trade them in.

GROSS, and the word is doing work: D6-D8 holds 221 names and pays turnover a
six-name book does not. This module prices none of it. It is a construction
diagnostic and not a proposal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.validation import transfer as T


def _panel(n_dates: int = 30, n_names: int = 400, seed: int = 11,
           shape: str = "monotone") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2025-01-01", periods=n_dates, freq="21B"):
        sc = rng.normal(size=n_names)
        pct = sc.argsort().argsort() / (n_names - 1)
        if shape == "monotone":
            y = (pct - 0.5) * 0.10
        else:                                  # peaks at the 70th percentile
            y = (0.5 - abs(pct - 0.7)) * 0.10
        rows.append(pd.DataFrame({
            "date": d, "symbol": [f"S{i}" for i in range(n_names)],
            "score": sc, "y63": y + rng.normal(0, 0.002, n_names)}))
    return pd.concat(rows, ignore_index=True)


def _by_name(results):
    return {r.shape: r for r in results}


def test_a_book_holding_exactly_the_signal_transfers_it_fully():
    """The top-half shape is the long side of the signal's own view, so its
    transfer coefficient must be high. A near-zero reading here would mean the
    coefficient is measuring something else."""
    p = _panel()
    n = 400
    out = _by_name(T.evaluate(p, "y63", T.standard_shapes(n, 6)))
    assert out["top-half"].transfer_coefficient > 0.7


def test_a_six_name_book_transfers_almost_none_of_it():
    """The finding, as one assertion. Six names out of several hundred cannot
    express a view the signal holds on every name."""
    p = _panel()
    out = _by_name(T.evaluate(p, "y63", T.standard_shapes(400, 6)))
    assert out["top6"].transfer_coefficient < out["top-half"].transfer_coefficient
    assert out["top6"].avg_positions == 6


def test_the_shipped_shape_wins_when_the_ranking_really_is_monotone():
    """The detector must be able to vindicate the book, or its verdict on the
    real panel is just a property of the code."""
    out = _by_name(T.evaluate(_panel(shape="monotone"), "y63",
                              T.standard_shapes(400, 6)))
    assert out["top6"].gross_excess > out["D6-D8"].gross_excess


def test_D6_to_D8_wins_when_the_profile_peaks_in_the_middle():
    """Which is the shape the real out-of-sample panel has."""
    out = _by_name(T.evaluate(_panel(shape="hump"), "y63",
                              T.standard_shapes(400, 6)))
    assert out["D6-D8"].gross_excess > out["top6"].gross_excess
    assert out["D6-D8"].gross_excess > out["D10"].gross_excess


def test_long_short_shapes_are_marked_untradeable():
    """India has no retail borrow market worth the name. A table that ranks a
    short book alongside a long one without saying so is proposing something
    the engine cannot do."""
    out = _by_name(T.evaluate(_panel(), "y63", T.standard_shapes(400, 6)))
    assert out["D10-D1"].tradeable_long_only is False
    assert out["D10-D1"].kind == T.LONG_SHORT
    assert out["top6"].tradeable_long_only is True
    assert "borrow" in T.table(list(out.values()))


def test_excess_is_measured_against_the_equal_weight_cross_section():
    """Otherwise a rising market reads as skill on every long-only shape."""
    rng = np.random.default_rng(2)
    rows = []
    for d in pd.bdate_range("2025-01-01", periods=20, freq="21B"):
        # No relationship between score and return, and a large common drift.
        rows.append(pd.DataFrame({
            "date": d, "symbol": [f"S{i}" for i in range(300)],
            "score": rng.normal(size=300),
            "y63": 0.20 + rng.normal(0, 0.01, 300)}))
    p = pd.concat(rows, ignore_index=True)
    out = _by_name(T.evaluate(p, "y63", T.standard_shapes(300, 6)))
    assert abs(out["top-half"].gross_excess) < 0.005, (
        "a 20% common drift with no signal must not show up as excess"
    )


def test_a_thin_cross_section_is_skipped_rather_than_ranked():
    p = _panel(n_names=40)
    assert T.evaluate(p, "y63", T.standard_shapes(40, 6)) == []
    assert T.MIN_NAMES >= 100


def test_the_overlap_correction_is_applied_when_a_horizon_is_given():
    """Dates 5 sessions apart against a 63-session label are not independent,
    and the naive t is inflated by about sqrt(VIF)."""
    p = _panel()
    naive = _by_name(T.evaluate(p, "y63", T.standard_shapes(400, 6)))
    corrected = _by_name(T.evaluate(p, "y63", T.standard_shapes(400, 6),
                                    stride=5, horizon=63))
    assert abs(corrected["top-half"].t_stat) < abs(naive["top-half"].t_stat)


def test_the_table_says_the_figures_are_gross():
    """D6-D8 holds 221 names on the real panel and pays turnover a six-name
    book does not. A table that omits the word proposes a strategy."""
    text = T.table(T.evaluate(_panel(), "y63", T.standard_shapes(400, 6)))
    assert "GROSS" in text
    assert "does not price it" in text
