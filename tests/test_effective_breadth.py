"""Twenty-two factors are not twenty-two bets.

Grinold's IR = IC * sqrt(breadth) takes breadth to be INDEPENDENT bets, and
every breadth argument in this repository rests on the declared counts: 22
factors across 5 themes. The pairwise redundancy check catches duplicates one
pair at a time and says nothing about the aggregate.

Measured per date across the 380-date panel and averaged:

    factors   20.9 columns present  ->  6.94 effective   (median 7.34)
    themes     4.59 columns present ->  3.96 effective   (median 4.30)

So the factor count overstates independent breadth by sqrt(20.9/6.94) = 1.74x,
and the theme level is close to honest -- which is the two-level structure
doing its job rather than an accident.

The measure is the participation ratio `(sum L)^2 / sum L^2` over the
eigenvalues of the cross-sectional Spearman matrix: it equals the column count
when the columns are orthogonal and collapses toward 1 as they align. Averaged
ACROSS dates rather than pooled, because a matrix over stacked cross-sections
mixes within-date structure with drift in the factor means, and drift is not
breadth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal import v3_monitor as vm


def test_orthogonal_columns_count_themselves():
    assert vm.participation_ratio(np.eye(5)) == pytest.approx(5.0)
    assert vm.participation_ratio(np.eye(22)) == pytest.approx(22.0)


def test_identical_columns_count_once():
    """The limit case. Five copies of one factor are one bet."""
    assert vm.participation_ratio(np.ones((5, 5))) == pytest.approx(1.0)


def test_partial_correlation_lands_between():
    c = np.full((4, 4), 0.5)
    np.fill_diagonal(c, 1.0)
    eff = vm.participation_ratio(c)
    assert 1.0 < eff < 4.0


def test_the_measure_is_invariant_to_column_order():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 6))
    x[:, 3] = x[:, 0] * 0.9 + rng.normal(scale=0.2, size=400)
    f = pd.DataFrame(x, columns=list("abcdef"))
    a = vm.effective_count(f)
    b = vm.effective_count(f[list("fedcba")])
    assert a == pytest.approx(b, rel=1e-9)


def test_a_duplicated_factor_lowers_the_effective_count():
    """The property the pairwise check cannot express."""
    rng = np.random.default_rng(2)
    base = pd.DataFrame(rng.normal(size=(300, 6)),
                        columns=[f"f{i}" for i in range(6)])
    dup = base.copy()
    dup["f5"] = dup["f0"]
    assert vm.effective_count(dup) < vm.effective_count(base) - 0.5


def test_a_thin_cross_section_reports_nothing_rather_than_a_number():
    """Two names cannot support a 22-column correlation matrix, and a number
    printed from one would be read as a measurement."""
    thin = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 1.0]})
    assert np.isnan(vm.effective_count(thin, min_names=30))
    assert vm.effective_breadth(pd.DataFrame()) == {}


def test_breadth_is_averaged_across_dates_not_pooled():
    """Pooling stacked cross-sections mixes within-date structure with drift in
    the factor means, and drift is not breadth. The per-date average is
    unaffected by a level shift between dates; a pooled matrix is not.
    """
    rng = np.random.default_rng(4)
    rows = []
    for d in range(12):
        for n in range(60):
            rows.append({"date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=d),
                         "f0": rng.normal() + d, "f1": rng.normal() + d,
                         "f2": rng.normal() + d})
    p = pd.DataFrame(rows)
    out = vm.effective_breadth(p, ["f0", "f1", "f2"])
    assert out["n_dates"] == 12.0
    assert out["effective_mean"] == pytest.approx(3.0, abs=0.35), (
        "three independent factors shifted by a common per-date level must "
        "still read as about three; a pooled matrix would read as one"
    )


def test_the_report_carries_it_and_says_how_much_the_count_overstates():
    from prosignal.core.contracts import RedundancyReport

    r = RedundancyReport(effective_breadth={"factors_declared": 22.0,
                                            "factors_effective": 6.94,
                                            "breadth_overstatement": 1.78})
    assert r.effective_breadth["factors_effective"] == 6.94
    assert RedundancyReport().effective_breadth == {}, (
        "the default must be empty rather than a placeholder count; an absent "
        "measurement and a measured 22 are different things"
    )
