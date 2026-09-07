"""The second-largest weight in the model rests on an expired constraint.

Each theme's weight was capped at the share of names the theme can speak about,
measured once over the fit window and frozen. `V3_SEARCH.md` §6 records why,
and it was the right call: fitted without the constraint, `quality` took 40%+
of the composite while only 19% of names had fundamentals at all, which ranks
the 19% and the 81% by two different models and calls the result one score.
Quality's shipped 0.18991 IS its 0.1899 coverage cap.

The fundamentals feed has since caught up. Measured on the shipped panel,
`quality_sub` coverage runs 0.363 over the fit window and 0.837 over the last
year of data, against a declared 0.1899 -- a 4.4x drift. `min(0.40, 0.837)` is
0.40, so a refreshed cap would not cut quality at all, and the search record
says its pre-cap weight was 40%+.

So the weight is held down by a constraint that has expired, and the constant
recording that constraint says nothing about it.

These tests do NOT refit. Refreshing the cap roughly doubles the quality weight
and pushes momentum off its own cap; that is a model change which spends trials
and opens an epoch, and it is a decision to take against this measurement
rather than a consequence of it. What they pin is that the staleness is
measured, named, and cannot go quiet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.features import v3

SUBS = [t + "_sub" for t in v3.THEMES]


def _panel(coverage: dict, n_names: int = 200, n_dates: int = 10):
    rng = np.random.default_rng(3)
    rows = []
    for d in pd.bdate_range("2026-01-01", periods=n_dates, freq="21B"):
        for i in range(n_names):
            r = {"date": d, "symbol": f"S{i}"}
            for t in v3.THEMES:
                have = (i / n_names) < coverage.get(t, 1.0)
                r[t + "_sub"] = float(rng.normal()) if have else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def test_a_panel_matching_the_declared_coverage_is_not_stale():
    """The detector must be able to say 'fine', or its alarm means nothing."""
    p = _panel({t: th.coverage for t, th in v3.THEMES.items()})
    assert v3.stale_coverage_caps(p) == []


def test_the_quality_cap_is_flagged_when_coverage_catches_up():
    """The finding, as one assertion."""
    p = _panel({**{t: th.coverage for t, th in v3.THEMES.items()},
                "quality": 0.84})
    msgs = v3.stale_coverage_caps(p)
    assert len(msgs) == 1 and msgs[0].startswith("quality:")
    assert "no longer binds" in msgs[0], (
        "a cap that has drifted is one thing; a cap that has drifted PAST the "
        "structural 0.40 and stopped binding altogether is the finding, and "
        "the message has to separate them"
    )


def test_drift_below_the_tolerance_is_not_flagged():
    """A feed that moved a little has not invalidated the constraint, and an
    alarm that fires on noise is an alarm nobody reads."""
    p = _panel({**{t: th.coverage for t, th in v3.THEMES.items()},
                "quality": v3.THEMES["quality"].coverage * 1.3})
    assert v3.stale_coverage_caps(p) == []
    assert v3.COVERAGE_DRIFT_TOLERANCE >= 1.5


def test_the_drift_report_separates_declared_from_binding():
    p = _panel({**{t: th.coverage for t, th in v3.THEMES.items()},
                "quality": 0.84})
    d = v3.coverage_drift(p)["quality"]
    assert d["declared"] == pytest.approx(0.1899)
    assert d["measured"] == pytest.approx(0.84, abs=0.01)
    assert d["ratio"] > 4.0
    assert d["declared_binds"] == 1.0, "0.1899 is below the 0.40 cap"
    assert d["measured_binds"] == 0.0, "0.84 is above it"


def test_a_theme_that_was_never_capped_by_coverage_stays_unflagged():
    """Momentum's coverage is 0.9988 and its weight is set by the structural
    0.40 cap, not by coverage. It could halve and still not be cut by coverage,
    so drift there changes no weight and must not raise an alarm."""
    p = _panel({**{t: th.coverage for t, th in v3.THEMES.items()},
                "momentum": 0.60})
    assert all(not m.startswith("momentum") for m in v3.stale_coverage_caps(p))


def test_a_cap_that_STARTS_binding_is_flagged_too():
    """The other direction, and the more dangerous one: a theme weighted for a
    coverage it no longer has is over-weighted on the names it cannot see."""
    p = _panel({**{t: th.coverage for t, th in v3.THEMES.items()},
                "momentum": 0.20})
    msgs = [m for m in v3.stale_coverage_caps(p) if m.startswith("momentum")]
    assert len(msgs) == 1
    assert "does now" in msgs[0]


def test_the_shipped_weight_still_equals_the_declared_cap():
    """If this ever fails the weights were refitted, and the finding above has
    either been acted on or silently undone. Either way somebody must say
    which."""
    q = v3.THEMES["quality"]
    assert q.weight == pytest.approx(q.coverage, abs=2e-5), (
        "quality's shipped weight is its coverage cap; they have separated"
    )


def test_nothing_refits_on_import():
    """The guard measures. It must not quietly change the model."""
    assert v3.THEMES["quality"].weight == pytest.approx(0.18991)
    assert v3.THEMES["momentum"].weight == pytest.approx(0.40)
