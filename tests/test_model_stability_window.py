"""A statistic pooled over the whole panel describes several models at once.

`score_frame` re-caps the theme blend over the themes a name actually has. Per
name that is the right thing to do. Across TIME it means the composite is not
one function: the fundamentals feed reaches almost nobody at the start of the
panel and most of the universe at the end --

    quality_sub coverage   2018 0.0%   2020 1.5%   2021 23.3%
                           2023 50.2%  2026 86.4%

and mean themes per name rises 2.99 -> 4.86 over the same span. So a 2019 score
is a three-theme blend and a 2026 score is a five-theme blend, and any figure
quoted over the whole panel is a weighted average across those models with the
weighting set by when a vendor's coverage improved.

The fix is not to throw the early dates away. It is to say which model each row
describes: the ranking table now reports FULL_PANEL and STABLE_MODEL side by
side, where STABLE_MODEL starts at the first date from which every theme stays
above `STABLE_MODEL_FLOOR`. On the shipped panel that is 2023-07-21 -- 150 of
380 dates, and the honest sample size for a claim about the composite as it now
stands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal import v3_monitor as vm
from prosignal.validation import results as R

THEME_SUBS = [t + "_sub" for t in vm.THEMES]


def _panel(n_dates: int = 200, n_names: int = 40, seed: int = 3) -> pd.DataFrame:
    """A panel shaped like the real one: one theme's coverage ramps from zero.

    The ramp is the whole point -- a fixture where every theme is always
    present cannot distinguish a working detector from one that returns the
    first date unconditionally.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_dates, freq="21B")
    rows = []
    for i, d in enumerate(dates):
        ramp = i / (n_dates - 1)          # 0.0 -> 1.0 across the panel
        for s in range(n_names):
            sub = {c: float(rng.normal()) for c in THEME_SUBS}
            if rng.random() > ramp:       # quality arrives late, like the feed
                sub["quality_sub"] = np.nan
            rows.append({
                "date": d, "symbol": f"S{s:02d}",
                **sub,
                "n_themes": float(sum(np.isfinite(v) for v in sub.values())),
                "score": float(rng.normal()),
                "y21": float(rng.normal()) * 0.05,
            })
    return pd.DataFrame(rows)


def test_theme_availability_is_reported_per_date():
    av = vm.theme_availability(_panel())
    assert list(av.columns) == THEME_SUBS
    assert av["momentum_sub"].eq(1.0).all()
    assert av["quality_sub"].iloc[0] < av["quality_sub"].iloc[-1]


def test_the_stable_window_starts_where_the_thin_theme_clears_the_floor():
    p = _panel()
    start = vm.stable_model_window(p)
    assert start is not None
    av = vm.theme_availability(p)
    assert (av.loc[start:] >= vm.STABLE_MODEL_FLOOR).all().all()
    assert start > av.index[0], (
        "a detector that returns the first date has not detected anything; "
        "quality coverage starts at zero in this fixture"
    )


def test_it_is_the_date_from_which_it_STAYS_above_not_the_first_touch():
    """A single date clearing the bar and falling back is not the point at
    which the model settled, and taking it as one restores exactly the pooling
    this finding is about."""
    p = _panel()
    av = vm.theme_availability(p)
    first_touch = (av >= vm.STABLE_MODEL_FLOOR).all(axis=1).idxmax()
    start = vm.stable_model_window(p)
    assert start >= first_touch


def test_no_stable_window_when_a_theme_never_arrives():
    p = _panel()
    p["quality_sub"] = np.nan
    assert vm.stable_model_window(p) is None, (
        "with a theme permanently absent there is no span over which the "
        "model is the current model, and the reporting must say so rather "
        "than quietly fall back to the whole panel"
    )


def test_the_floor_the_document_names_is_the_floor_the_monitor_applies():
    """The number is mirrored into `results` for the prose; if the two drift
    the document describes a bar that was never applied."""
    assert R._STABLE_FLOOR == vm.STABLE_MODEL_FLOOR


def test_the_ranking_carries_the_window_it_was_measured_on():
    p = _panel()
    full = R._ranking_results(p, (21,), 21)
    assert full and full[0].window == "FULL_PANEL"
    assert np.isfinite(full[0].n_themes_mean)

    start = vm.stable_model_window(p)
    stable = R._ranking_results(
        p[p["date"] >= start], (21,), 21, window="STABLE_MODEL")
    assert stable and stable[0].window == "STABLE_MODEL"
    assert stable[0].n_dates < full[0].n_dates
    assert stable[0].n_themes_mean > full[0].n_themes_mean, (
        "the restricted window exists because it is a richer model; if the "
        "themes/name does not rise, it is not measuring what it claims"
    )
    assert stable[0].min_theme_coverage >= vm.STABLE_MODEL_FLOOR


def test_a_short_stable_window_is_not_published_as_a_measurement():
    assert R.MIN_STABLE_DATES >= 30


def test_the_deploy_reference_admits_which_span_it_covers():
    from prosignal.validation import v3_panel as vp
    assert vp.DEPLOY_REFERENCE["spans_variable_theme_coverage"] is True
