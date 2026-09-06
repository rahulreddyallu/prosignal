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


# ------------------------------------------------- the fit-window provenance
# A second and independent cut. The signs and weights were fitted over
# `v3.FIT_WINDOW`, which covers 293 of the shipped panel's 380 signal dates, so
# a figure pooled across the whole panel is neither an in-sample fit statistic
# nor an out-of-sample result. Every published table quoted the pooled number.
#
# Measured on the shipped panel at h=21, overlap-corrected:
#
#     OUT_OF_SAMPLE    87 dates   IC +0.0411   t +3.91
#     IN_SAMPLE       293 dates   IC +0.0579   t +6.82
#     STABLE_MODEL    150 dates   IC +0.0468   t +6.22
#     FULL_PANEL      380 dates   IC +0.0541   t +7.74
#
# The out-of-sample t clears Harvey-Liu-Zhu's 3.0 bar on 87 NON-OVERLAPPING
# observations, and the in-sample-to-out-of-sample decay is 29%.

def _dated_panel() -> pd.DataFrame:
    """A panel straddling the real fit window, so the split has both sides."""
    from prosignal.features import v3

    lo, hi = v3.FIT_WINDOW
    p = _panel(n_dates=300)
    # The panel is long -- one row per (date, name) -- so the dates are
    # remapped through the distinct values rather than assigned positionally.
    fresh = pd.bdate_range(str(lo), periods=p["date"].nunique(), freq="21B")
    p["date"] = p["date"].map(dict(zip(sorted(p["date"].unique()), fresh)))
    assert p["date"].max() > pd.Timestamp(hi), "fixture must reach past the fit"
    return p


def test_the_headline_window_is_the_out_of_sample_one():
    assert R.HEADLINE_WINDOW == "OUT_OF_SAMPLE"


def test_the_ranking_is_split_on_the_fit_window():
    from prosignal.features import v3

    names = [n for n, _ in R._ranking_windows(_dated_panel())]
    assert names[0] == R.HEADLINE_WINDOW, (
        "the out-of-sample row is reported first because it is the one a "
        "claim about the shipped model rests on"
    )
    assert "IN_SAMPLE" in names
    assert names[-1] == "FULL_PANEL", (
        "FULL_PANEL is reported last rather than dropped -- it is the longer "
        "record and the one every superseded figure came from"
    )

    lo, hi = v3.FIT_WINDOW
    by = dict(R._ranking_windows(_dated_panel()))
    assert (pd.to_datetime(by["OUT_OF_SAMPLE"]["date"]) > pd.Timestamp(hi)).all()
    assert (pd.to_datetime(by["IN_SAMPLE"]["date"]) <= pd.Timestamp(hi)).all()


def test_a_window_too_small_to_measure_is_not_reported():
    """A handful of dates is not an out-of-sample result, and printing one
    invites a comparison against noise."""
    p = _dated_panel()
    from prosignal.features import v3

    _, hi = v3.FIT_WINDOW
    after = [d for d in sorted(pd.to_datetime(p["date"]).unique())
             if d > pd.Timestamp(hi)]
    assert len(after) > 1
    p = p[~pd.to_datetime(p["date"]).isin(after[1:])]   # one OOS date left
    names = [n for n, _ in R._ranking_windows(p)]
    assert "OUT_OF_SAMPLE" not in names


def test_the_windows_do_not_silently_overlap_on_the_fit_split():
    by = dict(R._ranking_windows(_dated_panel()))
    ins = set(pd.to_datetime(by["IN_SAMPLE"]["date"]))
    oos = set(pd.to_datetime(by["OUT_OF_SAMPLE"]["date"]))
    assert not (ins & oos)


# ------------------------------------------------------- the decile profile
# `decile_monotonicity` compresses the whole shape into one rank correlation,
# and a profile that rises to D7 and falls away scores +0.33 there and looks
# healthy. The profile itself, measured on the shipped panel at h=63 as mean
# excess over each date's own cross-section:
#
#            D1     D2     D3     D4     D5     D6     D7     D8     D9    D10
#     IS  -2.77  -1.41  -0.93  -0.10  -0.22  +0.06  +0.70  +0.78  +1.44  +2.45
#     OOS -2.31  -1.27  -0.60  -0.65  +0.31  +1.23  +1.73  +0.64  +0.58  +0.36
#
# In sample it is monotone and D10 wins by a distance. Out of sample it PEAKS
# AT D7 and D10 is the sixth-best decile: D10-D6 is +2.39 in sample and -0.87
# out of it. The bottom generalises almost perfectly; the top does not
# generalise at all.
#
# The shipped book holds six names off the very top of D10.

def _wide_panel(n_dates: int = 40, n_names: int = 200) -> pd.DataFrame:
    """Cross-sections wide enough to actually cut into ten deciles.

    Carries two labels: one monotone in the score and one that rises to the
    70th percentile and falls away past it -- the shape the real panel has out
    of sample.
    """
    rng = np.random.default_rng(7)
    rows = []
    for d in pd.bdate_range("2020-01-01", periods=n_dates, freq="21B"):
        sc = rng.normal(size=n_names)
        pct = sc.argsort().argsort() / (n_names - 1)
        rows.append(pd.DataFrame({
            "date": d, "symbol": [f"S{i}" for i in range(n_names)],
            "score": sc,
            "y21": pct * 0.10 + rng.normal(0, 0.001, n_names),
            "y_hump": -abs(pct - 0.7) * 0.10 + rng.normal(0, 0.001, n_names),
        }))
    return pd.concat(rows, ignore_index=True)


def test_the_profile_reports_every_decile_not_just_the_top():
    out = R._ranking_results(_wide_panel(), (21,), 21)[0].decile_profile
    assert {f"d{i}" for i in range(1, 11)} <= set(out)
    assert out["n_dates"] > 0


def test_the_peak_decile_is_reported_rather_than_assumed():
    """The whole point. A report that only ever prints D10 cannot say that D10
    is not where the information is."""
    prof = R._decile_profile(_wide_panel(), "y21")
    assert 1 <= prof["peak_decile"] <= 10
    assert np.isfinite(prof["top_minus_d6"])


def test_top_minus_d6_is_d10_minus_d6():
    prof = R._decile_profile(_wide_panel(), "y21")
    assert prof["top_minus_d6"] == pytest.approx(prof["d10"] - prof["d6"])


def test_a_monotone_ranking_peaks_at_the_top_and_an_inverted_one_does_not():
    """The detector, exercised in both directions -- a peak-decile report that
    always says D10 would be indistinguishable from a broken one."""
    p = _wide_panel()
    assert R._decile_profile(p, "y21")["peak_decile"] == 10
    assert R._decile_profile(p, "y_hump")["peak_decile"] in (7, 8)
    assert R._decile_profile(p, "y21")["top_minus_d6"] > 0
    assert R._decile_profile(p, "y_hump")["top_minus_d6"] < 0


def test_a_thin_cross_section_produces_no_profile():
    """Ten names cannot be cut into ten deciles, and a decile of one name is a
    name."""
    p = _wide_panel().groupby("date").head(20)
    assert R._decile_profile(p, "y21") == {}
    assert R.MIN_NAMES_FOR_DECILES >= 100
