"""A sign that fails out of sample is a different failure from one that decays.

The signs ARE the model. `V3_SEARCH.md` records that two of them read backwards
against their own theme name and pins them with a test "so nobody 'corrects'
it" -- which is right, a measured sign is a measurement. It is also exactly why
a sign that fails off the data it was chosen on has to be found by something
other than reading it.

`review_factors` watches a ROLLING window. That answers "is this drifting" and
not "was this ever true out of sample", and nothing asked the second question.

Measured on the shipped panel at h=63 against `v3.FIT_WINDOW`, 78 out-of-sample
dates, raw Spearman per date and averaged: **20 of 22 factors hold their shipped
sign**, several strongly -- `deliv_z_21` at t +9.13, `ret_kurt_126` at t -6.42,
`mom_12_6` at t +6.64. That is the headline and it is a good one.

Two do not:

    mom_3_1     (momentum)  ships +1,  OOS IC -0.0248 at t -2.18
    net_margin  (quality)   ships -1,  OOS IC +0.0207 at t +2.52

Both are significant in the OTHER direction, which is not the same thing as
decaying to zero: the factor still carries information and the model has the
sign backwards. `net_margin` matters most, because the theme it sits in carries
18.99% of the composite and is itself out-of-sample indistinguishable from zero
(quality_sub +0.0103 at t +1.41 against an in-sample +0.0547 at t +7.31) -- and
because Q11 shows that theme's coverage cap has expired, so a refresh would
roughly DOUBLE its weight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal import v3_monitor as vm
from prosignal.features import v3


def _panel(sign_of: dict, n_dates: int = 20, n_names: int = 200, seed: int = 5):
    """A panel after the fit window closed, with each factor's realised sign
    set by `sign_of`."""
    rng = np.random.default_rng(seed)
    _, hi = v3.FIT_WINDOW
    dates = pd.bdate_range(pd.Timestamp(hi) + pd.Timedelta(days=30),
                           periods=n_dates, freq="21B")
    rows = []
    for d in dates:
        base = {"date": d, "symbol": [f"S{i}" for i in range(n_names)]}
        frame = pd.DataFrame(base)
        y = rng.normal(0, 0.05, n_names)
        for theme, spec in v3.THEMES.items():
            for factor, _ in spec.factors:
                realised = sign_of.get(factor, 0.0)
                frame[factor + "_r"] = (realised * y * 12.0
                                        + rng.normal(0, 1.0, n_names))
        frame["y63"] = y
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _agreeing() -> dict:
    return {f: s for _, spec in v3.THEMES.items() for f, s in spec.factors}


def test_a_panel_where_every_sign_holds_flags_nothing():
    """The detector must be able to say 'fine'."""
    out = vm.out_of_sample_signs(_panel(_agreeing()), "y63")
    assert not out.empty
    assert out["agrees"].all(), out[~out["agrees"]][["factor", "ic_oos"]]
    assert vm.flipped_signs(_panel(_agreeing()), "y63") == []


def test_a_flipped_sign_is_found_and_named():
    """The finding, as one assertion."""
    signs = _agreeing()
    signs["net_margin"] = -signs["net_margin"]
    msgs = vm.flipped_signs(_panel(signs), "y63")
    assert len(msgs) == 1
    assert msgs[0].startswith("net_margin (quality)")
    assert "OTHER direction" in msgs[0]


def test_a_decay_to_zero_is_not_reported_as_a_flip():
    """They are different failures and only one of them is actionable. A
    factor that stopped working needs dropping; a factor whose sign is
    backwards is still carrying information the model is using wrongly."""
    signs = _agreeing()
    signs["net_margin"] = 0.0
    frame = vm.out_of_sample_signs(_panel(signs), "y63")
    row = frame[frame["factor"] == "net_margin"].iloc[0]
    assert abs(row["t_oos"]) < vm.SIGN_FLIP_T
    assert row["flipped"] is False or not bool(row["flipped"])


def test_it_only_looks_after_the_fit_window_closed():
    """A sign measured on the dates it was chosen on agrees by construction."""
    lo, hi = v3.FIT_WINDOW
    p = _panel(_agreeing())
    p["date"] = pd.Timestamp(lo) + pd.Timedelta(days=30)   # all in-sample
    assert vm.out_of_sample_signs(p, "y63").empty


def test_a_thin_cross_section_is_skipped_rather_than_scored():
    p = _panel(_agreeing(), n_names=20)
    assert vm.out_of_sample_signs(p, "y63", min_names=50).empty


def test_the_flip_threshold_is_a_significance_bar_not_a_sign_test():
    """Half the factors will read the wrong way by chance on a short window.
    Flagging every one of them is an alarm nobody reads."""
    assert vm.SIGN_FLIP_T >= 2.0


def test_the_shipped_panel_still_holds_twenty_of_twenty_two_signs():
    """The real measurement, and the headline is that the engine is mostly
    sound. If this drops sharply, the model has stopped describing the data
    rather than one factor having turned."""
    from pathlib import Path

    cache = Path("/tmp/panel_cad.parquet")
    if not cache.is_file():
        pytest.skip("no cached panel in this checkout")
    frame = vm.out_of_sample_signs(pd.read_parquet(cache), "y63")
    if frame.empty:
        pytest.skip("the cached panel carries no out-of-sample dates")
    assert int(frame["agrees"].sum()) >= 18, (
        f"only {int(frame['agrees'].sum())} of {len(frame)} signs hold out of "
        f"sample:\n{frame[~frame['agrees']][['factor', 'ic_oos', 't_oos']]}"
    )
    flipped = set(frame[frame["flipped"]]["factor"])
    assert flipped == {"mom_3_1", "net_margin"}, (
        f"the set of out-of-sample sign flips has moved: {sorted(flipped)}"
    )
