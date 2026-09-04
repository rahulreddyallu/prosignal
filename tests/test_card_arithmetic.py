"""What the panel prints has to be checkable by the person reading it.

The theme table renders, per row, a weight, a sector percentile and a
contribution, under a heading that says "sums to the score". The contributions
did sum to the score. The weights did not produce them: `viewmodel` served
`Theme.weight` -- the fit-time vector -- beside a contribution computed from the
renormalised one, so every row was out by exactly 1/den, 1.2344x on a
four-theme name. A reader multiplying the two printed numbers got the wrong
answer, on every card, on every run.

Two tests already existed for "contributions sum to the score" and both passed
throughout. Nothing asserted that the DISPLAYED weight was the weight used,
which is why the defect survived 1,638 tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.presentation.viewmodel import (V3_THEME_LABELS, _contributions,
                                              _scorer_used)
from prosignal.features import v3


def _card(themes) -> dict:
    """A card in the shape `rundetail` serialises."""
    return {"factors": {
        name: {"standardised": z, "weight": w, "contribution": c,
               "available": True, "tier": "v3_theme", "members": []}
        for name, z, w, c in themes}}


# =============================================================================
# the arithmetic
# =============================================================================

def test_the_weight_shown_produces_the_contribution_shown():
    rows = _contributions(_card([
        ("momentum",  0.72539, 0.40000, 0.29016),
        ("ownership", 0.60622, 0.27710, 0.16798),
        ("risk",      0.94041, 0.16223, 0.15256),
        ("reversal",  0.82902, 0.16068, 0.13320),
    ])["factors"], tier="v3_theme")
    assert len(rows) == 4
    for r in rows:
        assert r["z"] * r["coefficient"] == pytest.approx(r["contribution"], abs=5e-4), (
            f"{r['factor']}: the card shows weight {r['coefficient']} beside "
            f"contribution {r['contribution']}, and they do not agree"
        )


def test_the_weights_on_a_card_sum_to_one():
    """The reader is entitled to read the weight column as a share of the
    score. Four themes summing to 0.81 says the other 19% went somewhere the
    card does not name."""
    rows = _contributions(_card([
        ("momentum",  0.72539, 0.40000, 0.29016),
        ("ownership", 0.60622, 0.27710, 0.16798),
        ("risk",      0.94041, 0.16223, 0.15256),
        ("reversal",  0.82902, 0.16068, 0.13320),
    ])["factors"], tier="v3_theme")
    assert sum(r["coefficient"] for r in rows) == pytest.approx(1.0, abs=1e-3)


def test_the_card_reconciles_end_to_end_from_a_real_score_frame():
    """The property under the real blend rather than under hand-typed numbers:
    take a scored frame with a theme missing, build the card the way Stage 4
    does, and check it adds up."""
    rng = np.random.default_rng(3)
    idx = [f"S{i:03d}" for i in range(150)]
    raw = pd.DataFrame({f: rng.uniform(-1, 1, len(idx)) for f in v3.ALL_FACTORS},
                       index=idx)
    raw[v3.THEMES["quality"].names] = np.nan          # the live case
    scored = v3.score_frame(raw)
    sym = idx[0]

    themes = [(t, scored.at[sym, t + "_sub"], scored.at[sym, t + "_w"],
               scored.at[sym, t + "_contrib"])
              for t in v3.THEMES if pd.notna(scored.at[sym, t + "_sub"])]
    rows = _contributions(_card(themes)["factors"], tier="v3_theme")

    # DISPLAY precision, not machine precision. `_contributions` rounds z to 3dp
    # and the weight to 5dp, so a reader multiplying the two printed numbers
    # carries a rounding budget of about 5e-4 -- half a unit in z's last place,
    # times a weight below one. The exact identity is pinned on the unrounded
    # blend in `test_v3_blend_is_capped`; what matters here is that the numbers
    # ON THE CARD reconcile to the precision they are shown at.
    assert sum(r["coefficient"] for r in rows) == pytest.approx(1.0, abs=1e-4)
    assert sum(r["contribution"] for r in rows) == pytest.approx(
        float(scored.at[sym, "score"]), abs=1e-4)
    for r in rows:
        assert r["z"] * r["coefficient"] == pytest.approx(r["contribution"], abs=1e-3)


# =============================================================================
# which scorer ranked
# =============================================================================

def test_a_v3_run_is_not_reported_as_the_deleted_fitted_model():
    """THE DEFECT: the v3 theme keyed `reversal` collides with a family of the
    same name in the Fama-MacBeth model deleted on 2026-09-03, so the old
    key-matching reported {"model": "cross-sectional", "validated": True} on
    every single v3 run."""
    out = _scorer_used([_card([("reversal", 0.5, 0.16, 0.08),
                               ("momentum", 0.5, 0.40, 0.20)])])
    assert out["model"] == "v3_composite"
    assert out["validated"] is False
    assert "cross-sectional" not in str(out.get("note") or "")


def test_the_v3_caveat_is_a_disclosure_and_not_an_alarm():
    """The ranking IS evidenced; the six-name book is not. Painting that red on
    every run teaches the reader to skip the box that also carries the real
    failures."""
    out = _scorer_used([_card([("momentum", 0.5, 0.40, 0.20)])])
    assert out["severity"] == "note"
    assert out["note"]


def test_an_unrecognised_scorer_still_fails_loudly():
    """The safe direction is preserved: an unknown key set must not be asserted
    to be either scorer."""
    card = {"factors": {"mystery": {"standardised": 0.1, "weight": 1.0,
                                    "contribution": 0.1, "available": True,
                                    "tier": None, "members": []}}}
    out = _scorer_used([card])
    assert out["model"] == "unknown"
    assert out["validated"] is False
    assert out.get("severity") == "alarm"


# =============================================================================
# naming
# =============================================================================

def test_the_screen_and_the_model_agree_about_what_a_theme_is_called():
    """These were two independent tables. The screen said "Low-margin tilt"
    while the scoring note written into the same run's record said "quality"."""
    assert V3_THEME_LABELS == {n: t.label for n, t in v3.THEMES.items()}
    assert V3_THEME_LABELS["quality"] == "Low-margin tilt"
