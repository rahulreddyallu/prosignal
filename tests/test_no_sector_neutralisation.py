"""Factors are ranked across the universe, not within sector.

TWO INDEPENDENT REASONS, both measured.

LOOKAHEAD. `_refresh_sector_map` pools the `Industry` column of TODAY's NSE
constituent files, so a name that has since delisted or left every index has no
sector and falls into `__RESID__`. Holding a sector label is therefore
correlated with having survived, and it is worth more than the signal is:
across 380 panel dates, names with a known sector out-returned names without by
+1.05% per 21 sessions (overlap-corrected t +3.64) and +3.36% per 63 (t +3.48).
The groups the ranking was computed inside were defined by that attribute.

COST. Re-scoring the panel both ways on dates after the fit window closed:

    horizon   sector-neutral        universe rank
        5     +0.0406 (t +3.88)     +0.0485 (t +3.79)
       21     +0.0427 (t +1.99)     +0.0562 (t +2.18)
       63     +0.0473 (t +1.34)     +0.0759 (t +1.64)

The direction reproduces what `features/v9r.py` had already recorded
independently (+0.0674 unneutralised against +0.0547 at h=21).

`sector_neutral_rank` is deliberately NOT deleted: it is the honest
implementation of the idea and a point-in-time sector source would make it
usable again. What changed is which one `score_frame` calls by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.features import v3

SYMBOLS = [f"S{i:02d}" for i in range(60)]


def _raw(seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {f: rng.normal(size=len(SYMBOLS)) for f in v3.ALL_FACTORS},
        index=SYMBOLS,
    )


def _sectors() -> dict:
    """Three real sectors big enough to rank within, plus an unclassified tail
    -- the shape the live map actually has."""
    out = {}
    for i, s in enumerate(SYMBOLS):
        out[s] = ["Banks", "IT", "Pharma"][i % 3] if i < 45 else "Unknown"
    return out


def test_the_shipped_default_is_universe_ranking():
    assert v3.SECTOR_NEUTRAL is False


def test_passing_a_sector_map_no_longer_changes_the_ranking():
    """The behavioural assertion. Sectors are still accepted -- the residual
    bucket report needs them -- and they no longer decide anything."""
    raw = _raw()
    with_sec = v3.score_frame(raw, _sectors())
    without = v3.score_frame(raw, None)
    pd.testing.assert_series_equal(
        with_sec["score"], without["score"], check_names=False
    )


def test_the_within_sector_ranking_is_still_reachable_on_request():
    """A point-in-time sector source would make this the right call again, so
    the capability must not have been deleted -- only un-defaulted."""
    raw = _raw()
    neutral = v3.score_frame(raw, _sectors(), sector_neutral=True)
    universe = v3.score_frame(raw, _sectors(), sector_neutral=False)
    assert not np.allclose(
        neutral["score"].to_numpy(), universe["score"].to_numpy()
    ), "forcing sector_neutral=True must actually change the ranking"


def test_sector_neutral_rank_itself_is_unchanged():
    """The function is not the problem; the map feeding it was."""
    vals = pd.Series(np.arange(len(SYMBOLS), dtype="float64"), index=SYMBOLS)
    sec = pd.Series(_sectors())
    out = v3.sector_neutral_rank(vals, sec)
    assert out.notna().all()
    assert out.max() <= 1.0 and out.min() >= -1.0


def test_the_residual_bucket_report_still_works():
    """It now measures what a sector map WOULD cover, which is the condition
    for turning neutralisation back on."""
    rb = v3.residual_bucket_size(pd.Index(SYMBOLS), _sectors())
    assert rb["unknown"] == 15
    assert rb["resid"] >= 15


def test_no_surviving_claim_that_the_score_is_sector_neutral():
    """A card that says "percentile in sector" over a universe rank is the
    same class of untrue message this audit exists to remove."""
    from pathlib import Path

    ui = (Path(__file__).resolve().parents[1]
          / "src/prosignal/static/index.html").read_text(encoding="utf-8")
    assert "percentile in sector" not in ui
    assert "in its sector" not in ui
