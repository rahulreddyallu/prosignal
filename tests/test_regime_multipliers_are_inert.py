"""A number that is printed and changes nothing must say so.

Stage 2 computes three factor multipliers from the regime read and scales the
FAMILY block with them. `_apply_ranking_policy` discards that block under
`ranking.source = v3_composite`, which is the shipped setting -- so on every
shipped run the multipliers are computed, written to the ledger, printed on the
regime table, and inert.

The run NOTE was already fixed this way: it is emitted only after the ranking
source is known, because "Regime 'range_lowvol' multipliers applied (momentum
x0.75)" on a run whose book was ordered by an unmodified v3 blend tells an
operator the engine leaned against momentum today. It did not. The same is true
of every other surface the number reaches, and those were not fixed.

Nothing here deletes the multipliers. They work on the `fitted_composite` path,
which is still selectable, and the regime read itself -- trend, volatility,
breadth, the entry gate -- is live on every path. What changes is that the
inertness travels with the number.
"""

from __future__ import annotations

import datetime as dt

import pytest

from prosignal.core.contracts import RegimeCompatibility, RegimeState
from prosignal.core.enums import TrendRegime, VolContext, VolTercile


def _state(**over) -> RegimeState:
    base = dict(
        as_of_date=dt.date(2026, 9, 5),
        trend_regime=TrendRegime.UPTREND,
        trend_slope_annualised=0.12,
        index_vs_fast_ma_pct=1.0,
        index_vs_slow_ma_pct=3.0,
        vol_tercile=VolTercile.LOW,
        vol_context=VolContext.STABLE,
        vix_level=12.0,
        vix_percentile=0.2,
        vix_change_pct=-1.0,
        vol_signal_confidence=0.9,
        breadth_pct_above_ma=60.0,
        breadth_state="Strong",
        breadth_divergence_flag=False,
        breadth_sample_size=400,
        regime_bucket="uptrend_lowvol",
        momentum_multiplier=0.75,
        quality_multiplier=1.0,
        sector_rs_multiplier=0.9,
    )
    base.update(over)
    return RegimeState(**base)


def test_the_default_is_that_the_multipliers_do_not_reach_the_book():
    """The shipped configuration is the one where they do not, so an unset flag
    must read that way. A default of True restores the misreading on every
    caller that forgets to set it."""
    assert _state().scores_the_shipped_book is False


def test_an_inert_state_carries_a_note_saying_why():
    note = _state().multiplier_note()
    assert note is not None
    assert "INERT" in note
    assert "v3_composite" in note
    assert "not tilted" in note or "not tilted by them" in note


def test_the_note_says_the_regime_read_itself_is_still_live():
    """Deleting the whole regime block would be the wrong correction: the
    entry gate is a hard market-wide halt and it works."""
    assert "entry gate" in _state().multiplier_note()


def test_a_state_that_does_reach_the_book_carries_no_note():
    assert _state(scores_the_shipped_book=True).multiplier_note() is None


def test_compatibility_is_not_gated_on_the_flag():
    """It reads the REGIME, which is live on every path, and uses the
    multiplier only as a compact encoding of that read. Suppressing it would
    remove a live signal to fix an inert one."""
    fav = _state(momentum_multiplier=1.0)
    assert fav.compatibility() is RegimeCompatibility.FAVORABLE
    assert fav.scores_the_shipped_book is False

    unfav = _state(momentum_multiplier=0.4)
    assert unfav.compatibility() is RegimeCompatibility.UNFAVORABLE

    blocked = _state(allow_new_entries=False)
    assert blocked.compatibility() is RegimeCompatibility.UNFAVORABLE


def test_stage2_sets_the_flag_from_the_ranking_source():
    """The flag has to be derived where the config is read. A caller that has
    to work it out is a caller that will forget to."""
    import inspect

    from prosignal.stages import stage2_regime

    src = inspect.getsource(stage2_regime)
    assert "scores_the_shipped_book=" in src
    assert "fitted_composite" in src, (
        "the flag must be decided by comparing ranking.source against the one "
        "source that actually consumes the family block"
    )


def test_the_ledger_records_whether_the_multiplier_scored_the_book():
    """A row carrying the number and not the flag reads, years later, as
    though the engine leaned against momentum that day."""
    import inspect

    from prosignal import ledger

    assert "momentum_multiplier_scored_the_book" in inspect.getsource(ledger)
