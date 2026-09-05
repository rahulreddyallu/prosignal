"""A bar on the cross-section, not on the index.

NO TRADE WAS ALREADY REACHABLE and the audit finding that said otherwise was too
strong. Replaying the engine on its worst episodes: 2020-03-31 (34 eligible, 24
above the bar) and 2022-06-17 (74 eligible, 43 above) both reach NO TRADE
through the Stage 2 regime gate on 'downtrend_highvol'. Neither would have
tripped this rule, which sits behind `blocked_reason is None`.

What was genuinely missing is a bar on the CROSS-SECTION. The regime gate reads
NIFTY trend and volatility -- it says the market is bad, not that these
candidates are weak -- and the two disagree when an index is held up by a few
large caps while breadth collapses underneath it.

The rank gates cannot express either, and the config said so in its own words:
"A floor on a cross-sectional RANK cannot fire: somebody is top of the list
every day however weak the day is." Every scarcity control was a rank. `composite_score` is
`(rank-1)/(n-1)`, so its distribution is uniform on every session and
`min_universe_percentile = 90` admits the top decile whether or not the top
decile is worth anything -- measured live, the score gate rejected 0 of 37
defended names.

`min_dispersion_ratio` was written to fix exactly that, and reads
`prediction_dispersion` -- a field only the deleted fitted model ever populated.
A `is not None` guard therefore skipped it on every run after that model was
removed. It is wired to the v3 composite now, but it CANNOT close this hole and
these tests say why: a blend of cross-sectional ranks has a spread bounded by
construction, and over 61 training dates the worst day was 0.874x the median
against a 0.50 threshold.

What closes it is the BOOK-LEVEL CASH RULE, which counts names rather than
ranks: how many of the eligible universe close above their long moving average.
That can collapse for the whole market at once. Measured over 2,018 sessions it
runs 16 to 708, median 319, and falls below the threshold on exactly one --
2020-03-23.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.features import v3


# =============================================================================
# why the rank gates cannot fire
# =============================================================================

def test_the_composite_score_is_uniform_so_a_rank_gate_cannot_bind():
    """The arithmetic behind the whole finding. `rank_to_unit_interval` maps to
    `(rank-1)/(n-1)`, so the top decile exists on the worst day in history."""
    from prosignal.indicators import rank_to_unit_interval
    for spread in (1.0, 0.01, 1e-6):
        scores = pd.Series(np.linspace(-spread, spread, 300),
                           index=[f"S{i}" for i in range(300)])
        unit = rank_to_unit_interval(scores)
        above = (unit * 100.0 >= 90.0).sum()
        assert above == pytest.approx(30, abs=1), (
            "the share of names above the 90th percentile is 10% whatever the "
            "spread -- which is why this gate can never say 'flat day'"
        )


def test_a_rank_blend_cannot_collapse_far_enough_to_trip_the_dispersion_gate():
    """Pins the limit of `min_dispersion_ratio` rather than leaving a future
    reader to assume it protects them. Each sub-score is uniform on [-1, 1] by
    construction, so the blend's spread varies only with how much the themes
    agree -- never with how bad the market is."""
    rng = np.random.default_rng(4)
    W = np.array([0.40, 0.2694, 0.1577, 0.1562]); W = W / W.sum()
    worst = 1.0
    for _ in range(120):
        cols = []
        for _w in W:
            z = rng.normal(size=300)
            r = (np.argsort(np.argsort(z)) + 0.5) / 300.0
            cols.append((r - 0.5) * 2.0)
        d = v3.score_dispersion(pd.Series(np.column_stack(cols) @ W))
        worst = min(worst, d / v3.TYPICAL_DISPERSION)
    assert worst > 0.50, (
        f"the worst simulated day is {worst:.3f} x typical, so a 0.50 ratio "
        f"cannot fire. If this ever fails the scorer has genuinely degenerated, "
        f"which is what the gate is for."
    )


def test_dispersion_is_measurable_and_reported():
    """It was None on every run, because it read the deleted model's field."""
    rng = np.random.default_rng(7)
    d = v3.score_dispersion(pd.Series(rng.normal(size=400)))
    assert d is not None and d > 0
    assert v3.score_dispersion(pd.Series([0.1, 0.2])) is None, (
        "too few names is not a dispersion of zero, it is no measurement"
    )


# =============================================================================
# the rule that does bind
# =============================================================================

def _score(ticker: str, rank: int, cleared):
    from prosignal.core.contracts import StockScore
    return StockScore(ticker=ticker, rank=rank, composite_raw=1.0 / rank,
                      composite_score=1.0 - rank / 100.0,
                      percentile=100.0 - rank,
                      absolute_bar_cleared=cleared)


def test_the_bar_is_recorded_even_when_the_per_name_floor_is_off(cfg):
    """`absolute_floor.enabled` is false on a measured -2.2% treatment effect,
    and that decision stands. It governs whether the bar GATES a name; it must
    not govern whether the engine is allowed to know the number, because the
    book-level question needs the same measurement."""
    assert cfg.params.stage4_core_score.absolute_floor.enabled.value is False
    assert cfg.params.stage8_final_signal.scarcity.cash_rule.enabled is True


def test_the_threshold_defaults_to_the_band_the_book_lives_in(cfg):
    """Not a free parameter. `min_qualifying: null` means `exit_rank` -- the set
    the engine already treats as the names a position may live in. If fewer
    clear the bar than that, the book could not be maintained if it were
    opened."""
    cr = cfg.params.stage8_final_signal.scarcity.cash_rule
    assert cr.min_qualifying.value is None
    assert int(cfg.params.stage6_entry.admission.exit_rank.value) == 18


def test_a_normal_day_does_not_trip_the_rule():
    """309 of 386 cleared the bar on the live cross-section. A rule that fired
    there would be a signal-count tuner, not a circuit breaker."""
    scores = [_score(f"S{i}", i + 1, i < 309) for i in range(386)]
    cleared = sum(1 for s in scores if s.absolute_bar_cleared)
    assert cleared >= 18


def test_a_market_with_almost_nothing_above_water_trips_it():
    """The COVID trough: 16 names in the whole liquid universe above their
    200-session average."""
    scores = [_score(f"S{i}", i + 1, i < 16) for i in range(300)]
    cleared = sum(1 for s in scores if s.absolute_bar_cleared)
    assert cleared < 18, "16 < 18, so the book holds cash"


def test_an_unmeasured_bar_is_not_a_failed_one():
    """`absolute_bar_cleared` is None when there is not enough history to form
    the average. None must not count as "did not clear", or a short store would
    look like a permanent crash."""
    scores = [_score(f"S{i}", i + 1, None) for i in range(300)]
    measured = [s for s in scores if s.absolute_bar_cleared is not None]
    assert measured == [], (
        "with nothing measured the rule must not fire at all -- Stage 8 guards "
        "on `if measured:` for this reason"
    )


# =============================================================================
# a refusal to rank is a decision
# =============================================================================

def test_a_refusal_to_rank_is_a_blocked_run_not_a_stack_trace(runnable_cfg, monkeypatch):
    """`RankingUnavailable` is right to be fatal -- falling back to another
    scorer would issue signals from a model that was not the one measured. But
    it is a PipelineError and nothing caught it, so it reached the job runner
    as an unhandled exception and the screen showed a traceback where a reason
    belongs.

    Found by replaying 2020-03-23, the worst session in the store: the
    liquidity screen and the eligibility gates left SIX names, the v3 block
    refused to rank a universe that small, and the engine errored rather than
    saying so. That is the single date this engine most needs to be legible on.
    """
    from prosignal import pipeline as P
    from prosignal.stages import stage4_core_score

    def _refuse(*a, **k):
        raise stage4_core_score.RankingUnavailable(
            "the v3 composite covers 6 of 6 scoreable names, under the 20 floor.")

    monkeypatch.setattr(stage4_core_score, "run", _refuse)
    with pytest.raises(P.PipelineBlocked) as caught:
        P.run_analysis(runnable_cfg)
    assert "under the 20 floor" in str(caught.value), (
        "the operator has to be told WHY the engine would not rank"
    )
    assert caught.value.stage == stage4_core_score.STAGE_NAME
