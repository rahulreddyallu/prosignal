"""Stage 8's limits govern what may be OPENED. They must not close a position.

Stage 6 admits a name inside `entry_rank` and holds it while it stays inside
`exit_rank`. That hysteresis was implemented and tested. Stage 8 then undid it
every session, because its sector cap, correlation cap and book-size cap were
applied to every name in score order as though each session were the first --
so a held name that drifted below a fresher one in its sector was "downgraded
to WATCH", and since the ledger's `signals_generated` is the ONLY record of the
book, that demotion deleted the position with no exit recorded and no way for
the next run to know it had existed.

Measured on the recorded ledger before the fix, restricted to ADJACENT sessions
so that backfill runs months apart are not counted as a position being held: of
54 held-name transitions only 12 stayed in the book, and 19 were demoted this
way. Book turnover was 89.3% at a median hold of ONE session, against a
strategy validated as patient.

The regime block was the same defect in a louder form: it returned an empty
`buys` list, which is not "stop buying" but "sell everything, silently".
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from prosignal.config.loader import load_config
from prosignal.core.contracts import (
    CoreScoreReport, EligibilityReport, EntryDecision, EntryReport,
    FalseSignalReport, RegimeState, StockDefenseResult, StockScore,
)
from prosignal.core.enums import (
    Decision, EntryStatus, TrendRegime, VolContext, VolTercile,
)
from prosignal.stages import stage8_final_signal as s8

AS_OF = dt.date(2026, 8, 25)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _regime(allow: bool = True) -> RegimeState:
    return RegimeState(
        as_of_date=AS_OF, trend_regime=TrendRegime.UPTREND,
        vol_tercile=VolTercile.LOW, vol_context=VolContext.STABLE,
        regime_bucket="uptrend/low", momentum_multiplier=1.0,
        quality_multiplier=1.0, sector_rs_multiplier=1.0,
        allow_new_entries=allow,
        block_reason=None if allow else "risk-off",
    )


def _scores(rows) -> CoreScoreReport:
    """rows: (ticker, model_rank, score, sector)."""
    return CoreScoreReport(
        as_of_date=AS_OF, weighting_mode="model", standardisation="zscore",
        universe_size=600,
        ranked_scores=[
            StockScore(ticker=t, rank=r, composite_score=s,
                       percentile=99.0, sector=sec)
            for t, r, s, sec in rows
        ],
    )


def _defense(rows) -> FalseSignalReport:
    return FalseSignalReport(
        as_of_date=AS_OF,
        per_stock={
            t: StockDefenseResult(ticker=t, score_before=s, score_after=s,
                                  final_status="PASS")
            for t, _, s, _ in rows
        },
    )


def _entries(rows, triggered) -> EntryReport:
    return EntryReport(
        as_of_date=AS_OF,
        decisions={
            t: EntryDecision(
                ticker=t,
                status=EntryStatus.TRIGGERED if t in triggered else EntryStatus.WATCHLIST,
                reference_price=100.0,
                reason="held, rank still inside the exit band" if t in triggered else "out of band",
            )
            for t, _, _, _ in rows
        },
    )


def _run(cfg, rows, triggered, *, held=(), allow_entries=True):
    """(buys, watch, no_trade). Stage 8 also returns its gate counts now; the
    tests that want them call `_run_full`."""
    return _run_full(cfg, rows, triggered, held=held,
                     allow_entries=allow_entries)[:3]


def _run_full(cfg, rows, triggered, *, held=(), allow_entries=True):
    return s8.run(
        regime=_regime(allow_entries),
        eligibility=EligibilityReport(as_of_date=AS_OF, universe_considered=600,
                                      eligible_universe=[r[0] for r in rows]),
        scores=_scores(rows),
        defense=_defense(rows),
        entries=_entries(rows, triggered),
        plans={},
        closes=pd.DataFrame(),
        config=cfg,
        held=list(held),
    )


def _tickers(recs):
    return [r.ticker for r in recs]


# ------------------------------------------------- the sector cap eviction
def test_the_sector_cap_does_not_evict_a_position_already_held(cfg):
    """Two fresh names outscore a held one in the same sector. The cap is 2.

    Applied in score order with no knowledge of the book, the held name is
    third in its sector and is demoted -- deleting the position. It must not be.
    """
    rows = [("FRESH1", 1, 0.99, "Metals"),
            ("FRESH2", 2, 0.98, "Metals"),
            ("HELD",  12, 0.97, "Metals")]
    buys, watch, _ = _run(cfg, rows, triggered={"FRESH1", "FRESH2", "HELD"},
                          held={"HELD"})
    assert "HELD" in _tickers(buys), (
        "the sector cap closed an open position; it governs entries only"
    )
    # The cap still bites -- on the NEW name, which is what it is for.
    assert "FRESH2" in _tickers(watch)
    assert _tickers(buys) == ["HELD", "FRESH1"]


def test_the_sector_cap_still_limits_new_entries(cfg):
    rows = [("A", 1, 0.99, "Metals"), ("B", 2, 0.98, "Metals"),
            ("C", 3, 0.97, "Metals")]
    buys, watch, _ = _run(cfg, rows, triggered={"A", "B", "C"})
    assert _tickers(buys) == ["A", "B"]
    assert _tickers(watch) == ["C"]
    assert "already 2 signal(s)" in " ".join(watch[0].why_this_signal_exists)


def test_the_book_size_cap_does_not_evict_a_held_position(cfg):
    """max_signals_per_run applied in score order pushed a drifting held name
    past the edge of the book. The book is the positions that exist."""
    cap = int(cfg.params.stage8_final_signal.portfolio.max_signals_per_run.value)
    rows = [(f"FRESH{i}", i, 0.99 - i * 0.001, f"S{i}") for i in range(1, cap + 1)]
    rows.append(("HELD", cap + 4, 0.97, "SHELD"))
    triggered = {t for t, _, _, _ in rows}
    buys, _, _ = _run(cfg, rows, triggered=triggered, held={"HELD"})
    assert "HELD" in _tickers(buys)
    assert len(buys) == cap, "the cap still bounds the book"


def test_a_held_name_stage_six_closed_is_not_carried(cfg):
    """The exit band is what closes a position. When Stage 6 stops triggering a
    held name, Stage 8 must let it go -- otherwise nothing ever exits."""
    rows = [("HELD", 40, 0.97, "Metals")]
    buys, watch, _ = _run(cfg, rows, triggered=set(), held={"HELD"})
    assert _tickers(buys) == []
    assert _tickers(watch) == ["HELD"]


def test_a_held_name_is_told_apart_from_a_fresh_one_on_the_card(cfg):
    rows = [("HELD", 12, 0.97, "Metals")]
    buys, _, _ = _run(cfg, rows, triggered={"HELD"}, held={"HELD"})
    why = " ".join(buys[0].why_this_signal_exists)
    assert "Held from a previous run" in why
    assert "do not close a position that is already open" in why


# ------------------------------------------------------- the regime block
def test_a_regime_block_stops_new_entries_and_keeps_the_book(cfg):
    """The loudest form of the defect. Returning [] here did not pause the
    strategy, it liquidated it: the next run rebuilt `held` from an empty
    record and every position vanished without an exit."""
    rows = [("HELD", 12, 0.97, "Metals"), ("FRESH", 1, 0.99, "Energy")]
    buys, watch, no_trade = _run(cfg, rows, triggered={"HELD", "FRESH"},
                                 held={"HELD"}, allow_entries=False)
    assert _tickers(buys) == ["HELD"], "a blocked regime must not close the book"
    assert "FRESH" in _tickers(watch)
    assert no_trade is not None, "the block must still be reported"
    assert "No new positions were opened" in no_trade.reason


def test_a_regime_block_on_an_empty_book_is_still_no_trade(cfg):
    rows = [("FRESH", 1, 0.99, "Energy")]
    buys, _, no_trade = _run(cfg, rows, triggered={"FRESH"}, allow_entries=False)
    assert buys == []
    assert no_trade is not None
    assert "No position was open to carry" in no_trade.reason


def test_a_blocked_new_entry_says_the_book_is_unaffected(cfg):
    rows = [("FRESH", 1, 0.99, "Energy")]
    _, watch, _ = _run(cfg, rows, triggered={"FRESH"}, allow_entries=False)
    assert "Positions already held are unaffected" in " ".join(
        watch[0].why_this_signal_exists
    )


def test_a_market_halt_also_keeps_the_book(cfg):
    rows = [("HELD", 12, 0.97, "Metals")]
    defense = _defense(rows)
    defense.market_halt = True
    defense.market_halt_reason = "feed integrity"
    buys, _, no_trade, _gates = s8.run(
        regime=_regime(True),
        eligibility=EligibilityReport(as_of_date=AS_OF, universe_considered=600),
        scores=_scores(rows), defense=defense,
        entries=_entries(rows, {"HELD"}), plans={},
        closes=pd.DataFrame(), config=cfg, held=["HELD"],
    )
    assert _tickers(buys) == ["HELD"]
    assert no_trade is not None and "defense halt" in no_trade.reason


# ------------------------------------------------------------- no regression
def test_with_no_book_the_stage_behaves_exactly_as_before(cfg):
    rows = [("A", 1, 0.99, "Metals"), ("B", 2, 0.98, "Energy")]
    buys, watch, no_trade = _run(cfg, rows, triggered={"A", "B"})
    assert _tickers(buys) == ["A", "B"]
    assert watch == [] and no_trade is None

def test_the_quality_gate_still_applies_to_a_held_name(cfg):
    """Holding is not immunity. The score gate is a statement about evidence,
    not a portfolio constraint, so a held name that falls through it leaves --
    and leaves visibly, which is the part that was missing."""
    floor = float(cfg.params.stage8_final_signal.scarcity.min_composite_score.value)
    rows = [("HELD", 12, floor - 0.1, "Metals")]
    buys, watch, _ = _run(cfg, rows, triggered={"HELD"}, held={"HELD"})
    assert buys == [] and watch == []


def test_a_candidate_with_no_risk_plan_does_not_take_down_the_run(cfg):
    """Stage 7 builds a plan only when Stage 6 produced a reference price, and
    Stage 6 declines to for a name with under 70 sessions of history or an ATR
    it cannot compute. `_card` guards every plan-derived field with
    `if plan else None` -- but `position_risk_category` was declared required on
    the contract, so that guard raised a ValidationError and one unplannable
    name failed the ENTIRE analysis. `api._card` had always read the field as
    optional, so the contract was the only thing asserting otherwise.
    """
    rows = [("NOPLAN", 1, 0.99, "Metals")]
    buys, watch, _ = _run(cfg, rows, triggered={"NOPLAN"})   # plans={} throughout
    assert _tickers(buys) == ["NOPLAN"]
    assert buys[0].position_risk_category is None
    assert buys[0].expected_holding_period == "unknown"
    assert buys[0].initial_stop is None


def test_the_book_shrinks_from_the_bottom_if_the_cap_is_lowered(cfg):
    """Only reachable when max_signals_per_run is cut below an existing book.
    The book must stay bounded, and the names it drops must be the weakest and
    must be told, rather than whichever one happened to sort last."""
    cap = int(cfg.params.stage8_final_signal.portfolio.max_signals_per_run.value)
    rows = [(f"HELD{i}", i, 0.99 - i * 0.001, f"S{i}") for i in range(1, cap + 3)]
    held = {t for t, _, _, _ in rows}
    buys, watch, _ = _run(cfg, rows, triggered=held, held=held)
    assert len(buys) == cap
    assert _tickers(watch) == [f"HELD{cap + 1}", f"HELD{cap + 2}"]
    assert "lowest-scoring position held" in " ".join(watch[0].why_this_signal_exists)


def test_no_name_appears_twice_across_the_two_passes(cfg):
    """The two-pass structure makes double-listing possible in a way the single
    loop did not. One ticker on both lists, or twice on one, would spend two
    slots of a five-slot screen on one position."""
    cap = int(cfg.params.stage8_final_signal.portfolio.max_signals_per_run.value)
    rows = [(f"N{i}", i, 0.99 - i * 0.001, f"S{i % 3}") for i in range(1, cap + 6)]
    every = {t for t, _, _, _ in rows}
    # A mixed state: some held and still triggered, some held and closed, some new.
    buys, watch, _ = _run(cfg, rows, triggered=every - {"N4", "N7"},
                          held={"N2", "N4", "N9"})
    names = _tickers(buys) + _tickers(watch)
    assert len(names) == len(set(names)), f"duplicated: {names}"
    assert set(_tickers(buys)).isdisjoint(_tickers(watch))


def test_the_funnel_reports_the_book_size_on_the_blocked_path_too(cfg):
    """A blocked regime can now hold a live book, so the funnel the no-trade
    report carries must still say how many positions there are."""
    rows = [("HELD", 12, 0.97, "Metals")]
    _, _, no_trade = _run(cfg, rows, triggered={"HELD"}, held={"HELD"},
                          allow_entries=False)
    assert no_trade.gate_summary["buys"] == 1
