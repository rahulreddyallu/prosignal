"""One slate, decided by the run, rendered by everything downstream.

There used to be three. The live screen sorted the engine's two lists by model
rank and took the top five; the history page re-derived a slate from
`signals_generated` in the order Stage 8 emitted them; and the outcome record
followed the book, which is a different list again. All three claimed to be
"what was shown", and they disagreed.

They also could not have agreed, because the ledger recorded `rank` -- the
display position among the defended survivors -- and never `model_rank`, which
is the only number admission turns on. The record could not answer the one
question the whole hysteresis design rests on: was this name inside the band?
"""

from __future__ import annotations

import datetime as dt

import pytest

from prosignal.core.contracts import LedgerRow, SlateEntry
from prosignal.ledger import Ledger
from prosignal.presentation.history import Day, _slate, load_days
from prosignal.presentation.viewmodel import build_view


def _entry(ticker, position, status="BUY", rank=None, carried=False):
    return SlateEntry(ticker=ticker, position=position, status=status,
                      model_rank=rank, carried=carried,
                      shown_since=dt.date(2026, 8, 20), reason="carried")


def _payload(slate, recs, watch=()):
    return {
        "as_of_date": "2026-08-25",
        "slate": [e.model_dump(mode="json") for e in slate],
        "recommendations": list(recs),
        "watchlist": list(watch),
    }


def _card(ticker, rank, **kw):
    d = {"ticker": ticker, "model_rank": rank, "score": 0.9, "percentile": 95.0,
         "decision": "BUY CANDIDATE", "strength": "High"}
    d.update(kw)
    return d


# ------------------------------------------------------- the view renders it
def test_the_view_renders_the_recorded_slate_and_does_not_recompute_it():
    """The recorded slate puts a rank-14 name above a rank-1 one, which no
    fresh computation would ever produce. If the order survives, the view is
    reading the record."""
    slate = [_entry("HELD", 1, rank=14, carried=True),
             _entry("FRESH", 2, rank=1)]
    view = build_view(_payload(slate, [_card("HELD", 14), _card("FRESH", 1)]))
    assert [p["ticker"] for p in view["picks"]] == ["HELD", "FRESH"]
    assert view["picks"][0]["carried"] is True
    assert view["picks"][1]["carried"] is False


def test_the_view_reports_how_many_names_are_being_carried():
    slate = [_entry("A", 1, rank=3, carried=True), _entry("B", 2, rank=5)]
    view = build_view(_payload(slate, [_card("A", 3), _card("B", 5)]))
    assert "1 of 2 was held from the previous run" in view["summary"]["note"]


def test_the_view_surfaces_departures_rather_than_dropping_them():
    slate = [_entry("A", 1, rank=3)]
    payload = _payload(slate, [_card("A", 3)])
    payload["slate_departures"] = [{"ticker": "GONE", "reason": "rank 22 left the band"}]
    view = build_view(payload)
    assert view["departures"] == payload["slate_departures"]
    assert "GONE" in view["summary"]["note"]


def test_a_slate_naming_a_ticker_the_payload_lacks_is_dropped_not_faked():
    slate = [_entry("REAL", 1, rank=1), _entry("PHANTOM", 2, rank=2)]
    view = build_view(_payload(slate, [_card("REAL", 1)]))
    assert [p["ticker"] for p in view["picks"]] == ["REAL"]


def test_a_payload_with_no_recorded_slate_still_renders():
    """Runs recorded before the slate was part of the record must not 500."""
    view = build_view({"recommendations": [_card("A", 1), _card("B", 2)],
                       "watchlist": []})
    assert [p["ticker"] for p in view["picks"]] == ["A", "B"]


def test_new_entries_blocked_reaches_the_view():
    payload = _payload([_entry("A", 1, rank=1)], [_card("A", 1)])
    payload["new_entries_blocked"] = "Market regime blocks new entries."
    assert build_view(payload)["new_entries_blocked"] == payload["new_entries_blocked"]


# -------------------------------------------------------------- the ledger
def test_the_ledger_records_the_slate_that_was_shown(tmp_path):
    led = Ledger(tmp_path)
    led.append(LedgerRow(
        trial_id="T", run_id="r1", date=dt.date(2026, 8, 24),
        logged_at=dt.datetime(2026, 8, 24, 18, 0),
        engine_version="e", schema_version="s", config_version="c",
        signals_generated=["A", "B"],
        slate_shown=[_entry("B", 1, rank=2).model_dump(mode="json"),
                     _entry("A", 2, rank=5).model_dump(mode="json")],
    ))
    assert [e["ticker"] for e in led.shown_slate()] == ["B", "A"]
    assert led.open_book() == ["A", "B"], "the book is a different list"


def test_the_previous_screen_is_empty_for_a_run_that_never_recorded_one(tmp_path):
    """Not an error: there is no previous screen, so the next is chosen fresh.
    Inferring it from `signals_generated` would silently substitute the book."""
    led = Ledger(tmp_path)
    led.append(LedgerRow(
        trial_id="T", run_id="r1", date=dt.date(2026, 8, 24),
        logged_at=dt.datetime(2026, 8, 24, 18, 0),
        engine_version="e", schema_version="s", config_version="c",
        signals_generated=["A", "B"],
    ))
    assert led.shown_slate() == []


def test_the_book_and_the_screen_come_from_one_read(tmp_path):
    led = Ledger(tmp_path)
    for day, book in ((22, ["OLD"]), (24, ["NEW"])):
        led.append(LedgerRow(
            trial_id="T", run_id=f"r{day}", date=dt.date(2026, 8, day),
            logged_at=dt.datetime(2026, 8, day, 18, 0),
            engine_version="e", schema_version="s", config_version="c",
            signals_generated=book,
            slate_shown=[_entry(book[0], 1, rank=1).model_dump(mode="json")],
        ))
    row = led.previous_run(before=dt.date(2026, 8, 25))
    assert row["signals_generated"] == ["NEW"]
    assert led.previous_run(before=dt.date(2026, 8, 24))["signals_generated"] == ["OLD"]


# -------------------------------------------------------------- the history
def test_history_shows_the_screen_that_was_shown_not_a_reconstruction():
    day = Day(date="2026-08-25", run_id="r", buys=["A", "B", "C"], watch=["D"],
              shown=[{"ticker": "C", "position": 1, "status": "BUY"},
                     {"ticker": "D", "position": 2, "status": "WATCH"}])
    assert _slate(day, 5) == {"C": "BUY", "D": "WATCH"}


def test_history_falls_back_to_model_rank_for_rows_recorded_before_the_slate():
    """The old reconstruction used ledger list order while the screen sorted by
    model rank, so the two disagreed about the same past run. The fallback now
    orders the way the screen would have."""
    day = Day(
        date="2026-08-25", run_id="r",
        buys=["LAST", "FIRST"], watch=[],
        detail={"LAST": {"model_rank": 9}, "FIRST": {"model_rank": 1}},
    )
    assert list(_slate(day, 5)) == ["FIRST", "LAST"]


def test_history_carries_the_recorded_slate_off_the_ledger_row():
    rows = [{
        "date": "2026-08-25", "run_id": "r", "logged_at": "2026-08-25T18:00:00",
        "signals_generated": ["A"], "watchlist_generated": [],
        "slate_shown": [{"ticker": "A", "position": 1, "status": "BUY"}],
    }]
    days = load_days(rows)
    assert days[0].shown == rows[0]["slate_shown"]


def test_the_ledger_records_the_model_rank_admission_turns_on():
    """`rank` is the display position among the defended survivors; `model_rank`
    is the place in the full eligible universe. Only the second decides
    anything, and only the first was ever written down -- so every
    reconstruction from the record silently used the wrong number, and the
    record could not say whether a name had been inside the band.
    """
    from prosignal.core.contracts import (
        FinalSignalOutput, Recommendation, RegimeState, RunContext,
    )
    from prosignal.core.enums import (
        Decision, RegimeCompatibility, StrengthBand, TrendRegime, VolContext,
        VolTercile,
    )
    from prosignal.ledger import row_from_output

    rec = Recommendation(
        ticker="AAA", decision=Decision.BUY_CANDIDATE,
        signal_strength_band=StrengthBand.HIGH,
        regime_compatibility=RegimeCompatibility.FAVORABLE,
        expected_holding_period="unknown",
        rank=3,            # third among the defended survivors
        model_rank=11,     # eleventh in the eligible universe -- the band's input
    )
    regime = RegimeState(
        as_of_date=dt.date(2026, 8, 25), trend_regime=TrendRegime.UPTREND,
        vol_tercile=VolTercile.LOW, vol_context=VolContext.STABLE,
        regime_bucket="uptrend/low", momentum_multiplier=1.0,
        quality_multiplier=1.0, sector_rs_multiplier=1.0,
    )
    output = FinalSignalOutput(
        run_id="r", trial_id="T", as_of_date=dt.date(2026, 8, 25),
        generated_at=dt.datetime(2026, 8, 25, 18, 0),
        engine_version="e", config_version="c", regime_state=regime,
        recommendations=[rec],
        slate=[_entry("AAA", 1, rank=11)],
    )
    context = RunContext(
        run_id="r", trial_id="T", as_of_date=dt.date(2026, 8, 25),
        started_at=dt.datetime(2026, 8, 25, 17, 0),
        engine_version="e", schema_version="s", config_version="c",
    )
    row = row_from_output(output, context, funnel={}, duration_ms=1.0)

    scored = row.stocks_scored[0]
    assert scored["model_rank"] == 11, "the number admission turns on"
    assert scored["rank"] == 3, "and the display position, which is not it"
    assert [e["ticker"] for e in row.slate_shown] == ["AAA"]
