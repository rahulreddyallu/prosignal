"""The dashboard leads with a fixed slate. These are the rules it follows.

The engine's own output is 8 admitted and 44 monitored on a typical run. Both
numbers are honest and neither is a decision, so the slate curates them without
touching the criteria that produced them. The invariant that matters most is
the one in test_a_buy_is_never_manufactured_to_fill_the_slate: a short list is
a true statement about the market and padding it would be a false one.
"""

from __future__ import annotations

import pytest

from prosignal.presentation.selection import BUY, SLOTS, WATCH, select_slate


def card(ticker: str, model_rank: int, *, score: float = 0.9,
         percentile: float = 95.0) -> dict:
    return {"ticker": ticker, "model_rank": model_rank, "score": score,
            "percentile": percentile}


def buys(n: int, start: int = 1) -> list:
    return [card(f"BUY{i}", i) for i in range(start, start + n)]


def watches(n: int, start: int = 20) -> list:
    return [card(f"WATCH{i}", i) for i in range(start, start + n)]


# ------------------------------------------------------- the ten cases
def test_case_1_ten_buys_gives_the_top_five():
    s = select_slate(buys(10), [])
    assert s.buy_count == 5 and s.watch_count == 0
    assert [p["ticker"] for p in s.picks] == ["BUY1", "BUY2", "BUY3", "BUY4", "BUY5"]


def test_case_2_five_buys_gives_five_buys():
    s = select_slate(buys(5), watches(10))
    assert (s.buy_count, s.watch_count) == (5, 0)


def test_case_3_four_buys_and_ten_watch_gives_four_plus_one():
    s = select_slate(buys(4), watches(10))
    assert (s.buy_count, s.watch_count) == (4, 1)
    assert s.picks[-1]["ticker"] == "WATCH20"
    assert s.picks[-1]["status"] == WATCH


def test_case_4_two_buys_and_twenty_watch_gives_two_plus_three():
    s = select_slate(buys(2), watches(20))
    assert (s.buy_count, s.watch_count) == (2, 3)
    assert [p["status"] for p in s.picks] == [BUY, BUY, WATCH, WATCH, WATCH]


def test_case_5_no_buys_gives_five_watch():
    s = select_slate([], watches(20))
    assert (s.buy_count, s.watch_count) == (0, 5)
    assert all(p["status"] == WATCH for p in s.picks)


def test_case_6_a_buy_is_never_manufactured_to_fill_the_slate():
    """Three admitted and one monitored is four real names. A fifth row would
    have to be invented, and an invented row on this screen is a position
    someone may take."""
    s = select_slate(buys(3), watches(1))
    assert len(s.picks) == 4
    assert (s.buy_count, s.watch_count) == (3, 1)
    assert "padded" in s.selection_note


def test_case_7_a_failed_data_quality_bar_withholds_the_slate():
    s = select_slate(buys(10), watches(10), data_quality_ok=False)
    assert s.is_empty
    assert s.withheld_reason
    assert "data-quality" in s.withheld_reason


def test_case_8_duplicates_are_collapsed_before_selection():
    dupes = [card("AAA", 1), card("AAA", 1), card("BBB", 2)]
    s = select_slate(dupes, [])
    assert [p["ticker"] for p in s.picks] == ["AAA", "BBB"]


def test_case_9_identical_ranks_break_deterministically():
    tied = [card("ZZZ", 3, percentile=90.0), card("AAA", 3, percentile=90.0),
            card("MMM", 3, percentile=90.0)]
    first = [p["ticker"] for p in select_slate(tied, []).picks]
    second = [p["ticker"] for p in select_slate(list(reversed(tied)), []).picks]
    assert first == second == ["AAA", "MMM", "ZZZ"]


def test_case_10_a_name_in_both_lists_takes_one_slot_as_a_buy():
    """A name can reach the payload from more than one path. Two rows for one
    ticker would spend two of five slots on the same position."""
    s = select_slate([card("DUAL", 1)], [card("DUAL", 1), card("OTHER", 2)])
    assert [p["ticker"] for p in s.picks] == ["DUAL", "OTHER"]
    assert s.picks[0]["status"] == BUY


# ------------------------------------------------------------- ordering
def test_the_slate_orders_on_model_rank_not_the_penalised_score():
    """The real failure this fixes. `composite_score` is the model percentile
    minus Stage 5 penalties, and on a live run the top 52 names span
    percentiles 90-100 while one penalty is -0.10 -- enough to move a name
    across the whole visible range. Sorting on it put seven WATCHLIST names
    above every BUY.
    """
    penalised = card("ADMITTED", 1, score=0.9000, percentile=100.0)
    clean = card("MONITORED", 31, score=0.9513, percentile=95.1)
    s = select_slate([penalised], [clean])
    assert [p["ticker"] for p in s.picks] == ["ADMITTED", "MONITORED"]
    assert s.picks[0]["score"] < s.picks[1]["score"], (
        "the fixture must keep the inversion, or this proves nothing"
    )


def test_a_name_without_a_model_rank_does_not_sort_to_the_top():
    s = select_slate([card("RANKED", 5), {"ticker": "UNRANKED", "score": 1.0}], [])
    assert [p["ticker"] for p in s.picks] == ["RANKED", "UNRANKED"]


def test_positions_are_numbered_from_one_across_both_statuses():
    s = select_slate(buys(2), watches(10))
    assert [p["slate_position"] for p in s.picks] == [1, 2, 3, 4, 5]


def test_the_full_ranked_lists_survive_for_the_deeper_views():
    """The slate is what the dashboard leads with, not all the engine found."""
    s = select_slate(buys(8), watches(44))
    assert len(s.picks) == 5
    assert len(s.ranked_buys) == 8 and len(s.ranked_watch) == 44


def test_an_entirely_empty_run_is_not_an_error():
    s = select_slate([], [])
    assert s.is_empty and s.withheld_reason is None
    assert "No names qualified" in s.selection_note


def test_slots_cannot_be_negative():
    with pytest.raises(ValueError):
        select_slate(buys(3), [], slots=-1)


def test_the_default_slate_is_five():
    assert SLOTS == 5
