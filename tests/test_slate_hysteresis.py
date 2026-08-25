"""The screen is a position, not a snapshot.

The engine admits a name inside `entry_rank` and holds it while it stays inside
the wider `exit_rank`. That band is the strategy that was validated, and until
these tests existed nothing held the SCREEN to it: the slate was recomputed
from scratch every session with no memory of the last one.

Measured on the recorded ledger before the fix: mean top-5 turnover 74.9%,
median 80%, and the median number of sessions a name survived on the screen was
ONE -- for a card quoting a hold of roughly fourteen. These tests fail if that
returns.
"""

from __future__ import annotations

import datetime as dt

from prosignal.presentation.selection import BUY, WATCH, select_slate


def card(ticker: str, model_rank: int, *, score: float = 0.9,
         percentile: float = 95.0) -> dict:
    return {"ticker": ticker, "model_rank": model_rank, "score": score,
            "percentile": percentile}


# ---------------------------------------------------------------- carrying
def test_a_shown_name_keeps_its_slot_while_inside_the_exit_band():
    """The defect, stated as a test. AAA is rank 12 today -- outside the entry
    band of 8, inside the exit band of 16. Without the band it is replaced by
    five fresher names; with it, it stays."""
    today = [card("AAA", 12)] + [card(f"NEW{i}", i) for i in range(1, 8)]
    s = select_slate(today, [], held_slate=["AAA"], exit_rank=16)
    assert "AAA" in [p["ticker"] for p in s.picks]
    assert s.carried_count == 1
    assert next(p for p in s.picks if p["ticker"] == "AAA")["carried"] is True


def test_a_name_leaves_only_when_it_leaves_the_band():
    today = [card("AAA", 17)] + [card(f"NEW{i}", i) for i in range(1, 8)]
    s = select_slate(today, [], held_slate=["AAA"], exit_rank=16)
    assert "AAA" not in [p["ticker"] for p in s.picks]
    assert [d.ticker for d in s.departures] == ["AAA"]
    assert "left the exit band of 16" in s.departures[0].reason


def test_the_boundary_is_inclusive_and_matches_stage_six():
    """`_admit` keeps a held name at `rank <= exit_rank`. One layer disagreeing
    by one on that comparison is a position closed a session early, every time."""
    inside = select_slate([card("AAA", 16)], [], held_slate=["AAA"], exit_rank=16)
    outside = select_slate([card("AAA", 17)], [], held_slate=["AAA"], exit_rank=16)
    assert [p["ticker"] for p in inside.picks] == ["AAA"]
    assert outside.picks == []


def test_carrying_never_invents_a_row_the_run_did_not_produce():
    """A held name that failed eligibility or defence has no card. Showing it
    would put a position on the screen the run does not stand behind."""
    s = select_slate([card("NEW1", 1)], [], held_slate=["GONE"], exit_rank=16)
    assert [p["ticker"] for p in s.picks] == ["NEW1"]
    assert [d.ticker for d in s.departures] == ["GONE"]
    assert "did not produce it" in s.departures[0].reason


def test_only_free_slots_are_filled():
    held = ["H1", "H2", "H3"]
    today = [card(h, 10 + i) for i, h in enumerate(held)] + \
            [card(f"NEW{i}", i) for i in range(1, 6)]
    s = select_slate(today, [], slots=5, held_slate=held, exit_rank=16)
    shown = [p["ticker"] for p in s.picks]
    assert set(held) <= set(shown)
    assert len([t for t in shown if t.startswith("NEW")]) == 2, (
        "three slots were held, so exactly two were open"
    )


def test_a_full_screen_of_held_names_admits_nobody():
    held = [f"H{i}" for i in range(1, 6)]
    today = [card(h, 10 + i) for i, h in enumerate(held)] + [card("NEW", 1)]
    s = select_slate(today, [], slots=5, held_slate=held, exit_rank=16)
    assert [p["ticker"] for p in s.picks] == sorted(held, key=lambda h: held.index(h))
    assert "NEW" not in [p["ticker"] for p in s.picks]
    assert s.carried_count == 5


def test_held_names_compete_for_slots_by_rank_not_by_yesterdays_order():
    held = ["WORST", "BEST"]
    today = [card("WORST", 15), card("BEST", 2)]
    s = select_slate(today, [], slots=1, held_slate=held, exit_rank=16)
    assert [p["ticker"] for p in s.picks] == ["BEST"]
    assert [d.ticker for d in s.departures] == ["WORST"]
    assert "the screen holds 1" in s.departures[0].reason


def test_a_carried_name_is_ordered_by_todays_rank_not_by_tenure():
    """A held name that drifted sits where its rank now says, not at the top
    because it happened to be there yesterday."""
    s = select_slate([card("HELD", 14), card("FRESH", 1)], [],
                     held_slate=["HELD"], exit_rank=16)
    assert [p["ticker"] for p in s.picks] == ["FRESH", "HELD"]
    assert [p["slate_position"] for p in s.picks] == [1, 2]


def test_a_carried_name_carries_todays_status_not_yesterdays():
    """It was admitted yesterday and is merely monitored today. Showing it as a
    BUY because it used to be one would be the display asserting a position the
    engine has stopped taking."""
    s = select_slate([], [card("AAA", 12)], held_slate=["AAA"], exit_rank=16)
    assert s.picks[0]["status"] == WATCH


def test_admitted_names_still_fill_before_monitored_ones():
    s = select_slate([card("BUY1", 9)], [card("WATCH1", 1)], slots=1,
                     held_slate=[], exit_rank=16)
    assert [p["ticker"] for p in s.picks] == ["BUY1"]
    assert s.picks[0]["status"] == BUY


# ------------------------------------------------------- no history, no band
def test_without_a_previous_screen_the_slate_is_the_plain_top_n():
    """A first run has nothing to carry, and that is the correct state rather
    than an error."""
    today = [card(f"N{i}", i) for i in range(1, 8)]
    s = select_slate(today, [], held_slate=None, exit_rank=16)
    assert [p["ticker"] for p in s.picks] == ["N1", "N2", "N3", "N4", "N5"]
    assert s.carried_count == 0 and s.departures == []


def test_without_an_exit_rank_nothing_is_carried():
    """A caller with no band to apply must not silently invent one."""
    s = select_slate([card("AAA", 12), card("NEW", 1)], [],
                     slots=1, held_slate=["AAA"], exit_rank=None)
    assert [p["ticker"] for p in s.picks] == ["NEW"]


def test_every_pick_says_why_it_is_on_the_screen():
    s = select_slate([card("HELD", 12), card("FRESH", 1)], [],
                     held_slate=["HELD"], exit_rank=16)
    reasons = {p["ticker"]: p["slate_reason"] for p in s.picks}
    assert "carried from the previous run" in reasons["HELD"]
    assert "still inside the exit band of 16" in reasons["HELD"]
    assert "new to the screen this run" in reasons["FRESH"]


def test_the_note_accounts_for_carries_and_departures():
    s = select_slate([card("HELD", 12), card("FRESH", 1)], [],
                     held_slate=["HELD", "GONE"], exit_rank=16)
    assert "held from the previous run" in s.selection_note
    assert "GONE" in s.selection_note


# ----------------------------------------------------------- the real defect
def test_turnover_collapses_when_the_band_is_applied():
    """The end-to-end statement. Ranks jitter within the band across sessions,
    which is exactly the cross-sectional churn that replaced the whole screen
    every day. Under the band the screen holds."""
    import random

    rng = random.Random(20260825)
    names = [f"S{i}" for i in range(60)]

    def session():
        ranks = list(range(1, len(names) + 1))
        rng.shuffle(ranks)
        # Twelve names cluster at the top and jitter among themselves; the rest
        # sit far below. Without a band the top five are a fresh draw each time.
        order = names[:12]
        rng.shuffle(order)
        cards = [card(t, i + 1) for i, t in enumerate(order)]
        cards += [card(t, 20 + i) for i, t in enumerate(names[12:])]
        return cards

    def churn(use_band: bool) -> float:
        prev, moved, total = [], 0, 0
        for _ in range(80):
            s = select_slate(session(), [], slots=5,
                             held_slate=prev if use_band else None,
                             exit_rank=16 if use_band else None)
            now = [p["ticker"] for p in s.picks]
            if prev:
                moved += len(set(now) - set(prev))
                total += len(now)
            prev = now
        return moved / total

    without = churn(False)
    with_band = churn(True)
    assert without > 0.5, f"fixture must actually churn, got {without:.1%}"
    assert with_band < 0.05, f"the band must hold the screen, got {with_band:.1%}"
