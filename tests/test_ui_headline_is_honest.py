"""The interface headline must say what number it is showing.

`avg_excess` is the average one POSITION is ahead of the index over the days it
was held. That is the right way to judge the selection -- a position's return
does not depend on how large the position was, so unlike the book-level
`mean_excess` it is not confounded by the risk budget.

It is not what an account running this engine is up. Position size is
`risk_budget / risk_per_share`, so the book runs about a fifth invested and the
rest sat in cash while the index compounded. The closed-record headline had
always said "average per position"; the OPEN one said only "vs the index", and
neither said anything about the cash.

Nothing here changes a number. It changes what the page claims one means.
"""

from __future__ import annotations

from pathlib import Path

import pytest

UI = (Path(__file__).resolve().parents[1]
      / "src/prosignal/static/index.html").read_text(encoding="utf-8")


def test_the_open_headline_says_per_position():
    assert "vs the index, per position" in UI, (
        "the open-positions headline reads as the account's return against "
        "the index; it is a per-position average"
    )
    assert "' vs the index'" not in UI, (
        "the bare label is back"
    )


def test_the_closed_headline_still_says_per_position():
    assert "average per position vs the index" in UI


def test_both_headlines_say_it_is_not_the_account_balance():
    assert "CASH_NOTE" in UI
    assert "not what the account earned" in UI
    assert "in cash" in UI


def test_the_cash_note_is_attached_to_the_closed_headline():
    """Declared and unused is the same as absent."""
    body = UI[UI.index("const CASH_NOTE"):]
    assert "effLine + CASH_NOTE" in UI, (
        "CASH_NOTE is defined but never rendered"
    )
    assert body  # the constant is defined before its use site is rendered


def test_no_leverage_confounded_book_figure_is_shown_as_a_headline():
    """`mean_excess` and the information ratio move with `risk_per_trade_pct`
    -- nine points across a sweep in which the ranking and the names are
    identical. Neither belongs on a card."""
    for token in ("information_ratio", "mean_excess"):
        assert token not in UI, (
            f"{token} reached the interface; it is confounded with the risk "
            f"budget and must not be presented as a result"
        )
