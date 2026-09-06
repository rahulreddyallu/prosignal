"""Conviction: turning a ranked universe into 0, 1 or 2 defensible BUYs.

The engine ranks hundreds of names. This package decides whether ANY of them
has evidence strong enough, broad enough, robust enough and cheap enough to
occupy one of two extremely scarce slots -- and answers NO TRADE when none
does.

    evidence      how many INDEPENDENT bets the supporting factors really are
    separation    whether the leader is ahead of the field or merely first
    robustness    whether the conclusion survives other defensible weightings
    economics     whether the edge survives the cost of capturing it
    independence  whether a second BUY is a second bet or the same one twice
    gate          the synthesis, and select_at_most_two

Nothing here emits a probability. See `gate` for why the conviction grade is
ordinal and what would have to be measured before it could be anything else.
"""

from . import economics, evidence, gate, independence, robustness, separation

__all__ = [
    "economics",
    "evidence",
    "gate",
    "independence",
    "robustness",
    "separation",
]
