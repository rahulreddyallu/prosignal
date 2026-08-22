"""Choose the handful of names the dashboard leads with.

The engine ranks the whole eligible universe and admits the top band. Neither
number is five, and neither was chosen with a reader in mind: 8 BUY and 44
WATCH is a report, not a decision. This module curates that output down to a
fixed slate WITHOUT touching the criteria that produced it. Nothing here can
promote a name the engine did not admit, and nothing here re-scores anything.

ORDERING. Names are ordered by the model's own rank, not by `composite_score`,
and the difference is not cosmetic. `composite_score` is the model's percentile
minus the Stage 5 defence penalties, and on a real run the top 52 names occupy
percentiles 90 to 100 while a single penalty is -0.10. One penalty therefore
moves a name across the entire visible range. Sorting the table that way put
seven WATCHLIST names above every BUY, which is what made the old screen read
as arbitrary: admission used one ordering and the display used another.

Model rank is the ordering the engine actually admits on, it is the ordering
that was validated out of sample, and it is the one the slate uses. The
penalties do not disappear -- they are the strongest material the evidence and
risk sections have, and they are shown there as reasons rather than silently
folded into a number that decides row order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

#: The dashboard leads with this many names. Not a maximum to be filled at any
#: cost -- if the engine produced fewer real candidates, the slate is shorter.
SLOTS = 5

BUY = "BUY"
WATCH = "WATCH"


@dataclass(frozen=True)
class Slate:
    """The names the dashboard leads with, and an account of how they got there."""

    picks: List[Dict[str, Any]] = field(default_factory=list)
    buy_count: int = 0
    watch_count: int = 0
    #: Everything the engine admitted or monitored, ranked, for the deeper views.
    ranked_buys: List[Dict[str, Any]] = field(default_factory=list)
    ranked_watch: List[Dict[str, Any]] = field(default_factory=list)
    #: Plain-language account of the fill, for the audit trail.
    selection_note: str = ""
    #: Set when the slate is deliberately empty rather than merely short.
    withheld_reason: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not self.picks


def _sort_key(card: Dict[str, Any]) -> tuple:
    """Deterministic ordering, with every tie broken explicitly.

    Two names can share a model rank when the underlying scores tie to the
    stored precision. Leaving that to sort stability would make row order
    depend on dict iteration upstream, so the run would not be reproducible
    from its own output. The chain is: model rank, then universe percentile
    descending, then the penalised score descending, then ticker.
    """
    rank = card.get("model_rank")
    # A name with no model rank cannot be ordered against one that has it, and
    # must not silently sort to the top.
    rank_key = rank if isinstance(rank, (int, float)) else float("inf")
    pct = card.get("percentile")
    pct_key = -(pct if isinstance(pct, (int, float)) else float("-inf"))
    score = card.get("score")
    score_key = -(score if isinstance(score, (int, float)) else float("-inf"))
    return (rank_key, pct_key, score_key, str(card.get("ticker") or ""))


def _dedupe(cards: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per ticker.

    A name can reach the payload from more than one path -- it can be admitted
    and also appear among the near misses of a gate it later cleared. Two rows
    for one ticker would spend two of five slots on the same position.
    """
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for card in cards:
        ticker = str(card.get("ticker") or "")
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        out.append(card)
    return out


def select_slate(
    recommendations: Sequence[Dict[str, Any]],
    watchlist: Sequence[Dict[str, Any]],
    *,
    slots: int = SLOTS,
    data_quality_ok: bool = True,
    withheld_reason: Optional[str] = None,
) -> Slate:
    """Curate the engine's output into the dashboard slate.

    `recommendations` are the names the engine admitted; `watchlist` are the
    names it scored and monitored but did not admit. Buys fill first, in model
    order, and the remainder is filled from the watchlist in the same order.
    """
    if slots < 0:
        raise ValueError("slots cannot be negative")

    if not data_quality_ok:
        # Refusing is a result. A slate assembled from data the engine itself
        # flagged as unsound would look exactly like a sound one.
        return Slate(
            selection_note="No slate was produced.",
            withheld_reason=(
                withheld_reason
                or "The run did not meet its own data-quality bar, so no "
                   "recommendation was produced."
            ),
        )

    buys = _dedupe(sorted(recommendations, key=_sort_key))
    buy_tickers = {str(c.get("ticker")) for c in buys}
    # A name cannot be both. If the engine admitted it, that is the status it
    # carries, and it takes one slot rather than two.
    watch = [c for c in _dedupe(sorted(watchlist, key=_sort_key))
             if str(c.get("ticker")) not in buy_tickers]

    picked_buys = buys[:slots]
    remaining = max(slots - len(picked_buys), 0)
    picked_watch = watch[:remaining]

    picks: List[Dict[str, Any]] = []
    for position, card in enumerate(picked_buys + picked_watch, start=1):
        row = dict(card)
        row["slate_position"] = position
        row["status"] = BUY if card in picked_buys else WATCH
        picks.append(row)

    note = _describe(len(picked_buys), len(picked_watch), len(buys), len(watch), slots)
    return Slate(
        picks=picks,
        buy_count=len(picked_buys),
        watch_count=len(picked_watch),
        ranked_buys=buys,
        ranked_watch=watch,
        selection_note=note,
    )


def _describe(n_buy: int, n_watch: int, total_buy: int, total_watch: int,
              slots: int) -> str:
    """Say how the slate was filled, in the language of the interface."""
    if n_buy == 0 and n_watch == 0:
        return "No names qualified or came close enough to monitor."
    parts: List[str] = []
    if n_buy:
        of = f" of {total_buy}" if total_buy > n_buy else ""
        parts.append(f"{n_buy} qualifying {'setup' if n_buy == 1 else 'setups'}{of}")
    else:
        parts.append("no setups met the qualifying criteria")
    if n_watch:
        of = f" of {total_watch}" if total_watch > n_watch else ""
        parts.append(
            f"{n_watch} closest {'candidate' if n_watch == 1 else 'candidates'}{of}"
        )
    filled = n_buy + n_watch
    tail = ""
    if filled < slots:
        tail = (
            f". Only {filled} {'name' if filled == 1 else 'names'} cleared "
            f"the engine's bar, so the list stops there rather than being "
            f"padded."
        )
    return " and ".join(parts) + tail
