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

HYSTERESIS. The slate is stateful, and this is the part that was missing. Stage
6 admits a name inside ``entry_rank`` and holds it while it stays inside the
wider ``exit_rank``; the strategy that was validated is therefore patient. The
slate was a fresh top-N snapshot taken each session with no memory of the last
one, so the screen turned over even when the engine's own view had not changed.
Measured over the recorded ledger: mean top-5 turnover 74.9%, median 80%, and
the median number of sessions a name survived on the screen was ONE. The card
was quoting a multi-week hold for a position the list would not contain
tomorrow.

The same band now governs the screen. A name that was shown stays shown while
its model rank is inside ``exit_rank``, and leaves when the engine would
actually close it. Free slots -- and only free slots -- are filled from the top
of today's ranking. Nothing here promotes a name the engine did not produce,
and a carried name carries today's status, not the one it had when it entered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

#: The dashboard leads with this many names. Not a maximum to be filled at any
#: cost -- if the engine produced fewer real candidates, the slate is shorter.
#:
#: THIS IS THE BOOK, and it is a fallback rather than the authority. The run
#: passes `slots` from `stage6_entry.admission.entry_rank`, which the loader
#: pins to `capital.max_open_positions`; this constant only applies to a caller
#: that has no config in hand.
#:
#: It read 5 while the engine opened 6, so the screen omitted one position of
#: every book it described -- and because the slate is RECORDED, the sixth name
#: was also missing from the ledger, the history page and the outcome record.
#: The card's own thesis text said "inside the top 6", which is how the
#: disagreement stayed invisible: the prose came from the config and the list
#: came from here.
SLOTS = 6

BUY = "BUY"
WATCH = "WATCH"


@dataclass(frozen=True)
class Departure:
    """A name that was on the screen yesterday and is not on it today."""

    ticker: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"ticker": self.ticker, "reason": self.reason}


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
    #: How many of the picks were carried from the previous screen rather than
    #: chosen fresh today. The turnover figure, available without a diff.
    carried_count: int = 0
    #: What left the screen since the previous run, and why. A name vanishing
    #: with no reason recorded is how a silent eviction looks from outside.
    departures: List[Departure] = field(default_factory=list)

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
    held_slate: Optional[Sequence[str]] = None,
    exit_rank: Optional[int] = None,
) -> Slate:
    """Curate the engine's output into the dashboard slate.

    `recommendations` are the names the engine admitted; `watchlist` are the
    names it scored and monitored but did not admit.

    `held_slate` is what the previous run put on the screen, and `exit_rank` is
    Stage 6's exit band. Given both, a name that was shown keeps its slot while
    its model rank is inside that band; only the slots it does not hold are
    filled, buys first and then the watchlist, in model order. Given neither --
    a first run, or a caller with no history to offer -- the slate is the plain
    top-N it always was.
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

    # Priority order for filling: admitted names before monitored ones, each
    # already in model order.
    pool = buys + watch
    by_ticker = {str(c.get("ticker")): c for c in pool}

    carried, departures = _carry(held_slate, by_ticker, exit_rank, slots)
    carried_tickers = {str(c.get("ticker")) for c in carried}

    # A name that left the screen this session cannot fill a slot on it in the
    # same session. Without this the exit band is decorative: a name drifting to
    # rank 17 is dropped by `_carry` and immediately re-selected by the fill,
    # relabelled "new to the screen", and the departure that was just recorded
    # describes nothing that happened.
    departed = {d.ticker for d in departures}
    fresh = [c for c in pool
             if str(c.get("ticker")) not in carried_tickers
             and str(c.get("ticker")) not in departed]
    chosen = carried + fresh[: max(slots - len(carried), 0)]

    # The screen stays ordered by model rank whichever way a name arrived, so a
    # carried name that has drifted sinks to where its rank now says it belongs
    # rather than sitting at the top because it happened to be there yesterday.
    chosen = sorted(chosen, key=_sort_key)

    picks: List[Dict[str, Any]] = []
    n_buy = n_watch = 0
    for position, card in enumerate(chosen, start=1):
        ticker = str(card.get("ticker"))
        row = dict(card)
        row["slate_position"] = position
        status = BUY if ticker in buy_tickers else WATCH
        row["status"] = status
        row["carried"] = ticker in carried_tickers
        row["slate_reason"] = _why_shown(card, ticker in carried_tickers, exit_rank)
        if status == BUY:
            n_buy += 1
        else:
            n_watch += 1
        picks.append(row)

    note = _describe(n_buy, n_watch, len(buys), len(watch), slots,
                     carried=len(carried), departures=departures)
    return Slate(
        picks=picks,
        buy_count=n_buy,
        watch_count=n_watch,
        ranked_buys=buys,
        ranked_watch=watch,
        selection_note=note,
        carried_count=len(carried),
        departures=departures,
    )


def _rank_of(card: Dict[str, Any]) -> Optional[int]:
    rank = card.get("model_rank")
    return int(rank) if isinstance(rank, (int, float)) else None


def _carry(
    held_slate: Optional[Sequence[str]],
    by_ticker: Dict[str, Dict[str, Any]],
    exit_rank: Optional[int],
    slots: int,
) -> tuple:
    """Which of yesterday's names keep their slot, and why the rest do not.

    Mirrors ``stage6_entry._admit``'s held branch deliberately: one band, one
    rule, stated in one place per layer. A name is kept while its model rank is
    inside ``exit_rank``. It is not kept when the engine stopped producing it at
    all -- there is no card to show, and inventing one would put a position on
    the screen the run did not stand behind.

    Held names are ranked against each other BEFORE the slot cap is applied.
    Taking them in the order the previous screen happened to list them would
    drop a name at rank 2 to keep one at rank 15, which is the opposite of what
    the band is for.
    """
    if not held_slate or exit_rank is None:
        return [], []

    eligible: List[Dict[str, Any]] = []
    departures: List[Departure] = []
    for ticker in held_slate:
        ticker = str(ticker)
        card = by_ticker.get(ticker)
        if card is None:
            departures.append(Departure(
                ticker,
                "the engine did not produce it this run, so there is nothing to "
                "show. It failed an eligibility, data-quality or defence check.",
            ))
            continue
        rank = _rank_of(card)
        if rank is None:
            departures.append(Departure(
                ticker, "no model rank this run, so the exit band cannot be applied"
            ))
            continue
        if rank > exit_rank:
            departures.append(Departure(
                ticker, f"rank {rank} has left the exit band of {exit_rank}"
            ))
            continue
        eligible.append(card)

    eligible.sort(key=_sort_key)
    carried = eligible[:slots]
    for card in eligible[slots:]:
        departures.append(Departure(
            str(card.get("ticker")),
            f"rank {_rank_of(card)} is still inside the band, but the screen "
            f"holds {slots} and higher-ranked held names filled it",
        ))
    return carried, departures


def _why_shown(card: Dict[str, Any], carried: bool, exit_rank: Optional[int]) -> str:
    """One line per row saying why it is on the screen at all."""
    rank = _rank_of(card)
    where = f"rank {rank}" if rank is not None else "unranked"
    if carried and exit_rank is not None:
        return (f"carried from the previous run; {where}, still inside the exit "
                f"band of {exit_rank}")
    return f"new to the screen this run at {where}"


def _describe(n_buy: int, n_watch: int, total_buy: int, total_watch: int,
              slots: int, *, carried: int = 0,
              departures: Optional[Sequence[Departure]] = None) -> str:
    """Say how the slate was filled, in the language of the interface."""
    if n_buy == 0 and n_watch == 0:
        if departures:
            return (
                f"No names qualified or came close enough to monitor. "
                f"{len(departures)} left the screen since the previous run."
            )
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
    note = " and ".join(parts) + tail
    if carried:
        note += (
            f" {carried} of {filled} {'was' if carried == 1 else 'were'} held "
            f"from the previous run rather than picked again today."
        )
    if departures:
        note += (
            f" {len(departures)} left the screen: "
            + "; ".join(f"{d.ticker} -- {d.reason}" for d in departures)
            + "."
        )
    return note
