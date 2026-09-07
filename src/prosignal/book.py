"""The book: turning a ranking into positions, and staying invested.

WHAT THIS REPLACES AND WHY. `stage7_risk._position_size` sized every position as
``risk_budget / risk_per_share``. With ``risk_per_trade_pct`` 1.0 and a stop of
``8 x ATR`` capped at 35%, the risk budget always bound ahead of the 16.7%
capital slot, so each position came out at 2.9-5.0% of capital and six slots
invested **18.9%**. Measured beta against the equal-weight eligible universe was
0.15-0.30, and the forgone return on the idle 81% at the universe's ~21% a year
is **-17.0%** annually against a measured net excess of **-17.6%**. The cash
drag was 97% of the underperformance. See docs/REBUILD_2026_09.md 3.3.

THE FIX IS NOT A TUNING. A risk budget divided by a stop distance is a position
sizer for a strategy whose edge is its exit. This engine's edge, such as it is,
is an ORDERING -- and an ordering is expressed by holding the names at the top
of it, in size, all the time. So the stop, the target, the invalidation level
and the risk-category fractions do not get better constants here; they are
absent, and 1/N takes their place.

WHY BREADTH RATHER THAN CONCENTRATION. Grinold (1989) gives IR = IC x sqrt(N)
and Clarke, de Silva & Thorley (2002) refine it to IR = TC x IC x sqrt(N), where
the transfer coefficient TC is the correlation between the weights actually held
and the alpha. A six-name book from a ~380-name universe is small on both terms
at once: little breadth, and a TC crushed by sizing that is uncorrelated with
the score. Sharma, Subramaniam & Sehgal (2021) report for Indian equities
specifically that decile corner portfolios give greater return differentials
than quintiles, which is external evidence for the granularity rather than
this engine's own.

THE DECILE IS NOT CHOSEN FROM THIS ENGINE'S BACKTEST, and that distinction is
load-bearing. The trial denominator behind v3 is unknown and unrecoverable
(docs/REBUILD_2026_09.md 3.1), so picking the book size whose in-sample number
looked best would be one more unrecorded look at a surface the model was already
selected on. The decile rests on the two citations above and on the arithmetic
in this docstring, both of which are external to the panel.

WHAT THIS MODULE DOES NOT DO. It does not decide WHO is eligible -- that is
stage 3, and it is upstream and unchanged. It does not rank -- that is stage 4.
It has no view on whether the ranking works. Given a ranking and a universe it
answers one question: which names, at what size, today.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

from .liquidity import LiquidityState, assess

__all__ = ["BookSpec", "Position", "TargetBook", "size_book", "build_book"]


@dataclass(frozen=True)
class BookSpec:
    """Everything the construction needs, and nothing about exits.

    ``entry_percentile`` / ``exit_percentile`` are fractions of the ELIGIBLE
    universe, not absolute ranks. A rank band fixed in names silently changes
    meaning as the universe grows: rank 6 of 380 is the top 1.6%, and rank 6 of
    200 is the top 3%. Expressing it as a fraction keeps the object constant.
    """

    capital: float
    #: Top fraction of the eligible universe a name must reach to ENTER.
    entry_percentile: float = 0.10
    #: Fraction it may drift to before it is SOLD. The gap is the buffer band.
    #: Without one, a name oscillating across the boundary pays a full round
    #: trip at every rebalance for no change in view -- which is why NSE, MSCI
    #: and FTSE all construct factor indices with one.
    exit_percentile: float = 0.20
    #: A position may not exceed this share of the name's trailing ADTV.
    max_participation_of_adtv: float = 0.01
    #: Floors on the held count, so a thin day cannot produce a 3-name book and
    #: a broad one cannot produce 200 unmanageable slivers.
    min_names: int = 15
    max_names: int = 50
    #: Names whose liquidity is not KNOWN_VALID are not bought. An unmeasured
    #: quantity must reduce size, never license the largest one; `liquidity`
    #: documents the defect this repeats the fix for.
    require_known_liquidity: bool = True


@dataclass(frozen=True)
class Position:
    ticker: str
    rank: int
    shares: int
    price: float
    value: float
    weight: float
    #: Which constraint set the size. "equal weight" is the healthy answer.
    binding: str
    held_before: bool


@dataclass
class TargetBook:
    positions: List[Position] = field(default_factory=list)
    invested_fraction: float = 0.0
    n_eligible: int = 0
    entries: List[str] = field(default_factory=list)
    exits: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def tickers(self) -> List[str]:
        return [p.ticker for p in self.positions]

    @property
    def turnover(self) -> float:
        """Share of the book traded this rebalance, by value."""
        if not self.positions:
            return 0.0
        traded = sum(p.value for p in self.positions if not p.held_before)
        return traded / max(sum(p.value for p in self.positions), 1e-9)


def _target_count(n_eligible: int, spec: BookSpec) -> int:
    raw = int(math.floor(n_eligible * spec.entry_percentile))
    return max(spec.min_names, min(spec.max_names, raw))


def size_book(
    chosen: Sequence[str],
    prices: Mapping[str, float],
    adtv: Mapping[str, Optional[float]],
    spec: BookSpec,
    held: Optional[Sequence[str]] = None,
) -> TargetBook:
    """Equal weight over an ALREADY-SELECTED set. Selection is the caller's.

    THIS IS THE FUNCTION THE PIPELINE USES, and the split matters. Selection in
    production is `stage6_entry._admit` (a per-name band, correct) plus
    `stage8_final_signal`'s two-pass fill, which processes held names before
    fresh ones and therefore gives incumbents priority. Both were checked on
    2026-09-07 and neither has the truncation defect that made this module's own
    buffer inert. So production keeps one selector, and this module supplies the
    one thing that was missing: a size that is a property of the SET.

    `build_book` below adds selection on top for research use, where the whole
    construction has to run outside the pipeline. It is not wired into a run,
    and `tests/test_book.py` pins that it stays that way until stages 5-8 are
    replaced -- two live selectors is the defect this repository keeps
    rediscovering, and it is not being introduced deliberately.
    """
    book = TargetBook(n_eligible=len(chosen))
    if not chosen:
        book.notes.append("nothing was selected; no book to size")
        return book
    return _allocate(list(chosen), prices, adtv, spec, set(held or ()),
                     {t: i for i, t in enumerate(chosen)}, book)


def build_book(
    ranking: Sequence[str],
    prices: Mapping[str, float],
    adtv: Mapping[str, Optional[float]],
    spec: BookSpec,
    held: Optional[Sequence[str]] = None,
) -> TargetBook:
    """The names to hold today, equal-weighted and fully invested.

    ``ranking`` is the eligible universe, best first. ``held`` is what the book
    owned before this rebalance, which is what makes the buffer band mean
    anything: a held name is kept while it stays inside ``exit_percentile``,
    and a new name must reach ``entry_percentile`` to displace nothing in
    particular -- selection is by rank, not by pairwise comparison.

    FULL INVESTMENT IS THE POST-CONDITION, not an aspiration. When a liquidity
    cap binds on one name, the capital it could not take is REDISTRIBUTED over
    the others rather than left in cash. Capping and banking the remainder is
    how a liquidity rule quietly becomes a market-timing rule, and it is a
    smaller version of exactly the defect this module replaces.
    """
    book = TargetBook(n_eligible=len(ranking))
    if not ranking:
        book.notes.append("no eligible names; nothing to hold")
        return book

    held_set = set(held or ())
    n_target = _target_count(len(ranking), spec)
    entry_cut = max(n_target, 1)
    exit_cut = max(int(math.floor(len(ranking) * spec.exit_percentile)), entry_cut)
    rank_of = {t: i for i, t in enumerate(ranking)}

    # INCUMBENTS HAVE PRIORITY FOR SLOTS. This is what hysteresis means and it
    # is easy to write a version that does not do it: merging keepers and new
    # names, sorting the union by rank and truncating to `n_target` looks
    # equivalent and is not. `entry_cut` equals `n_target`, so the top slots are
    # always full of names ranked inside it, and every incumbent that drifted
    # past `n_target` is truncated away no matter how wide `exit_percentile`
    # is. Measured on the real panel, that version turned over 62.3% of the book
    # per rebalance at an exit band of 20%, 30% AND 50% -- identical to three
    # significant figures, because the band was inert.
    #
    # The rule below is the one `stage6_entry` documents and the one NSE, MSCI
    # and FTSE construct factor indices with: a name ENTERS inside
    # `entry_percentile` and is RETAINED while inside `exit_percentile`, so a
    # new name waits for a slot rather than evicting an incumbent that is still
    # inside the wider band.
    keep = [t for t in ranking[:exit_cut] if t in held_set][:n_target]
    room = max(n_target - len(keep), 0)
    fresh = [t for t in ranking[:entry_cut] if t not in held_set][:room]
    chosen = sorted(keep + fresh, key=lambda x: rank_of[x])

    return _allocate(chosen, prices, adtv, spec, held_set, rank_of, book)


def _allocate(chosen, prices, adtv, spec, held_set, rank_of, book) -> TargetBook:
    """Equal weight over `chosen`, redistributing whatever a cap refuses.

    Shared by `size_book` and `build_book` so there is exactly one place where
    capital becomes positions.
    """
    # Tradability, before any capital is divided. A name that cannot be sized
    # must not consume a slot, or the book silently holds cash in its place.
    tradable: List[str] = []
    for t in chosen:
        px = prices.get(t)
        view = assess(adtv.get(t))
        if px is None or not (px > 0):
            book.notes.append(f"{t}: no price; not sized")
            continue
        if spec.require_known_liquidity and view.state is not LiquidityState.KNOWN_VALID:
            book.notes.append(f"{t}: liquidity is {view.describe()}; not sized")
            continue
        tradable.append(t)

    if not tradable:
        book.notes.append("no chosen name could be sized; the book is empty")
        return book

    # Equal weight, then redistribute whatever the liquidity caps refuse. Two
    # passes are enough in practice and the loop is bounded regardless: each
    # round either places all remaining capital or removes at least one name
    # from the uncapped set.
    caps = {t: (assess(adtv[t]).adtv_inr or 0.0) * spec.max_participation_of_adtv
            for t in tradable}
    alloc: Dict[str, float] = {}
    binding: Dict[str, str] = {}
    remaining = float(spec.capital)
    open_names = list(tradable)
    while open_names and remaining > 1.0:
        share = remaining / len(open_names)
        capped = [t for t in open_names if caps[t] < share]
        if not capped:
            for t in open_names:
                alloc[t] = alloc.get(t, 0.0) + share
                binding.setdefault(t, "equal weight")
            remaining = 0.0
            break
        for t in capped:
            alloc[t] = caps[t]
            binding[t] = "liquidity cap"
            remaining -= caps[t]
            open_names.remove(t)

    if remaining > 1.0:
        book.notes.append(
            f"Rs {remaining:,.0f} ({remaining / spec.capital:.1%}) could not be "
            f"placed: every held name is at its liquidity cap. This is the one "
            f"case where cash is the honest answer.")

    for t in sorted(alloc, key=lambda x: rank_of[x]):
        px = float(prices[t])
        shares = int(alloc[t] // px)
        if shares <= 0:
            book.notes.append(f"{t}: allocation below one share; dropped")
            continue
        value = shares * px
        book.positions.append(Position(
            ticker=t, rank=rank_of[t] + 1, shares=shares, price=px, value=value,
            weight=value / spec.capital, binding=binding.get(t, "equal weight"),
            held_before=t in held_set))

    book.invested_fraction = sum(p.value for p in book.positions) / spec.capital
    book.entries = [p.ticker for p in book.positions if not p.held_before]
    book.exits = sorted(held_set - {p.ticker for p in book.positions})
    return book
