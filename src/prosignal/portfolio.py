"""Turnover-aware portfolio construction with buffer bands.

Measured context:

    Nifty200 Momentum 30, live since its 2020-08-25 launch:  Sharpe 0.36
    Nifty 200 benchmark, same window:                        Sharpe 0.23
    This engine, walk-forward, 18-session holds:             DSR 0.2%

The index earns its premium at semi-annual rebalance with near-zero turnover
cost. The engine trades an 18-session median hold with a measured 0.38%
per-trade cost drag against a +0.42% mean net return, so costs consume roughly
90% of the gross edge.

Buffer bands: a name enters only inside `entry_rank` but is not sold until it
falls outside a wider `exit_rank`. Without a buffer, a name oscillating around
the rank-30 boundary is bought and sold repeatedly, paying a full round trip
each time for no change in exposure. NSE's own factor indices use this device.

Fixed-cadence rebalance: decisions are made on a schedule rather than whenever
a score crosses a threshold, since continuous re-evaluation turns a quarterly
premium into a weekly cost.

Scoring is unchanged here; only the frequency at which the book may change.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

__all__ = ["Position", "RebalanceDecision", "BufferedPortfolio"]


@dataclass
class Position:
    """One holding, tracked from entry to exit."""

    ticker: str
    entry_date: dt.date
    entry_price: float
    quantity: int
    entry_rank: int
    #: Rank at the most recent rebalance. Used for the exit-band test.
    current_rank: Optional[int] = None
    sessions_held: int = 0

    @property
    def value_at(self) -> float:
        return self.entry_price * self.quantity


@dataclass
class RebalanceDecision:
    """What one rebalance actually did, and why."""

    date: dt.date
    entries: List[str] = field(default_factory=list)
    exits: List[Tuple[str, str]] = field(default_factory=list)   # (ticker, reason)
    holds: List[str] = field(default_factory=list)
    turnover_fraction: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def names_changed(self) -> int:
        return len(self.entries) + len(self.exits)


class BufferedPortfolio:
    """A book that changes on a cadence, with hysteresis on the rank boundary.

    Parameters
    ----------
    entry_rank:
        A name must rank at or better than this to be BOUGHT.
    exit_rank:
        A held name is SOLD only when it ranks worse than this. Must be >=
        `entry_rank`; the gap between the two is the buffer.
    max_positions:
        Hard cap on concurrent holdings.
    """

    def __init__(
        self,
        entry_rank: int = 15,
        exit_rank: int = 30,
        max_positions: int = 15,
    ) -> None:
        if exit_rank < entry_rank:
            raise ValueError(
                f"exit_rank ({exit_rank}) must be >= entry_rank ({entry_rank}). "
                f"An exit band tighter than the entry band inverts the buffer and "
                f"produces MORE turnover than no buffer at all."
            )
        self.entry_rank = entry_rank
        self.exit_rank = exit_rank
        self.max_positions = max_positions
        self.positions: Dict[str, Position] = {}
        self.history: List[RebalanceDecision] = []

    # -- introspection ------------------------------------------------------
    @property
    def buffer_width(self) -> int:
        """Ranks of hysteresis. Zero means no buffer -- pure threshold churn."""
        return self.exit_rank - self.entry_rank

    def held(self) -> Set[str]:
        return set(self.positions)

    # -- the decision -------------------------------------------------------
    def rebalance(
        self,
        date: dt.date,
        ranked: Sequence[str],
        prices: Dict[str, float],
        position_value: float,
        eligible: Optional[Set[str]] = None,
    ) -> RebalanceDecision:
        """Apply the buffer rule to today's ranking.

        ``ranked`` is the full ordered universe, best first. ``eligible`` is the
        set that passed the hard gates; anything held but no longer eligible is
        exited regardless of rank, because eligibility is a tradability
        statement rather than an attractiveness one.
        """
        decision = RebalanceDecision(date=date)
        rank_of = {t: i + 1 for i, t in enumerate(ranked)}
        opening = set(self.positions)

        # ---- exits ---------------------------------------------------------
        for ticker in sorted(opening):
            pos = self.positions[ticker]
            rank = rank_of.get(ticker)

            if eligible is not None and ticker not in eligible:
                self._close(ticker, decision, "no longer eligible")
                continue
            if rank is None:
                self._close(ticker, decision, "dropped out of the ranked universe")
                continue

            pos.current_rank = rank
            if rank > self.exit_rank:
                self._close(
                    ticker, decision,
                    f"rank {rank} fell outside the exit band ({self.exit_rank})",
                )

        # ---- entries -------------------------------------------------------
        room = self.max_positions - len(self.positions)
        if room > 0:
            for ticker in ranked[: self.entry_rank]:
                if room <= 0:
                    break
                if ticker in self.positions:
                    continue
                if eligible is not None and ticker not in eligible:
                    continue
                price = prices.get(ticker)
                if not price or price <= 0:
                    continue
                qty = int(position_value / price)
                if qty <= 0:
                    continue
                self.positions[ticker] = Position(
                    ticker=ticker, entry_date=date, entry_price=price,
                    quantity=qty, entry_rank=rank_of.get(ticker, 0),
                    current_rank=rank_of.get(ticker),
                )
                decision.entries.append(ticker)
                room -= 1

        decision.holds = sorted(set(self.positions) - set(decision.entries))

        # Turnover as a fraction of the book: (entries + exits) / 2 / positions,
        # which is the convention that makes "100% turnover" mean the entire
        # book was replaced once.
        denominator = max(len(opening | set(self.positions)), 1)
        decision.turnover_fraction = decision.names_changed / (2.0 * denominator)

        # Report the buffer's effect whenever a name is held outside the entry
        # band -- NOT only when something exited. The buffer's whole value shows
        # up on the rebalances where nothing changes, which is exactly when a
        # naive threshold would have churned.
        if self.buffer_width > 0:
            kept = [
                t for t, p in self.positions.items()
                if p.current_rank and p.current_rank > self.entry_rank
            ]
            if kept:
                decision.notes.append(
                    f"{len(kept)} name(s) held despite ranking outside the entry "
                    f"band, because they remain inside the exit band. Without the "
                    f"buffer these would have been sold and very likely "
                    f"re-bought, paying a full round trip for no change in view."
                )

        self.history.append(decision)
        return decision

    def _close(self, ticker: str, decision: RebalanceDecision, reason: str) -> None:
        self.positions.pop(ticker, None)
        decision.exits.append((ticker, reason))

    # -- reporting ----------------------------------------------------------
    def annualised_turnover(self, sessions_per_rebalance: int) -> Optional[float]:
        """One-way turnover per year implied by the observed rebalances.

        This is the number that decides whether a premium survives: an edge of
        4%/yr against 120% annual turnover at 38 bps a side is gone.
        """
        if not self.history:
            return None
        mean_turnover = sum(d.turnover_fraction for d in self.history) / len(self.history)
        rebalances_per_year = 252.0 / max(sessions_per_rebalance, 1)
        return mean_turnover * rebalances_per_year

    def summary(self, sessions_per_rebalance: int) -> Dict[str, object]:
        turn = self.annualised_turnover(sessions_per_rebalance)
        return {
            "entry_rank": self.entry_rank,
            "exit_rank": self.exit_rank,
            "buffer_width": self.buffer_width,
            "max_positions": self.max_positions,
            "rebalances": len(self.history),
            "currently_held": len(self.positions),
            "mean_turnover_per_rebalance": (
                round(sum(d.turnover_fraction for d in self.history) / len(self.history), 4)
                if self.history else None
            ),
            "annualised_turnover": round(turn, 3) if turn is not None else None,
            "total_entries": sum(len(d.entries) for d in self.history),
            "total_exits": sum(len(d.exits) for d in self.history),
        }
