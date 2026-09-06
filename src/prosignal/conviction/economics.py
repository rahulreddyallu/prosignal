"""Does the edge survive the cost of capturing it?

THE NUMBER THAT MAKES THIS BIND. `docs/RESULTS_OF_RECORD.json` measures the
ranking's top-decile excess return on the shipped panel, per horizon:

    horizon   top-decile excess   corrected t
      21              +0.64%          2.49
      42              +1.20%          2.36
      63              +1.76%          2.21

Against that, the shipped cost model prices real candidates on the live run at
60 to 84 bps round-trip including impact. So:

  * at 21 sessions the measured edge is SMALLER than the cost of a typical
    candidate. The trade is negative before it starts.
  * at 63 sessions -- the shipped horizon -- a candidate costing 84 bps keeps
    about 92 bps of a 176 bps edge. Slightly more than half.

That is the whole argument for this module. Cost is not a rounding error to be
netted off at the end; on this strategy it is comparable in size to the entire
measured edge, and a candidate whose cost is high enough eliminates it.

WHAT THE REFERENCE EDGE IS, AND WHAT IT IS NOT. `top_decile_excess` is a
UNIVERSE-LEVEL, TOP-DECILE, HISTORICAL AVERAGE. It is not a forecast for this
name. The engine has no calibrated per-name expected return and this module
does not invent one -- inventing one is precisely the "fake probability" failure
this codebase already refuses elsewhere. What is computed is a RATIO:

    cost_burden = (round-trip cost + impact) / reference gross edge

"This trade spends 48% of the only edge the model has ever demonstrated." That
is a statement the evidence supports. "This trade will make 1.2%" is not.

WHY THE REFERENCE IS READ FROM THE RESULTS FILE. Hardcoding 1.76% here would
let the constant and the measurement drift apart silently, which is the failure
mode this repository has been bitten by more than once. If the file is absent or
does not cover the shipped horizon, the burden is NOT TESTABLE and the
candidate cannot claim to have passed an economic test that did not run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

__all__ = ["NetEdge", "reference_edge", "assess"]

#: Where the measured ranking table lives, relative to the repository root.
_RESULTS = Path("docs/RESULTS_OF_RECORD.json")

_cache: Dict[str, Optional[dict]] = {}


def _load_results(root: Optional[Path] = None) -> Optional[dict]:
    key = str(root or "")
    if key in _cache:
        return _cache[key]
    base = Path(root) if root else Path.cwd()
    path = base / _RESULTS
    if not path.exists():
        # Walk upward: the CLI may be invoked from a subdirectory.
        for parent in [Path.cwd(), *Path.cwd().parents]:
            cand = parent / _RESULTS
            if cand.exists():
                path = cand
                break
    try:
        data = json.loads(path.read_text())
    except Exception:
        data = None
    _cache[key] = data
    return data


def reference_edge(horizon_sessions: int,
                   root: Optional[Path] = None) -> tuple:
    """(gross_edge_fraction, t_stat, note) for the shipped horizon.

    Returns (None, None, reason) when the measurement is unavailable. The
    caller must treat that as NOT TESTABLE.
    """
    data = _load_results(root)
    if not data or "ranking" not in data:
        return None, None, ("docs/RESULTS_OF_RECORD.json is unavailable, so the "
                            "measured gross edge this trade is priced against "
                            "cannot be read")
    rows = data.get("ranking") or []
    exact = [r for r in rows if int(r.get("horizon", -1)) == int(horizon_sessions)]
    if not exact:
        have = ", ".join(str(r.get("horizon")) for r in rows)
        return None, None, (f"no measured edge at horizon {horizon_sessions} "
                            f"sessions; the results file covers {have}")
    r = exact[0]
    edge = r.get("top_decile_excess")
    if edge is None:
        return None, None, "the results file records no top-decile excess"
    return float(edge), r.get("top_decile_t_corrected"), None


@dataclass(frozen=True)
class NetEdge:
    """What is left of the measured edge after paying to capture it."""

    ticker: str
    horizon_sessions: int
    #: Measured top-decile excess at this horizon, as a fraction.
    gross_edge: Optional[float] = None
    gross_edge_t: Optional[float] = None
    #: Modelled round-trip cost for THIS candidate at THIS size, in bps.
    cost_bps: Optional[float] = None
    impact_bps: Optional[float] = None
    unavailable: Optional[str] = None

    @property
    def total_cost(self) -> Optional[float]:
        """Round-trip plus impact, as a fraction. Impact is already inside the
        round-trip figure the risk plan reports, so it is NOT added twice."""
        if self.cost_bps is None:
            return None
        return float(self.cost_bps) / 10_000.0

    @property
    def net_edge(self) -> Optional[float]:
        if self.gross_edge is None or self.total_cost is None:
            return None
        return float(self.gross_edge) - float(self.total_cost)

    @property
    def cost_burden(self) -> Optional[float]:
        """Share of the measured edge consumed by cost. >1.0 means it is gone."""
        if self.gross_edge is None or self.total_cost is None:
            return None
        if self.gross_edge <= 0:
            return float("inf")
        return float(self.total_cost) / float(self.gross_edge)

    def testable(self) -> bool:
        return self.unavailable is None and self.net_edge is not None

    def summary(self) -> str:
        if not self.testable():
            return f"Net economic edge NOT TESTABLE: {self.unavailable}"
        return (
            f"The ranking's measured top-decile excess at "
            f"{self.horizon_sessions} sessions is "
            f"{self.gross_edge * 100:+.2f}% (corrected t "
            f"{self.gross_edge_t:.2f}). This trade costs "
            f"{self.cost_bps:.0f} bps round-trip including "
            f"{self.impact_bps:.0f} bps impact, which is "
            f"{self.cost_burden:.0%} of it, leaving "
            f"{self.net_edge * 100:+.2f}%. That is a universe-level historical "
            f"average, not a forecast for this name."
        )


def assess(ticker: str, plan, horizon_sessions: int,
           root: Optional[Path] = None) -> NetEdge:
    """Price one candidate against the measured edge at the shipped horizon."""
    edge, t, why = reference_edge(horizon_sessions, root=root)
    if edge is None:
        return NetEdge(ticker, horizon_sessions, unavailable=why)
    if plan is None:
        return NetEdge(ticker, horizon_sessions, edge, t,
                       unavailable="no risk plan, so the trade has no modelled cost")
    cost = getattr(plan, "estimated_round_trip_cost_bps", None)
    impact = getattr(plan, "estimated_impact_bps", None)
    if cost is None:
        return NetEdge(ticker, horizon_sessions, edge, t,
                       unavailable="the risk plan carries no round-trip cost")
    return NetEdge(ticker, horizon_sessions, edge, t,
                   cost_bps=float(cost),
                   impact_bps=float(impact) if impact is not None else 0.0)
