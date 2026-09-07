"""Does the edge survive the cost of capturing it?

THE NUMBER THAT MAKES THIS BIND, and it got much worse when the results file
was rebuilt to separate in-sample from out-of-sample. `docs/RESULTS_OF_RECORD.
json` now records the SAME horizon under four windows:

    horizon 63     top-decile excess   corrected t
      OUT_OF_SAMPLE        +0.360%         0.29
      STABLE_MODEL         +1.202%         1.37
      FULL_PANEL           +1.941%         2.34
      IN_SAMPLE            +2.399%         2.44

An earlier version of this module quoted +1.76% at t 2.21 -- a number that no
longer exists in the file -- and read the arm POSITIONALLY, taking the first
row matching the horizon. It landed on OUT_OF_SAMPLE by luck. See
`DEFAULT_WINDOW`.

Against the honest arm, the shipped cost model prices real live candidates at
60 to 84 bps round-trip including impact. So at the shipped 63-session horizon:

    out-of-sample edge     36 bps
    typical candidate cost 60 to 84 bps
    net                    NEGATIVE, before the trade starts

The out-of-sample edge does not cover the cost of capturing it at ANY horizon
in the file. That is not this module failing -- it is this module reporting,
for the first time, what the measurement actually says. A gate calibrated
against +1.76% waves candidates through that a gate calibrated against +0.36%
refuses, and only one of those numbers is out of sample.

WHAT THE REFERENCE EDGE IS, AND WHAT IT IS NOT. `top_decile_excess` is a
UNIVERSE-LEVEL, TOP-DECILE, HISTORICAL AVERAGE. It is not a forecast for this
name. The engine has no calibrated per-name expected return and this module
does not invent one -- inventing one is precisely the "fake probability" failure
this codebase already refuses elsewhere. What is computed is a RATIO:

    cost_burden = (round-trip cost + impact) / reference gross edge

"This trade spends 233% of the only out-of-sample edge the model has ever
demonstrated" is a statement the evidence supports. "This trade will make 1.2%"
is not, and neither is any figure taken from the in-sample arm.

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

__all__ = ["NetEdge", "reference_edge", "assess", "DEFAULT_WINDOW"]

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


#: WHICH ARM OF THE RESULTS FILE A TRADE IS PRICED AGAINST.
#:
#: The file records the SAME horizon under four windows, and they disagree by
#: a factor of nearly seven. At 63 sessions: OUT_OF_SAMPLE +0.360% (t 0.29),
#: STABLE_MODEL +1.202% (t 1.37), FULL_PANEL +1.941% (t 2.34), IN_SAMPLE
#: +2.399% (t 2.44).
#:
#: This constant used to be absent, and the lookup took the FIRST row matching
#: the horizon. It happened to land on OUT_OF_SAMPLE, so the shipped behaviour
#: was correct -- by accident. Nothing in the code knew windows existed, and a
#: reordered file would have silently started pricing every trade against a
#: number 6.7x larger while every test still passed.
#:
#: OUT_OF_SAMPLE is the only honest default. An in-sample edge is what the
#: model was fitted to produce; charging a real cost against it and calling the
#: difference an economic edge is the arithmetic that makes every overfitted
#: strategy look profitable.
DEFAULT_WINDOW = "OUT_OF_SAMPLE"


def reference_edge(horizon_sessions: int,
                   root: Optional[Path] = None,
                   window: str = DEFAULT_WINDOW) -> tuple:
    """(gross_edge_fraction, t_stat, note) for the shipped horizon.

    Returns (None, None, reason) when the measurement is unavailable. The
    caller must treat that as NOT TESTABLE -- never as a pass.
    """
    data = _load_results(root)
    if not data or "ranking" not in data:
        return None, None, ("docs/RESULTS_OF_RECORD.json is unavailable, so the "
                            "measured gross edge this trade is priced against "
                            "cannot be read")
    rows = data.get("ranking") or []
    at_h = [r for r in rows if int(r.get("horizon", -1)) == int(horizon_sessions)]
    if not at_h:
        have = ", ".join(sorted({str(r.get("horizon")) for r in rows}))
        return None, None, (f"no measured edge at horizon {horizon_sessions} "
                            f"sessions; the results file covers {have}")

    # NAMED, NEVER POSITIONAL. A file carrying several windows and no way to
    # say which one is being read is a file that will eventually be read wrong.
    windows = [r for r in at_h if str(r.get("window", "")) == window]
    if not windows:
        available = ", ".join(sorted({str(r.get("window") or "unlabelled")
                                      for r in at_h}))
        if len(at_h) == 1 and at_h[0].get("window") is None:
            windows = at_h                     # single unlabelled arm: no ambiguity
        else:
            return None, None, (
                f"the results file records {len(at_h)} arms at horizon "
                f"{horizon_sessions} ({available}) and none is {window!r}. "
                f"Pricing a trade against an unnamed arm is how an in-sample "
                f"number becomes an economic claim.")

    r = windows[0]
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
           root: Optional[Path] = None,
           window: str = DEFAULT_WINDOW) -> NetEdge:
    """Price one candidate against the measured edge at the shipped horizon."""
    edge, t, why = reference_edge(horizon_sessions, root=root, window=window)
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
