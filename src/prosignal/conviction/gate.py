"""The final gate: 0, 1 or 2 BUYs, and why not #3.

WHAT THIS REPLACES. The shipped path decides a BUY with a rank band
(`model_rank <= entry_rank`) and an entry clock (`cadence_sessions = 21`).
Measured on the live run, the evidence gates ahead of them cut ZERO of 36
defended names, and the answer on the day was NO TRADE for one reason: session
2 of 21. Scarcity was a property of the calendar. Here it is a property of the
evidence, and the calendar has no vote.

THE COMPETITION. The bar RISES as a candidate approaches the slot:

    every defended survivor
        -> hard gates              (may this be bought at all?)
        -> shortlist by alpha      (is it near the top?)
        -> conviction evaluation   (is the evidence real, broad and robust?)
        -> economic gate           (does it survive its own cost?)
        -> separation gate         (is it actually ahead of the field?)
        -> slot 1
        -> independence            (is #2 a second bet, or the same one?)
        -> slot 2

Every stage can end in nothing. NO TRADE is the designed outcome of a day whose
evidence does not justify capital, and it is reported with the reason that
actually bound.

CONVICTION IS ORDINAL AND UNCALIBRATED, AND SAYS SO. The grade below is a
summary of measured components -- independent evidence count, separation in
sigma units, specification survival, net edge after cost. It is NOT a
probability and must never be rendered as one. No mapping from grade to
realised outcome frequency has been established on this engine, because the
selection-precision study that would establish it has not been run. Until it
has, the grade orders candidates against each other and claims nothing about
what happens next. `docs/RESULTS_OF_RECORD.json` is the only outcome
measurement in the repository and it is universe-level, not per-grade.

THRESHOLDS ARE UNVALIDATED. Every number in `Thresholds` is a starting
hypothesis chosen to be defensible a priori, not one fitted to make today's
output look good. They must be moved by the selection-precision study, and the
frequency of BUYs must be a CONSEQUENCE of them rather than a target they were
tuned toward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import economics, evidence, independence, robustness, separation

__all__ = [
    "Thresholds",
    "Candidate",
    "Verdict",
    "evaluate",
    "select_at_most_two",
    "NoTradeCause",
]


class NoTradeCause:
    """Why nothing was bought. These are NOT interchangeable."""

    NO_CANDIDATE = "no_candidate_cleared_conviction"
    NO_SEPARATION = "candidates_existed_without_separation"
    COST = "edge_did_not_survive_cost"
    DATA = "data_unavailable"
    UNIVERSE = "universe_insufficient"
    HALT = "market_wide_safety_halt"


@dataclass(frozen=True)
class Thresholds:
    """The final bar. Every value is UNVALIDATED -- see the module docstring."""

    #: How many of the top-ranked survivors are evaluated in full. Conviction
    #: evaluation costs a universe re-rank per candidate, so this is a compute
    #: bound, not a selection rule -- it must be wide enough that the eventual
    #: winner is always inside it.
    shortlist: int = 15

    #: MINIMUM INDEPENDENT EVIDENCE. Below this the candidate's supporting
    #: factors are one bet wearing many hats. Set at 2.0 because a single
    #: independent direction is, definitionally, one piece of evidence, and this
    #: engine's whole claim is multi-factor. Measured on the live cross-section
    #: the top-ranked name spans 1.53 -- so this bar bites immediately, which is
    #: the point.
    min_independent_evidence: float = 2.0

    #: No single direction may carry more than this share of the evidence.
    max_evidence_concentration: float = 0.85

    #: SEPARATION, in robust cross-sectional sigma units of the RAW score.
    #: A leader inside noise of the field is not a leader.
    min_gap_to_median: float = 1.00
    min_gap_to_next: float = 0.05

    #: SPECIFICATION SURVIVAL. The share of alternative theme weightings under
    #: which the name holds a top-10 place.
    min_robustness: float = 0.70

    #: ECONOMICS. Cost may not consume more than this share of the measured
    #: top-decile excess at the shipped horizon, and net edge must be positive.
    max_cost_burden: float = 0.60

    #: INDEPENDENCE for the second slot.
    max_residual_correlation: float = 0.35
    max_evidence_similarity: float = 0.80
    min_basket_enb: float = 1.70

    #: Hard cap. The production layer can never emit more than this.
    max_buys: int = 2


@dataclass
class Candidate:
    """One name, fully evaluated, with every component that decided it."""

    ticker: str
    rank: int
    sector: Optional[str] = None
    evidence: Optional[evidence.EvidenceProfile] = None
    separation: Optional[separation.Separation] = None
    robustness: Optional[robustness.RankEnvelope] = None
    economics: Optional[economics.NetEdge] = None
    #: Reasons it cannot be bought. Empty means it cleared everything.
    failures: List[str] = field(default_factory=list)
    #: Things worth saying that are not disqualifying.
    notes: List[str] = field(default_factory=list)
    #: Set once the name is placed, explaining independence from the first pick.
    independence: Optional[independence.Independence] = None

    @property
    def clears(self) -> bool:
        return not self.failures

    def quality(self) -> float:
        """Decision quality, for ordering candidates that all cleared.

        NOT an expected return and NOT comparable across days. It exists to
        break ties among names that have already passed every gate, and it
        weights the things the gates measure only as minima: broader evidence,
        more separation, more robustness, more surviving edge.
        """
        ev = self.evidence.members.effective if self.evidence and self.evidence.testable() else 0.0
        sep = (self.separation.gap_to_median or 0.0) if self.separation and self.separation.testable() else 0.0
        rob = self.robustness.survival if self.robustness and self.robustness.testable() else 0.0
        net = (self.economics.net_edge or 0.0) if self.economics and self.economics.testable() else 0.0
        # Scaled so no term can dominate by units alone. Net edge is in return
        # fractions and is multiplied up to the same order as the others.
        return float(ev + sep + 2.0 * rob + 100.0 * net)

    def grade(self) -> str:
        """Ordinal conviction. NOT a probability -- see the module docstring."""
        if not self.clears:
            return "-"
        q = self.quality()
        if q >= 9.0:
            return "A"
        if q >= 7.5:
            return "A-"
        if q >= 6.0:
            return "B+"
        return "B"


@dataclass
class Verdict:
    """The day's decision."""

    buys: List[Candidate] = field(default_factory=list)
    considered: List[Candidate] = field(default_factory=list)
    cause: Optional[str] = None
    reason: Optional[str] = None

    @property
    def is_no_trade(self) -> bool:
        return not self.buys

    def runner_up(self) -> Optional[Candidate]:
        """The best name that did NOT get a slot -- the 'why not #3' record."""
        taken = {c.ticker for c in self.buys}
        rest = [c for c in self.considered if c.ticker not in taken]
        if not rest:
            return None
        return sorted(rest, key=lambda c: (not c.clears, -c.quality()))[0]


# =============================================================================


def _momentum_share(score, themes: Sequence[str]) -> Optional[float]:
    """Share of the composite carried by the momentum theme."""
    total = 0.0
    mom = 0.0
    for k in themes:
        f = score.factors.get(k)
        if f is None or f.contribution is None or not np.isfinite(f.contribution):
            continue
        c = float(f.contribution)
        total += abs(c)
        if "momentum" in k.lower():
            mom += abs(c)
    if total <= 0:
        return None
    return mom / total


def _panic_state(regime) -> Tuple[bool, str]:
    """Daniel & Moskowitz (2016) panic state, from what Stage 2 measures.

    Momentum crashes are partly forecastable: they cluster after market
    declines, in high volatility, and land on the rebound. The paper's own
    dynamic strategy conditions on exactly the market state and variance this
    engine already computes. This does not BLOCK -- it raises the bar for a
    candidate whose case is mostly momentum, which is the specific exposure the
    crash destroys.
    """
    trend = str(getattr(regime, "trend_regime", "") or "")
    declining = "down" in trend.lower()
    pct = getattr(regime, "vix_percentile", None)
    high_vol = pct is not None and float(pct) >= 70.0
    if declining and high_vol:
        return True, (f"the market is in a declining trend with volatility at "
                      f"the {float(pct):.0f}th percentile of its trailing year")
    if getattr(regime, "transition_flag", False) and high_vol:
        return True, (f"the regime is in transition with volatility at the "
                      f"{float(pct):.0f}th percentile")
    return False, ""


def evaluate(score, ranked_scores, plan, regime, horizon_sessions: int,
             thresholds: Thresholds,
             theme_panel=None, member_panel=None,
             themes: Sequence[str] = ()) -> Candidate:
    """Run every conviction test on one candidate and record what failed."""
    cand = Candidate(ticker=score.ticker, rank=int(score.rank or 0),
                     sector=score.sector)

    # ---- independent evidence -------------------------------------------
    prof = evidence.evidence_for(score, ranked_scores,
                                 theme_panel=theme_panel,
                                 member_panel=member_panel)
    cand.evidence = prof
    if not prof.testable():
        cand.failures.append(
            f"independent evidence NOT TESTABLE ({prof.themes.unavailable or prof.members.unavailable})")
    else:
        eff = prof.members.effective
        if eff < thresholds.min_independent_evidence:
            cand.failures.append(
                f"{prof.members.nominal} supporting factors span only "
                f"{eff:.2f} independent directions, below the "
                f"{thresholds.min_independent_evidence:.2f} required -- they "
                f"are one bet counted many times")
        if prof.members.concentration > thresholds.max_evidence_concentration:
            cand.failures.append(
                f"{prof.members.concentration:.0%} of the evidence sits in a "
                f"single direction, above the "
                f"{thresholds.max_evidence_concentration:.0%} limit")

    # ---- separation ------------------------------------------------------
    sep = separation.measure(ranked_scores, score.ticker)
    cand.separation = sep
    if not sep.testable():
        cand.failures.append(f"separation NOT TESTABLE ({sep.unavailable})")
    else:
        if sep.gap_to_median is None or sep.gap_to_median < thresholds.min_gap_to_median:
            got = "n/a" if sep.gap_to_median is None else f"{sep.gap_to_median:.2f}"
            cand.failures.append(
                f"only {got} sigma above the median name, below the "
                f"{thresholds.min_gap_to_median:.2f} required")
        if sep.gap_to_next is not None and sep.gap_to_next < thresholds.min_gap_to_next:
            cand.failures.append(
                f"only {sep.gap_to_next:.3f} sigma clear of the next candidate, "
                f"below the {thresholds.min_gap_to_next:.3f} required -- the "
                f"ordering here is inside the noise")

    # ---- robustness ------------------------------------------------------
    env = robustness.envelope(ranked_scores, score.ticker)
    cand.robustness = env
    if not env.testable():
        cand.failures.append(f"robustness NOT TESTABLE ({env.unavailable})")
    elif env.survival < thresholds.min_robustness:
        cand.failures.append(
            f"holds a top-{env.survive_rank} place under only "
            f"{env.survival:.0%} of alternative theme weightings, below the "
            f"{thresholds.min_robustness:.0%} required (worst: rank "
            f"{env.worst_rank} under '{env.worst_spec}')")

    # ---- economics -------------------------------------------------------
    net = economics.assess(score.ticker, plan, horizon_sessions)
    cand.economics = net
    if not net.testable():
        cand.failures.append(f"net economic edge NOT TESTABLE ({net.unavailable})")
    else:
        if net.net_edge is not None and net.net_edge <= 0:
            cand.failures.append(
                f"cost of {net.cost_bps:.0f} bps exceeds the "
                f"{net.gross_edge * 100:.2f}% measured edge at "
                f"{horizon_sessions} sessions -- the trade is negative before "
                f"it starts")
        elif net.cost_burden is not None and net.cost_burden > thresholds.max_cost_burden:
            cand.failures.append(
                f"cost consumes {net.cost_burden:.0%} of the measured edge, "
                f"above the {thresholds.max_cost_burden:.0%} limit")

    # ---- momentum crash defense -----------------------------------------
    panic, why = _panic_state(regime)
    share = _momentum_share(score, themes)
    if panic and share is not None and share >= 0.50:
        cand.failures.append(
            f"{share:.0%} of this name's case is momentum and {why}. Momentum "
            f"crashes cluster in exactly this state (Daniel & Moskowitz 2016), "
            f"so a momentum-carried candidate is refused here rather than "
            f"downweighted")
    elif share is not None and share >= 0.50:
        cand.notes.append(
            f"{share:.0%} of the case is momentum; the regime is not a panic "
            f"state, so this is disclosed rather than penalised")

    return cand


def select_at_most_two(candidates: Sequence[Candidate], closes: pd.DataFrame,
                       thresholds: Thresholds,
                       themes: Sequence[str] = ()) -> Verdict:
    """Fill 0, 1 or 2 slots. NEVER more, and never to fill a slot.

    Ordering is by decision quality among names that cleared every gate. A name
    that failed a gate is not promoted because the names above it failed too --
    that is the "forced slot filling" this design exists to prevent.
    """
    considered = list(candidates)
    verdict = Verdict(considered=considered)

    cleared = sorted([c for c in considered if c.clears],
                     key=lambda c: -c.quality())
    if not cleared:
        verdict.cause = NoTradeCause.NO_CANDIDATE
        n = len(considered)
        if n:
            best = sorted(considered, key=lambda c: -c.quality())[0]
            verdict.reason = (
                f"{n} candidate(s) were evaluated in full and none cleared the "
                f"conviction gate. The closest was {best.ticker} at rank "
                f"{best.rank}, refused because: {best.failures[0]}")
        else:
            verdict.reason = ("no candidate reached conviction evaluation")
        return verdict

    first = cleared[0]
    verdict.buys.append(first)

    # ---- the second slot has to EARN its place --------------------------
    for cand in cleared[1:]:
        if len(verdict.buys) >= thresholds.max_buys:
            break
        ind = independence.assess_pair(
            first.ticker, cand.ticker, closes,
            sector_a=first.sector, sector_b=cand.sector,
            themes=themes)
        cand.independence = ind
        if not ind.testable():
            cand.failures.append(
                f"cannot be shown independent of {first.ticker}: "
                f"{ind.unavailable}. An unmeasured correlation is not a low "
                f"one, so the second slot stays empty")
            continue
        if ind.residual_correlation is not None and \
                abs(ind.residual_correlation) > thresholds.max_residual_correlation:
            cand.failures.append(
                f"residual correlation {ind.residual_correlation:+.2f} with "
                f"{first.ticker} after removing the market, above the "
                f"{thresholds.max_residual_correlation:.2f} limit -- this is "
                f"the same trade twice")
            continue
        if ind.evidence_similarity is not None and \
                ind.evidence_similarity > thresholds.max_evidence_similarity:
            cand.failures.append(
                f"its evidence overlaps {first.ticker}'s at "
                f"{ind.evidence_similarity:.2f}, above the "
                f"{thresholds.max_evidence_similarity:.2f} limit -- both names "
                f"are being selected by the same themes")
            continue
        if ind.basket_enb is not None and ind.basket_enb < thresholds.min_basket_enb:
            cand.failures.append(
                f"a book of {first.ticker} and {cand.ticker} spans only "
                f"{ind.basket_enb:.2f} of 2.00 independent bets, below the "
                f"{thresholds.min_basket_enb:.2f} required")
            continue
        verdict.buys.append(cand)

    if len(verdict.buys) == 1 and len(cleared) > 1:
        verdict.reason = (
            f"One position. {len(cleared) - 1} other name(s) cleared the "
            f"conviction gate but none could be shown to be a second, "
            f"independent opportunity alongside {first.ticker}.")
    return verdict


def to_record(verdict: "Verdict") -> List[Dict[str, object]]:
    """Flatten a verdict into the rows the ledger keeps.

    EVERY candidate evaluated, cleared or refused. The refused half is the
    half that makes this worth recording: the selection-precision study asks
    whether the names the gate turned down went on to beat the ones it took,
    and that question is unanswerable from a record of winners.

    Values are plain floats and strings so the row survives a JSON round trip
    without the reader needing this module to interpret it.
    """
    taken = {c.ticker for c in verdict.buys}
    rows: List[Dict[str, object]] = []
    for c in verdict.considered:
        ev = c.evidence.members if c.evidence and c.evidence.testable() else None
        th = c.evidence.themes if c.evidence and c.evidence.testable() else None
        sp = c.separation if c.separation and c.separation.testable() else None
        rb = c.robustness if c.robustness and c.robustness.testable() else None
        ec = c.economics if c.economics and c.economics.testable() else None
        rows.append({
            "ticker": c.ticker,
            "model_rank": c.rank,
            "sector": c.sector,
            "bought": c.ticker in taken,
            "cleared": c.clears,
            "conviction_grade": c.grade(),
            "decision_quality": round(c.quality(), 4),
            # evidence
            "factors_supporting": ev.nominal if ev else None,
            "independent_evidence": round(ev.effective, 3) if ev else None,
            "evidence_concentration": round(ev.concentration, 3) if ev else None,
            "redundancy_ratio": round(ev.redundancy_ratio, 3) if ev else None,
            "themes_independent": round(th.effective, 3) if th else None,
            # separation
            "gap_to_next_sigma": round(sp.gap_to_next, 4) if sp and sp.gap_to_next is not None else None,
            "gap_to_median_sigma": round(sp.gap_to_median, 4) if sp and sp.gap_to_median is not None else None,
            "gap_to_fifth_sigma": round(sp.gap_to_fifth, 4) if sp and sp.gap_to_fifth is not None else None,
            # robustness
            "spec_survival": round(rb.survival, 3) if rb else None,
            "spec_median_rank": rb.median_rank if rb else None,
            "spec_worst_rank": rb.worst_rank if rb else None,
            "spec_worst": rb.worst_spec if rb else None,
            # economics
            "gross_edge": round(ec.gross_edge, 5) if ec else None,
            "cost_bps": ec.cost_bps if ec else None,
            "net_edge": round(ec.net_edge, 5) if ec and ec.net_edge is not None else None,
            "cost_burden": round(ec.cost_burden, 4) if ec and ec.cost_burden is not None else None,
            # independence, present only on a name that reached the second slot
            "residual_correlation": (round(c.independence.residual_correlation, 4)
                                     if c.independence and c.independence.testable()
                                     and c.independence.residual_correlation is not None else None),
            "basket_enb": (round(c.independence.basket_enb, 4)
                           if c.independence and c.independence.testable()
                           and c.independence.basket_enb is not None else None),
            # why not
            "failures": list(c.failures),
            "notes": list(c.notes),
        })
    return rows
