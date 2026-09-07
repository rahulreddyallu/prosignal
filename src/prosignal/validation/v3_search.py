"""The v3 factor search, enumerated so the Deflated Sharpe can charge for it.

WHAT WAS WRONG. The trial registry held 99 configurations, every one of them
from a research command written AFTER the v3 composite shipped -- estimator
arms, spread bands, CPCV folds, the 2026-08 execution re-audit. Not one row
came from the search that chose the 22 factors, the five themes, the
combination method, the weight caps, the quality floor or the book. The engine
was charging its headline result for the tuning done after the model existed
and nothing for the search that produced it.

`validation.search_budget.cumulative_trials_logged` was the escape hatch --
"everything before this registry existed" -- and it shipped at 20. The search
record puts the true figure two orders of magnitude higher.

WHY IT COULD NOT BE READ OFF ANYTHING. `research/V3_SEARCH.md` was deleted in
commit f1b2a9a ("Consolidate v3-v9 into one engine and delete what no longer
chooses anything"), along with `work/v3/` and `research/v3/`. Deleting the
search code is defensible -- it no longer chooses anything. Deleting the RECORD
of the search removed the only evidence of how much data-dredging the shipped
model rests on, and the count went with it. The document is restored at
`research/V3_SEARCH.md` and this module is its machine-readable form; every
entry cites the section it is taken from and `tests/test_v3_search_trials.py`
checks the citations against the file.

WHAT IS COUNTED. A configuration whose out-of-sample score was LOOKED AT and
which could have been chosen -- the registry's own rule. So:

  * each factor at each horizon it was screened at, because a factor that
    cleared at h=63 and not at h=21 was a per-horizon decision;
  * each theme at each candidate horizon, because the reversal theme was moved
    off a 42-session label BECAUSE it scored -3.96 there;
  * every arm of a grid that was compared -- combination methods, floor gates,
    cost buckets.

WHAT IS NOT. The 40 permuted-label draws (a null, not a candidate), the
pairwise correlations of the redundancy pass (label-free; the |rho| >= 0.80
THRESHOLD is counted as the one choice it is), and the placebo alignments
inside each screen (the critical value for a single trial, not extra trials).

THE COUNT IS A FLOOR AND SAYS SO. Two grids are described in the record without
their arm counts -- the cap and the floor in the level-2 blend -- and are
entered at 1 each with `exact=False`. Where the true number is unknown the
conservative direction is the one that charges MORE, but inventing arms is
still inventing, so they are entered at the documented minimum and flagged. The
honest statement is "at least 512", not "512".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

#: The command these are recorded under. Distinct from every live research
#: command so `by_command` shows the search as its own line rather than
#: blending it into work that is still being done.
COMMAND = "v3 factor search (reconstructed)"

#: Provenance of every row below.
SOURCE_DOC = "research/V3_SEARCH.md"
SOURCE_COMMIT = "f1b2a9a^"


@dataclass(frozen=True)
class SearchGroup:
    """One comparison, and how many configurations it looked at."""

    section: str
    label: str
    n: int
    #: False when the record describes the grid but not its size, so `n` is the
    #: documented minimum rather than the arm count.
    exact: bool = True
    note: str = ""


#: Factors built per theme, from section 4. Each was screened at four horizons
#: (10 / 21 / 42 / 63) -- the record reports the clear/no-clear verdict per
#: horizon, so each (factor, horizon) is a look.
FACTORS_BUILT: Dict[str, int] = {
    "momentum": 24, "risk": 19, "quality": 15, "liquidity": 9,
    "value": 8, "reversal": 10, "ownership": 6, "seasonality": 2,
}
SCREEN_HORIZONS: Tuple[int, ...] = (10, 21, 42, 63)

#: Themes, and the horizons each was oriented against in section 6.
THEME_HORIZONS: Tuple[int, ...] = (10, 21, 42, 63)
THEMES_ORIENTED: Tuple[str, ...] = (
    "momentum", "quality", "ownership", "risk", "reversal")

#: Section 6, level 1.
COMBINATION_METHODS: Tuple[str, ...] = ("equal", "icw", "ridge", "xgb")

#: Section 7. The record's own table.
FLOOR_GATES: Tuple[str, ...] = (
    "none", "npos3", "trend_npos3", "trend_npos4", "strict")

#: Section 8, the book sweep's cost buckets.
COST_BUCKETS: Tuple[str, ...] = (
    "0-2%", "2-3%", "3-4%", "4-6%", "6-9%", "9%+")

#: Section 8. Three books were specified and their statistics compared.
BOOKS: Tuple[str, ...] = ("holdout(10/20/30/weekly)",
                          "research(12/24/48/10s)",
                          "live(6/6/18/21s)")


def groups() -> List[SearchGroup]:
    """Every comparison in the record, in the order the record makes them."""
    out: List[SearchGroup] = []
    for theme, n in FACTORS_BUILT.items():
        out.append(SearchGroup(
            "4", f"factor screen: {theme} x {len(SCREEN_HORIZONS)} horizons",
            n * len(SCREEN_HORIZONS),
            note="93 factors built, 33 cleared at some horizon"))
    out.append(SearchGroup(
        "4", "both-halves sign stability on the 33 survivors x 2 halves",
        33 * 2, note="cut volume_shock_5 and dist_50dma on a sign flip"))
    out.append(SearchGroup(
        "5", "redundancy threshold |rho| >= 0.80", 1,
        note="the correlations are label-free; the threshold is the choice"))
    out.append(SearchGroup(
        "6", "theme orientation: 5 themes x 4 horizons",
        len(THEMES_ORIENTED) * len(THEME_HORIZONS),
        note="reversal was moved off h=42 because it scored t -3.96 there"))
    out.append(SearchGroup(
        "6", "within-theme combination: 4 methods x 5 themes",
        len(COMBINATION_METHODS) * len(THEMES_ORIENTED),
        note="equal / icw / ridge / xgb, compared on the same folds"))
    out.append(SearchGroup(
        "6", "level-2 weight cap", 1, exact=False,
        note="the record reports 0.40 and the uncapped outcome; the grid "
             "between them is not written down"))
    out.append(SearchGroup(
        "6", "level-2 weight floor", 1, exact=False,
        note="the record reports 0.06 and its drawdown effect; the grid is "
             "not written down"))
    out.append(SearchGroup(
        "6", "coverage cap on / off", 2,
        note="both arms are tabulated in full"))
    out.append(SearchGroup(
        "7", "absolute quality floor gates", len(FLOOR_GATES),
        note="none / npos3 / trend_npos3 / trend_npos4 / strict"))
    out.append(SearchGroup(
        "7", "floor applied to entries only vs the whole population", 2))
    out.append(SearchGroup(
        "8", "book cost sweep", len(COST_BUCKETS)))
    out.append(SearchGroup(
        "8", "book specification", len(BOOKS),
        note="only the holdout book was ever evaluated on a sealed window"))
    out.append(SearchGroup(
        "13", f"dividend yield screen x {len(SCREEN_HORIZONS)} horizons",
        len(SCREEN_HORIZONS),
        note="post-deploy; rejected at every horizon"))
    return out


def labels() -> List[str]:
    """One registry label per configuration looked at.

    Expanded rather than recorded as a count, because the registry is
    content-addressed by (command, label) and a single row reading "372 factor
    screens" would be idempotent with a later row reading the same thing while
    describing different work.
    """
    out: List[str] = []
    for g in groups():
        for i in range(g.n):
            suffix = "" if g.n == 1 else f" [{i + 1}/{g.n}]"
            out.append(f"§{g.section} {g.label}{suffix}")
    return out


def total() -> int:
    return sum(g.n for g in groups())


def is_floor() -> bool:
    """True when any group's arm count was not recorded, so `total` is a lower
    bound. Callers must print "at least" rather than an equals sign."""
    return any(not g.exact for g in groups())


def summary() -> str:
    lines = [f"the v3 factor search, reconstructed from {SOURCE_DOC} "
             f"(restored from {SOURCE_COMMIT})", ""]
    for g in groups():
        mark = "" if g.exact else "   [FLOOR: arm count not recorded]"
        lines.append(f"  §{g.section:<3} {g.n:>4}  {g.label}{mark}")
    lines += ["", f"  {'AT LEAST' if is_floor() else 'TOTAL'} {total():>4} "
                  f"configurations looked at before the model shipped"]
    return "\n".join(lines)
