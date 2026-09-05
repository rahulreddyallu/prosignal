"""ProSignal v4 -- the v3 composite with seven factors removed. THE SHIPPED SCORER.

v4 IS v3 MINUS SEVEN FACTORS. Nothing else moves: the same five themes, the same
frozen theme weights, the same signs, the same two-level blend, the same
sector-neutral ranks, the same coverage renormalisation. It calls
`v3.score_frame` with a different theme table rather than reimplementing the
blend, so there is no second scorer to drift.

WHY SEVEN. The 2026-09-05 factor audit removed each of the twenty-two factors in
turn, rebuilt the composite and re-scored 380 weekly dates with the population
held fixed. An independent split-half selection -- fit the drop set on one half
of the panel, apply it to the other -- nominated seven factors in BOTH halves:

    mom_2_0          own rank IC +0.0001 at t 0.02. Not weak, ABSENT: it is
                     2-month momentum with no skip, so the short-term reversal
                     window Jegadeesh (1990) says to skip sits inside it.
    mom_3_1          correlates +0.61 with prox_52w; adds nothing after it.
    mom_accel        correlates +0.74 with voladj_mom_6_1.
    voladj_mom_6_1   removing it RAISED composite IC at t +2.25.
    ulcer_120        filed under `risk` and correlating +0.69..+0.78 oriented
                     with prox_52w. The 40% momentum cap is applied per theme
                     and cannot see a momentum factor living in another one, so
                     this was momentum exposure carried twice at no cap.
    resid_rev_21     removing it RAISED composite IC at t +2.14; 7.6 round trips
                     a year against the shipped 89.6 bps.
    deliv_chg_5      the fast member of the theme that carries the model, and
                     the only one of the three that adds nothing (dIC +0.0012).
                     15.5 round trips a year, ~13.9%/yr of drag on its own.

WHAT THE EVIDENCE IS, stated as narrowly as it deserves. On 45 purged and
embargoed CPCV folds (`research/v3/experiments/epoch_2026_09.py`), against the
incumbent, on the same folds and the same population:

    composite rank IC        +0.0541 -> +0.0607
    delta                    +0.00658 at Newey-West t +2.37
    folds where it improved  96% of 45
    5th percentile of folds  +0.00024   (even the worst folds improved)
    split-half, first half   +0.00498 (t +1.24)
    split-half, second half  +0.00817 (t +2.15)

and the selection was re-derived from scratch after the research panel was
rebuilt -- a different date range, 16 more dates and a `quality` theme with more
than twice the coverage -- returning EXACTLY the same seven, with no additions
and no removals. That invariance is the strongest thing in the file.

WHAT THIS IS NOT. It is not a sealed-holdout result. v3's two sealed windows were
earned by the twenty-two-factor set and DO NOT TRANSFER to this one; both are
spent and neither can be reused. CPCV measures stability across sub-periods, not
selection out-of-sample, and the seven were chosen on this panel. The forward
test is what will grade this, and it has to be re-registered for the new epoch.

AND IT DOES NOT FIX THE BOOK. Measured at six names on a 21-session cadence, the
change is roughly neutral and inside the noise -- the audit's central finding
stands untouched: the RANKING has an edge and the concentrated book does not
inherit it. Read the shortlist as drawn from an evidenced ranking. The
concentration remains an operator's risk choice.

WHAT WAS CONSIDERED AND REJECTED, so the absence is a decision on the record:

  momentum-only    The best book number in the entire audit: +21.4%/yr gross at
                   t +2.51, positive in all three sub-periods. REJECTED. It was
                   chosen by reading book results across 25 model x exit-band
                   combinations, it contradicts the IC evidence (momentum's
                   marginal dIC is +0.0007 at t +0.08 -- nothing), its
                   sub-period t-statistics decline monotonically 2.88 -> 1.94 ->
                   0.93, and it discards the only out-of-sample evidence this
                   engine has. Chui, Ranganathan, Rohit & Veeraraghavan (2023)
                   find Indian momentum concentrates in the MOST liquid names;
                   this universe has a median ADTV of Rs 23 crore, which is the
                   wrong end of that result to bet everything on.
  equal weights    dIC +0.0051 but t +1.41 and a 5th percentile of -0.0039.
  cost-only prune  Removing only the four factors that cannot fund their own
                   turnover is dIC ~0 (t -0.05, 53% of folds). The case for
                   those removals is execution, not information, and two of the
                   four are already in the seven.
  new factors      Six with published Indian evidence -- betting-against-beta,
                   idiosyncratic vol and skew, max drawdown, slow delivery,
                   delivered-value z -- all point-in-time at 90-99% coverage.
                   NONE adds at NW t >= 2.0, and three predict standalone while
                   adding nothing blended, which is what an axis the shipped set
                   already carries looks like. See `candidates_2026_09.py`.

TO REVERT: set `stage4_core_score.ranking.source` back to `v3_composite`. This
module removes nothing from v3, which is untouched and still selectable.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional, Tuple

import pandas as pd

from .v3 import THEMES as V3_THEMES, Theme, score_frame as _v3_score_frame

__all__ = ["THEMES", "REMOVED", "ALL_FACTORS", "FACTOR_THEME", "MIN_THEMES",
           "MIN_LOOKBACK_SESSIONS", "SPEC_SHA256", "score_frame"]

#: The seven, frozen. Order is alphabetical so the hash below cannot depend on
#: how the list happened to be typed.
REMOVED: Tuple[str, ...] = (
    "deliv_chg_5", "mom_2_0", "mom_3_1", "mom_accel", "resid_rev_21",
    "ulcer_120", "voladj_mom_6_1",
)

#: v3's table with those seven filtered out. Weights, horizons, coverages and
#: signs are COPIED, never recomputed -- refitting them on the panel that chose
#: the prune would be a second fit on the same data.
THEMES: Dict[str, Theme] = {
    name: Theme(weight=th.weight, horizon=th.horizon, coverage=th.coverage,
                factors=tuple((f, s) for f, s in th.factors if f not in REMOVED))
    for name, th in V3_THEMES.items()
    if any(f not in REMOVED for f, _ in th.factors)
}

FACTOR_THEME: Dict[str, str] = {f: t for t, th in THEMES.items() for f in th.names}
ALL_FACTORS: Tuple[str, ...] = tuple(FACTOR_THEME)

#: Unchanged from v3: a name still needs three of five themes to be scored.
MIN_THEMES = 3
#: `prox_52w` survives the prune and still reads 273 sessions.
MIN_LOOKBACK_SESSIONS = 274

#: Hash of the specification -- the theme table as it will be used. Any change to
#: a weight, a sign, a horizon or the factor list changes it, which is what makes
#: "the model that ran" checkable after the fact rather than asserted.
SPEC_SHA256 = hashlib.sha256(
    repr(sorted((t, th.weight, th.horizon, th.coverage, th.factors)
                for t, th in THEMES.items())).encode()
).hexdigest()


def score_frame(raw: pd.DataFrame, sectors: Optional[Dict[str, str]] = None,
                min_themes: int = MIN_THEMES) -> pd.DataFrame:
    """Score one cross-section under the v4 specification.

    Delegates to `v3.score_frame` with this module's theme table. The emitted
    frame therefore carries the same columns the rest of the engine already
    reads -- `<factor>_r`, `<theme>_sub`, `<theme>_contrib`, `n_themes`, `score`,
    `score_rank` -- minus the seven removed factors' rank columns, which is
    exactly the difference and nothing else.
    """
    return _v3_score_frame(raw, sectors, min_themes=min_themes, themes=THEMES)
