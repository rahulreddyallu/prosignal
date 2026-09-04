"""The v9R CORE scorer -- nine factors, equal risk contribution, unneutralised.

This is the model the sealed 2012-2017 window measured. It is NOT the shipped
scorer: `stage4_core_score.ranking.source` still defaults to `v3_composite`, and
switching is a deliberate decision recorded in `config/parameters.yaml`, not a
side effect of this file existing.

WHAT IT IS. Nine of the twenty-two factors in `features/v3_factors.py`, chosen by a
long-side gate run on 2018-2026 only, blended at weights that equalise each
factor's contribution to composite variance.

WHY IT DIFFERS FROM v3, point by point, each on a measurement:

1. NINE FACTORS, NOT TWENTY-TWO. The other thirteen either fail a long-only
   quintile gate or fail breakeven turnover. The whole `reversal` theme -- 10.98%
   of v3's nominal weight -- is uninvestable: `rev_1w` turns over 20.2x a year
   against a breakeven of 2.2x, and `price_vs_vwap_20` and `max5_21` are negative
   before costs.

2. EQUAL RISK CONTRIBUTION, NOT EQUAL NOMINAL WEIGHT. Seven of the nine are
   momentum and correlate 0.50-0.75, so equal nominal weight lets them reinforce
   one another: measured, the largest factor's share of composite variance drifts
   to 0.078 above target. ERC brings that to 0.008.

3. NO COVERAGE RENORMALISATION. v3 renormalises theme weights over the themes a
   name actually has, so a thin theme donates its weight to the thick ones.
   Measured on 2026-08-28: momentum's 40% nominal became 47.0% effective while
   quality's 19.0% collapsed to 4.8%. Here a missing factor scores the
   cross-sectional mean -- which for a centred rank is zero -- and contributes
   nothing rather than reweighting its neighbours.

4. UNNEUTRALISED. v3 ranks within sector. The sector map covers 754 symbols, so
   46.7% of a live cross-section lands in one residual bucket that is not a
   sector, and ranking within it is not neutralisation. Measured, unneutralised
   composite rank IC is higher at every horizon (+0.0674 against +0.0547 at h=21).

FIDELITY. `tests/test_v9r_score.py` transcribes the research rank convention and
checks this module reproduces it to machine precision. The research tree itself was
removed in the 2026-09-03 cleanup, so that transcription IS the reference now --
which is why the test carries the algorithm rather than importing it.
The rank convention below is the researched one -- a 0-based ordinal rank scaled to
[-1, 1] -- and NOT pandas' `rank(pct=True)`, which averages ties and normalises
differently. The two agree closely; "closely" is not the standard a scorer that
earned a sealed number is held to.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["FACTORS", "WEIGHTS", "COVERAGE_FLOOR", "BOOK",
           "rank_pct", "score_frame", "SPEC_SHA256"]

#: The nine CORE factors, frozen. Selected by LS-GATE on 2018-2026 and evaluated
#: once on 2012-2017. Order is fixed so the weight vector cannot silently re-pair.
FACTORS: Tuple[str, ...] = (
    "voladj_mom_12_1",
    "voladj_mom_6_1",
    "mom_accel",
    "ret_kurt_126",
    "prox_52w_now",
    "intraday_mom_126",
    "prox_52w",
    "mom_2_0",
    "mom_consist_126",
)

#: Equal-risk-contribution weights, capped at each factor's coverage and
#: renormalised. Fitted on the training window and frozen before the sealed window
#: was opened; they are part of the pre-registered specification.
WEIGHTS: Dict[str, float] = {
    "voladj_mom_12_1": 0.0964,
    "voladj_mom_6_1": 0.0848,
    "mom_accel": 0.1126,
    "ret_kurt_126": 0.1816,
    "prox_52w_now": 0.0934,
    "intraday_mom_126": 0.0941,
    "prox_52w": 0.0885,
    "mom_2_0": 0.1380,
    "mom_consist_126": 0.1108,
}

#: A name below this much coverage-weighted factor presence is not scored. It is
#: NOT a renormalisation: the factors that are present still carry their own
#: weights, and the missing ones contribute zero.
COVERAGE_FLOOR: float = 0.70

#: The book the sealed window measured. Mirrored here for reading; the live book
#: is `capital.max_open_positions` and `stage6_entry.admission` in the config, and
#: this copy has no authority over them.
BOOK = {"slots": 20, "rebalance_sessions": 42, "weighting": "equal",
        "fill": "next session close"}

#: sha256 of the pre-registration that fixed this specification before the sealed
#: window was computed. The document itself was removed with the research tree in
#: the 2026-09-03 cleanup; `docs/MODEL_v9R.md` carries what it said. The hash stays
#: because it is what makes the sealed result a pre-registered one rather than a
#: number chosen afterwards.
SPEC_SHA256 = "d2dfba4f9a1e4ee1ed24d5cb6429307fdcf53bddb531979b25c2dbb78100f877"


def rank_pct(values: pd.Series) -> pd.Series:
    """Cross-sectional ordinal rank scaled to [-1, 1], NaN preserved.

    Ties are broken by first occurrence rather than averaged. That is the
    convention the sealed measurement used, and matching it exactly matters more
    than which convention is nicer: a scorer that earned an out-of-sample number
    has to compute what it was measured computing.
    """
    v = pd.to_numeric(values, errors="coerce")
    ok = v.notna().to_numpy()
    out = np.full(len(v), np.nan, dtype="float64")
    n = int(ok.sum())
    if n == 0:
        return pd.Series(out, index=values.index, dtype="float64")
    if n == 1:
        out[ok] = 0.0
        return pd.Series(out, index=values.index, dtype="float64")
    a = v.to_numpy("float64")
    order = np.argsort(np.where(ok, a, np.inf), kind="stable")
    rk = np.empty(len(a), dtype="float64")
    rk[order] = np.arange(len(a), dtype="float64")
    out[ok] = (rk[ok] / (n - 1.0)) * 2.0 - 1.0
    return pd.Series(out, index=values.index, dtype="float64")


def score_frame(raw: pd.DataFrame,
                factors: Optional[Tuple[str, ...]] = None,
                weights: Optional[Dict[str, float]] = None,
                coverage_floor: float = COVERAGE_FLOOR) -> pd.DataFrame:
    """Score one cross-section. Rows are symbols; columns are the raw factors.

    Returns the per-factor ranks, the coverage-weighted presence, and `score`.
    A name below `coverage_floor` gets a NaN score rather than a score built from
    whatever happened to be present.
    """
    factors = tuple(factors or FACTORS)
    weights = dict(weights or WEIGHTS)
    if raw is None or raw.empty:
        return pd.DataFrame()

    missing = [f for f in factors if f not in raw.columns]
    if missing:
        raise KeyError(
            f"the v9R scorer names {missing}, which the factor frame does not "
            f"provide. Scoring on a subset would be a different model from the "
            f"one the sealed window measured."
        )

    w = np.array([float(weights[f]) for f in factors], dtype="float64")
    if not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("v9R weights must be finite and sum above zero")
    w = w / w.sum()

    out = pd.DataFrame(index=raw.index)
    R = np.empty((len(raw), len(factors)), dtype="float64")
    for j, f in enumerate(factors):
        r = rank_pct(raw[f])
        out[f + "_r"] = r
        R[:, j] = r.to_numpy()

    present = np.isfinite(R)
    coverage = (present * w).sum(axis=1)
    # MISSING SCORES THE CROSS-SECTIONAL MEAN, which for a centred rank is zero.
    # It does not redistribute its weight, which is the defect this replaces.
    contrib = np.where(present, R, 0.0) * w
    score = contrib.sum(axis=1)

    out["coverage"] = coverage
    out["n_factors"] = present.sum(axis=1)
    out["score"] = np.where(coverage >= float(coverage_floor), score, np.nan)
    out["score_rank"] = out["score"].rank(ascending=False, method="first")
    return out


def attribution(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """FACTOR / RANK / WEIGHT / CONTRIBUTION for one name, summing to its score."""
    scored = score_frame(raw)
    if scored.empty or symbol not in scored.index:
        return pd.DataFrame(columns=["FACTOR", "RANK", "WEIGHT", "CONTRIB"])
    w = {f: WEIGHTS[f] / sum(WEIGHTS.values()) for f in FACTORS}
    rows: List[dict] = []
    for f in FACTORS:
        r = scored.loc[symbol, f + "_r"]
        rows.append({"FACTOR": f, "RANK": r, "WEIGHT": w[f],
                     "CONTRIB": (0.0 if not np.isfinite(r) else r) * w[f]})
    frame = pd.DataFrame(rows).sort_values("CONTRIB", ascending=False)
    return frame.reset_index(drop=True)
