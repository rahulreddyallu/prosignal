"""The two-level composite.

LEVEL 1  Within each theme, combine that theme's factors into one sub-score.
LEVEL 2  Blend the theme sub-scores, with weights set by each theme's validated
         contribution and CAPPED so no theme can swamp the rest.

WHY THE CAP IS NOT A COMPROMISE. Momentum has eight factors that clear the
screen; ownership has two and quality one. An unconstrained fit on validation
hands momentum most of the weight, and the resulting "composite" is momentum
with a decorative fringe -- which backtests slightly better and carries the
whole book on one bet. The cap is priced in validation like anything else, and
what it costs there is the premium being paid for not having a single point of
failure.

EVERY SUB-SCORE IS RE-RANKED WITHIN DATE before blending, so themes are on one
scale. Without it a theme whose factors happen to be more dispersed dominates
the sum for a reason that has nothing to do with information.

A NAME IS SCORED ON THE THEMES IT HAS. Weights renormalise over the themes with
a value, so a name with no fundamentals is not pushed toward the middle by a
zero -- it is scored on the four themes it does have, and how many that was is
carried on the row.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import core, themes as TH, guard as G

#: A name needs this many of a theme's factors before it gets a sub-score.
MIN_FACTORS_PER_THEME = 1


def rank_block(df, cols):
    """Sector-neutral within-date ranks for a list of factors, in [-1, 1]."""
    G.assert_no_lookahead(cols)
    R = core.sector_neutral_ranks(df, cols)
    R.columns = cols
    return R


def orient(R, y, cols):
    """Sign of each factor, from the training slice only."""
    sg = {}
    yy = y.to_numpy("float64")
    for c in cols:
        v = R[c].to_numpy("float64")
        m = np.isfinite(v) & np.isfinite(yy)
        if m.sum() < 500:
            sg[c] = 1.0
            continue
        s = np.corrcoef(v[m], yy[m])[0, 1]
        sg[c] = 1.0 if not np.isfinite(s) or s >= 0 else -1.0
    return sg


def theme_subscore(R, cols, signs, dates, method="equal", weights=None,
                   min_factors=MIN_FACTORS_PER_THEME):
    """One sub-score per (row) for a single theme, then re-ranked within date."""
    if not cols:
        return None
    M = R[cols].to_numpy("float64")
    S = np.array([signs[c] for c in cols], dtype="float64")
    W = (np.ones(len(cols)) if weights is None
         else np.array([weights.get(c, 0.0) for c in cols], dtype="float64"))
    if np.abs(W).sum() <= 0:
        W = np.ones(len(cols))
    ok = np.isfinite(M)
    n = ok.sum(axis=1)
    num = np.nansum(np.where(ok, M * S * W, 0.0), axis=1)
    den = np.where(ok, np.abs(W), 0.0).sum(axis=1)
    raw = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    raw = np.where(n >= min_factors, raw, np.nan)
    s = pd.Series(raw, index=R.index)
    # one scale for every theme
    return s.groupby(dates).transform(lambda x: (x.rank(pct=True) - 0.5) * 2.0)


def cap_weights(raw: dict, cap: float, floor: float = 0.0,
                coverage: dict | None = None) -> dict:
    """Normalise to 1, then cap, redistributing the excess to the uncapped.

    Iterated because redistributing can push another theme over the cap.
    """
    w = {k: max(float(v), 0.0) for k, v in raw.items()}
    # A FLOOR AS WELL AS A CAP. The cap stops one theme swamping the rest; it
    # does not stop a theme that merely scored badly on one training window from
    # falling to zero and taking its diversification with it. A theme that
    # cleared the placebo screen and held its sign in both halves has earned a
    # seat, and the floor is what keeps it in the composite when a single fold
    # dislikes it.
    if floor > 0 and w:
        n = len(w)
        floor = min(float(floor), 1.0 / n)
        tot0 = sum(w.values())
        if tot0 > 0:
            w = {k: v / tot0 for k, v in w.items()}
            slack = 1.0 - floor * n
            w = {k: floor + slack * v for k, v in w.items()}
    tot = sum(w.values())
    if tot <= 0:
        n = len(w) or 1
        return {k: 1.0 / n for k in w}
    w = {k: v / tot for k, v in w.items()}
    # A PER-THEME CAP, NOT ONE NUMBER. A theme's weight is also capped at its
    # COVERAGE, and the reason is structural rather than statistical: weights
    # renormalise over the themes a name actually has, so a theme carried at 45%
    # while only 19% of names have it means those two populations are ranked
    # against each other by materially different models. Fitted on the whole
    # training window that is exactly what happened -- `quality`, on 19%
    # coverage and with every one of its observations after 2022, took the cap.
    caps = {}
    for k in w:
        c = 1.0 if cap is None else float(cap)
        if coverage is not None and k in coverage:
            c = min(c, max(float(coverage[k]), 1e-6))
        caps[k] = max(c, 1e-6)
    if all(c >= 1.0 for c in caps.values()):
        return w
    lo = 1.0 / max(len(w), 1)
    caps = {k: max(v, min(lo, 1.0)) if v >= lo else v for k, v in caps.items()}
    if sum(caps.values()) < 1.0:                 # caps cannot all be honoured
        tot = sum(caps.values())
        return {k: v / tot for k, v in caps.items()}
    for _ in range(80):
        over = {k for k, v in w.items() if v > caps[k] + 1e-12}
        if not over:
            break
        excess = sum(w[k] - caps[k] for k in over)
        for k in over:
            w[k] = caps[k]
        free = [k for k in w if k not in over]
        pool = sum(w[k] for k in free)
        if not free or pool <= 0:
            break
        for k in free:
            w[k] += excess * w[k] / pool
    return w


def blend(sub: dict, weights: dict, dates, min_themes: int = 2):
    """Level 2. Weights renormalise over the themes each NAME actually has."""
    names = [t for t in sub if sub[t] is not None]
    if not names:
        return None, None
    M = np.column_stack([sub[t].to_numpy("float64") for t in names])
    W = np.array([weights.get(t, 0.0) for t in names], dtype="float64")
    ok = np.isfinite(M)
    num = np.nansum(np.where(ok, M * W, 0.0), axis=1)
    den = np.where(ok, W, 0.0).sum(axis=1)
    n_th = ok.sum(axis=1)
    raw = np.where((den > 0) & (n_th >= min_themes), num / np.maximum(den, 1e-12), np.nan)
    score = pd.Series(raw, index=sub[names[0]].index)
    return score, pd.Series(n_th, index=score.index)


def positive_themes(sub: dict, floor: float = 0.0):
    """How many of a name's available themes are ABOVE the cross-section median.

    This is the absolute quality floor's input. A rank composite cannot have a
    fixed threshold -- somebody is always top of the list -- so "good enough"
    is defined as broad rather than high: a name has to be on the right side of
    several themes at once, and on a day when nothing is, nothing qualifies.
    """
    names = [t for t in sub if sub[t] is not None]
    M = np.column_stack([sub[t].to_numpy("float64") for t in names])
    pos = (M > floor) & np.isfinite(M)
    return pd.Series(pos.sum(axis=1), index=sub[names[0]].index)
