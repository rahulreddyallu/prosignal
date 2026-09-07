"""Is the second BUY a second OPPORTUNITY, or the same trade twice?

THE FAILURE THIS PREVENTS. Two names can clear every pairwise test the engine
already runs -- different sectors, correlation under 0.70 -- and still be one
position. Both high-momentum, both high-beta, both riding the same theme the
model happens to be pricing this month. Stage 8's existing checks look at names
two at a time and at sector labels; neither can see that.

THREE READS, kept separate because they fail differently.

  1. PRICE. Realised return correlation over the shipped lookback, and then the
     correlation of the RESIDUALS after removing the market. Two names can
     correlate at 0.75 purely because the market moved; what matters for a
     two-name book is whether they still move together once that is taken out.
     Raw correlation alone rejects perfectly good pairs in a trending market
     and accepts bad ones in a flat one.

  2. EVIDENCE. Whether the two candidates are being selected BY THE SAME THING.
     If both are carried by the momentum theme and nothing else, the book has
     one bet on momentum, not two bets on two companies. Measured as the cosine
     similarity of the two names' theme-contribution vectors.

  3. BASKET. Meucci's Effective Number of Bets over the two-name book, using
     the same entropy index `conviction.evidence` applies to factors. ENB near
     2.0 means two genuinely independent positions; ENB near 1.0 means one
     position held twice. This is the number that answers the question directly
     and the other two explain WHY it came out where it did.

The second slot has to EARN its place. The default when independence cannot be
measured is to refuse the second name, not to take it: an unmeasured
correlation is not a low one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["Independence", "assess_pair"]


def _returns(closes: pd.DataFrame, a: str, b: str,
             lookback: int) -> Optional[tuple]:
    if closes is None or closes.empty:
        return None
    for sym in (a, b):
        if sym not in closes.columns:
            return None
    frame = closes[[a, b]].tail(lookback + 1)
    rets = frame.pct_change().dropna()
    if len(rets) < 20:
        return None
    return rets[a].to_numpy(dtype=float), rets[b].to_numpy(dtype=float)


def _market(closes: pd.DataFrame, lookback: int) -> Optional[np.ndarray]:
    """Equal-weight return of the cross-section -- the market leg.

    The MEAN OF THE RETURNS, not the return of the mean price. The second is
    price-weighted and jumps whenever the set of names with a print changes;
    this repository has already been bitten by exactly that substitution once.
    """
    if closes is None or closes.empty or closes.shape[1] < 5:
        return None
    rets = closes.tail(lookback + 1).pct_change().dropna(how="all")
    if len(rets) < 20:
        return None
    return rets.mean(axis=1).to_numpy(dtype=float)


def _corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 20:
        return None
    a, b = x[ok], y[ok]
    if a.std() <= 0 or b.std() <= 0:
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    return r if np.isfinite(r) else None


def _residualise(y: np.ndarray, m: np.ndarray) -> Optional[np.ndarray]:
    """y minus its market beta times the market. OLS, no intercept concerns."""
    ok = np.isfinite(y) & np.isfinite(m)
    if int(ok.sum()) < 20:
        return None
    yy, mm = y[ok], m[ok]
    var = float(((mm - mm.mean()) ** 2).mean())
    if var <= 0:
        return None
    beta = float(((yy - yy.mean()) * (mm - mm.mean())).mean() / var)
    out = np.full_like(y, np.nan, dtype=float)
    out[ok] = yy - beta * mm
    return out


def _theme_vector(score, themes: Sequence[str]) -> np.ndarray:
    v = []
    for k in themes:
        f = score.factors.get(k) if score is not None else None
        c = None if f is None else f.contribution
        v.append(float(c) if c is not None and np.isfinite(c) else 0.0)
    return np.asarray(v, dtype=float)


def _cosine(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return None
    return float(np.clip(a @ b / (na * nb), -1.0, 1.0))


def _basket_enb(rho: float) -> float:
    """ENB of an equal-weighted two-name book with correlation rho.

    For two equally weighted names the diversification distribution has a
    closed form: the principal directions are the sum and difference, carrying
    (1 + rho)/2 and (1 - rho)/2 of the variance. The entropy index follows.
    """
    r = float(np.clip(rho, -0.999, 0.999))
    p = np.array([(1.0 + r) / 2.0, (1.0 - r) / 2.0])
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


@dataclass(frozen=True)
class Independence:
    """Whether two candidates are two bets."""

    a: str
    b: str
    same_sector: Optional[bool] = None
    raw_correlation: Optional[float] = None
    residual_correlation: Optional[float] = None
    evidence_similarity: Optional[float] = None
    #: Effective number of bets in the two-name book, from residual
    #: correlation. 2.0 is fully independent, 1.0 is one position twice.
    basket_enb: Optional[float] = None
    unavailable: Optional[str] = None

    def testable(self) -> bool:
        return self.unavailable is None and self.residual_correlation is not None

    def summary(self) -> str:
        if not self.testable():
            return (f"Independence of {self.b} from {self.a} NOT TESTABLE: "
                    f"{self.unavailable}")
        sector = ("the same sector" if self.same_sector
                  else "different sectors")
        sim = ("n/a" if self.evidence_similarity is None
               else f"{self.evidence_similarity:.2f}")
        return (
            f"{self.b} against {self.a}: {sector}, raw correlation "
            f"{self.raw_correlation:+.2f}, residual correlation after the "
            f"market {self.residual_correlation:+.2f}, evidence overlap {sim}. "
            f"The two-name book spans {self.basket_enb:.2f} of a possible 2.00 "
            f"independent bets."
        )


def assess_pair(a: str, b: str, closes: pd.DataFrame,
                score_a=None, score_b=None,
                sector_a: Optional[str] = None, sector_b: Optional[str] = None,
                lookback: int = 126,
                themes: Optional[Sequence[str]] = None) -> Independence:
    """Measure whether ``b`` is a second opportunity alongside ``a``."""
    same_sector = None
    if sector_a and sector_b and sector_a != "Unknown" and sector_b != "Unknown":
        same_sector = (sector_a == sector_b)

    pair = _returns(closes, a, b, lookback)
    if pair is None:
        return Independence(
            a, b, same_sector,
            unavailable=f"fewer than 20 overlapping sessions for {a} and {b} in "
                        f"the {lookback}-session window")
    ra, rb = pair
    raw = _corr(ra, rb)

    mkt = _market(closes, lookback)
    if mkt is None or len(mkt) != len(ra):
        # NOT TESTABLE rather than falling back to the raw correlation. A raw
        # correlation reported in the residual field would let two names that
        # move together only because the market moved read as one bet, and two
        # that genuinely share a driver read as independent in a flat market.
        # Neither error is acceptable for the gate that admits the second BUY.
        return Independence(
            a, b, same_sector, raw,
            unavailable="the market leg could not be built, so the correlation "
                        "cannot be separated from shared market exposure")

    res_a, res_b = _residualise(ra, mkt), _residualise(rb, mkt)
    if res_a is None or res_b is None:
        return Independence(a, b, same_sector, raw,
                            unavailable="the market beta would not estimate")
    residual = _corr(res_a, res_b)
    if residual is None:
        return Independence(a, b, same_sector, raw,
                            unavailable="the residual correlation would not "
                                        "estimate")

    sim = None
    if themes:
        sim = _cosine(_theme_vector(score_a, themes),
                      _theme_vector(score_b, themes))

    return Independence(a, b, same_sector, raw, residual, sim,
                        _basket_enb(residual))
