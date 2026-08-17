"""Stage 4 -- Cross-sectional factor scoring.

Ranks the WHOLE eligible universe rather than testing one stock against
thresholds. That is the point of a cross-sectional model: "strong momentum" is
only meaningful relative to the other things you could have bought today.

Pipeline order, which is not negotiable (see indicators/crosssection.py):

    winsorise -> standardise -> sector-neutralise -> weight -> composite -> rank

Factor families here are deliberately few and deliberately independent. Adding
RSI, MACD and a moving-average cross would not add three confirmations -- they
all encode the same trend information as momentum, and the redundancy check at
the end of this stage measures exactly that rather than assuming it.

Quality is dropped when point-in-time fundamentals are absent, and the remaining
weights renormalise. That is stated on every card. Computing it from
current-vintage fundamentals would be lookahead.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ._cfg import bv, fv, iv, v
from ..core.calendar import TradingCalendar
from ..core.contracts import (
    CoreScoreReport,
    EligibilityReport,
    FactorScore,
    RedundancyReport,
    RegimeState,
    StockScore,
)
from ..core.logging import get_logger
from ..data.store import DataStore
from ..data.types import DATE, SYMBOL
from ..indicators import (
    momentum_skip,
    rank_to_unit_interval,
    sector_neutralise,
    spearman_pairs,
    standardise,
    trailing_return,
    winsorise,
)

__all__ = ["run", "STAGE_NAME"]

STAGE_NAME = "stage4_core_score"
log = get_logger(__name__)


def run(
    eligibility: EligibilityReport,
    store: DataStore,
    calendar: TradingCalendar,
    regime: RegimeState,
    config,
    as_of: Optional[dt.date] = None,
) -> CoreScoreReport:
    p = config.params
    cfg = p.stage4_core_score
    as_of = as_of or eligibility.as_of_date
    symbols = list(eligibility.eligible_universe)
    notes: List[str] = []

    if not symbols:
        return CoreScoreReport(
            as_of_date=as_of, weighting_mode=str(v(cfg.weighting_mode)),
            standardisation=str(v(cfg.standardisation)), universe_size=0,
            notes=["no eligible symbols to score"],
        )

    need = iv(cfg.factors.momentum_12_1.lookback_sessions) + int(
        cfg.factors.momentum_12_1.skip_sessions.value
    ) + 10
    window = calendar.trailing_window(as_of, need)
    start = window[0] if window else calendar.first

    prices = store.read_prices(symbols=symbols, start=start, end=as_of)
    prices = prices.copy()
    prices[DATE] = pd.to_datetime(prices[DATE]).dt.normalize()
    closes = prices.pivot_table(index=DATE, columns=SYMBOL, values="close", aggfunc="last").sort_index()

    sectors = dict(eligibility.sector_map)

    # ---- raw factor values -------------------------------------------------
    raw: Dict[str, pd.Series] = {}
    dropped: Dict[str, str] = {}

    if bv(cfg.factors.momentum_12_1.enabled):
        raw["momentum_12_1"] = _momentum(closes, cfg.factors.momentum_12_1)

    if bv(cfg.factors.sector_relative_strength.enabled):
        rs = _relative_strength(closes, sectors, store, as_of, p, cfg.factors.sector_relative_strength)
        if rs is not None:
            raw["sector_relative_strength"] = rs
        else:
            dropped["sector_relative_strength"] = "benchmark index unavailable"

    # Quality: only with point-in-time fundamentals carrying filing dates.
    fundamentals = store.read_fundamentals()
    if bv(cfg.factors.quality.enabled):
        if fundamentals is None or fundamentals.empty:
            dropped["quality"] = (
                "no point-in-time fundamentals with filing dates. Computing "
                "quality from current-vintage data would be lookahead, so the "
                "factor is dropped and the remaining weights renormalise."
            )
        else:
            q = _quality(fundamentals, symbols, as_of, cfg.factors.quality)
            if q is None:
                dropped["quality"] = "too few quality components available"
            else:
                raw["quality"] = q

    if not raw:
        return CoreScoreReport(
            as_of_date=as_of, weighting_mode=str(v(cfg.weighting_mode)),
            standardisation=str(v(cfg.standardisation)), universe_size=len(symbols),
            dropped_factors=dropped, notes=["every factor was dropped; no score computable"],
        )

    for name, reason in dropped.items():
        notes.append(f"{name} dropped: {reason}")

    # ---- winsorise -> standardise -> neutralise ----------------------------
    method = str(v(cfg.standardisation))
    wins = fv(cfg.winsorize_pct)
    standardised: Dict[str, pd.Series] = {}
    for name, series in raw.items():
        s = winsorise(series, wins, 100.0 - wins)
        s = standardise(s, method=method)
        if bv(cfg.sector_neutral):
            s = sector_neutralise(s, sectors)
        standardised[name] = s

    # ---- weights, renormalised over surviving factors ----------------------
    weights = _weights(cfg, list(standardised))
    # Regime multipliers scale each family's contribution.
    mult = {
        "momentum_12_1": regime.momentum_multiplier,
        "quality": regime.quality_multiplier,
        "sector_relative_strength": regime.sector_rs_multiplier,
    }
    effective = {n: weights[n] * mult.get(n, 1.0) for n in standardised}
    total = sum(effective.values())
    if total > 0:
        effective = {n: w / total for n, w in effective.items()}
    notes.append(
        f"Regime '{regime.regime_bucket}' multipliers applied "
        f"(momentum x{regime.momentum_multiplier:.2f}, "
        f"sector-RS x{regime.sector_rs_multiplier:.2f}), then weights renormalised."
    )

    # ---- composite ---------------------------------------------------------
    frame = pd.DataFrame(standardised)
    composite_raw = sum(frame[n].fillna(frame[n].median()) * w for n, w in effective.items())
    composite_unit = rank_to_unit_interval(composite_raw)
    percentile = composite_unit * 100.0
    order = composite_raw.sort_values(ascending=False)

    scores: List[StockScore] = []
    for rank, sym in enumerate(order.index, start=1):
        factors = {}
        for name in standardised:
            factors[name] = FactorScore(
                name=name,
                raw_value=_f(raw[name].get(sym)),
                standardised=_f(standardised[name].get(sym)),
                weight=round(effective[name], 4),
                available=pd.notna(raw[name].get(sym)),
                evidence_tier=_TIER.get(name),
                citation=_CITE.get(name),
            )
        scores.append(
            StockScore(
                ticker=str(sym),
                sector=sectors.get(sym),
                factors=factors,
                composite_raw=float(composite_raw.get(sym, 0.0)),
                composite_score=float(composite_unit.get(sym, 0.0)),
                percentile=float(percentile.get(sym, 0.0)),
                rank=rank,
            )
        )

    redundancy = _redundancy(frame, cfg)
    if redundancy.breaches:
        notes.append(
            f"Redundancy: {len(redundancy.breaches)} factor pair(s) exceed "
            f"|rho|={redundancy.cutoff}. They are not independent evidence."
        )

    log.info("stage 4 complete", extra={"scored": len(scores), "factors": list(effective)})
    return CoreScoreReport(
        as_of_date=as_of,
        weighting_mode=str(v(cfg.weighting_mode)),
        standardisation=method,
        effective_weights={k: round(v, 4) for k, v in effective.items()},
        dropped_factors=dropped,
        ranked_scores=scores,
        redundancy=redundancy,
        universe_size=len(symbols),
        notes=notes,
    )


# =============================================================================
_TIER = {
    "momentum_12_1": "OOO high",
    "sector_relative_strength": "OO medium",
    "quality": "OO medium",
}
_CITE = {
    "momentum_12_1": "Jegadeesh & Titman (1993); Asness, Moskowitz & Pedersen (2013)",
    "sector_relative_strength": "Moskowitz & Grinblatt (1999) industry momentum",
    "quality": "Asness, Frazzini & Pedersen (2019) quality-minus-junk",
}


def _f(v) -> Optional[float]:
    return None if v is None or pd.isna(v) else float(v)


def _momentum(closes: pd.DataFrame, cfg) -> pd.Series:
    lb = iv(cfg.lookback_sessions)
    sk = iv(cfg.skip_sessions)
    return pd.Series(
        {s: momentum_skip(closes[s].dropna(), lb, sk) for s in closes.columns},
        dtype="float64",
    )


def _relative_strength(closes, sectors, store, as_of, params, cfg) -> Optional[pd.Series]:
    """Blend of market-relative and sector-relative return over several horizons."""
    bench = store.index_series(str(v(params.stage2_regime.benchmark_index)), "close", end=as_of)
    if bench.empty:
        return None
    horizons = [int(h) for h in v(cfg.horizons_sessions)]
    mkt_w = fv(cfg.market_relative_weight)

    # sector index returns, when a matching NSE sector index exists
    sector_returns: Dict[str, Dict[int, float]] = {}
    for sector in {v for v in sectors.values() if v}:
        series = store.index_series(f"Nifty {sector}", "close", end=as_of)
        if series.empty:
            continue
        sector_returns[sector] = {
            h: trailing_return(series, h) for h in horizons
        }

    out: Dict[str, float] = {}
    for sym in closes.columns:
        px = closes[sym].dropna()
        parts = []
        for h in horizons:
            r = trailing_return(px, h)
            b = trailing_return(bench, h)
            if r is None or b is None:
                continue
            market_rel = r - b
            sec = sectors.get(sym)
            sec_ret = sector_returns.get(sec, {}).get(h) if sec else None
            if sec_ret is None:
                parts.append(market_rel)
            else:
                parts.append(mkt_w * market_rel + (1 - mkt_w) * (r - sec_ret))
        out[sym] = float(np.mean(parts)) if parts else np.nan
    return pd.Series(out, dtype="float64")


def _quality(fundamentals: pd.DataFrame, symbols, as_of, cfg) -> Optional[pd.Series]:
    """Composite quality from whatever point-in-time components are present.

    Every row must carry a filing date at or before ``as_of`` -- a fundamental
    figure is not usable before it was public, and the median disclosure lag on
    NSE is about 21 days (see DATA_SOURCES.md).
    """
    f = fundamentals.copy()
    if "filing_date" not in f.columns:
        return None
    f["filing_date"] = pd.to_datetime(f["filing_date"], errors="coerce").dt.date
    f = f[f["filing_date"].notna() & (f["filing_date"] <= as_of)]
    if f.empty:
        return None
    f = f.sort_values("filing_date").groupby(SYMBOL).tail(1).set_index(SYMBOL)

    comps = cfg.components
    spec = [
        ("return_on_equity", comps.return_on_equity),
        ("gross_profitability", comps.gross_profitability),
        ("accrual_intensity", comps.accrual_intensity),
        ("debt_to_equity", comps.debt_to_equity),
        ("interest_coverage", comps.interest_coverage),
    ]
    available = [(n, c) for n, c in spec if bv(c.enabled) and n in f.columns]
    if len(available) < iv(cfg.min_components_required):
        return None

    total = 0.0
    acc = pd.Series(0.0, index=f.index, dtype="float64")
    for name, c in available:
        vals = pd.to_numeric(f[name], errors="coerce")
        z = standardise(winsorise(vals, 2.0, 98.0), method="zscore")
        if not bv(c.higher_is_better):
            z = -z
        w = fv(c.weight)
        acc = acc.add(z * w, fill_value=0.0)
        total += w
    if total <= 0:
        return None
    return (acc / total).reindex(symbols)


def _weights(cfg, surviving: List[str]) -> Dict[str, float]:
    """Weights renormalised over factors that actually survived.

    ``equal_weight`` is the shipped default because the research program's own
    prediction is that rank-IC weighting's in-sample gains are usually
    overfitting.
    """
    mode = str(v(cfg.weighting_mode))
    if mode == "equal_weight":
        w = 1.0 / len(surviving)
        return {n: w for n in surviving}

    bands = {
        "momentum_12_1": v(cfg.factors.momentum_12_1.weight_band),
        "quality": v(cfg.factors.quality.weight_band),
        "sector_relative_strength": v(cfg.factors.sector_relative_strength.weight_band),
    }
    mids = {n: (float(bands[n][0]) + float(bands[n][1])) / 2.0 for n in surviving if n in bands}
    total = sum(mids.values())
    return {n: v / total for n, v in mids.items()} if total > 0 else {
        n: 1.0 / len(surviving) for n in surviving
    }


def _redundancy(frame: pd.DataFrame, cfg) -> RedundancyReport:
    """Measure factor overlap rather than assuming it."""
    cutoff = fv(cfg.redundancy.max_abs_spearman)
    pairs = spearman_pairs(frame) if frame.shape[1] >= 2 else {}
    breaches = [
        (k.split("|")[0], k.split("|")[1], round(v, 4))
        for k, v in pairs.items()
        if abs(v) > cutoff
    ]
    return RedundancyReport(
        pairwise_spearman={k: round(v, 4) for k, v in pairs.items()},
        breaches=breaches,
        cutoff=cutoff,
        action_taken=str(v(cfg.redundancy.on_breach)),
        notes=(
            []
            if pairs
            else ["fewer than two factors survived; correlation not measurable"]
        ),
    )
