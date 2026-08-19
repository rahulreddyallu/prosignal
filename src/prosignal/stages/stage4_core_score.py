"""Stage 4 -- Cross-sectional factor scoring.

Ranks the whole eligible universe rather than testing one stock against fixed
thresholds: "strong momentum" is only meaningful relative to the alternatives
available on the same day.

Order (see indicators/crosssection.py):

    winsorise -> standardise -> sector-neutralise -> weight -> composite -> rank

Factor families are few and chosen to be independent. RSI, MACD and a
moving-average cross would not add three confirmations, since all encode the
same trend information as momentum; the redundancy check at the end of this
stage measures that rather than assuming it.

Quality is dropped when point-in-time fundamentals are absent and the remaining
weights renormalise, which is stated on the card. Computing it from
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
from ..features import compute_features
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

#: A factor must be measurable for at least this share of the eligible universe
#: to be used at all. Below it, median-filling the remainder means most names
#: are ranked by a number that was never computed for them.
_MIN_FACTOR_COVERAGE = 0.60
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
    closes = prices.pivot_table(index=DATE, columns=SYMBOL, values="close", aggfunc="last", observed=True).sort_index()

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

    # Value and quality, from point-in-time fundamentals. Both are gated on
    # filing_date <= as_of inside features.compute_features, which is the whole
    # reason they are trustworthy: the measured NSE disclosure lag is 9-45 days,
    # so keying on period end instead would leak that window.
    fundamentals = store.read_fundamentals()
    if fundamentals is None or fundamentals.empty:
        reason = (
            "no point-in-time fundamentals stored. Run `prosignal data "
            "fundamentals` to ingest them from NSE quarterly filings. Computing "
            "these from current-vintage data would be lookahead, so they are "
            "dropped rather than approximated."
        )
        if bv(cfg.factors.value.enabled):
            dropped["value"] = reason
        if bv(cfg.factors.quality.enabled):
            dropped["quality"] = reason
    else:
        last_close = closes.iloc[-1].to_dict() if not closes.empty else {}
        feats = compute_features(
            fundamentals, last_close, as_of,
            max_age_days=int(iv(cfg.max_fundamental_age_days)),
        )
        if feats.empty:
            note = "fundamentals stored but none were public as of this date"
            if bv(cfg.factors.value.enabled):
                dropped["value"] = note
            if bv(cfg.factors.quality.enabled):
                dropped["quality"] = note
        else:
            feats = feats.set_index(SYMBOL)

            if bv(cfg.factors.value.enabled):
                metric = str(v(cfg.factors.value.metric))
                series = feats[metric].reindex(symbols).astype("float64")
                coverage = float(series.notna().mean())
                if coverage >= _MIN_FACTOR_COVERAGE:
                    raw["value"] = series
                else:
                    dropped["value"] = (
                        f"{metric} available for only {coverage:.0%} of the "
                        f"eligible universe, below the {_MIN_FACTOR_COVERAGE:.0%} "
                        f"floor. A factor scored on a minority of names ranks "
                        f"the rest by median fill, which is not a ranking."
                    )

            if bv(cfg.factors.quality.enabled):
                q, q_note = _quality_from_features(feats, symbols, cfg.factors.quality)
                if q is not None:
                    raw["quality"] = q
                else:
                    dropped["quality"] = q_note

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
    # Value tracks the quality multiplier: both are fundamental, slow-moving and
    # behave as crash stabilisers, which is the opposite of how momentum behaves
    # in a turn. Giving value the momentum multiplier would dampen it exactly
    # when it is most useful.
    mult = {
        "momentum_12_1": regime.momentum_multiplier,
        "quality": regime.quality_multiplier,
        "value": regime.quality_multiplier,
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
    # Each name is scored ONLY on the factors actually measured for it, with its
    # own weights renormalised over what is available.
    #
    # The previous implementation median-filled missing factors before
    # weighting. That makes an absent factor look AVERAGE rather than UNKNOWN,
    # and it put an imputed value score into the composite of the top-ranked
    # name -- 37% of the universe was affected on the value factor. A name
    # should be ranked on its own evidence or not ranked at all.
    frame = pd.DataFrame(standardised)
    weight_vector = pd.Series(effective, dtype="float64")

    available = frame.notna()
    available_weight = available.mul(weight_vector, axis=1).sum(axis=1)
    weighted_sum = frame.fillna(0.0).mul(weight_vector, axis=1).sum(axis=1)

    min_name_cov = fv(cfg.min_name_factor_coverage)
    scoreable = available_weight >= min_name_cov
    composite_raw = (weighted_sum / available_weight.where(available_weight > 0))[scoreable]

    unscoreable = [str(s_) for s_ in frame.index[~scoreable]]
    if unscoreable:
        notes.append(
            f"{len(unscoreable)} name(s) carried less than {min_name_cov:.0%} of "
            f"factor weight and were left unscored rather than imputed: "
            f"{', '.join(unscoreable[:8])}"
            + (" ..." if len(unscoreable) > 8 else "")
        )
    # ---- fitted cross-sectional model -------------------------------------
    # The hand-set factor weights above rank the universe with an excess return
    # over an equal-weight benchmark of +0.14%/month at t = 0.30 -- not
    # distinguishable from zero. A ridge fit over the same factors plus the
    # standard liquidity and risk controls reaches +1.11%/month at t = 3.44 on
    # 8.8 years of purged walk-forward. The factors are unchanged; only the
    # weighting moves from judgement to measurement.
    #
    # The fit uses history ending one full label horizon before the decision
    # date, so nothing in training overlaps today. When there is too little
    # history the model abstains and the hand-weighted composite stands, which
    # is stated on the card rather than substituted quietly.
    model_scores, model, model_unavailable = _cross_sectional_model(
        store, symbols, as_of, cfg
    )
    if model_scores is not None:
        aligned = model_scores.reindex(composite_raw.index).dropna()
        if len(aligned) >= max(int(0.6 * len(composite_raw)), 20):
            composite_raw = aligned
            notes.append(
                f"Ranking from the fitted cross-sectional model ({model.summary()}). "
                f"Measured against the hand-weighted composite over 8.8 years of "
                f"purged walk-forward: IC +0.052 (t 3.64) versus +0.025 (t 1.28)."
            )
        else:
            notes.append(
                f"Cross-sectional model covered only {len(aligned)} of "
                f"{len(composite_raw)} scored names; hand-weighted composite retained."
            )
    else:
        notes.append(f"Cross-sectional model unavailable: {model_unavailable}")

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
    "value": "OOO high (strongest India-specific evidence)",
    "momentum_12_1": "OOO high",
    "sector_relative_strength": "OO medium",
    "quality": "OO medium",
}
_CITE = {
    "value": "Fama & French (1993, 2015); FF5 replications on CNX 500 / NSE 500",
    "momentum_12_1": "Jegadeesh & Titman (1993); Asness, Moskowitz & Pedersen (2013)",
    "sector_relative_strength": "Moskowitz & Grinblatt (1999) industry momentum",
    "quality": "Asness, Frazzini & Pedersen (2019); FF5 profitability factor, India-replicated",
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


def _quality_from_features(feats, symbols, cfg):
    """Composite quality from whichever components cleared the coverage floor.

    Components are z-scored INDIVIDUALLY before weighting, so a component
    measured in multiples (interest coverage, which ranges 1 to 100) cannot
    swamp one measured as a fraction (net margin, 0 to 0.3). Weighting raw
    values would make the composite almost entirely interest coverage.
    """
    comps = cfg.components
    used, total = [], 0.0
    acc = pd.Series(0.0, index=symbols, dtype="float64")

    for name, comp in comps.items():
        if not bv(comp.enabled) or name not in feats.columns:
            continue
        series = feats[name].reindex(symbols).astype("float64")
        if float(series.notna().mean()) < _MIN_FACTOR_COVERAGE:
            continue
        z = standardise(winsorise(series, 2.0, 98.0), method="zscore")
        if not bv(comp.higher_is_better):
            z = -z
        weight = fv(comp.weight)
        acc = acc.add(z.fillna(0.0) * weight, fill_value=0.0)
        total += weight
        used.append(name)

    need = iv(cfg.min_components_required)
    if len(used) < need:
        return None, (
            f"only {len(used)} quality component(s) cleared the coverage floor "
            f"({used or 'none'}), need {need}. Banks and financials file a "
            f"different Ind-AS schema, so their line items are absent."
        )
    return acc / total, ""


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
        "value": v(cfg.factors.value.weight_band),
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


def _cross_sectional_model(store, symbols, as_of, cfg):
    """Fit the ridge ranker on history strictly before ``as_of``.

    Failure is reported, never swallowed: a model that could not be fitted must
    leave the hand-weighted composite visibly in charge.
    """
    from ..features import crossmodel as cm

    cache = store.curated / "crosssec_model.json"
    try:
        sessions = store.price_sessions()
        cached = cm.load_cached(cache, as_of)

        # Cheap path: a recent fit only needs today's features, which is one
        # date of history instead of a thousand. The large read is what pushed
        # peak RSS past the instance limit, so it happens on refit days only.
        need = (cm.MIN_LOOKBACK + 10) if cached else (cm.MAX_TRAIN_SESSIONS + cm.HORIZON + 5)
        start = sessions[-need] if len(sessions) > need else sessions[0]
        px = store.read_prices(
            symbols=list(symbols), start=start, end=as_of,
            columns=[DATE, SYMBOL, "close", "turnover"],
        )
        if px.empty:
            return None, None, "no price rows"
        px[DATE] = pd.to_datetime(px[DATE]).dt.normalize()
        close = px.pivot_table(index=DATE, columns=SYMBOL, values="close", aggfunc="last").sort_index()
        turnover = px.pivot_table(index=DATE, columns=SYMBOL, values="turnover", aggfunc="last").sort_index()
        del px

        # Value and quality: the only inputs not derived from price and volume.
        # Read once and passed to both paths so the fit and the live scoring see
        # the same filings.
        fundamentals = store.read_fundamentals()
        max_age = int(iv(cfg.max_fundamental_age_days))

        if cached is not None:
            feats = cm.today_features(close, turnover, as_of,
                                      fundamentals=fundamentals,
                                      max_fundamental_age_days=max_age)
            if feats is None:
                return None, None, "no symbol had a complete feature set today"
            return cm.score_with(cached, feats), cached, None

        scores, model, reason = cm.fit_predict(
            close, turnover, as_of,
            fundamentals=fundamentals, max_fundamental_age_days=max_age,
        )
        if model is not None:
            cm.save_cache(cache, model, as_of)
        return scores, model, reason
    except Exception as exc:  # a modelling failure must not take the run down
        log.warning("cross-sectional model failed", extra={"error": str(exc)})
        return None, None, f"{type(exc).__name__}: {exc}"
