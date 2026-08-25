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
from ..core.errors import PipelineError
from ..core.logging import get_logger
from ..data.store import DataStore
from ..data.types import DATE, SYMBOL
from ..features import compute_features
from ..features.crosssec import liquidity_mask
from ..features.refit_gate import review_refit
from ..features.crossmodel import (
    contributions as cm_contributions,
    standardised_features as cm_standardised,
)
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
                if coverage >= _min_coverage(cfg):
                    raw["value"] = series
                else:
                    dropped["value"] = (
                        f"{metric} available for only {coverage:.0%} of the "
                        f"eligible universe, below the {_min_coverage(cfg):.0%} "
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
    (model_scores, model, model_unavailable, model_features,
     refit_verdict) = _cross_sectional_model(store, symbols, as_of, cfg,
                                            p.universe)
    if refit_verdict is not None and not refit_verdict.accepted:
        notes.append(
            f"Refit held back: {refit_verdict.summary()}. The previous "
            f"coefficients remain live and this needs manual review."
        )
    if model_unavailable and _is_model_failure(model_unavailable) and not _model_optional(cfg):
        # The hand-weighted composite this would fall back to was measured at
        # -0.047%/month excess over an equal-weight benchmark, t = -0.11. Quietly
        # substituting it produces a watchlist that looks exactly like a healthy
        # one while being scored by something known not to work.
        raise ModelUnavailable(
            f"the cross-sectional model could not score this run "
            f"({model_unavailable}). Falling back to the hand-weighted composite "
            f"would issue signals from a scorer measured at t = -0.11 against an "
            f"equal-weight benchmark. Set stage4_core_score.allow_composite_fallback "
            f"to true to accept that explicitly."
        )

    model_contrib = None
    model_z = None
    if model_scores is not None:
        aligned = model_scores.reindex(composite_raw.index).dropna()
        if len(aligned) >= max(int(0.6 * len(composite_raw)), 20):
            composite_raw = aligned
            # The card must explain the number it prints. Once the model ranks,
            # the hand-weighted composite's factors no longer describe the
            # calculation, so the evidence comes from the fitted coefficients.
            if model is not None and model_features is not None:
                try:
                    model_contrib = cm_contributions(model, model_features)
                    model_z = cm_standardised(model, model_features)
                except Exception as exc:
                    log.warning("model attribution unavailable",
                                extra={"error": str(exc)})
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
        if model_contrib is not None and sym in model_contrib.index:
            # Attribution from the fit: contribution = coefficient x z-score,
            # and the terms sum back to the score. Ordered by absolute size, so
            # the card leads with what actually moved this name.
            row = model_contrib.loc[sym]
            zrow = model_z.loc[sym] if model_z is not None else None
            for name in row.abs().sort_values(ascending=False).index:
                contribution = _f(row.get(name))
                factors[name] = FactorScore(
                    name=name,
                    raw_value=contribution,
                    standardised=_f(zrow.get(name)) if zrow is not None else None,
                    weight=round(float(model.coef.get(name + "_r", 0.0)), 5),
                    available=pd.notna(row.get(name)),
                    evidence_tier="model",
                    citation=_MODEL_CITE.get(name),
                )
        else:
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
        if float(series.notna().mean()) < _min_coverage(cfg):
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


#: Sources for the fitted factors, so a card citing them can be checked.
_MODEL_CITE = {
    "mom_12_1": "Jegadeesh & Titman (1993)",
    "mom_6_1": "Jegadeesh & Titman (1993)",
    "mom_3_1": "Jegadeesh & Titman (1993)",
    "reversal_1m": "Jegadeesh (1990)",
    "vol_60": "Ang, Hodrick, Xing & Zhang (2006)",
    "downside_vol": "Ang, Chen & Xing (2006)",
    "beta_120": "Frazzini & Pedersen (2014)",
    "idio_vol": "Ang, Hodrick, Xing & Zhang (2006)",
    "amihud": "Amihud (2002)",
    "turnover_ratio": "Datar, Naik & Radcliffe (1998)",
    "rel_strength": "Moskowitz & Grinblatt (1999)",
    "dist_200dma": "Moskowitz & Grinblatt (1999)",
    "trend_r2": "trend quality; no single source",
    "max_dd_120": "tail risk; no single source",
    "prox_52w": "George & Hwang (2004)",
    "max5_21": "Bali, Cakici & Whitelaw (2011)",
    "resid_mom": "Blitz, Huij & Martens (2011)",
    "deliv_pct": "NSE delivery data; India-specific, no standard reference",
    "deliv_trend": "NSE delivery data; India-specific, no standard reference",
    "earnings_yield": "Basu (1977)",
    "book_to_price": "Fama & French (1992)",
    "ebitda_to_ev": "capital-structure-neutral value; no single source",
    "fcf_yield": "cash-based value; no single source",
    "sales_to_price": "Barbee, Mukherji & Raines (1996)",
    "net_margin": "Novy-Marx (2013)",
    "interest_coverage": "Altman (1968)",
    "earnings_growth": "Novy-Marx (2013)",
    "earnings_stability": "Sloan (1996)",
    "roe": "Novy-Marx (2013)",
    "roce": "Novy-Marx (2013)",
    "gross_margin": "Novy-Marx (2013)",
    "ebit_margin": "Novy-Marx (2013)",
    "accruals": "Sloan (1996)",
    "fcf_conversion": "Sloan (1996)",
    "revenue_growth": "Lakonishok, Shleifer & Vishny (1994)",
    "ebitda_growth": "Lakonishok, Shleifer & Vishny (1994)",
    "margin_expansion": "Novy-Marx (2013)",
    "debt_to_equity": "Altman (1968)",
    "net_debt_to_ebitda": "Altman (1968)",
    "earnings_acceleration": "Chan, Jegadeesh & Lakonishok (1996)",
}


def _cross_sectional_model(store, symbols, as_of, cfg, universe):
    """Fit the ridge ranker on history strictly before ``as_of``.

    Failure is reported, never swallowed: a model that could not be fitted must
    leave the hand-weighted composite visibly in charge.
    """
    from ..features import crossmodel as cm

    cache = store.curated / "crosssec_model.json"
    try:
        sessions = store.price_sessions()
        refit_every = iv(cfg.model_refit_every_sessions)
        cached = cm.load_cached(cache, as_of, refit_every)

        # Cheap path: a recent fit only needs today's features, which is one
        # date of history instead of a thousand. The large read is what pushed
        # peak RSS past the instance limit, so it happens on refit days only.
        need = ((cm.MIN_LOOKBACK + 10) if cached
                else int(iv(cfg.model_max_train_sessions)) + int(iv(cfg.model_horizon_sessions)) + 5)
        start = sessions[-need] if len(sessions) > need else sessions[0]
        # On a REFIT the training panel must span every name the universe screen
        # would have admitted on each past date -- not the names it admits
        # today. Restricting the read to today's universe is what made the panel
        # a projection of today's survivors backwards: measured against the
        # screen resolved per date, 27% of the names eligible on 2024-08-12 are
        # absent from today's set, and they were excluded for what happened
        # afterwards.
        #
        # The cheap cached path scores today only, so it keeps the narrow read
        # that the instance's memory budget was designed around. The wide read
        # is 3.8s and 31 MB as float32, and it happens on refit days.
        refitting = cached is None
        px = store.read_prices(
            symbols=None if refitting else list(symbols), start=start, end=as_of,
            columns=[DATE, SYMBOL, "close", "turnover"],
        )
        if px.empty:
            return None, None, "no price rows", None, None
        px[DATE] = pd.to_datetime(px[DATE]).dt.normalize()
        close = px.pivot_table(index=DATE, columns=SYMBOL, values="close",
                               aggfunc="last", observed=True).sort_index()
        turnover = px.pivot_table(index=DATE, columns=SYMBOL, values="turnover",
                                  aggfunc="last", observed=True).sort_index()
        del px

        # Value and quality: the only inputs not derived from price and volume.
        # Read once and passed to both paths so the fit and the live scoring see
        # the same filings.
        # Statements rather than the NSE filings table: that feed stopped in
        # March 2025 and its columns had gone constant, so the fundamental block
        # was contributing nothing to the score while still occupying five of
        # the model's columns.
        fundamentals = store.read_statements()
        max_age = int(iv(cfg.max_fundamental_age_days))

        # Delivered quantity as a share of traded volume. Read over the same
        # window as the prices so the two panels align date for date; a name
        # with no print ranks neutral rather than dropping out.
        #
        # A name with no print and the whole feed being gone are different
        # things and used to be handled the same way. deliv_pct carries the
        # largest coefficient in the fit, and crosssec lists it as
        # neutral-when-missing, so swallowing a read failure here scored every
        # name as though its delivered share were exactly average: measured,
        # that replaces a third of the top decile and costs 18% of the IC while
        # the run reports nothing. Per-name gaps stay neutral; an empty or
        # unreadable panel is a failure and is raised, which the outer handler
        # turns into a non-benign reason and Stage 4 into ModelUnavailable.
        dl = store.read_delivery(
            symbols=None if refitting else list(symbols), start=start, end=as_of)
        if dl is None or dl.empty or "deliv_pct" not in dl.columns:
            raise PipelineError(
                STAGE_NAME,
                "the delivery panel is empty or has no deliv_pct column. "
                "deliv_pct is the model's largest coefficient and ranks neutral "
                "when absent, so continuing would score every name as average on "
                "it and report a normal-looking watchlist. Run "
                "`prosignal data ingest` to refresh sec_bhavdata_full.",
            )
        dl[DATE] = pd.to_datetime(dl[DATE]).dt.normalize()
        delivery = dl.pivot_table(
            index=DATE, columns=SYMBOL, values="deliv_pct",
            aggfunc="last", observed=True
        ).sort_index()
        del dl

        covered = float(delivery.notna().any(axis=0).mean()) if not delivery.empty else 0.0
        floor = fv(cfg.min_name_factor_coverage)
        if covered < floor:
            raise PipelineError(
                STAGE_NAME,
                f"delivery covers {covered:.0%} of the universe, below the "
                f"{floor:.0%} floor. Below it the factor ranks most of the "
                f"universe neutral, which is not a ranking.",
            )

        if cached is not None:
            feats = cm.today_features(close, turnover, as_of,
                                      fundamentals=fundamentals,
                                      max_fundamental_age_days=max_age,
                                      delivery=delivery)
            if feats is None:
                return None, None, "no symbol had a complete feature set today", None, None
            return cm.score_with(cached, feats), cached, None, feats, None

        # The screen as it stood on every training date. `symbols` is the same
        # screen resolved for TODAY, and it stays the scoring universe.
        eligible = liquidity_mask(
            close, turnover,
            min_adtv_inr=float(fv(universe.pit_min_adtv_inr)),
            lookback_sessions=int(iv(universe.pit_adtv_lookback_sessions)),
            max_names=int(iv(universe.pit_max_names)),
            min_history_sessions=int(iv(universe.min_history_sessions)),
            min_price_inr=float(fv(universe.min_price_inr)),
        )
        scores, model, reason = cm.fit_predict(
            close, turnover, as_of,
            fundamentals=fundamentals, max_fundamental_age_days=max_age,
            horizon=int(iv(cfg.model_horizon_sessions)),
            alpha=float(fv(cfg.model_ridge_alpha)),
            max_train_sessions=int(iv(cfg.model_max_train_sessions)),
            min_train_rows=int(iv(cfg.model_min_train_rows)),
            delivery=delivery,
            eligible=eligible,
            score_symbols=list(symbols),
        )
        if model is not None:
            # A refit is proposed, not installed. This is the one path where a
            # bad upstream date reaches every future decision at once without
            # failing anything, so the new coefficients are compared against the
            # live ones before they replace them.
            previous, previous_end = cm.read_cached_coefficients(cache)
            verdict = review_refit(model.coef, previous, previous_end)
            if verdict.accepted:
                cm.archive_cache(cache)
                cm.save_cache(cache, model, as_of)
            else:
                log.warning("refit rejected; previous coefficients stay live",
                            extra={"verdict": verdict.summary(),
                                   "sign_flips": verdict.sign_flips,
                                   "magnitude_jumps": verdict.magnitude_jumps})
                held = cm.load_cached(cache, as_of, refit_every)
                if held is not None:
                    feats = cm.today_features(close, turnover, as_of,
                                              fundamentals=fundamentals,
                                              max_fundamental_age_days=max_age,
                                              delivery=delivery)
                    if feats is not None:
                        return (cm.score_with(held, feats), held, None,
                                feats, verdict)
                # Nothing usable to hold on to, so the run cannot quietly
                # continue on a fit that failed review.
                return None, None, f"refit rejected: {verdict.summary()}", None, verdict
            live = cm.today_features(close, turnover, as_of,
                                     fundamentals=fundamentals,
                                     max_fundamental_age_days=max_age,
                                     delivery=delivery)
        else:
            live = None
        return scores, model, reason, live, None
    except Exception as exc:  # a modelling failure must not take the run down
        log.warning("cross-sectional model failed", extra={"error": str(exc)})
        return None, None, f"{type(exc).__name__}: {exc}", None, None


class ModelUnavailable(PipelineError):
    """The cross-sectional model could not score, and fallback is not allowed."""

    code = "MODEL_UNAVAILABLE"

    def __init__(self, message: str, **context) -> None:
        super().__init__(STAGE_NAME, message, **context)


def _min_coverage(cfg) -> float:
    """Single source of truth for factor coverage.

    A module constant used to shadow this at three call sites while a fourth
    read the config. They agreed, so editing parameters.yaml changed one of the
    four and nothing reported the divergence.
    """
    return float(fv(cfg.min_name_factor_coverage))


def _model_optional(cfg) -> bool:
    return bool(bv(getattr(cfg, "allow_composite_fallback", False)))


#: Reasons that mean "not enough data yet" rather than "something broke".
_BENIGN_REASONS = (
    "sessions of history",
    "usable training rows",
    "no price rows",
    "complete feature set",
    "could not be computed",
)


def _is_model_failure(reason: str) -> bool:
    """Distinguish a broken model from one that legitimately cannot run yet.

    A fresh store with too little history is an expected state: the composite
    scores it and the card says the model was unavailable. An exception inside
    the model is not expected, and falling back there hands the run to a scorer
    measured at t = -0.11 while looking exactly like a healthy result.
    """
    text = str(reason).lower()
    return not any(b in text for b in _BENIGN_REASONS)
