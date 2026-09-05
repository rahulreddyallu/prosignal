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
    FactorMember,
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
from ..features import v3 as v3feat
from ..features import v3_factors as v3fac
from ..features.crosssec import liquidity_mask
from ..features.exits import rules_from_config
from ..features.labels import BarrierSpec
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



class RankingUnavailable(PipelineError):
    """The configured ranking column is not in the live feature frame.

    Deliberately fatal. The alternative -- fall back to the fitted composite --
    restores exactly the scorer `stage4_core_score.ranking` exists to retire,
    and it would do so silently, on a run that looks completely normal. A run
    that cannot rank the way it is configured to rank has not produced a
    ranking; it has produced a different strategy wearing the same name.
    """

    def __init__(self, message: str, **context) -> None:
        super().__init__(stage=STAGE_NAME, message=message, **context)


def _apply_ranking_policy(composite_raw, model_features, cfg, notes,
                          v3_scored=None, sectors=None):
    """Return the series the book is ordered by, and say which one it is.

    `composite_raw` arrives holding whatever ranked upstream -- the fitted
    model's scores when it covered enough names, the hand-weighted composite
    otherwise. Under `fitted_composite` that is left alone and this is a no-op.

    Under `measured_factor` the ranking becomes one already-ranked column and
    the composite continues to be computed, recorded, attributed on the card and
    monitored by `research decay`. That separation is the point: the fitted
    coefficients remain the engine's running measurement of what each theme is
    worth, and a future window that finds the composite ordering its own top
    decile positively is what would send the ranking back to it.
    """
    source = str(getattr(cfg.ranking, "source", "fitted_composite"))
    if source == "fitted_composite":
        return composite_raw, "fitted_composite"

    if source == "v9r_core":
        # THE MODEL THE SEALED 2012-2017 WINDOW MEASURED. Nine factors, equal
        # risk contribution, unneutralised, no coverage renormalisation. It
        # returned +9.50% net active at Newey-West t +1.87 against a
        # pre-registered bar of 2.0 -- POSITIVE AND UNDERPOWERED, which is a
        # FAILED ship gate, not a passed one. See docs/MODEL_v9R.md.
        # It is selectable so it can be shadow-run against the incumbent; it is
        # not the default and must not become one on this evidence alone.
        if v3_scored is None or "score" not in getattr(v3_scored, "columns", []):
            raise RankingUnavailable(
                "stage4_core_score.ranking.source is 'v9r_core' and the scorer "
                "did not build. Falling back would issue signals from a model "
                "that was not the one measured.")
        ranked = v3_scored["score"].dropna()
        covered = ranked.reindex(composite_raw.index).dropna()
        floor = max(int(0.6 * len(composite_raw)), 20)
        if len(covered) < floor:
            raise RankingUnavailable(
                f"the v9R composite covers {len(covered)} of "
                f"{len(composite_raw)} scoreable names, under the {floor} floor.")
        return covered, "v9r_core"

    if source == "v3_composite":
        # THE SHIPPED SCORER. Twenty-two factors in five themes, combined within
        # theme and then blended with weights capped at 40%, floored at 6%, and
        # additionally capped at each theme's coverage. Selected on 2018-11 to
        # 2024-10; evaluated ONCE on each of two sealed windows, one of which
        # (2021-07 to 2022-12) no search had touched and for which the whole
        # pipeline was re-run on data ending 2021-02. Numbers in CHANGELOG.md.
        if v3_scored is None or "score" not in getattr(v3_scored, "columns", []):
            raise RankingUnavailable(
                "stage4_core_score.ranking.source is 'v3_composite' and the v3 "
                "block did not build. Falling back to another scorer would "
                "issue signals from a model that was not the one measured.")
        ranked = v3_scored["score"].dropna()
        covered = ranked.reindex(composite_raw.index).dropna()
        floor = max(int(0.6 * len(composite_raw)), 20)
        if len(covered) < floor:
            raise RankingUnavailable(
                f"the v3 composite covers {len(covered)} of "
                f"{len(composite_raw)} scoreable names, under the {floor} "
                f"floor. A ranking built on a minority of the universe is a "
                f"ranking of that minority.")
        nth = v3_scored["n_themes"].reindex(covered.index)
        # THE WEIGHTS THIS RUN USED, and the LABELS rather than the dict keys.
        # The note quoted `Theme.weight` -- the fit-time vector, correct for a
        # name with all five themes and for 8.8% of this universe -- and named
        # the themes by their internal keys, so it said "quality 19%" for a
        # theme whose two factors both ship at -1 and which the screen has
        # called "Low-margin tilt" since the labels were made honest. The note
        # is written into the run record, so both halves outlived the screen.
        _wmean = {
            t: v3_scored[t + "_w"].dropna().mean()
            for t in v3feat.THEMES if t + "_w" in v3_scored.columns
        }
        _shown = ", ".join(
            f"{th.label} {_wmean.get(t, th.weight):.0%}"
            for t, th in v3feat.THEMES.items())
        notes.append(
            f"Book ordered by the v3 composite: {len(v3feat.ALL_FACTORS)} factors "
            f"in {len(v3feat.THEMES)} themes, at the mean weights this "
            f"cross-section actually blended them at ({_shown}). Each theme is "
            f"combined on its own, then blended with weights RE-CAPPED at 40% "
            f"over the themes each name has -- a name missing one does not hand "
            f"its share to whichever theme is largest. Median name scored on "
            f"{nth.median():.0f} of {len(v3feat.THEMES)} themes. "
            f"The two sealed holdouts (rank IC +0.049 t 3.69 on 2025-03..2026-08, "
            f"+0.036 t 3.83 on 2021-07..2022-12) measured the blend BEFORE that "
            f"re-cap and are not re-run: both windows are spent. The RANKING is "
            f"what generalised there; a ten-name book at these transaction costs "
            f"did not -- see CHANGELOG.md.")
        # AND HOW MUCH OF THE UNIVERSE "SECTOR-NEUTRAL" ACTUALLY COVERS. The
        # card says the score is a sector-neutral rank; for the residual bucket
        # it is a rank against a pool of unrelated industries. Reported rather
        # than fixed, because raising sector coverage is a data job (D-019) and
        # a claim that is true for 61% of the book should not be silent about
        # the other 39%.
        try:
            _rb = v3feat.residual_bucket_size(covered.index, sectors)
            if _rb["resid"]:
                notes.append(
                    f"Sector-neutral for {len(covered) - _rb['resid']} of "
                    f"{len(covered)} names. The other {_rb['resid']} "
                    f"({_rb['resid'] / max(len(covered), 1):.0%}) are ranked "
                    f"inside one residual bucket -- {_rb['unknown']} with no "
                    f"sector in the map and {_rb['folded']} folded in from "
                    f"{len(_rb.get('folded_sectors') or [])} sectors too small "
                    f"to rank within ({v3feat.MIN_SECTOR_NAMES} names needed). "
                    f"Inside that bucket a sector tilt is not neutralised.")
        except Exception as exc:                    # never fail a run to report
            log.warning("residual-bucket report did not run",
                        extra={"error": str(exc)})

        # THE DOMINANCE CHECK RUNS ON EVERY RUN, not only when somebody types a
        # research command. A theme that has taken over the ranking is a
        # property of today's scores and needs no forward outcome to see, so
        # leaving it to a quarterly command would mean the one thing the brief
        # asked to watch live -- "if one theme starts dominating live even
        # though it was capped in training" -- was watched four times a year.
        # It FLAGS. No weight moves and nothing is disabled.
        try:
            from .. import v3_monitor as v3mon
            notes.extend(v3mon.review_cross_section(v3_scored.loc[covered.index]))
        except Exception as exc:                    # never fail a run to report
            log.warning("theme influence check did not run",
                        extra={"error": str(exc)})
        return covered, source

    column = str(cfg.ranking.column)
    if model_features is None or column not in getattr(model_features, "columns", []):
        available = (", ".join(sorted(getattr(model_features, "columns", [])))
                     if model_features is not None else "no feature frame")
        raise RankingUnavailable(
            f"stage4_core_score.ranking.source is {source!r} and asks for "
            f"{column!r}, which the live feature frame does not have "
            f"({available}). Falling back to the fitted composite would restore "
            f"the scorer this setting retires -- measured at negative alpha in "
            f"every one of its 144 trade-level configurations -- so the run "
            f"stops instead. Fix the column name or set ranking.source to "
            f"fitted_composite to accept that scorer explicitly."
        )
    # KEYED BY SYMBOL, not by position. `features_for_date` ends with
    # `.reset_index(drop=True)`, so the live feature frame carries a RangeIndex
    # and keeps the ticker in a `symbol` COLUMN -- which is why the first
    # version of this reindexed a symbol-indexed composite against 0..451 and
    # matched nothing. It failed loudly, which is the only reason it is a
    # two-line fix rather than a book ranked by whatever survived the join.
    ranked = model_features[column]
    if "symbol" in getattr(model_features, "columns", []):
        ranked = pd.Series(ranked.to_numpy(),
                           index=model_features["symbol"].astype(str),
                           name=column)
    ranked = ranked[~ranked.index.duplicated(keep="last")].dropna()
    if ranked.empty:
        raise RankingUnavailable(
            f"the ranking column {column!r} is present but empty for every "
            f"scoreable name, so there is nothing to order the book by."
        )
    covered = ranked.reindex(composite_raw.index).dropna()
    floor = max(int(0.6 * len(composite_raw)), 20)
    if len(covered) < floor:
        raise RankingUnavailable(
            f"the ranking column {column!r} covers {len(covered)} of "
            f"{len(composite_raw)} scoreable names, under the {floor} floor. A "
            f"ranking built on a minority of the universe is a ranking of that "
            f"minority, and the names it cannot see would be dropped without "
            f"ever being compared."
        )
    notes.append(
        f"Book ordered by {column} ({source}), not by the fitted composite. "
        f"The composite is still fitted, recorded and shown -- it explains the "
        f"themes behind a name -- but it does not choose. Measured over 4,877 "
        f"trade-level configurations against an equal-weight benchmark of the "
        f"same eligible universe, this column returned positive alpha in 98.1% "
        f"of its 960 configurations while the fitted composite returned "
        f"negative alpha in all 144 of its own."
    )
    return covered, source




def build_v3_block(store, calendar, symbols, as_of, sectors, cfg, v9r_mode=False,
                   degraded=None):
    """The v3 two-level thematic score for ``as_of``.

    Returns (raw factor values, scored frame, error). Reads only sessions at or
    before ``as_of``; the fundamental block is as-of joined on DISCLOSURE dates,
    never on period ends.
    """
    # Feed failures that silently change the model, collected for the run
    # notes rather than left in a log file.
    _degraded = degraded if degraded is not None else []
    need = v3fac.LOOKBACK_SESSIONS + 15
    window = calendar.trailing_window(as_of, need)
    start = window[0] if window else calendar.first
    px = store.read_prices(symbols=symbols, start=start, end=as_of)
    if px is None or px.empty:
        return None, None, "no prices in the v3 lookback window"
    px = px.copy()
    px[DATE] = pd.to_datetime(px[DATE]).dt.normalize()

    def piv(frame, col):
        if col not in frame.columns:
            return None
        return frame.pivot_table(index=DATE, columns=SYMBOL, values=col,
                                 aggfunc="last", observed=True).sort_index()

    close = piv(px, "close")
    if close is None or close.empty:
        return None, None, "the store served no close for the v3 window"
    open_, vwap, turnover = piv(px, "open"), piv(px, "vwap"), piv(px, "turnover")

    # DELIVERY COMES FROM ITS OWN TABLE. `prices.deliv_pct` is a write-time
    # column that is empty across this store; reading it here would make the
    # whole ownership theme -- 19% of the composite -- silently neutral.
    deliv = None
    try:
        dl = store.read_delivery(symbols=symbols, start=start, end=as_of)
        if dl is not None and not dl.empty and "deliv_pct" in dl.columns:
            dl = dl.copy()
            dl[DATE] = pd.to_datetime(dl[DATE]).dt.normalize()
            deliv = piv(dl, "deliv_pct")
            if deliv is not None:
                deliv = deliv.reindex(index=close.index)
    except Exception as exc:
        # SURFACED, not just logged. Losing delivery removes a theme carrying
        # 19% nominal and about 20% of the live spread, and the blend then
        # re-caps over what is left -- momentum goes from 40% to its cap on
        # every name. That is a materially different model, and the only
        # previous symptom was a log line nobody reads and a slightly different
        # ranking. `_degraded` is carried into the run notes.
        _degraded.append(
            f"DELIVERY UNAVAILABLE ({exc}). The ownership theme is absent from "
            f"this run and its weight was redistributed. This is not the model "
            f"the holdouts measured.")
        log.warning("delivery unavailable; the ownership theme will be absent",
                    extra={"error": str(exc)})

    # THE EQUAL-WEIGHT RETURN, which is not what `close.mean(axis=1)` gives.
    #
    # That was the mean PRICE LEVEL across the names being scored, differenced.
    # Two things are wrong with it. It is price-weighted -- a Rs 30,000 name
    # moves it a thousand times more than a Rs 30 one, which is the opposite of
    # equal weight. And `mean` skips NaN, so on any date where the set of names
    # with a print changes, the mean jumps for a reason that is not a return:
    # measured on the live window, the implied "return" has sd 0.0156 on
    # composition-change dates against 0.0110 on stable ones, a 42% inflation.
    #
    # The equal-weight return is the mean of the per-name RETURNS. Only
    # `resid_rev_21` consumes it -- it is the market leg of that factor's beta
    # and residual -- so this moves one factor of twenty-two, and it moves it
    # onto the definition its own docstring already claimed.
    bench_ret = (close / close.shift(1) - 1.0).mean(axis=1)

    fund = None
    try:
        from ..features import pit_fundamentals as pitf
        recs = pitf.build_records(store=store)
        if recs is not None and not recs.empty:
            fsyms = [c for c in close.columns if c in set(recs["symbol"])]
            if fsyms:
                blk = pitf.asof_panel(recs, close.index, fsyms)
                fund = {k: v.reindex(columns=close.columns)
                        for k, v in blk.items()
                        if k in ("ttm_revenue", "ttm_net_profit", "fund_age_days")}
    except Exception as exc:
        _degraded.append(
            f"POINT-IN-TIME FUNDAMENTALS UNAVAILABLE ({exc}). The quality theme "
            f"is absent from this run and its weight was redistributed.")
        log.warning("point-in-time fundamentals unavailable; the quality theme "
                    "will be absent and the remaining weights renormalise",
                    extra={"error": str(exc)})

    raw = v3fac.factor_frame(close, open_, vwap, turnover, deliv, bench_ret, fund)
    if raw.empty:
        return None, None, "the v3 factor frame came back empty"
    raw = raw.reindex([x for x in symbols if x in raw.index])
    if v9r_mode:
        # The v9R CORE scorer reads NINE of the same twenty-two factors and does
        # not take `sectors`: it ranks unneutralised, because the sector map
        # covers 754 symbols and 46.7% of a live cross-section otherwise ranks
        # inside one residual bucket that is not a sector.
        from ..features import v9r as v9rfeat
        return raw, v9rfeat.score_frame(raw), None
    scored = v3feat.score_frame(raw, sectors, min_themes=int(iv(cfg.ranking.v3_min_themes)))
    return raw, scored, None




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
                q, q_note = _quality_from_features(feats, symbols, cfg.factors.quality,
                                                   _min_coverage(cfg))
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

    # Held until the ranking source is known, for the same reason the regime
    # note is. These describe the FAMILY block. Under `v3_composite` that block
    # does not rank, and its `quality` and the v3 theme keyed `quality` are
    # different things computed from different sources -- so "quality dropped:
    # no point-in-time fundamentals" went into the record of a run whose v3
    # quality theme had scored 34 names, which reads as the theme being absent
    # when it was present and carrying 19%.
    _dropped_notes = [f"{name} dropped: {reason}" for name, reason in dropped.items()]

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
    # SAID ONLY WHERE IT IS TRUE. This note went onto every run, and under
    # `v3_composite` the multipliers it describes scale the FAMILY block --
    # which `_apply_ranking_policy` then discards. An operator reading "regime
    # 'range_lowvol' multipliers applied (momentum x0.75)" on a run whose book
    # was ordered by an unmodified v3 blend is being told the engine leaned
    # against momentum today. It did not.
    #
    # Deferred rather than deleted: the note is correct on the `fitted_composite`
    # path, which is still selectable, so it is emitted after the ranking source
    # is known instead of before.
    _regime_note = (
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
    # ---- the fitted cross-sectional model, REMOVED -------------------------
    # Until this cleanup the engine fitted a Fama-MacBeth cross-sectional model on
    # every run, attributed it on the card, and then ranked on the v3 composite
    # anyway. The fit was a diagnostic nobody ordered a book by.
    #
    # It cost about 3,000 lines across crossmodel / famamacbeth / linear /
    # metalabel / refit_gate, all of it executing daily on the live signal path.
    # `composite_raw` above is built from the FAMILY factors and never depended on
    # it, so removing the fit leaves the scoreable universe exactly as it was.
    #
    # If a fitted ranker is wanted again, it should be re-derived and re-validated
    # rather than restored: the coefficients it carried were fitted against the
    # engine's own exit geometry, which is the defect `research/` recorded before
    # the v3 composite replaced it as the ranking.
    model_scores = model = model_features = refit_verdict = None
    model_unavailable = None

    model_contrib = None
    model_z = None

    v3_raw = v3_scored = None
    _src = str(getattr(cfg.ranking, "source", ""))
    if _src in ("v3_composite", "v9r_core"):
        _degraded_feeds: List[str] = []
        v3_raw, v3_scored, v3_err = build_v3_block(
            store, calendar, symbols, as_of, sectors, cfg,
            v9r_mode=(_src == "v9r_core"), degraded=_degraded_feeds)
        notes.extend(_degraded_feeds)
        if v3_err:
            notes.append(f"v3 block unavailable: {v3_err}")

    composite_raw, ranking_source = _apply_ranking_policy(
        composite_raw, model_features, cfg, notes, v3_scored=v3_scored,
        sectors=sectors)

    # The family block's regime multipliers and dropped factors moved the
    # ranking only if the family block IS the ranking. `fitted_composite` is
    # the one source where it is: every other branch of `_apply_ranking_policy`
    # REPLACES `composite_raw` and keeps only its index, as a population filter.
    # Listing the sources that discard it would have to be kept in step with
    # that function; naming the single source that does not, does not.
    if ranking_source == "fitted_composite":
        notes.extend(_dropped_notes)
        notes.append(_regime_note)

    # THE ABSOLUTE FLOOR, computed here and enforced at entry. Names that fail
    # it stay in the ranking -- they are holdable and they belong on a watchlist
    # -- and cannot be bought. When fewer clear it than there are slots, the
    # book holds cash, which is how NO TRADE happens.
    _fl = getattr(cfg, "absolute_floor", None)
    _floor_on = bool(bv(_fl.enabled)) if _fl is not None else False
    _floor_ma = int(iv(_fl.above_ma_sessions)) if _fl is not None else 200
    _floor_min_themes = int(iv(_fl.min_positive_themes)) if _fl is not None else 3
    _floor_scope = str(v(_fl.applies_to)) if _fl is not None else "entries"
    # COMPUTED WHETHER OR NOT THE FLOOR IS ON. The floor decides whether this
    # GATES a name; it does not decide whether the engine is allowed to know.
    # Stage 8's book-level cash rule asks a different question of the same
    # number -- not "may this name be bought" but "can the market supply a book
    # at all" -- and gating it behind `absolute_floor.enabled` meant the one
    # measurement that can distinguish a bad day from an ordinary one was not
    # taken on any day the floor was off, which is every day since 2026-09-02.
    _above_ma = {}
    if not closes.empty:
        ma = closes.rolling(_floor_ma, min_periods=int(_floor_ma * 0.75)).mean()
        if len(ma):
            _above_ma = (closes.iloc[-1] > ma.iloc[-1]).to_dict()

    composite_unit = rank_to_unit_interval(composite_raw)
    percentile = composite_unit * 100.0
    # STABLE. pandas defaults to quicksort, which is not, so two names on the
    # same score would take whichever order the algorithm happened to produce --
    # and rank decides the entry band, so a tie is a coin toss over what gets
    # bought. Ties are rare with 22 continuous factors (386 distinct scores over
    # 386 names on 2026-09-03) and cost nothing to make deterministic.
    order = composite_raw.sort_values(ascending=False, kind="stable")

    scores: List[StockScore] = []
    for rank, sym in enumerate(order.index, start=1):
        factors = {}
        if ranking_source == "v3_composite" and v3_scored is not None \
                and sym in v3_scored.index:
            # THE CARD EXPLAINS THE NUMBER IT PRINTS, AT BOTH LEVELS. Theme rows
            # carry the sub-score and the weight it was blended at and sum to
            # the composite; factor rows carry the ranks the theme was built
            # from. "momentum +0.42" does not say which momentum moved.
            for tname, th in sorted(
                    v3feat.THEMES.items(),
                    key=lambda kv: -(abs(_f(v3_scored.at[sym, kv[0] + "_contrib"]) or 0.0))):
                # THE WEIGHT THIS NAME WAS BLENDED AT. `th.weight` is the
                # frozen fit-time number and it is correct only for a name
                # carrying all five themes -- 8.8% of the live universe. The
                # blend re-caps per name, so serving the frozen figure here put
                # a weight on the card that did not multiply its own z into its
                # own contribution, uniformly out by 1/den. `score_frame` now
                # emits the weight it used and this reads it.
                _have = pd.notna(v3_scored.at[sym, tname + "_sub"])
                _w = _f(v3_scored.at[sym, tname + "_w"]) \
                    if tname + "_w" in v3_scored.columns else None
                if _w is None:
                    # A theme the name does not have was blended at ZERO, not at
                    # its fit-time weight. The card drops unavailable rows so
                    # this is invisible there, but the row goes into the ledger,
                    # where "quality, weight 0.18991, contribution null" reads
                    # as a theme that was carried and produced nothing rather
                    # than one that was absent.
                    _w = th.weight if _have else 0.0
                factors[tname] = FactorScore(
                    name=tname,
                    raw_value=None,
                    standardised=_f(v3_scored.at[sym, tname + "_sub"]),
                    weight=round(_w, 5),
                    contribution=_f(v3_scored.at[sym, tname + "_contrib"]),
                    available=pd.notna(v3_scored.at[sym, tname + "_sub"]),
                    evidence_tier="v3_theme",
                    citation=(f"{th.label} -- theme sub-score, oriented at "
                              f"{th.horizon} sessions"),
                    members=[
                        FactorMember(
                            name=fn,
                            rank=_f(v3_scored.at[sym, fn + "_r"]),
                            available=bool(v3_raw is not None and sym in v3_raw.index
                                           and pd.notna(v3_raw.at[sym, fn])),
                            description=f"sign {int(th.signs[fn]):+d}")
                        for fn in th.names],
                )
            # AND THE 26-FACTOR FITTED MODEL, KEPT. It no longer decides -- the
            # v3 composite orders the book -- but it is still fitted, still
            # monitored, and it is a second independent reading of the same
            # name. Dropping it from the card when the ranking source changed
            # made a working model invisible and left the reader unable to see
            # when the two disagree, which is exactly the moment worth seeing.
            # Tier `model_secondary` so the card can say it did not choose.
            if model_contrib is not None and sym in model_contrib.index:
                mrow = model_contrib.loc[sym]
                mz = model_z.loc[sym] if model_z is not None else None
                for name in mrow.abs().sort_values(ascending=False).index:
                    if name in factors:      # never shadow a theme row
                        continue
                    factors[name] = FactorScore(
                        name=name,
                        raw_value=_f(mrow.get(name)),
                        standardised=_f(mz.get(name)) if mz is not None else None,
                        weight=round(float(
                            model.coef.get(name + "_f",
                                           model.coef.get(name + "_r", 0.0))), 5),
                        available=pd.notna(mrow.get(name)),
                        evidence_tier="model_secondary",
                        citation=_MODEL_CITE.get(name),
                        members=_members_for(name, sym, model_features),
                    )
        elif model_contrib is not None and sym in model_contrib.index:
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
                    # Families are `_f`, individual factors `_r`. Appending
                    # only `_r` asked for `mom_f_r` and got zero.
                    weight=round(float(
                        model.coef.get(name + "_f",
                                       model.coef.get(name + "_r", 0.0))), 5),
                    available=pd.notna(row.get(name)),
                    evidence_tier="model",
                    citation=_MODEL_CITE.get(name),
                    # WHAT THE THEME IS MADE OF. One coefficient is fitted per
                    # theme over the average of its members' ranks, so the
                    # theme is what carries weight -- but "lottery -1.81 sd"
                    # does not say which lottery moment moved, and a reader
                    # cannot check the theme against the measurement without
                    # them. The members are the 26 ranked columns; the five
                    # themes are how they are priced.
                    members=_members_for(name, sym, model_features),
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
        adm, reason = True, None
        if _floor_on and v3_scored is not None and sym in v3_scored.index:
            npos = v3_scored.at[sym, "n_themes_positive"]
            above = bool(_above_ma.get(sym, False))
            npos_ok = bool(pd.notna(npos) and npos >= _floor_min_themes)
            adm = above and npos_ok
            if not adm:
                bits = []
                if not above:
                    bits.append(f"below its {_floor_ma}-session average")
                if not npos_ok:
                    bits.append(f"only {0 if pd.isna(npos) else int(npos)} themes "
                                f"above the median, {_floor_min_themes} required")
                reason = ("fails the absolute floor: " + " and ".join(bits)
                          + ". It can be held, not opened.")
        scores.append(
            StockScore(
                ticker=str(sym),
                sector=sectors.get(sym),
                factors=factors,
                entry_admissible=adm,
                entry_block_reason=reason,
                # The bar as MEASURED, independent of whether it gates.
                absolute_bar_cleared=(bool(_above_ma[sym])
                                      if sym in _above_ma else None),
                composite_raw=float(composite_raw.get(sym, 0.0)),
                composite_score=float(composite_unit.get(sym, 0.0)),
                percentile=float(percentile.get(sym, 0.0)),
                rank=rank,
            )
        )

    if _floor_on:
        n_ok = sum(1 for sc in scores if sc.entry_admissible)
        notes.append(
            f"Absolute floor ({_floor_scope}): {n_ok} of {len(scores)} ranked "
            f"names may be OPENED -- above their {_floor_ma}-session average and "
            f"on the right side of at least {_floor_min_themes} themes. The rest "
            f"stay ranked and holdable. If fewer clear it than there are slots, "
            f"the book holds cash: that is the NO TRADE state, and it is reached "
            f"by the names failing a test rather than by a view on the market.")
        if _floor_scope == "population":
            keep = {sc.ticker for sc in scores if sc.entry_admissible}
            scores = [sc for sc in scores if sc.ticker in keep]

    # Measured on the MODEL's own features, not on the hand-weighted composite's.
    # It ran on `frame` -- the legacy composite's factor block -- so the
    # seventeen columns that actually rank the universe were never checked
    # against each other. On the live universe they are not independent:
    # amihud/turnover_ratio at -0.87 are one factor measured from two sides, and
    # resid_mom/mom_6_1 at +0.77 with prox_52w at +0.60 make the momentum block
    # roughly one bet carrying three coefficients.
    # Measured on the columns that actually CARRY COEFFICIENTS, which since the
    # family refactor means the families, not their members.
    #
    # Pointing it at the individual `_r` factors reported the wrong thing
    # loudly. Measured live on 2026-08-25 those columns breach |rho| = 0.6 five
    # times -- mom_6_1/resid_mom +0.72, downside_vol/idio_vol +0.69,
    # mom_6_1/prox_52w +0.64, max_dd_120/prox_52w +0.63, prox_52w/resid_mom
    # +0.62 -- and every one of those pairs sits INSIDE a single family. That
    # collinearity is the reason the families exist; averaging them is the
    # engine's answer to it. Reporting it as a breach on every run is a
    # detector that fires constantly and teaches its reader to ignore it.
    #
    # The question the report is for is whether the things given INDEPENDENT
    # coefficients are independent. On the live model the families read
    # delivery/lottery -0.38 at the widest, which is the aggregation working.
    model_block = None
    member_block = None
    if model_features is not None and not model_features.empty:
        from ..features.families import UNSCORED_CONTROLS, UNSCORED_DIAGNOSTICS
        fitted = [c for c in (getattr(model, "features", None) or [])
                  if c in model_features.columns]
        if len(fitted) >= 2:
            model_block = model_features[fitted].rename(columns=lambda c: c[:-2])
        # Kept as context rather than as the verdict: it is what justifies the
        # aggregation, so it belongs in the report, not in the breach list.
        _ignore = set(UNSCORED_DIAGNOSTICS) | set(UNSCORED_CONTROLS)
        cols = [c for c in model_features.columns
                if c.endswith("_r") and c not in _ignore]
        if len(cols) >= 2:
            member_block = model_features[cols].rename(columns=lambda c: c[:-2])

    redundancy = _redundancy(model_block if model_block is not None else frame,
                             cfg, members=member_block)
    if redundancy.breaches:
        pairs = ", ".join(f"{a}/{b} {r:+.2f}" for a, b, r in redundancy.breaches[:4])
        notes.append(
            f"Redundancy: {len(redundancy.breaches)} factor pair(s) exceed "
            f"|rho|={redundancy.cutoff}. They are not independent evidence. "
            f"{pairs}"
        )

    log.info("stage 4 complete", extra={"scored": len(scores), "factors": list(effective)})
    return CoreScoreReport(
        as_of_date=as_of,
        # POPULATED FROM WHATEVER RANKED, not from a model that no longer
        # exists. `model` is None on every path but `fitted_composite`, so these
        # were None on every shipped run -- and Stage 8's dispersion gate is
        # guarded by `dispersion is not None`, which meant the one control able
        # to say "the scorer degenerated today" was skipped, silently, from the
        # moment the fitted model was deleted.
        prediction_dispersion=(
            v3feat.score_dispersion(v3_scored["score"])
            if (ranking_source == "v3_composite" and v3_scored is not None
                and "score" in getattr(v3_scored, "columns", []))
            else (float(getattr(model, "dispersion", 0.0))
                  if model is not None else None)),
        typical_dispersion=(
            v3feat.TYPICAL_DISPERSION if ranking_source == "v3_composite"
            else (float(getattr(model, "train_dispersion", 0.0))
                  if model is not None else None)),
        # The meta-label veto is gone (AUC 0.4996, shipped disabled), so there is
        # no win probability and nothing to explain its absence. The card reads
        # None for both rather than carrying a field that can only say "off".
        win_probability=None,
        win_probability_unavailable=None,
        weighting_mode=str(v(cfg.weighting_mode)),
        standardisation=method,
        # THE WEIGHTS THAT RANKED THE BOOK, when something ranked it.
        #
        # This served `effective` -- the FAMILY block's weights, regime
        # multipliers and all. Under `v3_composite` that block is computed and
        # then discarded by `_apply_ranking_policy`; only its INDEX survives, as
        # a population filter. So the field read
        # {'momentum_12_1': 0.4688, 'sector_relative_strength': 0.5312} on a run
        # that ordered its book by twenty-two factors in five themes: two
        # numbers, summing to one, describing nothing that chose anything.
        effective_weights=_reported_weights(ranking_source, v3_scored, effective),
        dropped_factors=dropped,
        ranked_scores=scores,
        redundancy=redundancy,
        universe_size=len(symbols),
        notes=notes,
    )



def _reported_weights(ranking_source, v3_scored, family_effective) -> Dict[str, float]:
    """The blend weights the REPORT should carry: the ones that ranked the book.

    Under `v3_composite` the weights are per-name -- the blend re-caps over the
    themes each name has -- so a single vector is a summary, not the thing
    itself. The mean over scored names is the honest summary and it is what the
    dominance question is asked of ("is momentum running hotter than its cap
    allows?"), so that is what is reported, keyed by theme.

    Falls back to the family block's weights only when the family block is what
    ranked, which is the `fitted_composite` path.
    """
    if ranking_source in ("v3_composite",) and v3_scored is not None:
        out: Dict[str, float] = {}
        for tname in v3feat.THEMES:
            col = tname + "_w"
            if col not in getattr(v3_scored, "columns", []):
                continue
            w = v3_scored[col].dropna()
            if len(w):
                out[tname] = round(float(w.mean()), 4)
        if out:
            return out
    return {k: round(val, 4) for k, val in (family_effective or {}).items()}


def _members_for(family: str, symbol, features) -> List[FactorMember]:
    """The measured factors underneath one fitted theme, for one name.

    Reads the SAME frame the model scored from, so a member shown here and the
    average the coefficient multiplied cannot disagree.
    """
    from ..features.crosssec import FEATURES
    from ..features.families import FAMILIES

    if features is None or features.empty or family not in FAMILIES:
        return []
    row = features[features["symbol"] == symbol]
    if row.empty:
        return []
    row = row.iloc[0]
    out: List[FactorMember] = []
    for col in FAMILIES[family]:
        if col not in features.columns:
            continue
        value = row.get(col)
        bare = col[:-2] if col.endswith("_r") else col
        described = FEATURES.get(bare)
        out.append(FactorMember(
            name=bare,
            rank=_f(value),
            available=pd.notna(value),
            description=(described[1] if described else _FUND_DESC.get(bare)),
        ))
    return out


#: The fundamental members carry no entry in `crosssec.FEATURES`, which
#: describes the price and volume block only.
_FUND_DESC = {
    "earnings_yield": "trailing earnings over price",
    "book_to_price": "book value over price",
    "ebitda_to_ev": "operating profit over enterprise value",
    "fcf_yield": "free cash flow over price",
    "sales_to_price": "revenue over price",
    "gross_profitability": "gross profit over assets (Novy-Marx 2013)",
    "cash_op_profitability": "cash operating profit over assets",
    "roce": "return on capital employed",
    "accruals": "the non-cash share of earnings (Sloan 1996)",
    "asset_growth": "year-on-year change in total assets",
    "net_issuance": "shares issued net of buybacks, bonus-adjusted",
}


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


def _quality_from_features(feats, symbols, cfg, min_coverage: float):
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
        # PASSED IN, not re-derived from `cfg`. This called `_min_coverage(cfg)`
        # while `cfg` here is the QUALITY FACTOR's config and `_min_coverage`
        # reads `stage4_core_score.min_name_factor_coverage` -- an attribute
        # that block does not have. Every call raised AttributeError.
        #
        # It never fired because the only caller is guarded by `if not
        # feats.empty`, and `fundamentals.parquet` has been frozen since
        # 2025-03-11, so no live date has had public fundamentals to reach it.
        # Repairing the fundamentals feed would have turned a silent dead branch
        # into a crash on the daily run. Found by replaying 2020-03-23, where
        # the fundamentals DO exist.
        if float(series.notna().mean()) < min_coverage:
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


def _redundancy(frame: pd.DataFrame, cfg,
                members: Optional[pd.DataFrame] = None) -> RedundancyReport:
    """Measure overlap between the things that carry coefficients.

    ``frame`` is the fitted block -- families under the current model.
    ``members`` is the individual factors underneath them, reported as context
    because their collinearity is what the aggregation exists to absorb, and a
    reader who sees only the clean family numbers should be able to check that.
    """
    cutoff = fv(cfg.redundancy.max_abs_spearman)
    pairs = spearman_pairs(frame) if frame.shape[1] >= 2 else {}
    breaches = [
        (k.split("|")[0], k.split("|")[1], round(v, 4))
        for k, v in pairs.items()
        if abs(v) > cutoff
    ]
    notes: List[str] = []
    if not pairs:
        notes.append(
            "fewer than two scored columns carried enough values to correlate; "
            "overlap not measurable this run"
        )
    if members is not None and members.shape[1] >= 2:
        inner = spearman_pairs(members)
        worst = sorted(inner.items(), key=lambda kv: -abs(kv[1]))[:4]
        if worst:
            notes.append(
                "Within-family overlap (absorbed by averaging, not a breach): "
                + ", ".join(f"{k.replace('|', '/')} {v:+.2f}" for k, v in worst)
            )
    return RedundancyReport(
        pairwise_spearman={k: round(v, 4) for k, v in pairs.items()},
        breaches=breaches,
        cutoff=cutoff,
        action_taken=str(v(cfg.redundancy.on_breach)),
        notes=notes,
    )


#: Sources for the fitted factors, so a card citing them can be checked.
_MODEL_CITE = {
    "mom_12_1": "Jegadeesh & Titman (1993)",
    "mom_6_1": "Jegadeesh & Titman (1993)",
    "mom_3_1": "Jegadeesh & Titman (1993)",
    "resid_reversal": "Blitz, Huij, Lansdorp & Martens (2013)",
    # Families. The card names the family, so the citation has to as well.
    "mom": "Jegadeesh & Titman (1993); George & Hwang (2004); Blitz, Huij & Martens (2011)",
    "reversal": "Blitz, Huij, Lansdorp & Martens (2013)",
    "lottery": "Bali, Cakici & Whitelaw (2011); Ang, Hodrick, Xing & Zhang (2006)",
    "skew": "Bali, Cakici & Whitelaw (2011); Boyer, Mitton & Vorkink (2010)",
    "beta": "Frazzini & Pedersen (2014); Agarwalla, Jacob, Varma & Vasudevan (2014) for India",
    "drawdown": "no literature analogue; drawdown depth is not a standard cross-sectional factor",
    "delivery": "NSE delivered-quantity data; no direct analogue outside India",
    "value": "Fama & French (1992); Basu (1977)",
    "quality": "Novy-Marx (2013); Sloan (1996); Cooper, Gulen & Schill (2008)",
    "idio_vol": "Ang, Hodrick, Xing & Zhang (2006)",
    "idio_skew": "Bali, Cakici & Whitelaw (2011)",
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




#: Reasons that mean "not enough data yet" rather than "something broke".
_BENIGN_REASONS = (
    "sessions of history",
    "usable training rows",
    "no price rows",
    "complete feature set",
    "could not be computed",
)


