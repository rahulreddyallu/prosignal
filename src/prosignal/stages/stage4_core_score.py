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
from typing import Dict, List, Optional, Tuple

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
from ..features.crosssec import liquidity_mask
from ..features import v3 as v3feat
from ..features import v4 as v4feat
from ..features import v3_factors as v3fac
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

__all__ = ["run", "STAGE_NAME", "theme_effective_weights"]

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
                          v3_scored=None):
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

    if source == "v4_composite":
        # THE SHIPPED SCORER since the 2026-09-05 epoch. v3 minus seven factors
        # that an independent split-half nominated in BOTH halves of the panel;
        # weights, signs, themes and blend are otherwise v3's, untouched.
        # dIC +0.0066 at Newey-West t +2.37 over 45 purged, embargoed CPCV folds,
        # 96% of folds improved, fifth percentile still positive. That is
        # STABILITY evidence, not a sealed holdout -- v3's two windows were
        # earned by the 22-factor set and do not transfer. See features/v4.py.
        if v3_scored is None or "score" not in getattr(v3_scored, "columns", []):
            raise RankingUnavailable(
                "stage4_core_score.ranking.source is 'v4_composite' and the v4 "
                "block did not build. Falling back to another scorer would "
                "issue signals from a model that was not the one measured.")
        ranked = v3_scored["score"].dropna()
        covered = ranked.reindex(composite_raw.index).dropna()
        floor = max(int(0.6 * len(composite_raw)), 20)
        if len(covered) < floor:
            raise RankingUnavailable(
                f"the v4 composite covers {len(covered)} of "
                f"{len(composite_raw)} scoreable names, under the {floor} "
                f"floor. A ranking built on a minority of the universe is a "
                f"ranking of that minority.")
        nth = v3_scored["n_themes"].reindex(covered.index)
        eff = theme_effective_weights(v3_scored.loc[covered.index])
        notes.append(
            f"Book ordered by the v4 composite: {len(v4feat.ALL_FACTORS)} factors "
            f"in {len(v4feat.THEMES)} themes -- v3 minus "
            f"{', '.join(v4feat.REMOVED)}. WEIGHTS AS APPLIED TODAY "
            f"(declared -> effective, coverage): "
            f"{', '.join(f'{t} {v4feat.THEMES[t].weight:.0%}->{eff[t][0]:.0%} ({eff[t][1]:.0%})' for t in v4feat.THEMES)}. "
            f"Median name scored on {nth.median():.0f} of {len(v4feat.THEMES)} "
            f"themes. THE EVIDENCE IS STABILITY, NOT A SEALED HOLDOUT: composite "
            f"rank IC +0.0541 -> +0.0607, delta +0.0066 at Newey-West t +2.37 "
            f"across 45 purged and embargoed CPCV folds, 96% of folds improved. "
            f"v3's two sealed windows were earned by the 22-factor set and do "
            f"NOT transfer to this one. The forward test re-registered with this "
            f"epoch is what will grade it. The RANKING is what improved; the "
            f"concentrated book is unchanged and still unevidenced.")
        try:
            from .. import v3_monitor as v3mon
            notes.extend(v3mon.review_cross_section(v3_scored.loc[covered.index]))
        except Exception as exc:
            log.warning("theme influence check did not run",
                        extra={"error": str(exc)})
        _resid = v3_scored.attrs.get("sector_residual_share")
        if isinstance(_resid, float) and _resid == _resid:
            notes.append(
                f"Sector neutralisation covers {1.0 - _resid:.0%} of the ranked "
                f"names; the other {_resid:.0%} are ranked inside ONE residual "
                f"group against each other, which is not neutralisation.")
        return covered, source

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
        # WHAT THE BLEND ACTUALLY APPLIED, measured on this cross-section. The
        # note used to print the DECLARED weights, which no name is scored at:
        # weights renormalise over the themes a name has, so a theme covering a
        # fifth of the universe donates four fifths of its weight to its
        # neighbours. Declared 40/19/19/11/11 runs at roughly 48/4/21/13/13.
        eff = theme_effective_weights(v3_scored.loc[covered.index])
        notes.append(
            f"Book ordered by the v3 composite: {len(v3feat.ALL_FACTORS)} factors "
            f"in {len(v3feat.THEMES)} themes. WEIGHTS AS APPLIED TODAY "
            f"(declared -> effective, coverage): "
            f"{', '.join(f'{t} {v3feat.THEMES[t].weight:.0%}->{eff[t][0]:.0%} ({eff[t][1]:.0%})' for t in v3feat.THEMES)}. "
            f"Effective is the population average after weights renormalise over "
            f"the themes each name has; it is what ranked the book, and the "
            f"declared number is not. Each theme is combined on its own and "
            f"blended with weights capped at 40%, floored at 6% and additionally "
            f"capped at the theme's coverage. Median name scored on "
            f"{nth.median():.0f} of {len(v3feat.THEMES)} themes. Sealed holdouts, "
            f"one run each: rank IC +0.049 (t 3.69) on 2025-03..2026-08 and "
            f"+0.036 (t 3.83) on 2021-07..2022-12, with every theme positive out "
            f"of sample on both. The RANKING is what generalised; a ten-name book "
            f"at these transaction costs did not -- see CHANGELOG.md.")
        _resid = v3_scored.attrs.get("sector_residual_share")
        if isinstance(_resid, float) and _resid == _resid:
            notes.append(
                f"Sector neutralisation covers {1.0 - _resid:.0%} of the ranked "
                f"names. The other {_resid:.0%} -- no NSE Industry, or a sector "
                f"holding fewer than {v3feat.MIN_SECTOR_NAMES} of today's names "
                f"-- are ranked inside ONE residual group against each other, "
                f"which is not neutralisation. Widening the sector map shrinks "
                f"this; changing how the residual is ranked would be a model "
                f"change and is not done here.")
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




def build_v3_block(store, calendar, symbols, as_of, sectors, cfg,
                   v9r_mode=False, v4_mode=False):
    """The v3 two-level thematic score for ``as_of``.

    Returns (raw factor values, scored frame, error). Reads only sessions at or
    before ``as_of``; the fundamental block is as-of joined on DISCLOSURE dates,
    never on period ends.
    """
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
        log.warning("delivery unavailable; the ownership theme will be absent",
                    extra={"error": str(exc)})

    # The market as it stood: equal-weight return of the names being scored.
    bench = close.mean(axis=1)
    bench_ret = bench / bench.shift(1) - 1.0

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
    if v4_mode:
        # v4 IS v3 MINUS SEVEN FACTORS and shares its blend, so everything
        # downstream -- the card, the monitor, the ledger -- reads the same
        # columns. `features/v4.py` carries what the seven were and why.
        scored = v4feat.score_frame(raw, sectors,
                                    min_themes=int(iv(cfg.ranking.v3_min_themes)))
    else:
        scored = v3feat.score_frame(raw, sectors, min_themes=int(iv(cfg.ranking.v3_min_themes)))
    # THE RESIDUAL BUCKET, MEASURED WHERE IT BITES. `sector_neutral_rank` sends
    # every name with no usable sector -- and every name in a sector too small
    # to rank inside -- to one `__RESID__` group and ranks it against the other
    # residuals. That is not neutralisation, and the share it covers is a
    # property of the sector map on the day, so it is measured per run rather
    # than quoted. Stage 4 attaches it to the frame for the note to read.
    scored.attrs["sector_residual_share"] = _residual_share(raw.index, sectors)
    return raw, scored, None


def _residual_share(index, sectors) -> float:
    """Share of scored names that `sector_neutral_rank` cannot neutralise.

    Counts both halves of the problem: names the map has no sector for, and
    names whose sector holds fewer than `MIN_SECTOR_NAMES` of today's
    cross-section, because both land in the same residual group.
    """
    if not sectors:
        return 1.0
    sec = pd.Series({s: sectors.get(s) for s in index}, dtype="object")
    bad = sec.isna() | sec.astype(str).isin(("", "Unknown", "nan", "None"))
    counts = sec[~bad].value_counts()
    small = set(counts[counts < v3feat.MIN_SECTOR_NAMES].index)
    return float((bad | sec.isin(small)).mean()) if len(sec) else 1.0




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
    if _src in ("v3_composite", "v4_composite", "v9r_core"):
        v3_raw, v3_scored, v3_err = build_v3_block(
            store, calendar, symbols, as_of, sectors, cfg,
            v9r_mode=(_src == "v9r_core"), v4_mode=(_src == "v4_composite"))
        if v3_err:
            notes.append(f"v3 block unavailable: {v3_err}")

    composite_raw, ranking_source = _apply_ranking_policy(
        composite_raw, model_features, cfg, notes, v3_scored=v3_scored)

    # WHICH THEME TABLE THE CARD AND THE MONITOR READ. v4 is v3 minus seven
    # factors, so the removed factors have no `_r` column and iterating v3's
    # table would raise on the first name. One lookup, used everywhere below.
    active_themes = (v4feat.THEMES if ranking_source == "v4_composite"
                     else v3feat.THEMES)

    # THE ABSOLUTE FLOOR, computed here and enforced at entry. Names that fail
    # it stay in the ranking -- they are holdable and they belong on a watchlist
    # -- and cannot be bought. When fewer clear it than there are slots, the
    # book holds cash, which is how NO TRADE happens.
    _fl = getattr(cfg, "absolute_floor", None)
    _floor_on = bool(bv(_fl.enabled)) if _fl is not None else False
    _floor_ma = int(iv(_fl.above_ma_sessions)) if _fl is not None else 200
    _floor_min_themes = int(iv(_fl.min_positive_themes)) if _fl is not None else 3
    _floor_scope = str(v(_fl.applies_to)) if _fl is not None else "entries"
    _above_ma = {}
    if _floor_on and not closes.empty:
        ma = closes.rolling(_floor_ma, min_periods=int(_floor_ma * 0.75)).mean()
        if len(ma):
            _above_ma = (closes.iloc[-1] > ma.iloc[-1]).to_dict()

    composite_unit = rank_to_unit_interval(composite_raw)
    percentile = composite_unit * 100.0
    order = composite_raw.sort_values(ascending=False)

    scores: List[StockScore] = []
    for rank, sym in enumerate(order.index, start=1):
        factors = {}
        if ranking_source in ("v3_composite", "v4_composite") \
                and v3_scored is not None \
                and sym in v3_scored.index:
            # THE CARD EXPLAINS THE NUMBER IT PRINTS, AT BOTH LEVELS. Theme rows
            # carry the sub-score and the weight it was blended at and sum to
            # the composite; factor rows carry the ranks the theme was built
            # from. "momentum +0.42" does not say which momentum moved.
            # EFFECTIVE WEIGHT, NOT DECLARED. The blend renormalises over the
            # themes THIS name actually has, so a name missing `quality` runs
            # momentum at 40/(1-0.1899) = 49.4%, not 40%. The card printed the
            # declared number and the reader had no way to see the difference;
            # `contribution` was already correct, which made the two printed
            # numbers fail to multiply out. Both are carried now.
            _present = [t for t in active_themes
                        if pd.notna(v3_scored.at[sym, t + "_sub"])]
            _den = sum(active_themes[t].weight for t in _present) or 1.0
            for tname, th in sorted(
                    active_themes.items(),
                    key=lambda kv: -(abs(_f(v3_scored.at[sym, kv[0] + "_contrib"]) or 0.0))):
                _eff = (th.weight / _den) if tname in _present else 0.0
                factors[tname] = FactorScore(
                    name=tname,
                    raw_value=None,
                    standardised=_f(v3_scored.at[sym, tname + "_sub"]),
                    weight=round(_eff, 5),
                    nominal_weight=round(th.weight, 5),
                    contribution=_f(v3_scored.at[sym, tname + "_contrib"]),
                    available=pd.notna(v3_scored.at[sym, tname + "_sub"]),
                    evidence_tier="v3_theme",
                    citation=f"theme sub-score, oriented at {th.horizon} sessions",
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

    # THE MONITOR FOLLOWS THE SCORER. `model_features` is None on every run
    # since the fitted ranker was removed, so the old call fell through to the
    # legacy `frame` and never saw a factor the book is ordered by.
    redundancy = None
    if ranking_source in ("v3_composite", "v4_composite", "v9r_core"):
        redundancy = _v3_redundancy(v3_scored, cfg, themes=active_themes)
    if redundancy is None:
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
        prediction_dispersion=(float(getattr(model, "dispersion", 0.0))
                               if model is not None else None),
        typical_dispersion=(float(getattr(model, "train_dispersion", 0.0))
                            if model is not None else None),
        # The meta-label veto is gone (AUC 0.4996, shipped disabled), so there is
        # no win probability and nothing to explain its absence. The card reads
        # None for both rather than carrying a field that can only say "off".
        win_probability=None,
        win_probability_unavailable=None,
        weighting_mode=str(v(cfg.weighting_mode)),
        standardisation=method,
        effective_weights={k: round(v, 4) for k, v in effective.items()},
        dropped_factors=dropped,
        ranked_scores=scores,
        redundancy=redundancy,
        universe_size=len(symbols),
        notes=notes,
    )



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


def theme_effective_weights(v3_scored: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """``{theme: (effective weight, coverage)}`` averaged over the scored names.

    The v3 blend divides each theme's weight by the total weight of the themes
    the NAME has, so a theme that is absent does not score zero -- it hands its
    weight to its neighbours. Averaged over the cross-section that makes the
    declared weight vector wrong in both directions at once: thin themes read
    far lower than declared and thick ones far higher.

    Reported per run rather than quoted from a study, because coverage moves
    with the fundamentals feed and the number is only true of the day it was
    measured on.
    """
    out: Dict[str, Tuple[float, float]] = {}
    if v3_scored is None or getattr(v3_scored, "empty", True):
        return {t: (th.weight, float("nan")) for t, th in v3feat.THEMES.items()}
    present = {t: v3_scored[t + "_sub"].notna().to_numpy()
               for t in v3feat.THEMES if t + "_sub" in v3_scored.columns}
    if not present:
        return {t: (th.weight, float("nan")) for t, th in v3feat.THEMES.items()}
    names = list(present)
    M = np.column_stack([present[t] for t in names])
    W = np.array([v3feat.THEMES[t].weight for t in names], dtype="float64")
    den = (M * W).sum(axis=1)
    ok = den > 0
    for j, t in enumerate(names):
        eff = np.where(M[:, j] & ok, W[j] / np.maximum(den, 1e-12), 0.0)
        out[t] = (float(eff[ok].mean()) if ok.any() else float("nan"),
                  float(M[:, j].mean()))
    for t, th in v3feat.THEMES.items():
        out.setdefault(t, (th.weight, float("nan")))
    return out


def _v3_blocks(v3_scored: pd.DataFrame,
               themes: Optional[dict] = None) -> Tuple[Optional[pd.DataFrame],
                                                       Optional[pd.DataFrame]]:
    """The two levels of the shipped scorer, ready for the redundancy check.

    Returns ``(themes, factors)``: the five theme sub-scores, which are what the
    blend weights are applied to, and the twenty-two factor ranks underneath
    them, SIGN-ORIENTED so a -1 factor lining up with a +1 factor reads as the
    positive alignment it is rather than a spurious negative.

    Orientation is the whole point. Unoriented, `ulcer_120` (sign -1) against
    `prox_52w` (sign +1) reports -0.78 and looks like diversification; oriented,
    it reports +0.78 and is the same bet twice.
    """
    if v3_scored is None or getattr(v3_scored, "empty", True):
        return None, None
    table = v3feat.THEMES if themes is None else themes
    theme_block = pd.DataFrame(index=v3_scored.index)
    for tname in table:
        col = tname + "_sub"
        if col in v3_scored.columns:
            theme_block[tname] = v3_scored[col]
    factors = pd.DataFrame(index=v3_scored.index)
    for tname, th in table.items():
        for fname, sign in th.factors:
            col = fname + "_r"
            if col in v3_scored.columns:
                factors[fname] = v3_scored[col] * sign
    return (theme_block if theme_block.shape[1] >= 2 else None,
            factors if factors.shape[1] >= 2 else None)


def _v3_redundancy(v3_scored: pd.DataFrame, cfg,
                   themes: Optional[dict] = None) -> Optional[RedundancyReport]:
    """Redundancy for the scorer that actually orders the book.

    WHY THIS EXISTS SEPARATELY. `_redundancy` was written for the fitted model
    and is fed `model_features`, which has been `None` on every run since the
    fitted ranker was removed -- so it fell through to the legacy standardised
    frame and NO SHIPPED v3 FACTOR PAIR HAS EVER BEEN CHECKED. A monitor that
    inspects an empty frame reports no breaches, which reads exactly like a
    clean bill of health.

    WHAT COUNTS AS A BREACH HERE IS NOT WHAT COUNTS THERE. Two factors inside
    one theme are SUPPOSED to overlap: the theme averages them, and the average
    is what carries a weight. Two factors in DIFFERENT themes are a different
    matter -- the 40% cap is applied per theme, so a momentum factor sitting in
    the `risk` theme is momentum exposure the cap cannot see. Measured on the
    research panel, `prox_52w`/`ulcer_120` correlate +0.78 across the
    momentum/risk boundary, which makes real momentum exposure 48% of the blend
    plus most of risk's 13%.

    So: cross-theme factor pairs above the cutoff are BREACHES, within-theme
    pairs are reported as absorbed, and the theme sub-scores are checked in
    their own right because they are what the weights multiply.
    """
    table = v3feat.THEMES if themes is None else themes
    theme_of = {f: t for t, th in table.items() for f in th.names}
    theme_block, factors = _v3_blocks(v3_scored, themes=table)
    if theme_block is None and factors is None:
        return None
    cutoff = fv(cfg.redundancy.max_abs_spearman)
    notes: List[str] = []

    theme_pairs = spearman_pairs(theme_block) if theme_block is not None else {}
    breaches = [(k.split("|")[0], k.split("|")[1], round(rho, 4))
                for k, rho in theme_pairs.items() if abs(rho) > cutoff]

    factor_pairs = spearman_pairs(factors) if factors is not None else {}
    cross, within = [], []
    for key, rho in factor_pairs.items():
        a, b = key.split("|")
        if abs(rho) <= cutoff:
            continue
        same = theme_of.get(a) == theme_of.get(b)
        (within if same else cross).append((a, b, round(rho, 4)))

    for a, b, rho in sorted(cross, key=lambda r: -abs(r[2])):
        breaches.append((a, b, rho))
        notes.append(
            f"CROSS-THEME OVERLAP: {a} ({theme_of[a]}) and {b} "
            f"({theme_of[b]}) correlate {rho:+.2f} oriented. The 40% "
            f"cap is applied per theme and cannot see this; the exposure is "
            f"carried twice.")
    if within:
        worst = sorted(within, key=lambda r: -abs(r[2]))[:4]
        notes.append(
            "Within-theme overlap (absorbed by the theme average, not a "
            "breach): " + ", ".join(f"{a}/{b} {rho:+.2f}" for a, b, rho in worst))
    if not theme_pairs and not factor_pairs:
        notes.append("fewer than two v3 columns carried enough values to "
                     "correlate; overlap not measurable this run")

    combined = {f"theme:{k}": round(v, 4) for k, v in theme_pairs.items()}
    combined.update({k: round(v, 4) for k, v in factor_pairs.items()})
    return RedundancyReport(
        pairwise_spearman=combined,
        breaches=breaches,
        cutoff=cutoff,
        action_taken=str(v(cfg.redundancy.on_breach)),
        notes=notes,
    )


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


