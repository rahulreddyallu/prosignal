"""Cross-sectional return-rank model used by Stage 4.

A ridge regression on fourteen cross-sectionally ranked features, refitted on
every run from history strictly before the decision date, predicting the rank of
the forward 21-session return.

It is here rather than in a research folder because it was measured against the
incumbent composite under purged walk-forward on 8.8 years and won:

    composite (momentum + sector RS)  IC +0.025  t 1.28   excess +0.14%/mo t 0.30
    ridge                             IC +0.052  t 3.64   excess +1.11%/mo t 3.44

The composite's excess return over an equal-weight benchmark is not
distinguishable from zero. Elastic Net additionally beats the composite
head-to-head on a scale-invariant loss at p = 0.045.

Training uses only dates whose 21-session label window closed before ``as_of``,
so no observation used in the fit can know anything about the decision date.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import json

from ..core.logging import get_logger
from ..core.memory import release_memory
from .crosssec import FEATURES, MIN_LOOKBACK, build_panel
from .exits import ExitRules, rules_from_config
from .labels import BarrierSpec
from .fundamental_factors import available_as_of, build_fundamental_panel, winsorise
from .fundamentals import FEATURE_NAMES as FUND_NAMES, compute_features
from .famamacbeth import (FMResult, SIGNIFICANCE_FLOOR, fama_macbeth,
                          gated_shrink, is_degenerate)
from .metalabel import MetaModel, fit_meta_out_of_sample, shortlist
from .linear import predict, ridge_fit

__all__ = ["CrossSectionalModel", "fit_predict", "load_cached", "save_cache",
           "score_with", "today_features", "contributions", "standardised_features"]

log = get_logger(__name__)

#: Fundamental factors, ranked cross-sectionally like everything else. These
#: are the only inputs not derived from price and volume.
#:
#: The value family only. Measured on the point-in-time universe over the 18
#: periods where statement data exists, adding each family to the price
#: baseline in turn:
#:
#:     baseline (price)      IC +0.0568 (t 1.81)  excess +0.088% (t 0.09)
#:     + value               IC +0.0623 (t 1.80)  excess +0.913% (t 0.91)
#:     + quality             IC +0.0466 (t 1.59)  excess -0.287% (t -0.33)
#:     + growth              IC +0.0505 (t 1.86)  excess -0.239% (t -0.28)
#:     + leverage            IC +0.0541 (t 1.80)  excess +0.089% (t 0.09)
#:     fundamentals alone    IC +0.0072 (t 0.55)
#:
#: Value is the only family that improves both measures; quality and growth
#: make the model worse. None of it is significant -- 18 periods is what the
#: statement history supports, and t = 1.80 on IC and 0.91 on excess do not
#: establish an edge. The family is carried because it is strictly better than
#: what it replaces, not because it is proven: the previous fundamental block
#: was five NSE-derived columns whose data stopped in March 2025, leaving them
#: with a standard deviation of 0.024 against 0.57 for the price factors and
#: one of them exactly constant. Dead columns are worse than weak ones.
#:
#: Earlier releases quoted IC +0.121 (t 5.09) and Diebold-Mariano p = 0.001 for
#: this block. Those were measured on a universe built from today's NIFTY 200
#: membership projected backwards, which is worth +5.00% per 63 sessions on its
#: own. They do not survive a point-in-time universe and are withdrawn.
FUNDAMENTAL_FEATURES = [
    # value
    "earnings_yield", "book_to_price", "ebitda_to_ev", "fcf_yield",
    "sales_to_price",
    # quality and profitability. Novy-Marx (2013) for gross profitability;
    # Fama & French (2018) concede cash-based operating profitability dominates
    # their own accrual measure. Accruals is Sloan (1996), asset growth is
    # Cooper, Gulen & Schill (2008), net issuance catches QIPs and promoter
    # dilution. The last three enter the quality family NEGATED -- see
    # NEGATED_IN_FAMILY.
    "gross_profitability", "cash_op_profitability", "roce",
    "accruals", "asset_growth", "net_issuance",
    # Scale. Every other factor picks size up implicitly; carrying it lets the
    # fit price it rather than absorb it.
    "log_mcap",
]

#: Members whose natural sign is the opposite of what the family means. The
#: family is built so a HIGHER composite is a BETTER name, and the fit is free
#: to price it either way -- but a member entering with the wrong sign would
#: cancel its neighbours rather than reinforce them.
NEGATED_IN_FAMILY = frozenset({
    "accruals_r", "asset_growth_r", "net_issuance_r",
})

FEATURE_COLUMNS = [f + "_r" for f in FEATURES] + [f + "_r" for f in FUNDAMENTAL_FEATURES]

#: Ridge penalty. Fixed rather than searched per run: tuning it against the same
#: history that scores it is how a validated result becomes an overfit one.
#:
#: 20,000 rather than 10: with 24 correlated factors a light penalty lets the
#: fit chase noise. Chosen on the selection period alone and then left; on the
#: holdout the same feature set scored IC +0.045 at this value against +0.015
#: at 10.
ALPHA = 20_000.0          # default only; stage 4 passes the configured value

#: Label horizon in sessions, matching the holding period the engine plans for.
#:
#: Measured on a holdout never used for selection, the model is better at every
#: horizon tested up to 63 sessions and the curve is still rising:
#:
#:     H=21  IC +0.044 (t 2.18)  excess +1.11%/period  DSR 0.645  net +7.50%/yr
#:     H=42  IC +0.078 (t 3.38)  excess +2.40%/period  DSR 0.670  net +11.45%/yr
#:     H=63  IC +0.098 (t 3.72)  excess +4.06%/period  DSR 0.875  net +14.19%/yr
#:
#: Longer horizons win twice: the cross-sectional signal is stronger and the
#: turnover charge is smaller. The holding period below must match, or the
#: engine would exit before the return it is forecasting has accrued.
HORIZON = 63              # default only; stage 4 passes the configured value

#: Minimum training rows. Below this the fit is noise and the model abstains
#: rather than returning a confident-looking number from nothing.
MIN_TRAIN_ROWS = 600

#: A factor scored on a minority of NAMES ranks the rest by neutral fill, which
#: is not a ranking. Stage 4 already states this rule and enforces it on the
#: hand-weighted composite -- the fitted model, which is the one that actually
#: ranks, imputed instead.
#:
#: Measured WITHIN A DATE, which is the question the rule asks. A single number
#: over the whole panel conflated two different failures and reported the wrong
#: one: after the fundamentals ingest took symbol coverage from 26% to 100%, the
#: value factors read 40% "coverage" and were dropped for ranking too few names
#: -- when in fact they rank 73% of the universe on every date they exist.
MIN_FACTOR_COVERAGE = 0.60

#: The other failure. A factor the feed cannot serve for the earlier part of the
#: panel is not badly covered, it is ABSENT, and fitting it beside factors that
#: span the whole period means either imputing it for those dates -- the bug
#: this pair of tests exists to prevent -- or shortening the panel for everyone.
#:
#: Measured after the ingest: the value and quality factors exist on 35 of 88
#: panel dates. yfinance serves about five years of statements and TTM needs
#: four quarters of it, so the first usable date is 2023-06 against a panel
#: starting 2018-12. Shortening the panel to 35 overlapping dates -- roughly
#: eight independent 63-session windows -- to fit seven coefficients instead of
#: five is a worse trade than dropping two families.
#:
#: This is a DATA-DEPTH bar, not a quality bar. A fundamentals source reaching
#: back as far as the price history clears it and the families enter on their
#: own.
MIN_FACTOR_DATE_SPAN = 0.60

#: Training window in sessions. Bounded rather than "all history" for memory:
#: the full 2,206-session panel peaked at 615 MB on a 512 MB instance. 1,500
#: sessions still yields ~45 training periods for 14 features, which is the
#: span the earliest validated walk-forward folds trained on.
MAX_TRAIN_SESSIONS = 3000

#: Refit cadence. The coefficients are a slow-moving object -- refitting daily
#: costs 17 seconds and 300 MB to move them marginally, and invites the fit to
#: chase noise. Cached coefficients are reused until this many sessions pass.
REFIT_EVERY_SESSIONS = 21


class CrossSectionalModel:
    """A fitted ridge model plus the provenance needed to reproduce it."""

    def __init__(self, coef: Dict[str, float], n_train: int, train_end: dt.date,
                 features: Optional[List[str]] = None) -> None:
        self.coef = coef
        self.n_train = n_train
        self.train_end = train_end
        #: The columns this model was fitted on, IN FIT ORDER -- `mu` and `sd`
        #: are positional and align to it. Not always every column in
        #: FEATURE_COLUMNS: a factor the feed cannot serve for most of the
        #: universe is dropped rather than neutral-filled, so the model has to
        #: say which ones it actually used.
        self.features = list(features) if features else list(coef.keys())

    def summary(self) -> str:
        top = sorted(self.coef.items(), key=lambda kv: -abs(kv[1]))[:4]
        parts = ", ".join(f"{k.removesuffix('_f').removesuffix('_r')} {v:+.3f}"
                          for k, v in top)
        how = getattr(self, "estimator", "ridge")
        if how == "fama_macbeth":
            live = sum(1 for v in self.coef.values() if abs(v) > 1e-12)
            n_dates = getattr(self, "fm_n_dates", 0)
            head = (f"Fama-MacBeth over {n_dates} cross-sections, "
                    f"{live} of {len(self.coef)} themes cleared |t| >= "
                    f"{SIGNIFICANCE_FLOOR:g}")
        else:
            head = f"ridge on {len(self.coef)} cross-sectional features"
        return (f"{head}, {self.n_train} training rows to {self.train_end}; "
                f"strongest: {parts}")



def _meta_blob(meta) -> Optional[Dict[str, object]]:
    """The veto classifier, flattened for the cache."""
    if meta is None:
        return None
    return {
        "features": list(meta.features),
        "coef": [float(c) for c in meta.coef],
        "intercept": float(meta.intercept),
        "mu": [float(x) for x in meta.mu],
        "sd": [float(x) for x in meta.sd],
        "n_train": int(meta.n_train),
        "base_rate": float(meta.base_rate),
    }


def _meta_from_blob(blob) -> Optional["MetaModel"]:
    if not blob:
        return None
    try:
        return MetaModel(
            features=list(blob["features"]),
            coef=np.array(blob["coef"], dtype="float64"),
            intercept=float(blob["intercept"]),
            mu=np.array(blob["mu"], dtype="float64"),
            sd=np.array(blob["sd"], dtype="float64"),
            n_train=int(blob["n_train"]),
            base_rate=float(blob["base_rate"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_cached(path, as_of: dt.date,
                refit_every_sessions: Optional[int] = None,
                estimator: Optional[str] = None) -> Optional[CrossSectionalModel]:
    """Coefficients from a recent fit, or None when absent or stale.

    ``refit_every_sessions`` comes from the config. It defaulted to the module
    constant and nothing passed it, so stage4_core_score.model_refit_every_sessions
    was declared, validated on every startup and ignored -- editing it changed
    nothing. The two happened to agree at 21, which is why it was invisible.
    """
    try:
        if not path.is_file():
            return None
        every = int(refit_every_sessions if refit_every_sessions is not None
                    else REFIT_EVERY_SESSIONS)
        blob = json.loads(path.read_text(encoding="utf-8"))
        fitted = dt.date.fromisoformat(blob["fitted_for"])
        if (as_of - fitted).days > every * 2:
            return None
        # A stored model may legitimately carry FEWER factors than the code
        # declares: one the feed could not serve for most of the universe is
        # dropped rather than neutral-filled. What must not happen is a stored
        # factor the code no longer knows, which means the definitions moved.
        stored = list(blob.get("features") or blob["coef"].keys())
        # Families now, individual factors before. Either is a legitimate stored
        # shape; a name from NEITHER means the definitions moved and the stored
        # coefficients describe a different model.
        known = set(FEATURE_COLUMNS) | set(FAMILY_COLUMNS)
        if not set(stored) <= known:
            return None                      # feature set changed; refit
        if not set(stored) <= set(FAMILY_COLUMNS):
            return None                      # pre-family fit; refit on families
        if sorted(stored) != sorted(blob["coef"]):
            return None                      # blob is internally inconsistent
        # A model fitted by a different estimator is a different model. Blobs
        # written before the field existed were ridge fits.
        if estimator is not None and str(blob.get("estimator", "ridge")) != str(estimator):
            return None
        m = CrossSectionalModel(
            coef=blob["coef"], n_train=int(blob["n_train"]),
            train_end=dt.date.fromisoformat(blob["train_end"]),
            features=stored,
        )
        m.dispersion = float(blob.get("dispersion") or 0.0)
        m.train_dispersion = float(blob.get("train_dispersion") or 0.0)
        m.mu = np.array(blob["mu"], dtype="float64")
        m.sd = np.array(blob["sd"], dtype="float64")
        m.intercept = float(blob["intercept"])
        m.estimator = str(blob.get("estimator", "ridge"))
        m.meta = _meta_from_blob(blob.get("meta"))
        m.meta_prob = None                 # scored per run, never cached
        m.fm_t_stat = dict(blob.get("fm_t_stat") or {})
        m.fm_lambda = dict(blob.get("fm_lambda") or {})
        m.fm_n_dates = int(blob.get("fm_n_dates", 0) or 0)
        return m
    except (OSError, ValueError, KeyError):
        return None


def read_cached_coefficients(path) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """Coefficients currently live, and the date they were trained to.

    Deliberately does not go through load_cached: staleness and feature-set
    checks are right for scoring and wrong here, where the question is only
    what a proposed refit would be replacing.
    """
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
        return dict(blob.get("coef") or {}), blob.get("train_end")
    except (OSError, ValueError, KeyError, TypeError):
        return None, None


def archive_cache(path, keep: int = 10) -> Optional[str]:
    """Copy the live coefficients aside before they are overwritten.

    Without this a bad refit is unrecoverable: the file it replaced is gone and
    the only way back is a full retrain, which reproduces whatever upstream
    problem caused the bad fit in the first place.
    """
    if not path.is_file():
        return None
    versions = path.parent / f"{path.stem}_versions"
    versions.mkdir(parents=True, exist_ok=True)
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
        stamp = str(blob.get("fitted_for") or "unknown")
    except (OSError, ValueError):
        stamp = "unknown"
    target = versions / f"{path.stem}_{stamp}.json"
    target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    existing = sorted(versions.glob(f"{path.stem}_*.json"))
    for old_file in existing[:-keep]:
        try:
            old_file.unlink()
        except OSError:
            pass
    return str(target)


def save_cache(path, model: CrossSectionalModel, as_of: dt.date) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "fitted_for": as_of.isoformat(),
        "train_end": model.train_end.isoformat(),
        "n_train": model.n_train,
        # IN FIT ORDER. `mu` and `sd` are positional and align to it, so a dict
        # whose iteration order changed would silently mis-standardise.
        "features": list(model.features),
        "dispersion": float(getattr(model, "dispersion", 0.0) or 0.0),
        "train_dispersion": float(getattr(model, "train_dispersion", 0.0) or 0.0),
        "coef": model.coef,
        # The estimator is part of what this model IS, not a note about how it
        # was made. A ridge blob reloaded under Fama-MacBeth would score today
        # with coefficients no current code path would produce.
        "estimator": str(getattr(model, "estimator", "ridge")),
        "fm_t_stat": {k: float(v) for k, v in
                      (getattr(model, "fm_t_stat", {}) or {}).items()
                      if v == v},
        "fm_lambda": {k: float(v) for k, v in
                      (getattr(model, "fm_lambda", {}) or {}).items()},
        "fm_n_dates": int(getattr(model, "fm_n_dates", 0) or 0),
        # The veto travels with the model. Without this the cheap path -- 20 of
        # every 21 sessions -- reloads a model with no classifier, every new
        # entry scores as unknown, and a gate whose rule is "unknown is not
        # approved" silently refuses the whole book.
        "meta": _meta_blob(getattr(model, "meta", None)),
        "dropped_for_coverage": getattr(model, "dropped_for_coverage", {}) or {},
        "mu": list(map(float, model.mu)),
        "sd": list(map(float, model.sd)),
        "intercept": model.intercept,
    }), encoding="utf-8")


def apply_family_multipliers(
    frame: pd.DataFrame, multipliers: Optional[Dict[str, float]]
) -> pd.DataFrame:
    """Scale family columns before they are scored.

    Stage 2 measures the regime and produces a momentum multiplier -- 0.5 in a
    momentum-crash bucket, 1.0 in a clean uptrend. It was applied ONLY to the
    hand-weighted composite's weights, so for the fitted model, which is the one
    that actually ranks, the entire regime layer was decorative: the multiplier
    was computed, logged, written to the ledger, printed on the card, and never
    reached a score.

    The families are rank averages in [-1, 1], so scaling one by 0.5 halves what
    it can contribute, which is what the multiplier means.

    This is also the engine's analogue of volatility-scaled momentum exposure
    (Barroso & Santa-Clara 2015; Daniel & Moskowitz 2016). It is regime-bucket
    scaling rather than a continuous inverse-volatility weight, because the
    latter needs a momentum FACTOR-return series the engine does not build.
    """
    if not multipliers:
        return frame
    out = frame
    for family, m in multipliers.items():
        col = family + "_f"
        if col in out.columns and m is not None and float(m) != 1.0:
            if out is frame:
                out = frame.copy()
            out[col] = out[col] * float(m)
    return out


def _bare(column: str) -> str:
    """Drop the rank or family suffix. `_r` was stripped and `_f` was not, so a
    family arrived downstream still called `mom_f`, and a coefficient lookup
    that appends `_r` to it asked for `mom_f_r` and got zero -- the card printed
    a factor moving the score by 0.039 at a coefficient of +0.00000."""
    for suffix in ("_r", "_f"):
        if column.endswith(suffix):
            return column[: -len(suffix)]
    return column


def prediction_dispersion(raw: pd.Series) -> float:
    """Gap between the top decile's predicted rank and the median's.

    In predicted-rank units, where the fitted label spans -1 to +1. Near zero
    means the model ordered the universe without distinguishing it, and that
    day's ranking is noise wearing a shortlist.

    This exists because the SCORE cannot express it. The score is a
    cross-sectional rank, so its distribution is uniform every single day and an
    absolute threshold on it -- `min_universe_percentile = 90` -- admits the top
    10% whether or not the top 10% is any better than the middle.
    """
    clean = pd.Series(raw).dropna()
    if len(clean) < 20:
        return 0.0
    return float(clean.quantile(0.90) - clean.quantile(0.50))


def score_with(model: CrossSectionalModel, features: pd.DataFrame,
               multipliers: Optional[Dict[str, float]] = None) -> pd.Series:
    """Apply stored coefficients to today's features."""
    features = apply_family_multipliers(features, multipliers)
    cols = model.features
    x = features.reindex(columns=cols).fillna(0.0).to_numpy("float64")
    coef = np.array([model.coef[c] for c in cols], dtype="float64")
    raw = ((x - model.mu) / model.sd) @ coef + model.intercept
    s = pd.Series(raw, index=features["symbol"].to_numpy())
    # The cheap path scores today from a cached model, and the veto has to come
    # with it -- otherwise 20 of every 21 sessions produce no probability at
    # all, and a gate whose rule is "unknown is not approved" refuses the book.
    meta = getattr(model, "meta", None)
    if meta is not None:
        scored = features.copy()
        scored["_meta_score"] = raw
        try:
            model.meta_prob = pd.Series(meta.predict_proba(scored),
                                        index=features["symbol"].to_numpy())
        except (KeyError, ValueError) as exc:
            model.meta_prob = None
            model.meta_unavailable = f"the veto could not score today's names: {exc}"
    return ((s.rank(pct=True) - 0.5) * 2.0).sort_values(ascending=False)


def contributions(model: CrossSectionalModel, features: pd.DataFrame) -> pd.DataFrame:
    """Per-factor contribution to each symbol's score.

    The score is a sum of standardised factors times fitted coefficients, so
    each term is directly attributable and the terms add back to the score.
    This is what the card must cite: quoting the hand-weighted composite's
    factors beside a number the model produced describes a calculation that did
    not happen.
    """
    cols = model.features
    x = features.reindex(columns=cols).fillna(0.0).to_numpy("float64")
    z = (x - model.mu) / np.where(model.sd == 0, 1.0, model.sd)
    coef = np.array([model.coef[c] for c in cols], dtype="float64")
    return pd.DataFrame(
        z * coef,
        index=features["symbol"].to_numpy(),
        columns=[_bare(c) for c in cols],
    )


def standardised_features(model: CrossSectionalModel, features: pd.DataFrame) -> pd.DataFrame:
    """The z-scores the coefficients multiply, for the same columns."""
    cols = model.features
    x = features.reindex(columns=cols).fillna(0.0).to_numpy("float64")
    z = (x - model.mu) / np.where(model.sd == 0, 1.0, model.sd)
    return pd.DataFrame(
        z,
        index=features["symbol"].to_numpy(),
        columns=[_bare(c) for c in cols],
    )


# =============================================================================
# Factor families
# =============================================================================
#
# Seventeen coefficients over a set this collinear is not estimable, and the
# near-uniform coefficient band was the model saying so. Measured on the live
# universe:
#
#     amihud / turnover_ratio   -0.869    one factor measured from two sides
#     resid_mom / mom_6_1       +0.770
#     resid_mom / prox_52w      +0.601
#
# Ridge does not pick a winner among collinear inputs, it spreads the penalty
# across the block, so three momentum coefficients that each look small carry an
# effective weight of roughly three times any one of them.
#
# The members are averaged as ranks FIRST and one coefficient is fitted per
# family. Five or six coefficients over several hundred names and a decade of
# cross-sections is estimable; seventeen is not.
#
# LIQUIDITY IS NOT HERE, deliberately. The illiquidity premium is real but it is
# compensation FOR trading costs, and a manual executor pays that cost rather
# than collecting it -- a positive amihud loading walks the book into names
# where realised slippage exceeds forecast alpha. It belongs in the universe
# screen as a floor, which is where `universe.pit_min_adtv_inr` already puts it.
FAMILIES: Dict[str, Tuple[str, ...]] = {
    # Three names for one bet. Averaged, not fitted separately.
    "mom": ("mom_6_1_r", "prox_52w_r", "resid_mom_r"),
    # Reversal is the opposite side of the same axis and stays on its own: it is
    # a different horizon, and folding it into `mom` would net out against it.
    "reversal": ("resid_reversal_r",),
    # Lottery demand. In India this is stronger than the US literature suggests,
    # because the marginal buyer is retail. Signs are aligned so that a HIGHER
    # composite means MORE lottery-like, and the fit is free to price it
    # negatively.
    "lottery": ("max5_21_r", "idio_vol_r", "idio_skew_r", "downside_vol_r"),
    # What is left of risk once the lottery moments are taken out.
    "risk": ("beta_120_r", "max_dd_120_r"),
    # Delivered share of traded volume. No clean analogue outside India.
    "delivery": ("deliv_pct_r", "deliv_trend_r"),
    "value": ("earnings_yield_r", "book_to_price_r", "ebitda_to_ev_r",
              "fcf_yield_r", "sales_to_price_r"),
    # Quality is a SLOW factor: a modest gross edge that turns over slowly and
    # therefore sits far below breakeven turnover, which is where most of a
    # gross edge is otherwise lost.
    "quality": ("gross_profitability_r", "cash_op_profitability_r", "roce_r",
                "accruals_r", "asset_growth_r", "net_issuance_r"),
}

#: Computed, reported, and NOT scored. `log_mcap` is carried so the size of what
#: the model is ranking is visible, and so `research factors` can measure it --
#: but it does not get a coefficient.
#:
#: Measured over 17 dates it reads IC -0.2297 at a hit rate of 0/17, which is
#: not a factor, it is three independent 63-session windows in which small caps
#: happened to win. Giving that a family coefficient equal in weight to momentum
#: would rebuild by hand exactly the small-cap tilt the point-in-time panel fix
#: took out. The unintended-sector-bet problem it was raised against is solved
#: by ranking within sector, which is done.
UNSCORED_CONTROLS = ("log_mcap_r",)

#: Families whose members are all price-derived, so they are always computable.
FAMILY_COLUMNS = [f + "_f" for f in FAMILIES]


def build_families(frame: pd.DataFrame, available: Sequence[str]) -> List[str]:
    """Average each family's available members into one column, in place.

    Returns the family columns that could be built. A family with no available
    member is not built at all rather than filled -- the same rule the factors
    themselves follow.
    """
    built: List[str] = []
    have = set(available)
    for family, members in FAMILIES.items():
        present = [m for m in members if m in have and m in frame.columns]
        if not present:
            continue
        block = frame[present]
        flip = [m for m in present if m in NEGATED_IN_FAMILY]
        if flip:
            block = block.copy()
            block[flip] = -block[flip]
        frame[family + "_f"] = block.mean(axis=1)
        built.append(family + "_f")
    return built


def share_count_adjustment(
    actions: Optional[pd.DataFrame], since: pd.Timestamp, until: pd.Timestamp,
) -> Optional[pd.Series]:
    """Factor the share COUNT was multiplied by between two dates, per symbol.

    Net issuance is the year-on-year change in shares outstanding, and the raw
    count cannot tell a placement from a bonus: a 1:1 bonus doubles the count
    and dilutes nobody. Dividing the raw change by this factor leaves only the
    part that was actually issued.

    The store's `ratio` is the PRICE adjustment factor -- 0.5 for a 1:1 bonus --
    so the share count moved by its reciprocal. Dividends do not change the
    count and are skipped.
    """
    if actions is None or actions.empty:
        return None
    frame = actions.copy()
    frame["ex_date"] = pd.to_datetime(frame["ex_date"])
    frame = frame[(frame["ex_date"] > since) & (frame["ex_date"] <= until)]
    frame = frame[frame["action_type"].astype(str) != "dividend"]
    frame = frame[pd.to_numeric(frame["ratio"], errors="coerce").fillna(0) > 0]
    if frame.empty:
        return None
    return frame.groupby("symbol")["ratio"].apply(lambda r: float(1.0 / r.prod()))


def _attach_fundamentals(
    panel: pd.DataFrame, statements: Optional[pd.DataFrame],
    close: pd.DataFrame, max_age_days: Optional[int],
    actions: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge point-in-time fundamentals onto each panel date.

    Availability is derived from the SEBI LODR filing deadline rather than a
    filing date, because the statement feed carries period end only. Using the
    deadline is the conservative direction: companies file early, so the model
    never sees a figure before the market did.

    Names without a usable statement rank neutral rather than being dropped,
    which is what Stage 4 does when a factor is unavailable. Coverage is
    roughly two thirds of the universe -- market capitalisation needs a share
    count, and the feed does not carry one for every name.
    """
    cols = [f + "_r" for f in FUNDAMENTAL_FEATURES]

    def _blank(frame: pd.DataFrame) -> pd.DataFrame:
        for c in cols:
            frame[c] = np.nan
        return frame

    if statements is None or statements.empty:
        return _blank(panel)

    st = statements.copy()
    st["period_end"] = pd.to_datetime(st["period_end"])
    if "available_on" not in st.columns:
        st["available_on"] = available_as_of(
            st["period_end"],
            st["kind"] if "kind" in st.columns else "annual",
            st["filing_date"] if "filing_date" in st.columns else None,
        )
    if st.empty:
        return _blank(panel)

    shares = st.dropna(subset=["Ordinary Shares Number"])[
        ["symbol", "period_end", "available_on", "Ordinary Shares Number"]
    ] if "Ordinary Shares Number" in st.columns else pd.DataFrame()

    frames = []
    for d in panel["date"].unique():
        ts = pd.Timestamp(d)
        prices = close.loc[:ts]
        if prices.empty or shares.empty:
            continue
        sh = (shares[shares["available_on"] <= ts]
              .sort_values("period_end").groupby("symbol", observed=True).tail(1)
              .set_index("symbol")["Ordinary Shares Number"])
        px = prices.iloc[-1].dropna()
        mcap = (px.reindex(sh.index) * sh).dropna()
        if len(mcap) < 20:
            continue
        # Staleness is enforced inside the TTM, per symbol, against this panel
        # date. Filtering the statement rows here instead removed the older
        # quarters a trailing-twelve-month sum is built from, so the names the
        # cutoff was meant to keep current were the ones it made uncomputable.
        feats = build_fundamental_panel(
            st, mcap, ts.date(), enabled=FUNDAMENTAL_FEATURES,
            max_age_days=max_age_days,
            share_adjustment=share_count_adjustment(
                actions, ts - pd.Timedelta(days=400), ts),
        )
        if feats is None or feats.empty:
            continue
        f = feats.reset_index().rename(columns={"index": "symbol"})
        f["date"] = ts
        frames.append(f)

    if not frames:
        return _blank(panel)

    fund = pd.concat(frames, ignore_index=True)
    keep = ["date", "symbol"] + [f for f in FUNDAMENTAL_FEATURES if f in fund.columns]
    panel = panel.merge(fund[keep], on=["date", "symbol"], how="left")
    for f in FUNDAMENTAL_FEATURES:
        col = f + "_r"
        if f in panel.columns:
            w = panel.groupby("date")[f].transform(winsorise)
            # NOT filled here. A missing filing is left as NaN so the caller can
            # SEE how much of the column is absent and decide. Filling it at 0.0
            # here made the gap invisible: every name without a statement landed
            # on the neutral rank, the five value columns became a constant for
            # three quarters of the universe, and the model reported seventeen
            # factors while twelve carried the ranking.
            panel[col] = w.groupby(panel["date"]).transform(
                lambda s: ((s.rank(pct=True, na_option="keep") - 0.5) * 2.0)
            )
        else:
            panel[col] = np.nan
    return panel


def prepare_features(panel: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    """Apply the coverage tests and build the families. One implementation.

    `fit_predict` and the research commands must agree about what the model IS,
    or the thing validated is not the thing that runs. They did not: CPCV passed
    every raw FEATURE_COLUMN straight to `dropna`, which deleted every row
    without a fundamental and cut a 70-date panel to 17 -- too few to build ten
    CPCV groups, so the run did not merely validate the wrong model, it could
    not complete at all.

    Returns the panel with family columns attached, the feature list to fit on,
    and what was dropped with the coverage figure that dropped it.
    """
    dropped: Dict[str, float] = {}
    features = list(FEATURE_COLUMNS)
    fund_cols = [f + "_r" for f in FUNDAMENTAL_FEATURES]
    n_dates = max(panel["date"].nunique(), 1) if "date" in panel.columns else 1

    for c in fund_cols:
        if c not in panel.columns:
            dropped[c] = 0.0
            continue
        if "date" not in panel.columns:
            continue
        per_date = panel.groupby("date")[c].apply(lambda x: float(x.notna().mean()))
        live = per_date[per_date > 0]
        span = len(live) / n_dates
        within = float(live.median()) if len(live) else 0.0
        if span < MIN_FACTOR_DATE_SPAN:
            dropped[c] = round(span, 4)
            log.info("factor dropped: absent for too much of the panel",
                     extra={"factor": c, "date_span": round(span, 3),
                            "within_date": round(within, 3)})
        elif within < MIN_FACTOR_COVERAGE:
            dropped[c] = round(within, 4)
            log.info("factor dropped: ranks too few names on the dates it exists",
                     extra={"factor": c, "within_date": round(within, 3)})

    features = [c for c in FEATURE_COLUMNS if c not in dropped]
    panel = panel.drop(columns=[c for c in dropped if c in panel.columns])
    # Above the floors a gap still ranks neutral, the same rule
    # `crosssec.NEUTRAL_WHEN_MISSING` applies to delivery.
    for c in fund_cols:
        if c in features and c in panel.columns:
            panel[c] = panel[c].fillna(0.0)

    fitted = build_families(panel, features)
    return panel, fitted, dropped



def fit_coefficients(
    panel: pd.DataFrame,
    features: Sequence[str],
    *,
    estimator: str = "fama_macbeth",
    alpha: float = ALPHA,
    horizon: int = HORIZON,
    step: int = 21,
    significance_floor: Optional[float] = None,
    shrink_toward: str = "zero",
    weights: Optional[np.ndarray] = None,
    window_dates: Optional[int] = None,
    target: str = "label_rank",
) -> Tuple[Optional[Dict[str, np.ndarray]], Optional["FMResult"], Optional[str]]:
    """Turn a training panel into scoring coefficients. ONE implementation.

    `fit_predict`, CPCV, the portfolio harness and the nested-CV search all have
    to agree about how a model is estimated, or what gets validated is not what
    gets traded. They did not: the harness called `ridge_fit` at six separate
    sites, so switching the production estimator to Fama-MacBeth would have left
    every validation number describing a model no longer in use -- the same
    divergence `prepare_features` was extracted to close on the feature side.

    Returns ``(fit, fm_result, reason_unavailable)``. The fit is always in the
    standardised parameterisation `linear.predict` expects, whichever estimator
    produced it.
    """
    cols = [c for c in features if c in panel.columns]
    if not cols:
        return None, None, "no usable feature columns"
    x = panel[cols].to_numpy("float64")
    y = panel[target].to_numpy("float64")

    if estimator == "ridge":
        return ridge_fit(x, y, alpha=alpha, weights=weights), None, None
    if estimator != "fama_macbeth":
        return None, None, f"unknown estimator {estimator!r}"

    fm = fama_macbeth(panel, cols, target=target, horizon=horizon, step=step,
                      window=window_dates)
    if fm is None:
        n = panel["date"].nunique() if "date" in panel.columns else 0
        return None, None, (f"Fama-MacBeth needs at least 3 usable "
                            f"cross-sections; {n} dates produced too few")
    floor = SIGNIFICANCE_FLOOR if significance_floor is None else float(significance_floor)
    lam = gated_shrink(fm, floor=floor, toward=shrink_toward)
    if is_degenerate(lam):
        strongest = max(fm.t_stat, key=lambda k: abs(fm.t_stat.get(k, 0.0)))
        return None, fm, (
            f"no factor theme cleared |t| >= {floor:g} on {fm.n_dates} "
            f"cross-sections; strongest was {strongest.removesuffix('_f')} at "
            f"t {fm.t_stat[strongest]:+.2f}"
        )
    # predict() computes ((x - mu)/sd) @ coef + intercept, so coef = lambda * sd
    # reproduces (x - mu) @ lambda exactly. The dropped constant moves no rank.
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return ({"coef": np.array([lam[c] for c in cols]) * sd, "mu": mu, "sd": sd,
             "intercept": float(y.mean())}, fm, None)


def fit_predict(
    close: pd.DataFrame,
    turnover: pd.DataFrame,
    as_of: dt.date,
    fundamentals: Optional[pd.DataFrame] = None,
    max_fundamental_age_days: Optional[int] = None,
    horizon: Optional[int] = None,
    alpha: Optional[float] = None,
    max_train_sessions: Optional[int] = None,
    min_train_rows: Optional[int] = None,
    delivery: Optional[pd.DataFrame] = None,
    eligible: Optional[pd.DataFrame] = None,
    score_symbols: Optional[Sequence[str]] = None,
    sectors: Optional[Dict[str, str]] = None,
    multipliers: Optional[Dict[str, float]] = None,
    actions: Optional[pd.DataFrame] = None,
    barriers: Optional["BarrierSpec"] = None,
    exit_rules: Optional["ExitRules"] = None,
    open_: Optional[pd.DataFrame] = None,
    high: Optional[pd.DataFrame] = None,
    low: Optional[pd.DataFrame] = None,
    uniqueness_weighting: bool = True,
    estimator: str = "fama_macbeth",
    significance_floor: Optional[float] = None,
    fm_window_dates: Optional[int] = None,
    shrink_toward: str = "zero",
    metalabel: bool = False,
    metalabel_top_k: int = 50,
    metalabel_l2: float = 1.0,
) -> Tuple[Optional[pd.Series], Optional[CrossSectionalModel], Optional[str]]:
    """Rank every symbol by predicted forward return.

    Returns ``(scores, model, reason_unavailable)``. Scores are in [-1, 1] and
    comparable only within this run, since the fit is refitted each time.

    ``eligible`` masks the TRAINING panel to the names the universe screen would
    have admitted on each date. ``score_symbols`` restricts what is ranked
    today, which is a different question: the training set has to be
    point-in-time, and today's ranking is over today's eligible universe.
    """
    H = int(horizon if horizon is not None else HORIZON)
    A = float(alpha if alpha is not None else ALPHA)
    MAXS = int(max_train_sessions if max_train_sessions is not None else MAX_TRAIN_SESSIONS)
    MINR = int(min_train_rows if min_train_rows is not None else MIN_TRAIN_ROWS)

    ts = pd.Timestamp(as_of)
    hist = close[close.index <= ts]
    if len(hist) > MAXS + H:
        hist = hist.iloc[-(MAXS + H):]
    if len(hist) < MIN_LOOKBACK + H + 60:
        return None, None, (
            f"{len(hist)} sessions of history; the cross-sectional model needs "
            f"{MIN_LOOKBACK + H + 60}"
        )

    # Training stops one full label horizon before as_of. A row dated later
    # would have a label running past the decision date, which is the leak this
    # model exists to avoid.
    train_close = hist.iloc[: len(hist) - H]
    train_turnover = turnover.reindex(train_close.index)
    panel = build_panel(train_close, train_turnover, horizon=H, step=21,
                        delivery=delivery, eligible=eligible, sectors=sectors,
                        barriers=barriers, exit_rules=exit_rules,
                        high=(high.reindex(train_close.index) if high is not None else None),
                        low=(low.reindex(train_close.index) if low is not None else None),
                        open_=(open_.reindex(train_close.index) if open_ is not None else None))
    features: List[str] = []
    dropped: Dict[str, float] = {}
    member_features: List[str] = []
    if not panel.empty:
        panel = _attach_fundamentals(panel, fundamentals, train_close,
                                     max_fundamental_age_days, actions=actions)
        member_features = [c for c in FEATURE_COLUMNS if c in panel.columns]
        panel, features, dropped = prepare_features(panel)
        panel = panel.dropna(subset=[c for c in features if c in panel.columns]
                             + ["label_rank"])
    if panel.empty or len(panel) < MINR:
        return None, None, (
            f"{0 if panel.empty else len(panel)} usable training rows; "
            f"{MINR} required"
        )

    # Fit on FAMILIES, not on the individual factors. See FAMILIES for the
    # measured collinearity that makes seventeen coefficients unestimable.
    if len(features) < 2:
        return None, None, (
            f"only {len(features)} factor famil(y/ies) could be built"
        )
    x = panel[features].to_numpy("float64")
    y = panel["label_rank"].to_numpy("float64")
    # Overlapping labels are not independent draws. See labels.average_uniqueness.
    w = None
    if uniqueness_weighting and "uniqueness" in panel.columns:
        w = panel["uniqueness"].to_numpy("float64")
        if not np.isfinite(w).any():
            w = None

    # The pooled ridge treats 33,000 rows as 33,000 independent draws. They are
    # 70 cross-sections, and within one date every name shares the same market.
    # Measured on a purged walk-forward, 50 out-of-sample dates:
    #
    #   arm                        IC    t(NW)   hit   top-decile     t
    #   ridge (was production)  +0.0021  +0.06   48%     -0.11%   -0.17
    #   equal weight 1/N        -0.0022  -0.07   50%     -0.38%   -0.79
    #   gate |t|>=2, then shrink +0.0516 +3.25   78%     +1.12%   +2.33
    #
    # The ridge could not beat the 1/N control it exists to justify. What broke
    # it: the pooled fit gave `lottery` a -0.0143 coefficient while lottery
    # measures IC +0.0485 in this universe, so the blend bet against it and
    # `mom_f - lottery_f` reads IC +0.0031 against `mom_f` alone at +0.0481.
    fit, fm, why = fit_coefficients(
        panel, features, estimator=estimator, alpha=A, horizon=H, step=21,
        significance_floor=significance_floor, shrink_toward=shrink_toward,
        weights=w, window_dates=fm_window_dates)
    if fit is None:
        # A refusal, not a crash. When no theme clears the floor the honest
        # answer is not to trade this refit -- scoring anyway emits a flat
        # ranking nothing downstream could tell from a real view.
        return None, None, why
    if dropped:
        log.info("factors dropped for coverage",
                 extra={"dropped": {k: round(v, 3) for k, v in dropped.items()},
                        "floor": MIN_FACTOR_COVERAGE, "kept": len(features)})

    # Features for the decision date itself, from the same builder, so training
    # and inference cannot drift apart in definition.
    # Today's ranking covers the names eligible TODAY. Training spans every name
    # that was ever eligible, which is a much wider matrix, and computing live
    # features across all of it would cost several times what the decision needs.
    live_cols = ([c for c in hist.columns if c in set(score_symbols)]
                 if score_symbols is not None else list(hist.columns))
    if not live_cols:
        return None, None, "no eligible symbol survived into the scoring universe"
    live_hist = hist[live_cols].tail(MIN_LOOKBACK + 5)
    live = build_panel(live_hist,
                       turnover.reindex(hist.index)[live_cols].tail(MIN_LOOKBACK + 5),
                       horizon=1, step=21, delivery=delivery, sectors=sectors)
    if live.empty:
        return None, None, "features could not be computed for the decision date"
    live = _attach_fundamentals(live, fundamentals, live_hist,
                                max_fundamental_age_days, actions=actions)
    latest = live[live["date"] == live["date"].max()]
    for c in [f + "_r" for f in FUNDAMENTAL_FEATURES]:
        if c in features and c in latest.columns:
            latest = latest.copy()
            latest[c] = latest[c].fillna(0.0)
    latest = latest.copy()
    build_families(latest, member_features)
    latest = latest.dropna(subset=[c for c in features if c in latest.columns])
    if latest.empty:
        return None, None, "no symbol had a complete feature set on the decision date"

    # What this model's dispersion normally looks like, measured on its own
    # training panel. An ABSOLUTE floor cannot work here: ridge at alpha=20000
    # shrinks predictions hard, so the achievable spread is small and entirely
    # a function of the penalty. Measured across 88 panel dates the whole range
    # was 0.0355 to 0.0607, and a floor set anywhere near the label's own scale
    # would block every single day. Changing alpha would move the range again.
    #
    # The ratio to the model's own median is scale-free and survives a penalty
    # change, which an absolute number does not.
    train_pred = predict(fit, x)
    by_date = pd.Series(train_pred, index=panel["date"].to_numpy())
    per_date = by_date.groupby(level=0).apply(prediction_dispersion)
    train_dispersion = float(per_date.median()) if len(per_date) else 0.0

    n_train = len(panel)
    train_end = pd.Timestamp(panel["date"].max()).date()

    # THE NO TRADE VETO. A second model, fitted only on the trades this engine
    # would actually have taken, predicting whether one reaches its profit
    # barrier before its stop. It cannot pick a name the primary did not; it can
    # only refuse one. Off by default -- see MetaLabelConfig for the measurement.
    meta_model = None
    meta_reason = None
    if metalabel:
        def _inner_primary(inner_train, frame):
            inner_fit, _fm2, _why2 = fit_coefficients(
                inner_train, features, estimator=estimator, alpha=A, horizon=H,
                step=21, significance_floor=significance_floor,
                shrink_toward=shrink_toward,
                weights=(inner_train["uniqueness"].to_numpy("float64")
                         if "uniqueness" in inner_train.columns else None))
            if inner_fit is None:
                return None
            return predict(inner_fit, frame[features].to_numpy("float64"))

        meta_model, meta_reason = fit_meta_out_of_sample(
            panel, list(features) + ["_meta_score"], _inner_primary,
            top_k=int(metalabel_top_k), l2=float(metalabel_l2))
        if meta_model is None:
            log.info("meta-label model unavailable; the veto is inert this refit",
                     extra={"reason": meta_reason})

    del panel, x, y
    latest = apply_family_multipliers(latest, multipliers)
    raw = predict(fit, latest[features].to_numpy("float64"))
    scores = pd.Series(raw, index=latest["symbol"].to_numpy())
    # The veto reads the primary score as one of its inputs, under the same
    # name it was trained with.
    latest["_meta_score"] = raw
    ranked = (scores.rank(pct=True) - 0.5) * 2.0
    # How far apart the model actually placed the universe today, BEFORE the
    # rank transform flattens it. The ranked score is uniform by construction
    # every day, so nothing downstream can tell a day the model had a view from
    # a day it did not.
    dispersion = prediction_dispersion(scores)

    model = CrossSectionalModel(
        coef=dict(zip(features, fit["coef"].tolist())),
        n_train=n_train,
        train_end=train_end,
        features=features,
    )
    # P(target before stop) for the names being ranked today, or None when the
    # veto could not be fitted. Computed here rather than in stage 8 so the
    # probability comes from the SAME feature frame the score did.
    model_meta_prob = None
    if meta_model is not None:
        try:
            model_meta_prob = pd.Series(
                meta_model.predict_proba(latest), index=latest["symbol"].to_numpy())
        except (KeyError, ValueError) as exc:
            meta_reason = f"the veto could not score today's names: {exc}"
            log.warning("meta-label scoring failed; the veto is inert",
                        extra={"reason": meta_reason})

    model.dropped_for_coverage = dropped
    model.meta_prob = model_meta_prob
    model.meta = meta_model
    model.meta_unavailable = meta_reason
    model.estimator = estimator
    if fm is not None:
        model.fm_t_stat = dict(fm.t_stat)
        model.fm_lambda = dict(fm.lam)
        model.fm_n_dates = fm.n_dates
        model.fm_nw_lags = fm.nw_lags
    model.dispersion = dispersion
    model.train_dispersion = train_dispersion
    model.mu, model.sd, model.intercept = fit["mu"], fit["sd"], fit["intercept"]
    release_memory()
    log.info(
        "cross-sectional model fitted",
        extra={"n_train": n_train, "train_end": str(model.train_end),
               "scored": len(ranked)},
    )
    return ranked.sort_values(ascending=False), model, None


def today_features(close: pd.DataFrame, turnover: pd.DataFrame, as_of: dt.date,
                   fundamentals: Optional[pd.DataFrame] = None,
                   max_fundamental_age_days: Optional[int] = None,
                   delivery: Optional[pd.DataFrame] = None,
                   sectors: Optional[Dict[str, str]] = None,
                   actions: Optional[pd.DataFrame] = None):
    """Features for the decision date only.

    The cheap path: one date rather than a full training panel, so a cached
    model scores today without the large historical read.
    """
    ts = pd.Timestamp(as_of)
    hist = close[close.index <= ts].tail(MIN_LOOKBACK + 5)
    if len(hist) < MIN_LOOKBACK:
        return None
    live = build_panel(hist, turnover.reindex(hist.index), horizon=1, step=21,
                       delivery=delivery, sectors=sectors)
    if live.empty:
        return None
    live = _attach_fundamentals(live, fundamentals, hist,
                                max_fundamental_age_days, actions=actions)
    latest = live[live["date"] == live["date"].max()].dropna(
        subset=[c for c in FEATURE_COLUMNS
                if c in live.columns and not c.endswith(
                    tuple(f + "_r" for f in FUNDAMENTAL_FEATURES))]).copy()
    if latest.empty:
        return None
    # The model is fitted on families, so the scoring frame has to carry them.
    # A fundamental the feed cannot serve is absent rather than zero, and
    # `build_families` averages over the members that ARE present -- so a value
    # family with two of five members is the mean of those two, not a mean
    # diluted by three imputed zeros.
    available = [c for c in FEATURE_COLUMNS if c in latest.columns
                 and latest[c].notna().any()]
    build_families(latest, available)
    return latest
