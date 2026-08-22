"""Corporate-action parsing, price adjustment, and unexplained-jump detection.

Three jobs, in dependency order:

1. :func:`parse_action_subject` turns NSE's free-text description ("Face Value
   Split From Rs.10/- To Rs.2/-", "Bonus 1:1") into a multiplicative price
   factor.

2. :func:`apply_adjustments` rewrites an unadjusted OHLCV series. Prices before
   an ex-date are multiplied by the factor and volumes divided by it, leaving
   rupee turnover invariant.

3. :func:`detect_unexplained_jumps` finds the opposite case: an overnight ratio
   near a clean fraction with no corporate action on file. Stage 1 rejects
   those names rather than admit a fabricated -80% return into a momentum score.

Convention: ``ratio`` multiplies PRE-ex-date prices. A 1:1 bonus doubles the
share count, so ``ratio = 0.5``.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.logging import get_logger
from .types import CORPORATE_ACTION_COLUMNS, DATE, SYMBOL, normalise_symbol

__all__ = [
    "parse_action_subject",
    "build_adjustment_factors",
    "apply_adjustments",
    "detect_unexplained_jumps",
    "plausible_price_factors",
    "merge_action_sources",
]

log = get_logger(__name__)


# =============================================================================
# 1. parsing
# =============================================================================

_NUM = r"(\d+(?:\.\d+)?)"

_SPLIT_PATTERNS = (
    # "Face Value Split From Rs.10/- To Rs.2/-"
    re.compile(rf"split.*?from\s*(?:rs\.?\s*)?{_NUM}.*?to\s*(?:rs\.?\s*)?{_NUM}", re.I),
    # "Stock Split 10:1" / "Split 10/1"
    re.compile(rf"split\D*{_NUM}\s*[:/]\s*{_NUM}", re.I),
)

_BONUS_PATTERNS = (
    # "Bonus 1:1" -> 1 new share for every 1 held
    re.compile(rf"bonus\D*{_NUM}\s*[:/]\s*{_NUM}", re.I),
    re.compile(rf"bonus.*?ratio\D*{_NUM}\s*[:/]\s*{_NUM}", re.I),
)

_RIGHTS_PATTERNS = (re.compile(rf"rights\D*{_NUM}\s*[:/]\s*{_NUM}", re.I),)

_DIVIDEND_PATTERN = re.compile(r"dividend", re.I)


def parse_action_subject(
    subject: str, face_value_hint: Optional[float] = None
) -> Tuple[str, float, str]:
    """Parse an NSE corporate-action description.

    Returns ``(action_type, price_factor, explanation)``.

    A ``price_factor`` of ``1.0`` means "no price rescaling required" -- that
    covers dividends and anything we could not confidently parse. Returning 1.0
    for an unparsed string is the safe direction: the series stays as published
    and :func:`detect_unexplained_jumps` will still catch a real rescaling that
    we failed to describe.
    """
    text = " ".join(str(subject or "").split())
    if not text:
        return "unknown", 1.0, "empty description"

    for pattern in _SPLIT_PATTERNS:
        m = pattern.search(text)
        if m:
            old_fv, new_fv = float(m.group(1)), float(m.group(2))
            if old_fv > 0 and new_fv > 0:
                # Splitting FV 10 -> 2 multiplies share count by 5; the price
                # factor is the inverse of that multiplication.
                factor = new_fv / old_fv
                if 0 < factor < 1.0 or factor > 1.0:
                    return "split", factor, f"face value {old_fv:g} -> {new_fv:g}"

    for pattern in _BONUS_PATTERNS:
        m = pattern.search(text)
        if m:
            new_shares, per_held = float(m.group(1)), float(m.group(2))
            if new_shares > 0 and per_held > 0:
                factor = per_held / (per_held + new_shares)
                return "bonus", factor, f"bonus {new_shares:g}:{per_held:g}"

    for pattern in _RIGHTS_PATTERNS:
        m = pattern.search(text)
        if m:
            # A rights issue's price effect depends on the subscription price,
            # which the subject line rarely carries. Flag it, adjust nothing,
            # and let the unexplained-jump detector decide.
            return "rights", 1.0, f"rights {m.group(1)}:{m.group(2)} (price effect not derivable)"

    if _DIVIDEND_PATTERN.search(text):
        return "dividend", 1.0, "dividend (no price rescaling applied)"

    return "other", 1.0, f"unparsed: {text[:80]}"


# =============================================================================
# 2. adjustment
# =============================================================================


def build_adjustment_factors(
    dates: Sequence[pd.Timestamp], actions: pd.DataFrame
) -> pd.Series:
    """Cumulative price factor per date for ONE symbol.

    For each date ``t`` the factor is the product of every action factor whose
    ex-date is strictly after ``t``. Multiply the published price by it to put
    the whole history on today's share basis.
    """
    idx = pd.DatetimeIndex(dates).normalize()
    factors = pd.Series(1.0, index=idx, dtype="float64")
    if actions is None or actions.empty:
        return factors

    usable = actions.dropna(subset=["ex_date", "ratio"])
    usable = usable[(usable["ratio"] > 0) & (usable["ratio"] != 1.0)]
    for _, row in usable.iterrows():
        ex_date = pd.Timestamp(row["ex_date"]).normalize()
        factors.loc[idx < ex_date] *= float(row["ratio"])
    return factors


def apply_adjustments(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    price_columns: Iterable[str] = ("open", "high", "low", "close", "prev_close", "last", "vwap"),
) -> pd.DataFrame:
    """Return a corporate-action-adjusted copy of a tidy OHLCV frame.

    Volume is divided by the same factor so that ``price * volume`` (rupee
    turnover) is invariant across the action -- the property that makes ADTV
    comparable either side of a split.

    The applied factor is retained in ``adj_factor`` so any downstream check
    can see exactly what was done, and ``turnover`` is deliberately left alone
    because rupee turnover is unaffected by a share-count change.
    """
    if prices is None or prices.empty:
        return prices

    out = prices.copy()
    out[DATE] = pd.to_datetime(out[DATE]).dt.normalize()
    out["adj_factor"] = 1.0

    if actions is None or actions.empty:
        return out

    acts = actions.copy()
    acts[SYMBOL] = acts[SYMBOL].map(normalise_symbol)
    acts["ex_date"] = pd.to_datetime(acts["ex_date"]).dt.normalize()
    by_symbol: Dict[str, pd.DataFrame] = {
        sym: grp for sym, grp in acts.groupby(SYMBOL, observed=True)
    }

    cols = [c for c in price_columns if c in out.columns]
    pieces: List[pd.DataFrame] = []
    for sym, grp in out.groupby(SYMBOL, sort=False, observed=True):
        sym_actions = by_symbol.get(sym)
        chunk = grp.copy()
        if sym_actions is not None and not sym_actions.empty:
            factors = build_adjustment_factors(chunk[DATE].to_numpy(), sym_actions)
            fac = factors.to_numpy()
            for col in cols:
                chunk[col] = chunk[col].to_numpy() * fac
            if "volume" in chunk.columns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    chunk["volume"] = np.where(fac > 0, chunk["volume"].to_numpy() / fac, np.nan)
            if "deliv_qty" in chunk.columns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    chunk["deliv_qty"] = np.where(
                        fac > 0, chunk["deliv_qty"].to_numpy() / fac, np.nan
                    )
            chunk["adj_factor"] = fac
        pieces.append(chunk)

    return pd.concat(pieces, ignore_index=True).sort_values([SYMBOL, DATE]).reset_index(drop=True)


# =============================================================================
# 3. unexplained-jump detection
# =============================================================================


#: Bonus ratios (a new shares for every b held) that Indian issuers actually
#: declare. Deliberately curated rather than generated: an exhaustive set of
#: a/b fractions is so dense that almost any price move sits within a few
#: percent of *something*, which would make the detector fire constantly and
#: teach the operator to ignore it.
COMMON_BONUS_RATIOS: Tuple[Tuple[int, int], ...] = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 10),
    (2, 1), (2, 3), (2, 5),
    (3, 1), (3, 2), (3, 5),
    (4, 1), (5, 1), (6, 1), (9, 1), (10, 1),
)

#: Face-value split pairs (old -> new) standard on NSE.
COMMON_SPLIT_FACE_VALUES: Tuple[Tuple[float, float], ...] = (
    (10, 1), (10, 2), (10, 5), (10, 0.5),
    (5, 1), (5, 2), (5, 0.5),
    (2, 1), (2, 0.5),
    (1, 0.5),
    (100, 1), (100, 2), (100, 5), (100, 10),
)


def plausible_price_factors() -> List[float]:
    """Price factors a genuine Indian split or bonus would produce.

    Restricted to ratios issuers really use. The trade-off is explicit: a
    sparser set means the detector occasionally misses an exotic ratio, but a
    denser set means it flags ordinary volatility as a corporate action. Since
    Stage 1 turns a hit into a hard rejection, a detector that fires on normal
    moves would quietly shrink the universe for no reason.
    """
    out = set()
    for new_shares, per_held in COMMON_BONUS_RATIOS:
        out.add(round(float(Fraction(per_held, per_held + new_shares)), 6))
    for old_fv, new_fv in COMMON_SPLIT_FACE_VALUES:
        out.add(round(new_fv / old_fv, 6))
    return sorted(v for v in out if 0.001 <= v <= 0.999)


def detect_unexplained_jumps(
    prices: pd.DataFrame,
    actions: Optional[pd.DataFrame],
    min_ratio_gap: float = 0.30,
    tolerance: float = 0.03,
    lookback_sessions: Optional[int] = None,
    max_circuit_move: float = 0.20,
) -> pd.DataFrame:
    """Find overnight price ratios that look like an unadjusted corporate action.

    Two independent rules, because they catch different things.

    CLEAN FACTOR. ``close_t / close_{t-1}`` sits within ``tolerance`` of a
    plausible split or bonus fraction and is at least ``min_ratio_gap`` from
    1.0. This is the signature of an unadjusted split: a 5:1 reads as -80%.

    BEYOND THE CIRCUIT. The move exceeds ``max_circuit_move``, the widest daily
    band the exchange permits. NSE caps a scrip at 20%, so a larger overnight
    move is not something the market can produce -- it is a corporate action or
    bad data, whatever ratio it happens to land on.

    The second rule exists because the first cannot see a demerger. When
    Vedanta separated on 2026-04-30 the parent fell 62.6%, a ratio of 0.374.
    The nearest clean factor is 0.40 and 0.374 is 6.4% away, outside any sane
    tolerance -- because a demerger's ratio is the market value of what was
    spun out, not a clean fraction. It passed Stage 1 with no flags while
    corrupting every feature that looks back past it.

    Returns one row per suspect ``(symbol, date)``. Stage 1 turns these into
    hard rejections.
    """
    empty = pd.DataFrame(
        columns=[SYMBOL, DATE, "ratio", "nearest_clean_factor", "prev_close", "close"]
    )
    if prices is None or prices.empty:
        return empty

    frame = prices[[SYMBOL, DATE, "close"]].copy()
    frame[DATE] = pd.to_datetime(frame[DATE]).dt.normalize()
    frame = frame.dropna(subset=["close"]).sort_values([SYMBOL, DATE])
    if lookback_sessions is not None and lookback_sessions > 0:
        frame = frame.groupby(SYMBOL, group_keys=False, observed=True).tail(lookback_sessions + 1)

    frame["prev_close"] = frame.groupby(SYMBOL, observed=True)["close"].shift(1)
    frame = frame.dropna(subset=["prev_close"])
    frame = frame[frame["prev_close"] > 0]
    if frame.empty:
        return empty

    frame["ratio"] = frame["close"] / frame["prev_close"]
    # Strictly beyond the band: a move sitting exactly ON the circuit limit is
    # the market hitting its cap, which is legitimate and common.
    beyond_circuit = (frame["ratio"] - 1.0).abs() > (max_circuit_move + 1e-9)
    frame["beyond_circuit"] = beyond_circuit
    suspects = frame[((frame["ratio"] - 1.0).abs() >= min_ratio_gap)
                     | beyond_circuit].copy()
    if suspects.empty:
        return empty

    clean = np.array(plausible_price_factors(), dtype="float64")
    # A reverse split (share consolidation) produces a ratio > 1.
    clean_all = np.concatenate([clean, 1.0 / clean])

    def nearest(r: float) -> Tuple[float, float]:
        diffs = np.abs(clean_all - r) / np.maximum(clean_all, 1e-12)
        i = int(np.argmin(diffs))
        return float(clean_all[i]), float(diffs[i])

    nearest_vals: List[float] = []
    rel_errors: List[float] = []
    for r in suspects["ratio"].to_numpy():
        val, err = nearest(r)
        nearest_vals.append(val)
        rel_errors.append(err)
    suspects["nearest_clean_factor"] = nearest_vals
    suspects["relative_error"] = rel_errors
    # Keep a row if it matches a clean factor OR the move is beyond anything
    # the exchange permits. Requiring both would reinstate the demerger blind
    # spot, since a demerger matches no clean factor by construction.
    suspects = suspects[(suspects["relative_error"] <= tolerance)
                        | suspects["beyond_circuit"]]
    if suspects.empty:
        return empty

    if actions is not None and not actions.empty:
        acts = actions.dropna(subset=["ex_date"]).copy()
        acts[SYMBOL] = acts[SYMBOL].map(normalise_symbol)
        acts["ex_date"] = pd.to_datetime(acts["ex_date"]).dt.normalize()
        known = {
            (row[SYMBOL], row["ex_date"]) for _, row in acts.iterrows()
        }
        # Allow +/-1 session of slack: ex-date conventions differ by source.
        def explained(row: pd.Series) -> bool:
            sym, day = row[SYMBOL], row[DATE]
            for offset in (-1, 0, 1):
                if (sym, day + pd.Timedelta(days=offset)) in known:
                    return True
            return False

        suspects = suspects[~suspects.apply(explained, axis=1)]

    if suspects.empty:
        return empty
    return suspects[
        [SYMBOL, DATE, "ratio", "nearest_clean_factor", "prev_close", "close"]
    ].reset_index(drop=True)


# =============================================================================
# merging sources
# =============================================================================


#: Sources ranked by authority. NSE issues the corporate action; a scraped
#: feed reports it second-hand and sometimes a day out.
_SOURCE_RANK = {"csv_import": 3, "nse_corporate_actions": 2, "yfinance": 1}

#: Two actions for one symbol this many days apart are the same event seen by
#: two sources, not two events. yfinance ex-dates drift by a day against NSE's
#: -- HAL's 2023 split is 09-28 at NSE and 09-29 at Yahoo -- and applying both
#: adjusts the series twice.
_SAME_EVENT_DAYS = 3


def dedupe_actions(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate descriptions of the same corporate action.

    Two independent failures are handled here, and both double-adjust a price
    series when missed:

    Same date, different label. NSE reports GAIL 2017-03-09 as "bonus" and
    yfinance as "split_or_bonus". A key that includes the label keeps both and
    multiplies 0.75 by itself.

    Adjacent dates, same event. A source whose ex-date is a day off looks like
    a second action the day after the first.

    Dividends are exempt from both: several can genuinely share a date and none
    of them moves the adjusted series.
    """
    if frame is None or frame.empty:
        return frame if frame is not None else pd.DataFrame(columns=CORPORATE_ACTION_COLUMNS)

    work = frame.copy()
    work["ex_date"] = pd.to_datetime(work["ex_date"]).dt.normalize()
    ratio = pd.to_numeric(work["ratio"], errors="coerce").fillna(1.0)
    work["_rank"] = work.get("source", pd.Series("", index=work.index)).map(
        lambda s: _SOURCE_RANK.get(str(s), 0)
    )

    neutral = work[ratio == 1.0].drop_duplicates(
        subset=[SYMBOL, "ex_date", "action_type"], keep="last"
    )

    adjusting = work[ratio != 1.0].sort_values([SYMBOL, "ex_date", "_rank"])
    kept: List[pd.Series] = []
    for _, group in adjusting.groupby(SYMBOL, sort=False, observed=True):
        cluster: List[pd.Series] = []

        def _flush() -> None:
            if cluster:
                # Highest-authority row in the cluster; ties break to the later
                # ex-date, which is the one the exchange actually published.
                best = max(cluster, key=lambda r: (r["_rank"], r["ex_date"]))
                kept.append(best)

        for _, row in group.sort_values("ex_date").iterrows():
            if cluster and (row["ex_date"] - cluster[-1]["ex_date"]).days > _SAME_EVENT_DAYS:
                _flush()
                cluster = []
            cluster.append(row)
        _flush()

    adjusting_out = pd.DataFrame(kept) if kept else adjusting.head(0)
    combined = pd.concat([adjusting_out, neutral], ignore_index=True)
    combined = combined.drop(columns=["_rank"], errors="ignore")
    return combined.sort_values([SYMBOL, "ex_date"]).reset_index(drop=True)


def merge_action_sources(*frames: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Combine corporate-action frames, preferring later arguments on conflict.

    Order your calls so the most trustworthy source comes last: a hand-curated
    CSV should override an automatically scraped one.
    """
    usable = [f for f in frames if f is not None and not f.empty]
    if not usable:
        return pd.DataFrame(columns=CORPORATE_ACTION_COLUMNS)
    combined = pd.concat(usable, ignore_index=True)
    combined[SYMBOL] = combined[SYMBOL].map(normalise_symbol)
    combined["ex_date"] = pd.to_datetime(combined["ex_date"]).dt.normalize()
    combined = combined.dropna(subset=[SYMBOL, "ex_date"])
    return dedupe_actions(combined)
