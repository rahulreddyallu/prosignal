"""Stage 6 -- Entry.

Admission is by rank, with hysteresis. A name enters while it sits inside
``entry_rank`` and is kept while it stays inside the wider ``exit_rank``; only
outside that does it leave. This is how NSE, MSCI and FTSE construct factor
indices, and the reason is arithmetic: without a buffer a name oscillating
around the boundary pays a full round trip at every rebalance for no change in
view.

Why not a price trigger. This stage used to require a pullback, a moving-average
reclaim or a breakout before converting a rank into a BUY, and measured that
way it destroyed most of the strategy. Portfolio-level, non-overlapping
cohorts, real risk-based sizing and 70 bps round trip:

                            selection SR   holdout SR   holdout ret/qtr
    price trigger (old)        -0.05          +0.46         +1.59%
    rank <= 8                  +0.77          +1.38         +4.24%
    rank <= 8, exit > 16       +0.86          +1.56         +5.55%

Grinold (1989) gives IR = IC x sqrt(breadth); Clarke, de Silva & Thorley (2002)
refine it to IR = TC x IC x sqrt(breadth), where the transfer coefficient TC is
the correlation between the weights actually held and the alpha. The trigger cut
average holdings from 8.0 to 5.6, which accounts for a 1.20x Sharpe difference.
The observed difference was 7.53x. Breadth explains 16% of it; the rest is TC --
the trigger admitted names on price structure, which is uncorrelated with the
model's ranking, so the book held was not the book the model ranked.

The triggers still run. They no longer gate: they describe what price structure
is present so the operator can judge execution, and the entry zone is still
quoted from ATR. A rank that qualifies is not converted into a market order at
any price.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ._cfg import bv, fv, iv, v
from ..core.contracts import EntryDecision, EntryReport
from ..core.enums import EntryStatus, TriggerType
from ..core.logging import get_logger
from ..indicators import atr, sma

__all__ = ["run", "STAGE_NAME"]

STAGE_NAME = "stage6_entry"
log = get_logger(__name__)


def run(
    candidates: List[str],
    frames: Dict[str, pd.DataFrame],
    config,
    as_of: dt.date,
    ranks: Optional[Dict[str, int]] = None,
    held: Optional[Sequence[str]] = None,
) -> EntryReport:
    """Decide which ranked candidates are admitted today.

    ``ranks`` maps ticker to its Stage 4 rank in the eligible universe.
    ``held`` is the previous run's book, which the exit band needs; an empty or
    absent book is the correct first-run state, not an error.
    """
    p = config.params
    cfg = p.stage6_entry
    decisions: Dict[str, EntryDecision] = {}
    admission = cfg.admission
    entry_rank = iv(admission.entry_rank)
    exit_rank = iv(admission.exit_rank)
    open_book = set(held or ())

    for sym in candidates:
        frame = frames.get(sym)
        if frame is None or len(frame) < 70:
            decisions[sym] = EntryDecision(
                ticker=sym, status=EntryStatus.NOT_TRIGGERED,
                reason="insufficient history to evaluate an entry",
            )
            continue
        decisions[sym] = _evaluate(
            sym, frame, cfg, p,
            rank=(ranks or {}).get(sym),
            entry_rank=entry_rank, exit_rank=exit_rank,
            is_held=sym in open_book,
        )

    log.info(
        "stage 6 complete",
        extra={"triggered": sum(1 for d in decisions.values() if d.status is EntryStatus.TRIGGERED),
               "watchlist": sum(1 for d in decisions.values() if d.status is EntryStatus.WATCHLIST)},
    )
    return EntryReport(as_of_date=as_of, decisions=decisions)


def _evaluate(sym, frame, cfg, params, rank=None, entry_rank=8,
              exit_rank=16, is_held=False) -> EntryDecision:
    closes = pd.Series(frame["close"].to_numpy(dtype="float64"))
    highs = pd.Series(frame["high"].to_numpy(dtype="float64"))
    lows = pd.Series(frame["low"].to_numpy(dtype="float64"))
    vols = pd.Series(frame["volume"].to_numpy(dtype="float64"))
    last = float(closes.iloc[-1])

    a = atr(frame["high"], frame["low"], frame["close"],
            iv(params.stage7_risk.atr.period_sessions),
            str(v(params.stage7_risk.atr.method))).dropna()
    if a.empty:
        return EntryDecision(ticker=sym, status=EntryStatus.NOT_TRIGGERED,
                             reason="ATR not computable")
    atr_v = float(a.iloc[-1])

    passed: List[str] = []
    failed: List[str] = []
    untestable: List[str] = []

    # -- confirmations shared by every trigger ------------------------------
    conf = cfg.confirmation
    vol_ok = True
    if bv(conf.require_volume_confirmation):
        lb = iv(conf.volume_lookback_sessions)
        base = float(vols.iloc[-(lb + 1):-1].mean())
        ratio = float(vols.iloc[-1]) / base if base > 0 else 0.0
        need = fv(conf.volume_multiple)
        if ratio >= need:
            passed.append(f"volume {ratio:.2f}x its {lb}-session average (need {need:g}x)")
        else:
            vol_ok = False
            failed.append(f"volume only {ratio:.2f}x its {lb}-session average, need {need:g}x")

    if bv(conf.require_delivery_confirmation):
        if "deliv_pct" in frame.columns and pd.notna(frame["deliv_pct"].iloc[-1]):
            dp = float(frame["deliv_pct"].iloc[-1])
            need = fv(conf.min_delivery_pct)
            (passed if dp >= need else failed).append(
                f"delivery {dp:.1f}% vs {need:g}% required"
            )
        else:
            untestable.append("delivery_confirmation")

    # -- triggers, in configured order --------------------------------------
    order = [str(t) for t in v(cfg.triggers.order)]
    trigger = TriggerType.NONE
    trigger_note = None

    for name in order:
        if name == "pullback" and bv(cfg.triggers.pullback.enabled):
            ok, note = _pullback(closes, highs, lows, atr_v, cfg.triggers.pullback)
            if ok:
                trigger, trigger_note = TriggerType.PULLBACK, note
                break
        elif name == "ma_reclaim" and bv(cfg.triggers.ma_reclaim.enabled):
            ok, note = _ma_reclaim(closes, cfg.triggers.ma_reclaim)
            if ok:
                trigger, trigger_note = TriggerType.MA_RECLAIM, note
                break
        elif name == "breakout" and bv(cfg.triggers.breakout.enabled):
            ok, note = _breakout(closes, highs, vols, cfg.triggers.breakout)
            if ok:
                trigger, trigger_note = TriggerType.BREAKOUT, note
                break

    if trigger_note:
        passed.append(trigger_note)

    # ---- admission: is this a trade at all? --------------------------------
    # A NAME ALREADY BELOW ITS OWN INVALIDATION LEVEL IS NOT AN ENTRY. Stage 7
    # lists thesis invalidation as exit condition #1, so opening here issues a
    # card whose stated "the thesis is dead below X" sits ABOVE the entry price
    # -- the position satisfies its first exit condition on the day it opens.
    #
    # This is the same predicate `exits.resolve_exits` uses to keep such rows
    # out of the LABEL, and it had no counterpart on the live path once Stage 6
    # stopped gating on price structure. The consequence was not cosmetic: the
    # exclusion also removes those rows from the training panel, so every
    # validation deriving rankings from it inherits the filter while the live
    # engine does not. Measured on the eligible universe, 21.8% of the
    # selection period and 26.9% of the holdout sits below the level.
    #
    # HELD NAMES ARE EXEMPT. A position already open is governed by the exit
    # band and by Stage 7's own exit hierarchy; letting an entry rule close it
    # would be the entry/holding confusion Stage 8 documents fixing.
    if not is_held and bv(cfg.admission.require_above_invalidation):
        blocked = _below_invalidation(closes, highs, lows, params)
        if blocked is not None:
            return EntryDecision(
                ticker=sym, status=EntryStatus.WATCHLIST, trigger_type=trigger,
                reference_price=round(last, 2),
                confirmations_passed=passed, confirmations_failed=failed,
                confirmations_not_testable=untestable,
                reason=blocked,
            )

    # ---- admission: rank, with hysteresis ---------------------------------
    # The trigger above is now description, not a gate. What decides admission
    # is where the name sits in the model's own ranking, and whether it is
    # already held.
    band = _admit(rank, entry_rank, exit_rank, is_held)
    if not band.admitted:
        return EntryDecision(
            ticker=sym, status=EntryStatus.WATCHLIST, trigger_type=trigger,
            reference_price=round(last, 2),
            confirmations_passed=passed, confirmations_failed=failed,
            confirmations_not_testable=untestable,
            reason=band.reason,
        )

    half = fv(cfg.entry_zone.half_width_atr) * atr_v
    lo, hi = last - half, last + half
    max_w = fv(cfg.entry_zone.max_width_pct) / 100.0 * last
    if (hi - lo) > max_w:
        lo, hi = last - max_w / 2, last + max_w / 2
    step = iv(cfg.entry_zone.round_to_paise) / 100.0
    rnd = lambda x: round(round(x / step) * step, 2) if step > 0 else round(x, 2)

    notes = [band.reason,
             f"Entry zone is +/-{fv(cfg.entry_zone.half_width_atr):g} ATR around "
             f"Rs {last:,.2f}. Beyond the upper bound the tested risk/reward no "
             f"longer holds -- do not chase."]
    if trigger is TriggerType.NONE:
        notes.append(
            "No pullback, reclaim or breakout is present. That is a note about "
            "execution, not a reason to decline: gating on it was measured at "
            "holdout Sharpe +0.46 against +1.56 for admitting on rank."
        )
    elif not vol_ok:
        notes.append(
            "Volume is below the confirmation multiple. Recorded, not gated."
        )
    return EntryDecision(
        ticker=sym, status=EntryStatus.TRIGGERED, trigger_type=trigger,
        entry_zone=(rnd(lo), rnd(hi)), reference_price=round(last, 2),
        confirmations_passed=passed, confirmations_failed=failed,
        confirmations_not_testable=untestable,
        notes=notes,
    )


def _below_invalidation(closes, highs, lows, params) -> Optional[str]:
    """Why this bar is not an entry, or None if it is one.

    Reads the level through `exits.tradeable_at_entry` -- the SAME function the
    label uses -- from `stage7_risk.thesis_invalidation`, so the rule that
    decides what the engine may buy and the rule that decides what it trains on
    cannot drift apart.

    An uncomputable level refuses the entry. A name without enough history for
    its own invalidation level has not cleared it, and treating "cannot check"
    as "passed" is the failure the eligibility stage states it exists to
    prevent.
    """
    from ..features.exits import ExitRules, invalidation_level, tradeable_at_entry

    c7 = params.stage7_risk
    n = iv(c7.thesis_invalidation.structure_ma_sessions)
    if len(closes) < n:
        return (f"fewer than {n} sessions of history, so the "
                f"{n}-session invalidation level cannot be computed. A level "
                f"that cannot be checked has not been cleared.")
    rules = ExitRules(
        invalidation_ma_sessions=n,
        invalidation_buffer_atr=fv(c7.thesis_invalidation.structure_buffer_atr),
        atr_period_sessions=iv(c7.atr.period_sessions),
        atr_method=str(v(c7.atr.method)),
    )
    a = atr(highs, lows, closes, rules.atr_period_sessions,
            rules.atr_method).dropna()
    if a.empty:
        return "ATR is not computable, so the invalidation level cannot be checked."
    ma_now = float(closes.tail(n).mean())
    atr_now = float(a.iloc[-1])
    last = float(closes.iloc[-1])
    if bool(tradeable_at_entry(last, ma_now, atr_now, rules)):
        return None
    level = invalidation_level(ma_now, atr_now, rules)
    return (
        f"Rs {last:,.2f} is already below the thesis-invalidation level of "
        f"Rs {level:,.2f} ({n}-session average less "
        f"{rules.invalidation_buffer_atr:g} ATR). Opening here would issue a "
        f"position that meets its own first exit condition on day one, and the "
        f"model was never fitted on names in this state."
    )


@dataclass(frozen=True)
class _Band:
    admitted: bool
    reason: str


def _admit(rank, entry_rank: int, exit_rank: int, is_held: bool) -> _Band:
    """Rank admission with hysteresis.

    Enter inside ``entry_rank``; once held, stay until outside ``exit_rank``.
    The gap between the two is the whole point -- a name drifting across a
    single boundary would otherwise be bought and sold at every rebalance for
    no change in view, paying a round trip each time.
    """
    if rank is None:
        return _Band(False, "no rank available for this name, so it cannot be admitted")
    if is_held:
        if rank <= exit_rank:
            return _Band(True, f"held, rank {rank} still inside the exit band of {exit_rank}")
        return _Band(False,
                     f"held, but rank {rank} has left the exit band of {exit_rank}")
    if rank <= entry_rank:
        return _Band(True, f"rank {rank}, inside the entry band of {entry_rank}")
    return _Band(False,
                 f"rank {rank}, outside the entry band of {entry_rank}. Not held, "
                 f"so the wider exit band of {exit_rank} does not apply.")


# =============================================================================
def _pullback(closes, highs, lows, atr_v, cfg) -> Tuple[bool, Optional[str]]:
    n = iv(cfg.support_ma_sessions)
    ma = sma(closes, n)
    if ma.dropna().empty:
        return False, None
    ma_v = float(ma.iloc[-1])
    last = float(closes.iloc[-1])
    if last < ma_v:
        return False, None
    dist_atr = (last - ma_v) / atr_v if atr_v > 0 else 99.0
    if dist_atr > fv(cfg.max_distance_atr):
        return False, None
    if bv(cfg.require_reversal_candle):
        rng = float(highs.iloc[-1]) - float(lows.iloc[-1])
        if rng <= 0:
            return False, None
        pos = (last - float(lows.iloc[-1])) / rng
        if pos < fv(cfg.min_close_position_in_range):
            return False, None
        return True, (f"pullback to within {dist_atr:.2f} ATR of the {n}-session "
                      f"average, closing in the top {(1-pos)*100:.0f}% of the range")
    return True, f"pullback to within {dist_atr:.2f} ATR of the {n}-session average"


def _ma_reclaim(closes, cfg) -> Tuple[bool, Optional[str]]:
    n = iv(cfg.ma_sessions)
    lb = iv(cfg.lookback_sessions)
    ma = sma(closes, n)
    if ma.dropna().empty or len(closes) < lb + 2:
        return False, None
    above = closes > ma
    if not bool(above.iloc[-1]):
        return False, None
    recent = above.iloc[-(lb + 1):-1]
    if bool(recent.all()) or recent.isna().all():
        return False, None
    return True, f"reclaimed the {n}-session average after trading below it within {lb} sessions"


def _breakout(closes, highs, vols, cfg) -> Tuple[bool, Optional[str]]:
    lb = iv(cfg.lookback_sessions)
    if len(highs) < lb + 2:
        return False, None
    prior_high = float(highs.iloc[-(lb + 1):-1].max())
    last = float(closes.iloc[-1])
    margin = (last - prior_high) / prior_high * 100.0
    if margin < fv(cfg.min_breakout_margin_pct):
        return False, None
    base = float(vols.iloc[-(lb + 1):-1].mean())
    ratio = float(vols.iloc[-1]) / base if base > 0 else 0.0
    if ratio < fv(cfg.min_volume_multiple):
        return False, None
    return True, (f"closed {margin:.2f}% above the {lb}-session high on {ratio:.1f}x "
                  f"average volume")
