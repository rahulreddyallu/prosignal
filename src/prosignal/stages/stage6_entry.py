"""Stage 6 -- Entry confirmation.

A high rank is not converted into an entry unless price is at a location where
a defined stop is close enough for the risk/reward to work.

Triggers are tried in configured order, first match wins:

    pullback     strength returning to support -- best risk/reward, since the
                 stop sits just under structure
    ma_reclaim   a stock reclaiming a reference level it had lost
    breakout     a move through prior resistance on real volume -- worst
                 risk/reward, since the stop is furthest away

No trigger means WATCHLIST, not BUY: the engine may rate a stock highly and
still decline to buy it today.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Tuple

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
) -> EntryReport:
    p = config.params
    cfg = p.stage6_entry
    decisions: Dict[str, EntryDecision] = {}

    for sym in candidates:
        frame = frames.get(sym)
        if frame is None or len(frame) < 70:
            decisions[sym] = EntryDecision(
                ticker=sym, status=EntryStatus.NOT_TRIGGERED,
                reason="insufficient history to evaluate an entry trigger",
            )
            continue
        decisions[sym] = _evaluate(sym, frame, cfg, p)

    log.info(
        "stage 6 complete",
        extra={"triggered": sum(1 for d in decisions.values() if d.status is EntryStatus.TRIGGERED),
               "watchlist": sum(1 for d in decisions.values() if d.status is EntryStatus.WATCHLIST)},
    )
    return EntryReport(as_of_date=as_of, decisions=decisions)


def _evaluate(sym, frame, cfg, params) -> EntryDecision:
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

    if trigger is TriggerType.NONE:
        return EntryDecision(
            ticker=sym, status=EntryStatus.WATCHLIST, trigger_type=TriggerType.NONE,
            reference_price=round(last, 2),
            confirmations_passed=passed, confirmations_failed=failed,
            confirmations_not_testable=untestable,
            reason=(
                "ranks well but no entry trigger is active. A good stock at a bad "
                "price is a bad trade -- held on the watchlist rather than chased."
            ),
        )

    passed.append(trigger_note)
    if not vol_ok:
        return EntryDecision(
            ticker=sym, status=EntryStatus.WATCHLIST, trigger_type=trigger,
            reference_price=round(last, 2),
            confirmations_passed=passed, confirmations_failed=failed,
            confirmations_not_testable=untestable,
            reason="trigger fired but volume confirmation failed",
        )

    half = fv(cfg.entry_zone.half_width_atr) * atr_v
    lo, hi = last - half, last + half
    max_w = fv(cfg.entry_zone.max_width_pct) / 100.0 * last
    if (hi - lo) > max_w:
        lo, hi = last - max_w / 2, last + max_w / 2
    step = iv(cfg.entry_zone.round_to_paise) / 100.0
    rnd = lambda x: round(round(x / step) * step, 2) if step > 0 else round(x, 2)

    return EntryDecision(
        ticker=sym, status=EntryStatus.TRIGGERED, trigger_type=trigger,
        entry_zone=(rnd(lo), rnd(hi)), reference_price=round(last, 2),
        confirmations_passed=passed, confirmations_failed=failed,
        confirmations_not_testable=untestable,
        notes=[f"Entry zone is +/-{fv(cfg.entry_zone.half_width_atr):g} ATR around "
               f"Rs {last:,.2f}. Beyond the upper bound the tested risk/reward no "
               f"longer holds -- do not chase."],
    )


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
