"""Stage 5 -- False-signal defense.

Looks for reasons a candidate is wrong. A screen that only accumulates
confirming evidence returns confirmations.

Four outcomes per check:

    PASS           the check ran and found nothing
    SCORE_PENALTY  the check ran and found something worth discounting
    HARD_REJECT    the check ran and found a disqualifying condition
    NOT_TESTABLE   the check could not run

NOT_TESTABLE is never upgraded to PASS. Untestable checks are printed on the
card, so a signal built on partial evidence says so.

Only checks whose inputs exist are implemented. Checks needing data the engine
cannot obtain report NOT_TESTABLE by construction rather than existing as dead
code -- see DATA_SOURCES.md.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ._cfg import bv, fv, iv, v
from ..core.calendar import TradingCalendar
from ..core.contracts import (
    CheckResult,
    CoreScoreReport,
    FalseSignalReport,
    RegimeState,
    StockDefenseResult,
)
from ..core.enums import CheckOutcome
from ..core.logging import get_logger
from ..data.store import DataStore
from ..data.types import DATE, SYMBOL
from ..indicators import atr, rate_of_change_pct, sigma_move, trailing_return

__all__ = ["run", "STAGE_NAME"]

STAGE_NAME = "stage5_false_signal"
log = get_logger(__name__)


def run(
    scores: CoreScoreReport,
    store: DataStore,
    calendar: TradingCalendar,
    regime: RegimeState,
    config,
    as_of: Optional[dt.date] = None,
) -> FalseSignalReport:
    p = config.params
    cfg = p.stage5_false_signal
    as_of = as_of or scores.as_of_date

    # Everything that could still become a signal has to be argued against.
    # A fixed count was tuned when the universe held 145 eligible names, where
    # 25 covered the top 17%. On the point-in-time universe it covers the top
    # 4%, which silently dropped names that clear the Stage 8 score gate: they
    # were never defended, so they could never be issued, and the funnel read
    # "survived defense 25" when 25 was the cap rather than the attrition.
    gate = p.stage8_final_signal.scarcity
    min_score = fv(gate.min_composite_score)
    min_pct = fv(gate.min_universe_percentile)
    eligible_for_signal = [
        sc for sc in scores.ranked_scores
        if sc.composite_score >= min_score and sc.percentile >= min_pct
    ]
    top_n = max(iv(cfg.top_n_to_defend), len(eligible_for_signal))
    candidates = scores.ranked_scores[:top_n]
    if not candidates:
        return FalseSignalReport(as_of_date=as_of)

    symbols = [c.ticker for c in candidates]
    need = max(iv(p.stage5_false_signal.beta_explained_move.beta_estimation_sessions), 260) + 10
    window = calendar.trailing_window(as_of, need)
    start = window[0] if window else calendar.first
    prices = store.read_prices(symbols=symbols, start=start, end=as_of).copy()
    prices[DATE] = pd.to_datetime(prices[DATE]).dt.normalize()
    grouped = {s: f.sort_values(DATE) for s, f in prices.groupby(SYMBOL, sort=False, observed=True)}

    bench = store.index_series(str(v(p.stage2_regime.benchmark_index)), "close", end=as_of)
    actions = store.read_corporate_actions()
    earnings = store.read_earnings_calendar()

    # -- market-wide checks --------------------------------------------------
    market: List[CheckResult] = []
    if bv(cfg.regime_transition.enabled):
        market.append(_regime_transition(regime, cfg))
    if bv(cfg.volatility_shock.enabled):
        market.append(_volatility_shock(store, as_of, cfg, p))
    if bv(cfg.momentum_crash.enabled):
        market.append(_momentum_crash_market(
            regime, cfg, v(p.stage2_regime.no_new_entry_buckets)))
    market_penalty = sum(c.penalty for c in market)
    halt = any(c.outcome is CheckOutcome.HARD_REJECT for c in market)

    per_stock: Dict[str, StockDefenseResult] = {}
    max_pen = fv(cfg.max_cumulative_penalty)

    for cand in candidates:
        sym = cand.ticker
        frame = grouped.get(sym)
        checks: List[CheckResult] = []

        # EVERY CHECK IS GATED ON ITS OWN `enabled` FLAG.
        #
        # Twelve of them are declared in parameters.yaml and none was read: the
        # checks ran unconditionally, so setting `enabled: false` on any of them
        # changed nothing. The liveness checker could not see it either --
        # `enabled` is a SHARED_LEAF_NAME consumed by Stage 4, Stage 6 and the
        # providers, so the name reads as live and twelve unread copies hid
        # behind it. That is the same defect as the duplicate
        # `min_win_probability`, twelve times over.
        def add(flag, fn, *args):
            if bv(flag.enabled):
                checks.append(fn(*args))

        if frame is None or len(frame) < 30:
            checks.append(_nt("insufficient_history", "fewer than 30 sessions in window"))
        else:
            add(cfg.low_volume_breakout, _low_volume_breakout, frame, cfg.low_volume_breakout)
            add(cfg.liquidity_distortion, _liquidity_distortion, frame, cfg.liquidity_distortion)
            add(cfg.gap_signal, _gap_signal, frame, cfg.gap_signal, p)
            add(cfg.news_spike, _news_spike, frame, cfg.news_spike)
            add(cfg.overextension, _overextension, frame, cfg.overextension, p)
            add(cfg.beta_explained_move, _beta_explained, frame, bench, cfg.beta_explained_move)
            add(cfg.corporate_action_distortion, _corporate_action, frame, actions,
                sym, as_of, cfg.corporate_action_distortion)

        add(cfg.earnings_distortion, _earnings_distortion, earnings, sym, as_of,
            calendar, cfg.earnings_distortion)

        # THE CAP IS A TEST ON THIS NAME'S OWN EVIDENCE, so it is measured on
        # this name's own penalties. `market_penalty` is identical for every
        # candidate -- it cannot reorder anything -- but adding it before the
        # threshold turned market weather into a per-stock rejection.
        #
        # The arithmetic: regime_transition contributes 0.10 and
        # volatility_shock 0.15, so a transitioning market with a VIX spike puts
        # 0.25 on every name before any stock-specific check runs, against a cap
        # of 0.35. Every per-stock penalty is 0.10 or larger, so in that state
        # the survival condition collapsed to "zero stock-level flags" -- one
        # below-average volume session was a hard rejection. Two market
        # conditions that say nothing about a particular stock became the
        # reason that stock was refused.
        #
        # The market penalty still lowers the SCORE, so a hostile tape is still
        # priced. It just no longer decides whether the case against one name is
        # overwhelming.
        stock_penalty = sum(c.penalty for c in checks)
        total = stock_penalty + market_penalty
        hard = [c for c in checks if c.outcome is CheckOutcome.HARD_REJECT]
        status = "CLEARED"
        if hard:
            status = "REJECTED"
        elif stock_penalty >= max_pen:
            status = "REJECTED"
        elif total > 0:
            status = "PENALIZED"

        before = cand.composite_score
        after = max(before - total, 0.0)
        per_stock[sym] = StockDefenseResult(
            ticker=sym, checks=checks, total_penalty=round(total, 4),
            score_before=before, score_after=round(after, 4), final_status=status,
        )

    log.info(
        "stage 5 complete",
        extra={"defended_top_n": len(per_stock),
               "rejected": sum(1 for r in per_stock.values() if r.final_status == "REJECTED")},
    )
    return FalseSignalReport(
        as_of_date=as_of, market_wide_checks=market,
        market_wide_penalty=round(market_penalty, 4), market_halt=halt,
        market_halt_reason=next((c.reason for c in market if c.outcome is CheckOutcome.HARD_REJECT), None),
        per_stock=per_stock,
    )


# =============================================================================
def _nt(name: str, reason: str) -> CheckResult:
    return CheckResult(check=name, outcome=CheckOutcome.NOT_TESTABLE, reason=reason)


def _ok(name: str, observed=None) -> CheckResult:
    return CheckResult(check=name, outcome=CheckOutcome.PASS, observed=observed or {})


def _pen(name, penalty, reason, observed=None, threshold=None, cite=None) -> CheckResult:
    return CheckResult(
        check=name, outcome=CheckOutcome.SCORE_PENALTY, penalty=float(penalty),
        reason=reason, observed=observed or {}, threshold=threshold, citation=cite,
    )


def _rej(name, reason, observed=None) -> CheckResult:
    return CheckResult(
        check=name, outcome=CheckOutcome.HARD_REJECT, reason=reason, observed=observed or {}
    )


# -- stock-level checks -------------------------------------------------------
def _low_volume_breakout(frame, cfg) -> CheckResult:
    """A move without participation is a move without conviction."""
    lb = iv(cfg.lookback_sessions)
    vol = pd.to_numeric(frame["volume"], errors="coerce").dropna()
    if len(vol) < lb + 1:
        return _nt("low_volume_breakout", "insufficient volume history")
    ratio = float(vol.iloc[-1]) / float(vol.iloc[-(lb + 1):-1].mean())
    need = fv(cfg.min_volume_multiple)
    obs = {"volume_ratio": round(ratio, 2), "required": need}
    if ratio < need:
        return _pen("low_volume_breakout", fv(cfg.penalty),
                    f"latest session volume is {ratio:.2f}x its {lb}-session average, "
                    f"below the {need:.2f}x participation this check requires", obs, need)
    return _ok("low_volume_breakout", obs)


def _liquidity_distortion(frame, cfg) -> CheckResult:
    """Today's turnover collapsing vs its own norm means the tape is thin."""
    to = pd.to_numeric(frame.get("turnover"), errors="coerce").dropna()
    if len(to) < 22:
        return _nt("liquidity_distortion", "turnover history unavailable")
    ratio = float(to.iloc[-1]) / float(to.iloc[-22:-1].mean())
    need = fv(cfg.min_session_turnover_vs_adtv)
    obs = {"turnover_vs_adtv": round(ratio, 3), "required": need}
    if ratio < need:
        return _rej("liquidity_distortion",
                    f"session turnover is {ratio:.1%} of its 21-session average, below "
                    f"{need:.0%}. Executing here would move the price, not meet it.", obs)
    return _ok("liquidity_distortion", obs)


def _gap_signal(frame, cfg, params) -> CheckResult:
    """A move delivered by an opening gap is not a move you could have joined."""
    a = atr(frame["high"], frame["low"], frame["close"],
            iv(params.stage7_risk.atr.period_sessions), str(v(params.stage7_risk.atr.method)))
    if a.dropna().empty:
        return _nt("gap_signal", "ATR not computable")
    atr_now = float(a.dropna().iloc[-1])
    if not np.isfinite(atr_now) or atr_now <= 0.0:
        # A halted or genuinely flat scrip has zero true range. Dividing by it
        # raised ZeroDivisionError and took the whole run down; the honest
        # answer is that this check cannot be evaluated for this name.
        return _nt("gap_signal", "ATR is zero; gap cannot be scaled")
    prev_close = float(frame["close"].iloc[-2])
    today_open = float(frame["open"].iloc[-1])
    gap = abs(today_open - prev_close)
    gap_atr = gap / atr_now
    lim = fv(cfg.max_gap_atr_multiple)
    obs = {"gap_atr_multiple": round(gap_atr, 2), "limit": lim}
    if gap_atr > lim:
        return _pen("gap_signal", fv(cfg.penalty),
                    f"opened {gap_atr:.2f} ATR away from the prior close. A gap of this "
                    f"size is price you could not have transacted at.", obs, lim)
    return _ok("gap_signal", obs)


def _news_spike(frame, cfg) -> CheckResult:
    """A single unexplained sigma-event tends to mean-revert, not continue.

    Two conditions, both from the config: an outsized move AND the volume that
    makes it an event rather than a thin-book print. The volume leg was
    declared in parameters.yaml and never read, so the check fired on the move
    alone -- 1.30% of stock-sessions against 1.01% with volume required, 2.7x
    its own specification once persistence is counted too, applying a 0.12
    penalty every time.
    """
    closes = pd.Series(frame["close"].to_numpy(dtype="float64"))
    rets = closes.pct_change(fill_method=None).dropna()
    if len(rets) < 70:
        return _nt("news_spike", "insufficient return history")
    sm = sigma_move(rets, window=63)
    if sm is None:
        return _nt("news_spike", "zero-variance return distribution")
    lim = fv(cfg.move_sigma)
    need_vol = fv(cfg.volume_multiple)

    ratio = None
    if "volume" in frame.columns and len(frame) >= 21:
        vols = pd.Series(frame["volume"].to_numpy(dtype="float64"))
        base = float(vols.iloc[-21:-1].mean())
        if base > 0:
            ratio = float(vols.iloc[-1]) / base
    obs = {"sigma_move": round(sm, 2), "limit": lim,
           "volume_multiple": None if ratio is None else round(ratio, 2),
           "volume_limit": need_vol}

    if abs(sm) <= lim:
        return _ok("news_spike", obs)
    if ratio is None:
        # No volume to judge by. The move alone is not enough to penalise on,
        # and inventing a ratio would be worse than saying so.
        return _nt("news_spike", "volume unavailable; the move alone does not "
                                 "distinguish an event from a thin print")
    if ratio < need_vol:
        return _ok("news_spike", obs)
    return _pen("news_spike", fv(cfg.penalty),
                f"latest session is a {sm:+.1f} sigma move on {ratio:.1f}x its "
                f"20-session average volume -- an event, not a trend continuation",
                obs, lim)


def _overextension(frame, cfg, params) -> CheckResult:
    """Buying after the move is the most common way momentum loses money."""
    a = atr(frame["high"], frame["low"], frame["close"],
            iv(params.stage7_risk.atr.period_sessions), str(v(params.stage7_risk.atr.method))).dropna()
    if a.empty:
        return _nt("overextension", "ATR not computable")
    h = iv(cfg.short_horizon_sessions)
    closes = pd.Series(frame["close"].to_numpy(dtype="float64"))
    if len(closes) < h + 1:
        return _nt("overextension", "insufficient history")
    move = float(closes.iloc[-1]) - float(closes.iloc[-(h + 1)])
    in_atr = move / float(a.iloc[-1])
    lim = fv(cfg.extended_atr_multiple)
    obs = {"move_in_atr": round(in_atr, 2), "horizon_sessions": h, "limit": lim}
    if in_atr > lim:
        return _pen("overextension", fv(cfg.penalty),
                    f"up {in_atr:.1f} ATR in {h} sessions. Entry here pays for a move "
                    f"that has already happened and sits far from any stop.", obs, lim)
    return _ok("overextension", obs)


def _beta_explained(frame, bench, cfg) -> CheckResult:
    """If the move is just market beta, it is not stock-specific evidence."""
    if bench is None or bench.empty:
        return _nt("beta_explained_move", "benchmark unavailable")
    px = pd.Series(frame["close"].to_numpy(dtype="float64"),
                   index=pd.DatetimeIndex(frame[DATE])).dropna()
    est = iv(cfg.beta_estimation_sessions)
    lb = iv(cfg.lookback_sessions)
    joined = pd.concat([px.rename("s"), bench.rename("b")], axis=1).dropna()
    if len(joined) < est + 5:
        return _nt("beta_explained_move", f"need {est} overlapping sessions, have {len(joined)}")

    r = joined.pct_change(fill_method=None).dropna().tail(est)
    var = float(r["b"].var())
    if var <= 0:
        return _nt("beta_explained_move", "zero benchmark variance")
    beta = float(r["s"].cov(r["b"]) / var)

    stock_ret = trailing_return(joined["s"], lb)
    bench_ret = trailing_return(joined["b"], lb)
    if stock_ret is None or bench_ret is None or abs(stock_ret) < 1e-9:
        return _nt("beta_explained_move", "return window unavailable")

    explained = (beta * bench_ret) / stock_ret
    thr = fv(cfg.explained_fraction_threshold)
    obs = {"beta": round(beta, 2), "stock_return_pct": round(stock_ret * 100, 2),
           "benchmark_return_pct": round(bench_ret * 100, 2),
           "fraction_explained": round(explained, 3), "threshold": thr}
    if explained > thr:
        over = min((explained - thr) / max(1 - thr, 1e-9), 1.0)
        return _pen("beta_explained_move", fv(cfg.max_penalty) * over,
                    f"{explained:.0%} of the {lb}-session move is explained by beta "
                    f"{beta:.2f} to the index. This is market exposure, not selection.",
                    obs, thr)
    return _ok("beta_explained_move", obs)


def _earnings_distortion(earnings, sym, as_of, calendar, cfg) -> CheckResult:
    """Results too close on either side of the decision date.

    Ahead of a print the move is a coin flip on a number nobody has; just after
    one the drift is the market repricing that number, not the factors this
    engine ranks on. Only company-filed dates count as confirmed -- an estimate
    projected from past quarters cannot support a hard rejection, so a name
    with nothing but an estimate is reported NOT_TESTABLE rather than passed.
    """
    if earnings is None or earnings.empty:
        return _nt("earnings_distortion", "earnings calendar empty")
    rows = earnings[earnings[SYMBOL] == sym]
    if rows.empty:
        return _nt("earnings_distortion", "no results date on file for this name")

    confirmed = rows[rows["confirmed"] == True] if "confirmed" in rows.columns else rows  # noqa: E712
    if confirmed.empty:
        return _nt(
            "earnings_distortion",
            "only estimated results dates on file; an estimate cannot support a rejection",
        )

    dates = pd.to_datetime(confirmed["earnings_date"]).dt.date
    ahead = int(iv(cfg.upcoming_earnings_sessions))
    behind = int(iv(cfg.recent_earnings_sessions))

    upcoming = [d for d in dates if d >= as_of]
    if upcoming:
        sessions = calendar.sessions_until(as_of, min(upcoming))
        if sessions <= ahead:
            return _rej(
                "earnings_distortion",
                f"results due in {sessions} session(s); the move will price an "
                f"earnings number this engine has no view on",
                {"sessions_to_results": sessions, "results_date": str(min(upcoming))},
            )

    past = [d for d in dates if d < as_of]
    if past:
        since = calendar.sessions_until(max(past), as_of)
        if since <= behind:
            return _pen(
                "earnings_distortion",
                fv(cfg.recent_earnings_penalty),
                f"results {since} session(s) ago; post-earnings drift is "
                f"repricing that print, not the ranked factors",
                {"sessions_since_results": since, "results_date": str(max(past))},
                threshold=behind,
            )
    return _ok("earnings_distortion", {"confirmed_dates": int(len(confirmed))})


def _corporate_action(frame, actions, sym, as_of, cfg) -> CheckResult:
    """Reject on an action that RESCALED the price, not on any action at all.

    The rejection reason is that the return series across the ex-date is not
    comparable. That is true of a split or a bonus, which rescale the price by
    a factor and are adjusted for in the store. It is NOT true of a dividend:
    `parse_action_subject` assigns dividends a price factor of 1.0 and
    `build_adjustment_factors` filters on `ratio != 1.0`, so nothing is
    rescaled and the only discontinuity is the ex-dividend drop itself --
    typically well inside one session's volatility for an Indian large cap.

    Rejecting on any row made this a dividend filter. Measured on 2026-08-25
    the 45-day window held 88 actions across 85 symbols and 81 of them were
    ordinary dividends; dividends are 5,058 of the 5,965 rows in the feed. Ex
    dates cluster hard -- 70 symbols in July 2026 against 11 in December -- so
    the effect was a seasonal cull that removed high-yield names from the book
    for a reason the price series does not actually have.
    """
    if actions is None or actions.empty:
        return _nt("corporate_action_distortion", "corporate actions feed empty")
    lb = iv(cfg.lookback_sessions)
    a = actions[actions[SYMBOL] == sym].copy()
    if a.empty:
        return _ok("corporate_action_distortion", {"actions_in_window": 0})
    # NaT is dropped BEFORE the comparison, not filtered inside it. `.dt.date`
    # on a coerced column leaves NaT as an object, and `object <= datetime.date`
    # raises rather than evaluating False -- so one unparseable ex_date in the
    # feed took down the whole run from inside the per-stock loop, which has no
    # handler. `notna()` sat in the same boolean expression as the comparison,
    # which does not help: pandas evaluates both operands before combining them.
    a["ex_date"] = pd.to_datetime(a["ex_date"], errors="coerce")
    a = a[a["ex_date"].notna()]
    if a.empty:
        return _ok("corporate_action_distortion",
                   {"actions_in_window": 0, "unparseable_ex_dates": True})
    a["ex_date"] = a["ex_date"].dt.date
    recent = a[(a["ex_date"] <= as_of)
               & (a["ex_date"] >= as_of - dt.timedelta(days=int(lb * 1.5)))]
    if recent.empty:
        return _ok("corporate_action_distortion", {"actions_in_window": 0})

    # Only a factor that moved the share count breaks comparability -- but the
    # columns needed to establish that are not guaranteed. `ratio` is canonical
    # (CORPORATE_ACTION_COLUMNS) and the NSE ingest fills it, while the
    # reference-CSV override ships `ratio_from,ratio_to` and no `ratio` at all,
    # so a feed arriving by that path has neither the column nor a scalar to
    # coerce. Reading it unguarded raised inside the per-stock loop, which has
    # no handler, and took the whole run down.
    types = (recent["action_type"].astype(str).str.lower()
             if "action_type" in recent.columns else None)
    ratio = (pd.to_numeric(recent["ratio"], errors="coerce")
             if "ratio" in recent.columns else None)

    if ratio is not None and ratio.notna().any():
        rescaling = recent[ratio.notna() & (ratio > 0) & (ratio != 1.0)]
        basis = "ratio"
    elif types is not None:
        # No usable ratio, so classify by KIND. A split, bonus or rights issue
        # rescales; a dividend does not. Anything unrecognised is treated as
        # rescaling, because an action we cannot classify is not one we have
        # cleared.
        rescaling = recent[~types.isin({"dividend"})]
        basis = "action_type"
    else:
        # Neither column. The check cannot run, and NOT_TESTABLE is never
        # upgraded to PASS -- this stage's contract.
        return _nt("corporate_action_distortion",
                   f"{len(recent)} action(s) in the window carry neither a "
                   f"ratio nor an action type, so a price rescaling cannot be "
                   f"distinguished from a dividend")

    obs = {"actions_in_window": int(len(recent)),
           "rescaling_actions": int(len(rescaling)),
           "classified_by": basis,
           "types": (sorted({str(x) for x in types.dropna()})
                     if types is not None else [])}
    if rescaling.empty:
        return _ok("corporate_action_distortion", obs)
    return _rej("corporate_action_distortion",
                f"{len(rescaling)} price-rescaling corporate action(s) "
                f"(split, bonus or rights) inside the {lb}-session lookback; "
                f"the return series across the ex-date is not comparable",
                obs)


# -- market-level checks ------------------------------------------------------
def _regime_transition(regime, cfg) -> CheckResult:
    if regime.transition_flag:
        return _pen("regime_transition", fv(cfg.regime_transition.penalty),
                    f"regime is changing ({', '.join(regime.transition_components)}). "
                    f"Historical relationships are least reliable mid-transition.",
                    {"components": regime.transition_components})
    return _ok("regime_transition", {"transition": False})


def _volatility_shock(store, as_of, cfg, params) -> CheckResult:
    from ..data.providers.nse_archives import INDIA_VIX_NAME
    vix = store.index_series(INDIA_VIX_NAME, "close", end=as_of)
    if vix.empty:
        return _nt("volatility_shock", "India VIX unavailable")
    lb = iv(cfg.volatility_shock.vix_spike_lookback_sessions)
    roc = rate_of_change_pct(vix, lb)
    if roc is None:
        return _nt("volatility_shock", "insufficient VIX history")
    lim = fv(cfg.volatility_shock.vix_spike_pct)
    obs = {"vix_change_pct": round(roc, 1), "limit": lim, "lookback": lb}
    if roc > lim:
        return _pen("volatility_shock", fv(cfg.volatility_shock.penalty),
                    f"India VIX up {roc:.0f}% in {lb} sessions", obs, lim)
    return _ok("volatility_shock", obs)


def _momentum_crash_market(regime, cfg, no_entry_buckets=()) -> CheckResult:
    """The Daniel & Moskowitz crash state.

    ONE list of hostile buckets, read from `stage2_regime.no_new_entry_buckets`
    rather than named again here. Both mechanisms block the same bucket -- Stage
    2 by refusing new entries and this by halting the market -- so a hardcoded
    literal meant the funnel attributed the block twice and editing the config
    moved only one of them.
    """
    if regime.regime_bucket in set(no_entry_buckets or ()) or \
            regime.regime_bucket == "uptrend_highvol_rebound":
        return _rej("momentum_crash",
                    "Daniel & Moskowitz momentum-crash state: prior decline, high "
                    "volatility, sharp rebound. Momentum historically inverts hardest here.",
                    {"bucket": regime.regime_bucket})
    return _ok("momentum_crash", {"bucket": regime.regime_bucket})
