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

from ._cfg import fv, iv, v
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

    top_n = iv(cfg.top_n_to_defend)
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
    market.append(_regime_transition(regime, cfg))
    market.append(_volatility_shock(store, as_of, cfg, p))
    market.append(_momentum_crash_market(regime, cfg))
    market_penalty = sum(c.penalty for c in market)
    halt = any(c.outcome is CheckOutcome.HARD_REJECT for c in market)

    per_stock: Dict[str, StockDefenseResult] = {}
    max_pen = fv(cfg.max_cumulative_penalty)

    for cand in candidates:
        sym = cand.ticker
        frame = grouped.get(sym)
        checks: List[CheckResult] = []

        if frame is None or len(frame) < 30:
            checks.append(_nt("insufficient_history", "fewer than 30 sessions in window"))
        else:
            checks.append(_low_volume_breakout(frame, cfg.low_volume_breakout))
            checks.append(_liquidity_distortion(frame, cfg.liquidity_distortion))
            checks.append(_gap_signal(frame, cfg.gap_signal, p))
            checks.append(_news_spike(frame, cfg.news_spike))
            checks.append(_overextension(frame, cfg.overextension, p))
            checks.append(_beta_explained(frame, bench, cfg.beta_explained_move))
            checks.append(_corporate_action(frame, actions, sym, as_of, cfg.corporate_action_distortion))

        checks.append(_earnings_distortion(earnings, sym, as_of, calendar,
                                           cfg.earnings_distortion))

        total = sum(c.penalty for c in checks) + market_penalty
        hard = [c for c in checks if c.outcome is CheckOutcome.HARD_REJECT]
        status = "CLEARED"
        if hard:
            status = "REJECTED"
        elif total >= max_pen:
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
    """A single unexplained sigma-event tends to mean-revert, not continue."""
    closes = pd.Series(frame["close"].to_numpy(dtype="float64"))
    rets = closes.pct_change(fill_method=None).dropna()
    if len(rets) < 70:
        return _nt("news_spike", "insufficient return history")
    sm = sigma_move(rets, window=63)
    if sm is None:
        return _nt("news_spike", "zero-variance return distribution")
    lim = fv(cfg.move_sigma)
    obs = {"sigma_move": round(sm, 2), "limit": lim}
    if abs(sm) > lim:
        return _pen("news_spike", fv(cfg.penalty),
                    f"latest session is a {sm:+.1f} sigma move against its own 63-session "
                    f"distribution -- an event, not a trend continuation", obs, lim)
    return _ok("news_spike", obs)


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
    if actions is None or actions.empty:
        return _nt("corporate_action_distortion", "corporate actions feed empty")
    lb = iv(cfg.lookback_sessions)
    a = actions[actions[SYMBOL] == sym].copy()
    if a.empty:
        return _ok("corporate_action_distortion", {"actions_in_window": 0})
    a["ex_date"] = pd.to_datetime(a["ex_date"], errors="coerce").dt.date
    recent = a[(a["ex_date"].notna()) & (a["ex_date"] <= as_of)
               & (a["ex_date"] >= as_of - dt.timedelta(days=int(lb * 1.5)))]
    if recent.empty:
        return _ok("corporate_action_distortion", {"actions_in_window": 0})
    return _rej("corporate_action_distortion",
                f"{len(recent)} corporate action(s) inside the {lb}-session lookback; "
                f"return series across the ex-date is not comparable",
                {"actions_in_window": int(len(recent))})


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


def _momentum_crash_market(regime, cfg) -> CheckResult:
    if regime.regime_bucket == "uptrend_highvol_rebound":
        return _rej("momentum_crash",
                    "Daniel & Moskowitz momentum-crash state: prior decline, high "
                    "volatility, sharp rebound. Momentum historically inverts hardest here.",
                    {"bucket": regime.regime_bucket})
    return _ok("momentum_crash", {"bucket": regime.regime_bucket})
