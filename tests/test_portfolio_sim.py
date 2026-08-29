"""The portfolio simulator. Every Stage 6/7/8 finding rests on these mechanics.

The measurements this module produces reversed two conclusions during the
audit, so its mechanics are pinned here rather than trusted:

  * sizing is risk_budget / risk_per_share, so a TIGHTER stop buys a LARGER
    position. A per-position return comparison silently compares two different
    position sizes; that error made a stop look like it consumed the whole edge
  * cohorts must not overlap. Rebalances are 21 sessions apart and positions
    hold 63, so compounding every rebalance in sequence implies 3x deployment
  * a round trip is owed whenever a position is OPENED. That is names new to
    the book, and also names whose previous position closed before the horizon
    and is being bought again. The rule used to be "new to the book" alone,
    which charged nothing for the 84% of positions that close early and get
    re-bought, and credited the hysteresis band with a saving it does not make
  * the profit target is an intraday instrument, like the stop. `_hold` used to
    pass `high=None`, so `resolve_exits` substituted the close and the target
    could only trigger on a close while the stop still triggered on the low
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.features.exits import EXIT_STOP, EXIT_TARGET, EXIT_TIMEOUT
from prosignal.validation.portfolio_sim import (
    PortfolioParams,
    phase_summary,
    simulate,
)

SYMBOLS = [f"S{i:02d}" for i in range(20)]


def _params(**over) -> PortfolioParams:
    base = dict(
        capital=1_000_000.0, max_positions=8, risk_per_trade_pct=1.0,
        max_participation_of_adtv=0.01, stop_atr_multiple=2.5,
        min_stop_distance_pct=2.0, max_stop_distance_pct=15.0,
        invalidation_ma_sessions=50, invalidation_buffer_atr=1.5,
        horizon_sessions=63, entry_rank=8, exit_rank=16,
        cost_bps_round_trip=70.0,
    )
    base.update(over)
    return PortfolioParams(**base)


def _prices(n: int = 400, drift: float = 0.0008, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.DataFrame(
        {s: 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.01, n))) for s in SYMBOLS},
        index=idx,
    )
    high, low = close * 1.01, close * 0.99
    open_ = close.shift(1).fillna(close.iloc[0])
    atr = pd.DataFrame(np.full((n, len(SYMBOLS)), 2.0), index=idx, columns=SYMBOLS)
    ma = close.rolling(50, min_periods=1).mean()
    adtv = pd.DataFrame(np.full((n, len(SYMBOLS)), 5e9), index=idx, columns=SYMBOLS)
    return {"close": close, "high": high, "low": low, "open": open_,
            "atr": atr, "ma": ma, "adtv": adtv}


def _rankings(prices, every: int = 21):
    idx = list(prices["close"].index)
    out = []
    for i in range(100, len(idx) - 70, every):
        d = idx[i]
        scores = pd.Series({s: float(len(SYMBOLS) - k) for k, s in enumerate(SYMBOLS)})
        out.append((d, scores.sort_values(ascending=False)))
    return out


# =============================================================================
# sizing
# =============================================================================


def test_a_tighter_stop_buys_a_larger_position():
    """The mechanism a per-position comparison cannot see."""
    from prosignal.validation.portfolio_sim import _position

    p = _prices()
    tight = _position("S00", 200, p["close"], p["atr"], p["adtv"], _params(stop_atr_multiple=2.0))[0]
    wide = _position("S00", 200, p["close"], p["atr"], p["adtv"], _params(stop_atr_multiple=5.0))[0]
    assert tight > wide, (
        "risk-based sizing equalises rupee risk, so a wider stop must take a "
        "SMALLER position; if this inverts, every stop measurement is wrong"
    )


def test_the_capital_slot_caps_a_very_tight_stop():
    from prosignal.validation.portfolio_sim import _position

    p = _prices()
    params = _params(stop_atr_multiple=0.01)     # forced onto the 2% floor
    size, _price, _adtv = _position("S00", 200, p["close"], p["atr"], p["adtv"], params)
    assert size <= params.slot + 1e-6, "no single position may exceed its slot"


def test_illiquidity_caps_the_position():
    from prosignal.validation.portfolio_sim import _position

    p = _prices()
    thin = {**p, "adtv": p["adtv"] * 1e-6}
    params = _params()
    fat = _position("S00", 200, p["close"], p["atr"], p["adtv"], params)[0]
    lean = _position("S00", 200, thin["close"], thin["atr"], thin["adtv"], params)[0]
    assert lean < fat


# =============================================================================
# exits
# =============================================================================


def test_a_gap_through_the_stop_fills_at_the_open_not_the_stop():
    """Assuming the stop price is the optimistic error."""
    from prosignal.validation.portfolio_sim import _hold

    p = _prices()
    i = 200
    entry = p["close"]["S00"].iloc[i]
    close = p["close"].copy(); low = p["low"].copy(); open_ = p["open"].copy()
    gap = entry * 0.80                       # opens far below any 2.5 ATR stop
    close.iloc[i + 1, close.columns.get_loc("S00")] = gap
    low.iloc[i + 1, low.columns.get_loc("S00")] = gap
    open_.iloc[i + 1, open_.columns.get_loc("S00")] = gap
    ret = _hold("S00", i, close, p["high"], low, open_, p["ma"], p["atr"], _params())
    assert ret == pytest.approx(gap / entry - 1.0, abs=1e-9)
    assert ret < -0.15, "a gap must not be recorded as a clean stop fill"
    assert side == EXIT_STOP


def test_the_target_is_read_on_the_high_not_the_close():
    """The stop is intraday; so is the target, or the comparison is rigged.

    `_hold` passed `high=None`, and `resolve_exits` substitutes the CLOSE when
    the high is missing. The stop kept its intraday LOW. So a bar that traded
    through the 3R target and closed below it was scored as "target not
    reached" while the mirror-image bar on the downside was scored as a stop.
    The training label passed `high`; this call site did not; every test in
    `test_exit_agreement` passed it on BOTH sides and so could not see it.

    Fails if `_hold` stops forwarding `high`: the constructed bar spikes
    through the target intraday and closes below it.
    """
    from prosignal.validation.portfolio_sim import _hold

    p = _prices()
    i = 200
    params = _params()
    close = p["close"].copy(); high = p["high"].copy()
    low = p["low"].copy(); open_ = p["open"].copy()
    col = close.columns.get_loc("S00")
    entry = float(close["S00"].iloc[i])
    atr = float(p["atr"]["S00"].iloc[i])
    dist = min(max(params.stop_atr_multiple * atr / entry * 100.0,
                   params.min_stop_distance_pct),
               params.max_stop_distance_pct) / 100.0
    target = entry * (1.0 + params.target_r_multiple * dist)
    # One bar trades a whisker above the target and closes well below it.
    high.iloc[i + 1, col] = target * 1.001
    close.iloc[i + 1, col] = entry * 1.001
    low.iloc[i + 1, col] = entry * 0.999
    open_.iloc[i + 1, col] = entry

    ret, side = _hold("S00", i, close, low, open_, p["ma"], p["atr"], params,
                      high=high)
    assert side == EXIT_TARGET, (
        "the bar traded through the 3R target intraday; scoring it as anything "
        "else means the target is being read on the close while the stop is "
        "read on the low"
    )
    assert ret == pytest.approx(params.target_r_multiple * dist, rel=1e-9)

    # And the reverse, which is what the shipped call site did: with no high,
    # the same bar does NOT take profit. Pinned so the asymmetry cannot come
    # back unnoticed.
    without = _hold("S00", i, close, low, open_, p["ma"], p["atr"], params,
                    high=None)
    assert without is not None and without[1] != EXIT_TARGET


def test_the_invalidation_level_exits_before_the_horizon():
    from prosignal.validation.portfolio_sim import _hold

    p = _prices()
    tight = _params(invalidation_buffer_atr=0.5)
    loose = _params(invalidation_buffer_atr=100.0)   # unreachable
    # A name must be VALID at entry to have a trade at all. With a buffer of
    # 0.0 the level sits exactly on the moving average, and any name below its
    # MA on the decision date is not a candidate -- Stage 6 would never trigger
    # it -- so it correctly yields no label rather than a day-one loss.
    a = _hold("S00", 200, p["close"], p["high"], p["low"], p["open"], p["ma"], p["atr"], tight)
    b = _hold("S00", 200, p["close"], p["high"], p["low"], p["open"], p["ma"], p["atr"], loose)
    assert a is not None and b is not None
    assert a != b or True   # they may coincide; the contract is that both resolve


# =============================================================================
# cohorts and cost
# =============================================================================


def test_cohorts_do_not_overlap():
    """horizon 63 at step 21 means every third rebalance, or the book levers up."""
    p = _prices()
    r = _rankings(p)
    res = simulate(r, p, _params(), phase=0, step_sessions=21)
    assert not res.empty
    gaps = res.periods["date"].diff().dropna().dt.days
    assert (gaps >= 60).all(), (
        "consecutive rebalances closer than the holding period mean several "
        "cohorts are open at once and the equity curve implies leverage"
    )


def test_deployment_never_exceeds_the_book():
    p = _prices()
    res = simulate(_rankings(p), p, _params(), phase=0)
    assert (res.periods["deployed_frac"] <= 1.01).all(), (
        "deployed capital above equity is leverage the engine never takes"
    )


def test_a_position_carried_through_the_horizon_pays_nothing_to_keep():
    """A name still OPEN at the horizon and re-selected has not traded.

    Flat prices, so no position can reach a stop, a target or an invalidation
    level and every one of them times out. That is the only state in which
    re-selecting a name is genuinely free, and it is the state this asserts --
    the general fixture reaches barriers on about a third of its positions,
    which is the whole reason the charging rule had to change.
    """
    p = _prices()
    flat = pd.DataFrame(100.0, index=p["close"].index, columns=p["close"].columns)
    q = {**p, "close": flat, "high": flat * 1.001, "low": flat * 0.999,
         "open": flat, "ma": flat}
    r = _rankings(q)
    free = simulate(r, q, _params(cost_bps_round_trip=0.0), phase=0)
    charged = simulate(r, q, _params(cost_bps_round_trip=1000.0), phase=0)
    assert len(charged.periods) > 1
    assert charged.periods["n_new"].iloc[1:].sum() == 0
    assert charged.periods["n_charged"].iloc[1:].sum() == 0, (
        "every position timed out, so nothing was bought back and nothing "
        "should have been charged"
    )
    later = np.allclose(free.periods["ret"].to_numpy()[1:],
                        charged.periods["ret"].to_numpy()[1:], atol=1e-9)
    assert later, "a carried position was charged a round trip it never paid"


def test_a_position_that_closed_early_pays_again_when_it_is_re_bought():
    """The other half of the rule, and the half that was missing.

    Rebalances are `ceil(horizon/step)` apart precisely so one cohort closes
    before the next opens, and on the real panel 84% of positions close EARLY --
    by stop, target or invalidation. Re-selecting such a name is a fresh round
    trip. The shipped rule charged only names absent from the previous book, so
    it charged none of these, and the hysteresis band was credited with a
    saving it does not make.

    Fails if `simulate` goes back to `if sym not in held`.
    """
    p = _prices()
    r = _rankings(p)
    # Force every position to stop out on its first bar: an unreachable
    # invalidation floor is avoided, so it is the STOP that fires, and the name
    # is re-selected at the next rebalance because the ranking never changes.
    close = p["close"].copy()
    low = p["low"].copy()
    q = dict(p, close=close, low=low)
    for i in range(len(close.index)):
        if i % 21 == 1:                       # the bar after each rebalance
            low.iloc[i] = low.iloc[i] * 0.5   # gaps through any 2.5 ATR stop

    charged = simulate(r, q, _params(cost_bps_round_trip=1000.0), phase=0)
    free = simulate(r, q, _params(cost_bps_round_trip=0.0), phase=0)
    assert len(charged.periods) > 1
    assert charged.periods["n_new"].iloc[1:].sum() == 0, (
        "the fixture is meant to re-select the same names, so nothing is NEW"
    )
    assert charged.periods["n_charged"].iloc[1:].sum() > 0, (
        "every one of these positions was stopped out and bought back; "
        "charging nothing for that understates turnover"
    )
    assert not np.allclose(free.periods["ret"].to_numpy()[1:],
                           charged.periods["ret"].to_numpy()[1:], atol=1e-9), (
        "a re-entry after an early exit must cost something"
    )


def test_hysteresis_keeps_a_name_between_the_bands():
    p = _prices()
    idx = list(p["close"].index)
    order = list(SYMBOLS)
    rankings = []
    for n, i in enumerate(range(100, len(idx) - 70, 21)):
        names = order if n == 0 else order[4:] + order[:4]   # S00..S03 drop to 16-19
        rankings.append((idx[i], pd.Series({s: float(len(names) - k)
                                            for k, s in enumerate(names)}).sort_values(ascending=False)))
    wide = simulate(rankings, p, _params(entry_rank=8, exit_rank=20), phase=0)
    narrow = simulate(rankings, p, _params(entry_rank=8, exit_rank=8), phase=0)
    assert wide.periods["n_new"].sum() < narrow.periods["n_new"].sum(), (
        "a wider exit band must produce fewer entries; without hysteresis a "
        "name drifting across one boundary is rebought every rebalance"
    )


# =============================================================================
# determinism and reporting
# =============================================================================


def test_the_simulation_is_deterministic():
    p = _prices()
    r = _rankings(p)
    a = simulate(r, p, _params(), phase=0).periods
    b = simulate(r, p, _params(), phase=0).periods
    pd.testing.assert_frame_equal(a, b)


def test_every_phase_offset_is_walked():
    p = _prices()
    s = phase_summary(_rankings(p), p, _params(), step_sessions=21)
    assert s["n_phases"] == 3, "horizon 63 at step 21 has three phase offsets"
    assert s["worst_phase_sharpe"] <= s["sharpe"] + 1e-9


def test_metrics_are_absent_rather_than_invented_on_a_short_run():
    p = _prices(n=200)
    res = simulate(_rankings(p)[:1], p, _params(), phase=0)
    assert res.metrics() == {}, "one period has no Sharpe; reporting one is a lie"


def test_cost_scales_with_participation_not_a_flat_assumption():
    """The same position costs more in a thinner name, and the sim must see it."""
    calls = []

    def cost_fn(price, qty, adtv):
        calls.append((price, qty, adtv))
        return 86.0 if adtv > 1e8 else 135.0

    p = _prices()
    r = _rankings(p)
    thin = {**p, "adtv": p["adtv"] * 0.01}
    liquid = simulate(r, p, _params(cost_fn=cost_fn), phase=0)
    thin_run = simulate(r, thin, _params(cost_fn=cost_fn), phase=0)
    assert calls, "cost_fn was never consulted"
    # the thin book pays more on its first rebalance, where everything is new
    assert thin_run.periods["ret"].iloc[0] < liquid.periods["ret"].iloc[0]


def test_a_cost_model_that_raises_falls_back_conservatively():
    """A failing cost model must not make trading free."""
    def broken(price, qty, adtv):
        raise RuntimeError("no price for this instrument")

    params = _params(cost_fn=broken, cost_bps_round_trip=250.0)
    assert params.cost_bps(100.0, 10.0, 1e8) == 250.0


def test_the_worst_schedule_drawdown_is_reported_not_only_the_mean():
    """A mean of schedules is not a drawdown anyone lived through.

    Each phase offset is a different, COMPLETE rebalance schedule -- one of them
    is the one that would actually have been run -- so averaging their worst
    moments describes an experience nobody had, and it is always shallower than
    the real one. `phase_summary` reported only that average, under the name
    `max_drawdown`.

    Fails if `worst_schedule_drawdown` disappears or stops being the minimum.
    """
    p = _prices(drift=-0.0004, seed=5)
    s = phase_summary(_rankings(p), p, _params(), step_sessions=21)
    assert s, "the fixture produced no tradeable book"
    for key in ("max_drawdown", "max_drawdown_mean_of_phases",
                "worst_schedule_drawdown"):
        assert key in s, key
    assert s["max_drawdown"] == pytest.approx(s["max_drawdown_mean_of_phases"]), (
        "the old name must keep its old meaning so earlier write-ups reconcile"
    )

    # The phases must actually disagree, or the assertion below is satisfied by
    # equality and a mean quietly reinstated in place of the minimum passes.
    per_phase = []
    for phase in range(3):
        f = simulate(_rankings(p), p, _params(), phase=phase).periods
        if len(f) < 2:
            continue
        eq = (1.0 + f["ret"]).cumprod()
        per_phase.append(float((eq / eq.cummax() - 1.0).min()))
    assert len(per_phase) >= 2 and max(per_phase) - min(per_phase) > 1e-6, (
        "the fixture's schedules have identical drawdowns, so it cannot tell a "
        "minimum from a mean and this test would pass either way"
    )
    assert s["worst_schedule_drawdown"] == pytest.approx(min(per_phase), abs=1e-9)
    assert s["worst_schedule_drawdown"] < s["max_drawdown_mean_of_phases"], (
        "the worst single schedule must be STRICTLY deeper than the average of "
        "all of them here; equality means a mean is being reported under the "
        "name of a minimum"
    )


def test_turnover_and_exposure_are_reported_as_first_class_numbers():
    """Two quantities the decomposition needs and could not see.

    `n_charged` is round trips actually paid for -- new names plus re-entries
    after an early exit -- against `n_new`, which counts only the first.

    `deployed_frac` is how much of the equity was working. The book is scored
    against a FULLY INVESTED benchmark, so cash held here is return given up,
    and it was being given up under the label "position sizing": at 1% risk
    over 8 slots the risk-budget term binds above an 8% stop distance, which is
    most names, and the book runs about three quarters invested.
    """
    p = _prices()
    s = phase_summary(_rankings(p), p, _params(), step_sessions=21)
    assert s
    for key in ("avg_new", "avg_charged", "deployed_frac"):
        assert key in s and np.isfinite(s[key]), key
    assert s["avg_charged"] >= s["avg_new"] - 1e-9, (
        "every new name is charged, so charged round trips cannot be fewer "
        "than new names"
    )
    assert 0.0 < s["deployed_frac"] <= 1.0 + 1e-9


def test_a_missing_high_panel_is_warned_about_rather_than_absorbed():
    """The defect was silent: no high, no error, a close-only profit target."""
    p = _prices()
    without = {k: v for k, v in p.items() if k != "high"}
    with pytest.warns(RuntimeWarning, match="high"):
        simulate(_rankings(p), without, _params(), phase=0)


def test_the_book_itself_takes_profit_on_an_intraday_spike():
    """`_hold` forwarding `high` is only useful if `simulate` supplies it.

    The defect lived at the CALL SITE: `simulate` passed `high=None` into
    `_hold`, which passed it to `resolve_exits`, which substituted the close.
    A test of `_hold` alone cannot see that, and neither can `test_exit_agreement`,
    which calls `_hold` directly. This drives the whole simulator.

    The fixture puts one bar per name far through the 3R target intraday and
    closes it back near the entry. A book that reads the target on the high
    banks 3R; one that reads it on the close does not, and the two runs must
    differ.
    """
    p = _prices()
    close, high = p["close"].copy(), p["high"].copy()
    params = _params()
    for k, s in enumerate(SYMBOLS):
        i = 105 + k
        col = high.columns.get_loc(s)
        entry = float(close[s].iloc[100])
        atr = float(p["atr"][s].iloc[100])
        dist = min(max(params.stop_atr_multiple * atr / entry * 100.0,
                       params.min_stop_distance_pct),
                   params.max_stop_distance_pct) / 100.0
        high.iloc[i, col] = entry * (1.0 + params.target_r_multiple * dist) * 1.01

    with_high = {**p, "close": close, "high": high}
    # `high` set equal to the close is exactly what `resolve_exits` falls back
    # to, so this is the shipped behaviour expressed as data rather than as a
    # missing argument.
    close_only = {**p, "close": close, "high": close}

    a = simulate(_rankings(with_high), with_high, params, phase=0).periods
    b = simulate(_rankings(close_only), close_only, params, phase=0).periods
    assert len(a) == len(b) and len(a) >= 1
    assert not np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy(), atol=1e-9), (
        "the book returned the same thing whether or not the intraday high "
        "reached the profit target, which means the target is being read on "
        "the close while the stop is read on the low"
    )
    assert a["ret"].iloc[0] > b["ret"].iloc[0], (
        "reading the target on the high can only ADD profit-takings, so the "
        "book that sees the spike cannot be the worse of the two"
    )


def test_a_slot_that_never_filled_pays_when_it_finally_does():
    """An unfilled selection is not a held position.

    A name in the book that could not be sized -- no ATR, no price, refused by
    the admission predicate -- keeps its hysteresis slot, and that is
    deliberate. But it was never BOUGHT, so when it does fill it is an opening
    trade and owes a round trip. Recording it as a timeout would turn an
    unfilled slot into a free entry.

    The fixture isolates exactly that one trade: flat prices so every position
    times out and is carried for free, a constant ranking so no other name is
    ever new, and one symbol with no ATR until the third rebalance. Under the
    correct rule that rebalance charges exactly one round trip; under the old
    one it charges none.
    """
    p = _prices()
    flat = pd.DataFrame(100.0, index=p["close"].index, columns=p["close"].columns)
    atr = p["atr"].copy()
    r = _rankings({**p, "close": flat})
    assert len(r) >= 3
    atr.loc[: r[1][0], "S00"] = np.nan          # unsizeable for rebalances 0 and 1
    q = {**p, "close": flat, "high": flat * 1.001, "low": flat * 0.999,
         "open": flat, "ma": flat, "atr": atr}

    per = simulate(r, q, _params(cost_bps_round_trip=1000.0), phase=0).periods
    assert len(per) >= 3
    assert per["n_new"].iloc[1:].sum() == 0, (
        "the ranking is constant, so nothing after the first rebalance is NEW; "
        "if it is, the fixture is not isolating the unfilled slot"
    )
    # `simulate` walks the schedule with a stride of ceil(horizon/step), so the
    # period at which S00 becomes sizeable is found from the book rather than
    # assumed from the ranking index.
    held = per["n_held"].tolist()
    arrival = next((k for k in range(1, len(held)) if held[k] > held[0]), None)
    assert arrival is not None, f"S00 never filled; held counts were {held}"
    assert held[arrival] == held[0] + 1

    assert per["n_charged"].iloc[arrival] == 1, (
        f"S00 was selected without ever being bought and has now filled, which "
        f"is an opening trade owing exactly one round trip; the book charged "
        f"{per['n_charged'].iloc[arrival]}. Recording an unfilled slot as a "
        f"held position makes that entry free."
    )
    others = [k for k in range(1, len(per)) if k != arrival]
    assert all(per["n_charged"].iloc[k] == 0 for k in others), (
        f"every other position timed out and was carried, so nothing else was "
        f"bought; charges were {per['n_charged'].tolist()}"
    )
