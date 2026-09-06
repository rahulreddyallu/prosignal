"""The conviction layer, and the behaviours it exists to guarantee.

These are the acceptance assertions for the 0-2 decision engine. Several are
written as MUTATION tests: they break the production rule deliberately and
assert the failure is caught, because a test that only ever sees correct input
proves the code runs, not that it decides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.conviction import (
    economics, evidence, gate, independence, robustness, separation,
)
from prosignal.core.contracts import FactorMember, FactorScore, StockScore


# =============================================================================
# helpers -- a synthetic cross-section we control exactly
# =============================================================================


def _score(ticker, rank, raw, themes, sector="Tech", members=None):
    """One StockScore with the given theme contributions."""
    fs = {}
    for name, (z, w) in themes.items():
        mem = [FactorMember(name=f"{name}_{i}", rank=float(r))
               for i, r in enumerate((members or {}).get(name, []))]
        fs[name] = FactorScore(name=name, standardised=z, weight=w,
                               contribution=z * w, members=mem)
    return StockScore(ticker=ticker, sector=sector, factors=fs,
                      composite_raw=raw, composite_score=0.0,
                      percentile=0.0, rank=rank)


def _universe(n=60, seed=0):
    """A ranked universe with a genuine leader and a noisy tail."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        raw = 3.0 - i * 0.05 + float(rng.normal(0, 0.01))
        themes = {
            "momentum": (float(rng.normal(0.5, 1.0)), 0.40),
            "quality": (float(rng.normal(0.0, 1.0)), 0.19),
            "ownership": (float(rng.normal(0.0, 1.0)), 0.19),
            "reversal": (float(rng.normal(0.0, 1.0)), 0.11),
            "risk": (float(rng.normal(0.0, 1.0)), 0.11),
        }
        members = {k: list(rng.normal(0, 0.5, 3)) for k in themes}
        out.append(_score(f"S{i:02d}", i + 1, raw, themes,
                          sector=f"Sec{i % 5}", members=members))
    return out


class _Plan:
    def __init__(self, cost=60.0, impact=10.0):
        self.estimated_round_trip_cost_bps = cost
        self.estimated_impact_bps = impact


class _Regime:
    trend_regime = "Range-bound"
    vix_percentile = 30.0
    transition_flag = False
    # Stage 8 reads these on the market-block path; a stub that omits them
    # fails with an AttributeError that says nothing about the test.
    allow_new_entries = True
    block_reason = None
    regime_bucket = "range_lowvol"


def _closes(tickers, n=200, seed=1, rho=0.0, pair_rho=0.0, pair=("S0", "S1")):
    """Price history with two separate common factors.

    ``rho`` loads every name on a MARKET factor -- which `independence` removes,
    so raising it must NOT make two names look like one bet.
    ``pair_rho`` loads only ``pair`` on a second, non-market factor. That is a
    shared driver the market leg cannot absorb, and it is what "the same trade
    twice" actually looks like.
    """
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.01, n)
    shared = rng.normal(0, 0.01, n)
    data = {}
    for t in tickers:
        idio = rng.normal(0, 0.01, n)
        load = pair_rho if t in pair else 0.0
        r = (np.sqrt(rho) * market
             + np.sqrt(load) * shared
             + np.sqrt(max(1.0 - rho - load, 0.0)) * idio)
        data[t] = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(data, index=pd.bdate_range("2024-01-01", periods=n))


# =============================================================================
# EVIDENCE -- five correlated indicators are not five confirmations
# =============================================================================


def test_perfectly_correlated_factors_count_as_one_bet():
    """Test 7. The whole reason this layer exists."""
    n = 200
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, n)
    # Five columns that are the SAME measurement with a whisper of noise.
    panel = np.column_stack([base + rng.normal(0, 0.01, n) for _ in range(5)])
    got = evidence.effective_independent_count(
        [1.0] * 5, panel, [f"f{i}" for i in range(5)])
    assert got.nominal == 5
    assert got.effective < 1.2, (
        f"five copies of one factor scored {got.effective:.2f} independent "
        f"bets; they are one")
    assert got.redundancy_ratio > 4.0


def test_orthogonal_factors_count_as_many_bets():
    n = 400
    rng = np.random.default_rng(4)
    panel = rng.normal(0, 1, (n, 5))
    got = evidence.effective_independent_count(
        [1.0] * 5, panel, [f"f{i}" for i in range(5)])
    assert got.effective > 4.0, (
        f"five independent factors scored only {got.effective:.2f}")


def test_a_factor_pointing_the_other_way_is_not_evidence_for():
    n = 200
    rng = np.random.default_rng(5)
    panel = rng.normal(0, 1, (n, 3))
    both = evidence.effective_independent_count([1.0, 1.0, 1.0], panel, list("abc"))
    one_against = evidence.effective_independent_count([1.0, 1.0, -1.0], panel, list("abc"))
    assert one_against.nominal == 2 < both.nominal == 3


def test_evidence_count_is_bounded_by_the_number_of_factors():
    n = 100
    rng = np.random.default_rng(6)
    for k in (2, 4, 8):
        panel = rng.normal(0, 1, (n, k))
        got = evidence.effective_independent_count([1.0] * k, panel,
                                                   [f"f{i}" for i in range(k)])
        assert 1.0 <= got.effective <= k + 1e-9


# =============================================================================
# SEPARATION -- rank 1 is not evidence of being ahead
# =============================================================================


def test_a_leader_inside_the_noise_has_no_separation():
    """Test 6. A tiny rank advantage must not become a BUY.

    NOTE ON WHAT "BUNCHED" MEANS. Separation is measured RELATIVE to the
    cross-section's own dispersion, so a perfectly uniform ramp always puts the
    leader about 1/15th of a sigma ahead whatever its absolute step size -- that
    is the correct answer for a uniform ramp, not a defect. The case this gate
    is for is the one seen live: the top names bunched together while the field
    below them spreads out. Ranks 7 to 10 of a real run were separated by 0.004
    to 0.013 sigma with the leader 1.9 sigma above the median.
    """
    bunched = [_score("A", 1, 1.0000, {"momentum": (1.0, 1.0)}),
               _score("B", 2, 0.9999, {"momentum": (1.0, 1.0)}),
               _score("C", 3, 0.9998, {"momentum": (1.0, 1.0)})]
    bunched += [_score(f"S{i}", i + 4, 0.5 - i * 0.02, {"momentum": (1.0, 1.0)})
                for i in range(40)]
    sep = separation.measure(bunched, "A")
    assert sep.testable()
    assert sep.gap_to_median > 1.0, "the leader IS above the field"
    assert sep.gap_to_next is not None and sep.gap_to_next < 0.05, (
        f"a 0.0001 lead over a spread field scored {sep.gap_to_next} sigma")


def test_a_genuine_leader_separates():
    names = [_score("LEAD", 1, 5.0, {"momentum": (1.0, 1.0)})]
    names += [_score(f"S{i}", i + 2, 1.0 - i * 0.01, {"momentum": (1.0, 1.0)})
              for i in range(40)]
    sep = separation.measure(names, "LEAD")
    assert sep.gap_to_median > 1.0 and sep.gap_to_next > 0.05


def test_separation_is_not_testable_on_a_degenerate_cross_section():
    same = [_score(f"S{i}", i + 1, 1.0, {"momentum": (1.0, 1.0)}) for i in range(20)]
    assert not separation.measure(same, "S0").testable()


# =============================================================================
# ROBUSTNESS -- a fragile candidate gets a lower grade
# =============================================================================


def test_a_one_theme_candidate_collapses_when_that_theme_is_dropped():
    """Test 9. Fragility must be visible."""
    uni = []
    # FRAGILE leads on momentum alone; the field leads on everything else.
    uni.append(_score("FRAGILE", 1, 3.0,
                      {"momentum": (5.0, 0.40), "quality": (-1.0, 0.19),
                       "ownership": (-1.0, 0.19), "reversal": (-1.0, 0.11),
                       "risk": (-1.0, 0.11)}))
    for i in range(30):
        uni.append(_score(f"S{i}", i + 2, 2.0 - i * 0.01,
                          {"momentum": (0.0, 0.40), "quality": (1.0, 0.19),
                           "ownership": (1.0, 0.19), "reversal": (1.0, 0.11),
                           "risk": (1.0, 0.11)}))
    env = robustness.envelope(uni, "FRAGILE")
    assert env.testable()
    assert env.survival < 1.0
    assert "momentum" in (env.worst_spec or ""), (
        f"dropping momentum should be the worst case for a momentum-only name, "
        f"got {env.worst_spec}")


def test_a_broad_candidate_survives_every_specification():
    uni = [_score("BROAD", 1, 3.0,
                  {"momentum": (2.0, 0.40), "quality": (2.0, 0.19),
                   "ownership": (2.0, 0.19), "reversal": (2.0, 0.11),
                   "risk": (2.0, 0.11)})]
    for i in range(30):
        uni.append(_score(f"S{i}", i + 2, 1.0 - i * 0.01,
                          {"momentum": (0.0, 0.40), "quality": (0.0, 0.19),
                           "ownership": (0.0, 0.19), "reversal": (0.0, 0.11),
                           "risk": (0.0, 0.11)}))
    env = robustness.envelope(uni, "BROAD")
    assert env.survival == 1.0 and not env.fragile


# =============================================================================
# ECONOMICS -- the edge has to survive the cost of capturing it
# =============================================================================


def test_the_measured_edge_is_read_from_the_results_file():
    edge, t, why = economics.reference_edge(63)
    assert why is None, why
    assert edge is not None and 0.0 < edge < 0.1
    assert t is not None


def test_an_unmeasured_horizon_is_not_testable():
    edge, _, why = economics.reference_edge(7)
    assert edge is None and why


def test_cost_can_exceed_the_measured_edge():
    """Test 11. At 21 sessions the measured edge is 64 bps."""
    net = economics.assess("X", _Plan(cost=200.0), 21)
    assert net.testable()
    assert net.net_edge < 0, "200 bps of cost must not leave a 64 bps edge positive"
    assert net.cost_burden > 1.0


def test_a_cheap_trade_at_a_long_horizon_keeps_most_of_the_edge():
    net = economics.assess("X", _Plan(cost=30.0), 63)
    assert net.net_edge > 0 and net.cost_burden < 0.25


def test_a_plan_without_a_cost_is_not_testable():
    assert not economics.assess("X", None, 63).testable()


# =============================================================================
# INDEPENDENCE -- the second slot has to be a second bet
# =============================================================================


def test_two_names_driven_by_one_factor_are_one_bet():
    """Test 10. A shared driver the market leg cannot absorb."""
    closes = _closes([f"S{i}" for i in range(30)], pair_rho=0.95, seed=9)
    ind = independence.assess_pair("S0", "S1", closes,
                                   sector_a="A", sector_b="B")
    assert ind.testable()
    assert ind.residual_correlation > 0.5
    assert ind.basket_enb < 1.7, (
        f"two names at residual rho={ind.residual_correlation:.2f} scored "
        f"{ind.basket_enb:.2f} independent bets")


def test_shared_MARKET_exposure_alone_does_not_make_two_names_one_bet():
    """The reason correlation is taken on residuals and not raw."""
    closes = _closes([f"S{i}" for i in range(30)], rho=0.95, seed=31)
    ind = independence.assess_pair("S0", "S1", closes)
    assert ind.testable()
    assert ind.raw_correlation > 0.5, "the raw correlation IS high"
    assert abs(ind.residual_correlation) < 0.35, (
        "once the market is removed these are two different companies")
    assert ind.basket_enb > 1.9


def test_independent_names_span_two_bets():
    closes = _closes([f"S{i}" for i in range(30)], rho=0.0, seed=10)
    ind = independence.assess_pair("S0", "S1", closes,
                                   sector_a="A", sector_b="B")
    assert ind.testable()
    assert ind.basket_enb > 1.9


def test_missing_history_is_not_testable_rather_than_independent():
    """An unmeasured correlation is not a low one."""
    closes = _closes(["S0", "S1"], n=5)
    assert not independence.assess_pair("S0", "S1", closes).testable()
    assert not independence.assess_pair("S0", "GONE", closes).testable()


# =============================================================================
# THE GATE -- 0, 1 or 2, and never for the wrong reason
# =============================================================================


def _evaluate_all(uni, plans=None, regime=None, th=None):
    th = th or gate.Thresholds()
    tp = evidence.theme_matrix(uni)
    mp = evidence.member_matrix(uni)
    plans = plans or {}
    return [gate.evaluate(s, uni, plans.get(s.ticker, _Plan()), regime or _Regime(),
                          63, th, theme_panel=tp, member_panel=mp, themes=tp[1])
            for s in uni[:th.shortlist]], tp[1]


def test_no_candidate_clearing_produces_no_trade():
    """Test 5. Zero is a first-class result."""
    flat = [_score(f"S{i}", i + 1, 1.0 - i * 0.0001,
                   {"momentum": (1.0, 1.0)}) for i in range(40)]
    cands, themes = _evaluate_all(flat)
    v = gate.select_at_most_two(cands, _closes([s.ticker for s in flat]),
                                gate.Thresholds(), themes=themes)
    assert v.is_no_trade
    assert v.cause == gate.NoTradeCause.NO_CANDIDATE
    assert v.reason and "closest" in v.reason.lower()


def test_never_more_than_two_buys():
    """Test 18. The hard cap."""
    uni = _universe(60, seed=11)
    cands, themes = _evaluate_all(uni)
    # Force everything to clear, then confirm the cap still binds.
    for c in cands:
        c.failures.clear()
    closes = _closes([s.ticker for s in uni], rho=0.0, seed=12)
    v = gate.select_at_most_two(cands, closes, gate.Thresholds(), themes=themes)
    assert len(v.buys) <= 2


def test_one_excellent_candidate_produces_one_buy_not_two():
    """Test 4. No slot filling."""
    uni = _universe(60, seed=13)
    cands, themes = _evaluate_all(uni)
    for c in cands:
        c.failures.clear() if c.ticker == "S00" else c.failures.append("weak")
    closes = _closes([s.ticker for s in uni], seed=14)
    v = gate.select_at_most_two(cands, closes, gate.Thresholds(), themes=themes)
    assert len(v.buys) == 1 and v.buys[0].ticker == "S00"


def test_a_failing_candidate_is_never_promoted_because_the_others_failed():
    """The promotion failure mode, stated directly."""
    uni = _universe(40, seed=15)
    cands, themes = _evaluate_all(uni)
    for c in cands:
        c.failures.append("deliberately refused")
    v = gate.select_at_most_two(cands, _closes([s.ticker for s in uni]),
                                gate.Thresholds(), themes=themes)
    assert v.buys == []


def test_the_second_slot_is_refused_when_independence_is_unmeasurable():
    uni = _universe(40, seed=16)
    cands, themes = _evaluate_all(uni)
    for c in cands:
        c.failures.clear()
    thin = _closes([s.ticker for s in uni], n=6, seed=17)
    v = gate.select_at_most_two(cands, thin, gate.Thresholds(), themes=themes)
    assert len(v.buys) == 1, "an unmeasured correlation must not fill slot 2"


def test_the_second_slot_is_refused_when_the_pair_is_one_bet():
    """Exactly two names are allowed to clear, and they share a driver.

    NOTE. It is not possible to build a universe where EVERY pair has a high
    residual correlation: the residual is taken against the equal-weight mean,
    so a driver every name loads on IS the market and is removed by
    construction. The pair has to be named, which is why the two survivors are
    pinned here rather than left to the gate to discover.
    """
    uni = _universe(40, seed=18)
    cands, themes = _evaluate_all(uni)
    winners = {"S00", "S01"}
    for c in cands:
        c.failures.clear() if c.ticker in winners else c.failures.append("refused")
    names = [s.ticker for s in uni]
    correlated = _closes(names, pair_rho=0.97, seed=19, pair=("S00", "S01"))
    v = gate.select_at_most_two(cands, correlated, gate.Thresholds(), themes=themes)
    assert len(v.buys) == 1, (
        f"two names sharing a non-market driver took both slots: "
        f"{[b.ticker for b in v.buys]}")
    assert len(v.buys) == 1 and v.buys[0].ticker in winners

    # And the control: the SAME two names, independent, take both slots.
    independent = _closes(names, seed=19)
    cands2, themes2 = _evaluate_all(uni)
    for c in cands2:
        c.failures.clear() if c.ticker in winners else c.failures.append("refused")
    v2 = gate.select_at_most_two(cands2, independent, gate.Thresholds(),
                                 themes=themes2)
    assert len(v2.buys) == 2, "two independent names should fill both slots"


def test_a_momentum_carried_name_is_refused_in_a_panic_state():
    """Daniel & Moskowitz. Test that the crash defense actually fires."""

    class Panic:
        trend_regime = "Downtrend"
        vix_percentile = 85.0
        transition_flag = False

    uni = [_score("MOM", 1, 3.0,
                  {"momentum": (3.0, 0.40), "quality": (0.05, 0.19)})]
    uni += [_score(f"S{i}", i + 2, 1.0 - i * 0.01,
                   {"momentum": (0.0, 0.40), "quality": (0.0, 0.19)})
            for i in range(30)]
    tp, mp = evidence.theme_matrix(uni), evidence.member_matrix(uni)
    calm = gate.evaluate(uni[0], uni, _Plan(), _Regime(), 63,
                         gate.Thresholds(), theme_panel=tp, member_panel=mp,
                         themes=tp[1])
    panicked = gate.evaluate(uni[0], uni, _Plan(), Panic(), 63,
                             gate.Thresholds(), theme_panel=tp, member_panel=mp,
                             themes=tp[1])
    assert any("momentum crash" in f.lower() or "Daniel" in f
               for f in panicked.failures), panicked.failures
    assert not any("momentum crash" in f.lower() for f in calm.failures)


def test_the_runner_up_is_recorded_with_a_reason():
    """The 'why not #3' diagnostic."""
    uni = _universe(40, seed=20)
    cands, themes = _evaluate_all(uni)
    v = gate.select_at_most_two(cands, _closes([s.ticker for s in uni]),
                                gate.Thresholds(), themes=themes)
    ru = v.runner_up()
    assert ru is not None
    assert ru.failures or ru.clears


def test_conviction_grade_is_ordinal_and_never_a_probability():
    uni = _universe(30, seed=21)
    cands, _ = _evaluate_all(uni)
    for c in cands:
        g = c.grade()
        assert g in {"A", "A-", "B+", "B", "-"}
        assert "%" not in g


# =============================================================================
# MUTATION -- break the rule, prove the test catches it
# =============================================================================


def test_mutating_the_evidence_floor_to_zero_admits_a_one_bet_candidate():
    """If this passes with the floor at 0 and fails at 2.0, the floor binds."""
    uni = []
    # One name whose five supporting factors are all the same measurement.
    same = 2.0
    # Members matter: with none, the count is NOT TESTABLE and the gate refuses
    # for a different reason. Every member here is the SAME measurement, which
    # is the case being tested.
    mem = {k: [0.9, 0.9, 0.9] for k in
           ("momentum", "quality", "ownership", "reversal", "risk")}
    uni.append(_score("ONEBET", 1, 3.0,
                      {"momentum": (same, 0.40), "quality": (same, 0.19),
                       "ownership": (same, 0.19), "reversal": (same, 0.11),
                       "risk": (same, 0.11)}, members=mem))
    for i in range(30):
        z = -0.5
        low = {k: [-0.4, -0.4, -0.4] for k in mem}
        uni.append(_score(f"S{i}", i + 2, 1.0 - i * 0.01,
                          {"momentum": (z, 0.40), "quality": (z, 0.19),
                           "ownership": (z, 0.19), "reversal": (z, 0.11),
                           "risk": (z, 0.11)}, members=low))
    tp, mp = evidence.theme_matrix(uni), evidence.member_matrix(uni)

    strict = gate.evaluate(uni[0], uni, _Plan(), _Regime(), 63,
                           gate.Thresholds(min_independent_evidence=2.0),
                           theme_panel=tp, member_panel=mp, themes=tp[1])
    loose = gate.evaluate(uni[0], uni, _Plan(), _Regime(), 63,
                          gate.Thresholds(min_independent_evidence=0.0,
                                          max_evidence_concentration=1.0),
                          theme_panel=tp, member_panel=mp, themes=tp[1])
    strict_ev = [f for f in strict.failures if "independent direction" in f]
    loose_ev = [f for f in loose.failures if "independent direction" in f]
    assert strict_ev, "the evidence floor did not fire on a one-bet candidate"
    assert not loose_ev, "the floor fired even when disabled -- it is not the gate"


def test_mutating_the_max_buys_cap_is_observable():
    uni = _universe(40, seed=22)
    cands, themes = _evaluate_all(uni)
    for c in cands:
        c.failures.clear()
    closes = _closes([s.ticker for s in uni], rho=0.0, seed=23)
    two = gate.select_at_most_two(cands, closes, gate.Thresholds(max_buys=2),
                                  themes=themes)
    one = gate.select_at_most_two(cands, closes, gate.Thresholds(max_buys=1),
                                  themes=themes)
    assert len(one.buys) == 1
    assert len(two.buys) >= len(one.buys)


# =============================================================================
# NULL -- noise must not produce confident selections
# =============================================================================


@pytest.mark.parametrize("seed", [101, 202, 303, 404, 505])
def test_pure_noise_rarely_clears_the_gate(seed):
    """Test 12 / §100. A cross-section with no structure.

    Scores, factors and members are all independent noise, so nothing is
    genuinely ahead of anything. The gate is allowed to fire occasionally -- a
    threshold that never fires on noise would never fire at all -- but a name
    must not clear on a cross-section this flat with any regularity.
    """
    rng = np.random.default_rng(seed)
    uni = []
    for i in range(60):
        themes = {k: (float(rng.normal()), w) for k, w in
                  [("momentum", 0.40), ("quality", 0.19), ("ownership", 0.19),
                   ("reversal", 0.11), ("risk", 0.11)]}
        members = {k: list(rng.normal(0, 1, 3)) for k in themes}
        raw = float(rng.normal())
        uni.append(_score(f"N{i:02d}", i + 1, raw, themes,
                          sector=f"Sec{i % 6}", members=members))
    uni.sort(key=lambda s: -s.composite_raw)
    for i, s in enumerate(uni):
        s.rank = i + 1
    cands, themes = _evaluate_all(uni)
    cleared = [c for c in cands if c.clears]
    assert len(cleared) <= 1, (
        f"{len(cleared)} names cleared the conviction gate on pure noise")


# =============================================================================
# HISTORY INDEPENDENCE -- tests 1, 2, 3 and 17
#
# These assert the ABSENCE of a coupling, which is the only way to state the
# requirement. "A stock that was not registered yesterday can be bought today"
# is not a behaviour to demonstrate on one example -- it is a property of a
# decision function that takes no history at all, and that is what is checked.
# =============================================================================


def test_the_conviction_gate_takes_no_historical_state():
    """Tests 1, 2 and 3, structurally.

    If `select_at_most_two` cannot see yesterday, then:
      - a name never seen before is as eligible as any other;
      - no registration, cadence or carried slate can suppress a candidate;
      - a name bought yesterday has no claim on a slot today.

    Demonstrating each on an example would prove the examples. Reading the
    signature proves the property.
    """
    import inspect

    params = set(inspect.signature(gate.select_at_most_two).parameters)
    forbidden = {"held", "open_book", "previous", "previous_slate", "ledger",
                 "registered", "registration", "history", "as_of", "clock",
                 "entries_open", "cadence"}
    assert not (params & forbidden), (
        f"the final selection can see historical state: {params & forbidden}")

    ev_params = set(inspect.signature(gate.evaluate).parameters)
    assert not (ev_params & forbidden), (
        f"conviction evaluation can see historical state: {ev_params & forbidden}")


def test_the_stage_takes_no_book_and_no_clock():
    from prosignal.stages import stage9_conviction
    import inspect

    params = set(inspect.signature(stage9_conviction.run).parameters)
    forbidden = {"held", "open_book", "entries", "entries_open", "clock",
                 "previous", "previous_slate"}
    assert not (params & forbidden), (
        f"stage 9 can see the book or the clock: {params & forbidden}")


def test_the_pipeline_does_not_let_the_cadence_gate_under_conviction():
    """The clock is still resolved and recorded. It must not decide."""
    src = (__import__("pathlib").Path(
        __import__("prosignal").__file__).parent / "pipeline.py").read_text()
    assert "entries_open = True if conviction_on else clock.is_entry_date" in src, (
        "the entry clock is gating again")
    assert "entry_clock=_clock_record(" in src, (
        "the clock must still be RECORDED even though it does not gate -- the "
        "recorded history was generated under it")


def test_thresholds_are_frozen_so_a_quiet_day_cannot_lower_the_bar():
    """Test 17. No forced daily signal."""
    th = gate.Thresholds()
    with pytest.raises(Exception):
        th.min_independent_evidence = 0.0        # frozen dataclass


def test_an_empty_day_and_a_broken_day_are_different_results():
    """Test 8 / §20. NOT_TESTABLE is not NO TRADE."""
    causes = {gate.NoTradeCause.NO_CANDIDATE, gate.NoTradeCause.DATA,
              gate.NoTradeCause.UNIVERSE, gate.NoTradeCause.HALT,
              gate.NoTradeCause.COST, gate.NoTradeCause.NO_SEPARATION}
    assert len(causes) == 6, "the no-trade causes must stay distinguishable"


def test_a_universe_too_small_to_rank_is_not_a_no_candidate_day():
    """A cross-section of four names has no field to be ahead of."""

    class _Scores:
        ranked_scores = [_score(f"S{i}", i + 1, 1.0, {"momentum": (1.0, 1.0)})
                         for i in range(4)]

    class _Defense:
        market_halt = False
        market_halt_reason = None
        per_stock = {}

    from prosignal.config.loader import load_config
    from prosignal.stages import stage9_conviction

    v = stage9_conviction.run(_Scores(), _Defense(), {}, _Regime(),
                              pd.DataFrame(), load_config())
    assert v.is_no_trade
    assert v.cause == gate.NoTradeCause.UNIVERSE, (
        "a four-name universe must report UNIVERSE, not 'nothing cleared'")


def test_a_market_halt_is_not_a_thin_evidence_day():
    class _Scores:
        ranked_scores = [_score(f"S{i}", i + 1, 1.0 - i * 0.1,
                                {"momentum": (1.0, 1.0)}) for i in range(30)]

    class _Halt:
        market_halt = True
        market_halt_reason = "the feed disagreed with itself"
        per_stock = {}

    from prosignal.config.loader import load_config
    from prosignal.stages import stage9_conviction

    v = stage9_conviction.run(_Scores(), _Halt(), {}, _Regime(),
                              pd.DataFrame(), load_config())
    assert v.cause == gate.NoTradeCause.HALT
    assert "feed disagreed" in v.reason


def test_the_record_keeps_the_refused_candidates():
    """§78. A book of winners cannot answer the question it exists for."""
    uni = _universe(40, seed=24)
    cands, themes = _evaluate_all(uni)
    v = gate.select_at_most_two(cands, _closes([s.ticker for s in uni]),
                                gate.Thresholds(), themes=themes)
    rows = gate.to_record(v)
    assert len(rows) == len(cands), "every evaluated candidate must be recorded"
    refused = [r for r in rows if not r["cleared"]]
    assert refused, "this fixture should refuse somebody"
    assert all(r["failures"] for r in refused), (
        "a refused candidate must record WHY")
    # And the record must survive a JSON round trip without this module.
    import json
    assert json.loads(json.dumps(rows)) == rows


def test_a_decision_stage_8_cannot_render_refuses_the_run():
    """Found by running it, not by testing it.

    Stage 9 selects from Stage 5's survivors; Stage 8 can only card a name that
    is in the defended set AND has an entry decision. When those populations
    diverged the old behaviour was a bare KeyError -- or, worse, `_build`
    returning None and the buy being dropped in silence, so the engine decided
    two and reported one. A decision that cannot be rendered is a broken
    engine, not a smaller book.
    """
    from prosignal.core.contracts import (
        CoreScoreReport, EligibilityReport, EntryReport, FalseSignalReport,
    )
    from prosignal.config.loader import load_config
    from prosignal.stages import stage8_final_signal

    cfg = load_config()
    uni = _universe(20, seed=77)
    scores = CoreScoreReport(as_of_date=__import__("datetime").date(2026, 9, 3),
                             weighting_mode="v3", standardisation="rank",
                             ranked_scores=uni, universe_size=len(uni))
    defense = FalseSignalReport(as_of_date=scores.as_of_date)
    verdict = gate.Verdict(
        buys=[gate.Candidate(ticker="NOT_DEFENDED", rank=1)],
        considered=[gate.Candidate(ticker="NOT_DEFENDED", rank=1)])

    with pytest.raises(ValueError) as err:
        stage8_final_signal.run(
            regime=_Regime(),
            eligibility=EligibilityReport(as_of_date=scores.as_of_date,
                                          universe_considered=len(uni)),
            scores=scores, defense=defense,
            entries=EntryReport(as_of_date=scores.as_of_date),
            plans={}, closes=pd.DataFrame(), config=cfg,
            conviction=verdict)
    assert "NOT_DEFENDED" in str(err.value)
    assert "silently shrink" in str(err.value)
