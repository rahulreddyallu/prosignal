"""The factor set, restructured: families, sector-neutral ranks, no liquidity.

Seventeen coefficients over a collinear set is not estimable, and the
near-uniform coefficient band was the model saying so. Measured on the live
universe, before this change:

    amihud / turnover_ratio   -0.907   one factor measured from two sides
    resid_mom / mom_6_1       +0.719
    downside_vol / idio_vol   +0.664
    max_dd_120 / prox_52w     +0.627

Standalone rank IC over 70 dates says the same thing a second way:

    factor            IC      ICIR       t
    prox_52w      +0.0660    +0.462   +3.87
    mom_6_1       +0.0549    +0.434   +3.63
    resid_mom     +0.0536    +0.418   +3.50
    amihud        +0.0087    +0.090   +0.75   <- carries nothing
    turnover_ratio -0.0155   -0.175   -1.47   <- carries nothing

WHAT AVERAGING IS AND IS NOT FOR. An earlier version of this file claimed
"every family beats its own best member on ICIR" and quoted mom +0.505 against
prox_52w +0.462, risk +0.488 against max_dd_120 +0.327, delivery +0.420 against
deliv_trend +0.388. Those were measured at #68, against the HORIZON RETURN. The
label was replaced at #75 by the engine's own exit geometry, and re-measured
against it on the same 69-date panel (magnitudes, since three themes are priced
negatively) three of the four reverse:

    family     |ICIR|    best member
    lottery     0.838    0.853  downside_vol   member wins
    mom         0.145    0.202  resid_mom      member wins
    risk        0.177    0.713  max_dd_120     member wins, heavily
    delivery    0.929    0.794  deliv_pct      family wins

The claim is withdrawn rather than restated, and the families are NOT
restructured on it, because standalone ICIR is not the criterion they exist to
serve. They exist for ESTIMABILITY: seventeen coefficients over a set this
collinear is not estimable, and the near-uniform coefficient band was the model
saying so. A family with a weaker standalone ICIR can still be the better thing
to fit one coefficient to.

`risk` WAS defended here as "the case worth understanding rather than fixing":
its members correlate -0.43 within date, so the average cancels the common
low-risk axis and keeps the residual, which is why it scored 0.177 against
max_dd_120's 0.713. That defence is now WITHDRAWN, and it is worth recording why
rather than deleting it.

It was defended on standalone ICIR measured against the BARRIER label. That
label has since been removed -- it made the target's magnitude proportional to
each name's volatility, so every ICIR in the table above was measured against a
target that no longer exists. A design defended on a criterion that has been
retired is not defended.

Re-examined on construction rather than on a number, the cancellation is not
defensible either: a high beta rank means RISKIER and a high max_dd rank means a
SHALLOWER drawdown and therefore SAFER, and both entered with a + sign. Two
significant signals -- beta alone t -3.67, max_dd alone t +4.69 -- were averaged
into a composite at t -0.93, which the significance gate then discarded for
being insignificant. The families exist for ESTIMABILITY, and the whole argument
for averaging (seventeen coefficients over a collinear set is not estimable)
does not apply to a PAIR THAT ANTICORRELATES. Those are two different bets.

They are now two themes, `beta` and `drawdown`, with one coefficient each.
Splitting is also what makes a literature prior claimable: the note in
famamacbeth.THEME_PRIOR_SIGN correctly refused a Frazzini-Pedersen prior for the
old composite, because (beta + max_dd)/2 is not the BAB portfolio. A `beta`
theme on its own is.

Deciding whether a family should be replaced by its best member needs an
out-of-sample ablation on the BOOK, not a standalone IC table. That experiment
has not been run.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from prosignal.features import crossmodel as cm
from prosignal.features.crosssec import (
    FEATURES, MIN_SECTOR_NAMES, sector_neutral_rank,
)


# ------------------------------------------------------------- the families
def test_liquidity_is_not_scored():
    """The illiquidity premium is compensation FOR trading costs, and a manual
    executor pays it rather than collecting it. It belongs in the universe
    screen, where `universe.pit_min_adtv_inr` already puts it."""
    scored = {m for members in cm.FAMILIES.values() for m in members}
    assert "amihud_r" not in scored
    assert "turnover_ratio_r" not in scored


def test_the_momentum_trio_is_one_family_not_three_coefficients():
    assert set(cm.FAMILIES["mom"]) == {"mom_6_1_r", "prox_52w_r", "resid_mom_r"}


def test_the_lottery_family_is_the_volatility_block_only():
    """THREE members, not four. India's lottery effect is driven by retail flow
    and is stronger than the US literature suggests, so the volatility moments
    are treated as one block -- and Bali, Cakici & Whitelaw (2011) find MAX
    subsumes idiosyncratic volatility, which is the same statement.

    `idio_skew` was the fourth member until it was measured against the family
    it was averaged into: rho +0.04 with downside_vol and +0.28 with max5_21,
    against 0.48-0.68 among the three that remain. It was near-orthogonal to its
    own family while carrying a quarter of its weight -- a second factor hidden
    inside a first. Skewness preference is a separate channel and is controlled
    for separately in that paper.
    """
    assert set(cm.FAMILIES["lottery"]) == {
        "max5_21_r", "idio_vol_r", "downside_vol_r"}
    assert "idio_skew_r" not in cm.FAMILIES["lottery"]


def test_skew_is_its_own_theme_rather_than_dropped():
    """Measured against the real forward return idio_skew reads t -0.94, so it
    has earned neither a place in `lottery` nor deletion. A single-member theme
    costs one coefficient, the significance gate is expected to zero it, and a
    theme that is visible and zeroed is more informative than one that is
    invisible and diluting its neighbours."""
    assert cm.FAMILIES["skew"] == ("idio_skew_r",)


def test_beta_and_drawdown_are_different_families():
    """The defect this test exists to prevent recurring. `risk` averaged
    beta_120_r and max_dd_120_r under a common sign, but a high beta rank is
    RISKIER while a high max_dd rank is a SHALLOWER drawdown and therefore
    SAFER. They correlate -0.42 within date, so the average cancelled the axis:
    beta alone t -3.67 and max_dd alone t +4.69 became a composite at t -0.93,
    which the |t| >= 2 gate then discarded for being insignificant.

    Families exist for ESTIMABILITY, and two anticorrelated members are not a
    collinear block."""
    families = {f: set(m) for f, m in cm.FAMILIES.items()}
    holding_beta = [f for f, m in families.items() if "beta_120_r" in m]
    holding_dd = [f for f, m in families.items() if "max_dd_120_r" in m]
    assert holding_beta == ["beta"]
    assert holding_dd == ["drawdown"]
    assert holding_beta != holding_dd
    assert "risk" not in cm.FAMILIES


def test_reversal_stays_out_of_the_momentum_family():
    """It is the opposite side of the same axis at a different horizon. Folding
    it in would net it out against momentum."""
    assert cm.FAMILIES["reversal"] == ("resid_reversal_r",)
    assert "resid_reversal_r" not in cm.FAMILIES["mom"]


def test_a_family_is_the_mean_of_the_members_that_are_present():
    frame = pd.DataFrame({
        "mom_6_1_r": [1.0, -1.0], "prox_52w_r": [0.0, 0.0],
        "resid_mom_r": [-1.0, 1.0],
    })
    built = cm.build_families(frame, list(frame.columns))
    assert "mom_f" in built
    assert frame["mom_f"].tolist() == [0.0, 0.0]


def test_a_family_with_no_available_member_is_not_built():
    """Not built, rather than built from nothing and reported as a factor --
    the same rule the individual factors follow."""
    frame = pd.DataFrame({"mom_6_1_r": [1.0]})
    built = cm.build_families(frame, ["mom_6_1_r"])
    assert "value_f" not in built and "value_f" not in frame.columns


def test_a_partial_family_averages_what_it_has_not_what_it_wishes_it_had():
    frame = pd.DataFrame({"earnings_yield_r": [1.0], "book_to_price_r": [0.0]})
    cm.build_families(frame, list(frame.columns))
    assert frame["value_f"].iloc[0] == pytest.approx(0.5), (
        "three absent members must not dilute the two that are present"
    )


# ------------------------------------------------------- sector neutrality
def test_ranks_are_taken_within_sector():
    """Ranking across the whole market compares a bank's leverage with an IT
    firm's, and every factor then carries an unintended sector bet."""
    values = pd.Series({f"S{i}": float(i) for i in range(40)})
    sectors = pd.Series({f"S{i}": ("A" if i < 20 else "B") for i in range(40)})
    out = sector_neutral_rank(values, sectors)
    # Each sector spans the full range rather than B sitting entirely above A.
    assert out[:20].max() == pytest.approx(1.0)
    assert out[20:].max() == pytest.approx(1.0)
    assert out[:20].min() < 0 and out[20:].min() < 0


def test_a_thin_sector_falls_back_to_the_universe_rank():
    """Three names give ranks of -1, 0 and +1 whatever the values were."""
    n = MIN_SECTOR_NAMES + 10
    values = pd.Series({f"S{i}": float(i) for i in range(n)})
    sectors = pd.Series({f"S{i}": ("TINY" if i < 3 else "BIG") for i in range(n)})
    out = sector_neutral_rank(values, sectors)
    universe = sector_neutral_rank(values)
    assert out[:3].tolist() == universe[:3].tolist()


def test_a_name_with_no_sector_is_ranked_against_the_universe():
    """Common here: the point-in-time universe reaches past any index
    constituent file, so sectors are genuinely absent for part of it."""
    values = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
    out = sector_neutral_rank(values, pd.Series({"A": None, "B": None, "C": None}))
    assert out.notna().all()


def test_no_sectors_at_all_is_the_universe_rank():
    values = pd.Series({f"S{i}": float(i) for i in range(20)})
    pd.testing.assert_series_equal(
        sector_neutral_rank(values), sector_neutral_rank(values, None))


# ------------------------------------------------ the new residual factors
def test_the_residual_block_is_registered():
    for name in ("idio_vol", "idio_skew", "resid_reversal"):
        assert name in FEATURES


def test_raw_reversal_is_gone():
    """Conventional short-term reversal carries dynamic exposure to the market
    and size factors; residual reversal avoids them."""
    assert "reversal_1m" not in FEATURES


def test_the_residual_moments_come_from_one_regression():
    """Four factors, one residual series. Computing them separately would let
    the definitions drift apart."""
    import inspect

    from prosignal.features import crosssec

    src = inspect.getsource(crosssec._features_at)
    block = src[src.index("mom_win = resid.tail(252)"):]
    for name in ("idio_vol", "idio_skew", "resid_reversal"):
        assert f'out["{name}"]' in block


# ------------------------------------------------- regime reaches the model
def test_regime_multipliers_reach_the_fitted_model():
    """They were applied ONLY to the hand-weighted composite's weights, so for
    the model that actually ranks the whole regime layer was decorative: the
    multiplier was computed, logged, written to the ledger, printed on the card,
    and never reached a score."""
    frame = pd.DataFrame({"mom_f": [1.0, -1.0], "lottery_f": [0.5, 0.5]})
    out = cm.apply_family_multipliers(frame, {"mom": 0.5})
    assert out["mom_f"].tolist() == [0.5, -0.5]
    assert out["lottery_f"].tolist() == [0.5, 0.5]
    assert frame["mom_f"].tolist() == [1.0, -1.0], "the input must not be mutated"


def test_a_neutral_multiplier_changes_nothing():
    frame = pd.DataFrame({"mom_f": [1.0]})
    assert cm.apply_family_multipliers(frame, {"mom": 1.0}) is frame
    assert cm.apply_family_multipliers(frame, None) is frame


def test_stage_4_passes_the_regime_into_the_model_path():
    import inspect

    from prosignal.stages import stage4_core_score as s4

    src = inspect.getsource(s4._cross_sectional_model)
    assert '"mom": float(regime.momentum_multiplier)' in src
    assert "multipliers=multipliers" in src
    assert "cm.score_with(cached, feats, multipliers)" in src


# ------------------------------------- one regime rule, on every scoring path
def test_the_guard_skips_when_no_stabiliser_is_priced():
    """Scaling momentum down is only meaningful when there is something to
    rotate INTO. With no defensive family priced, the weight goes to whatever
    else is -- on the shipped model `delivery`, which was never a stabiliser."""
    coef = {"mom_f": 0.02, "lottery_f": -0.08, "delivery_f": 0.03}
    applied, reason = cm.reachable_multipliers({"mom": 0.5}, coef)
    assert applied is None
    assert "stabiliser" in reason


def test_the_guard_lets_the_layer_act_when_a_stabiliser_exists():
    coef = {"mom_f": 0.02, "value_f": 0.04}
    applied, reason = cm.reachable_multipliers({"mom": 0.5, "value": 1.1}, coef)
    assert applied == {"mom": 0.5, "value": 1.1}
    assert reason is None


def test_every_scoring_path_applies_the_same_regime_rule():
    """THE divergence. The guard lived in `fit_predict` alone, so the regime
    layer behaved differently depending on which branch produced the ranking:
    guarded on a refit (1 session in 21), unguarded on the cached path (the
    other 20), and never applied at all when a refit was held back. Three
    behaviours for one rule, and the two that skipped the guard were the
    common ones.

    Invisible today only because every targeted family sits at coefficient
    zero, so all three produce the same score -- it activates the moment `mom`
    is priced again, which is exactly when the layer is meant to matter.
    """
    import inspect

    # A model with momentum priced and NO defensive family: the guard must bite.
    model = cm.CrossSectionalModel(
        coef={"mom_f": 0.02, "delivery_f": 0.03}, n_train=900,
        train_end=dt.date(2026, 1, 1), features=["mom_f", "delivery_f"],
    )
    model.mu = np.zeros(2)
    model.sd = np.ones(2)
    model.intercept = 0.0

    features = pd.DataFrame({
        "mom_f": [1.0, -1.0, 0.5, 0.2],
        "delivery_f": [0.3, 0.1, -0.4, 0.9],
        "symbol": ["A", "B", "C", "D"],
    })
    without = cm.score_with(model, features)
    with_mult = cm.score_with(model, features, {"mom": 0.5})
    assert without.equals(with_mult), (
        "the cached path applied a multiplier the refit path would have skipped"
    )
    assert model.regime_multipliers_applied is False

    # And the held-model path must pass them, so the same guard runs there too.
    src = inspect.getsource(
        __import__("prosignal.stages.stage4_core_score", fromlist=["x"])
        ._cross_sectional_model)
    assert "cm.score_with(held, feats, multipliers)" in src, (
        "a refit held back must still score under the regime rule, not without it"
    )


def test_a_held_model_is_scored_on_the_same_features_as_every_other_path():
    """The rejected-refit branch dropped `sectors` and `actions`, so a run that
    held its previous coefficients also ranked every factor universe-wide
    instead of within sector -- reintroducing the unintended sector bet -- and
    lost net issuance's bonus-vs-placement correction."""
    import inspect

    from prosignal.stages import stage4_core_score as s4

    src = inspect.getsource(s4._cross_sectional_model)
    held = src[src.index("held = cm.load_cached"):]
    held = held[:held.index("# NOTHING TO HOLD ON TO")]
    assert "sectors=sector_map" in held
    assert "actions=actions" in held


# -------------------------------------------------------- dispersion gate
def test_a_flat_day_has_no_dispersion():
    assert cm.prediction_dispersion(pd.Series([0.001] * 100)) == pytest.approx(0.0)


def test_a_day_with_a_view_does():
    assert cm.prediction_dispersion(pd.Series(np.linspace(-1, 1, 100))) > 0.3


def test_too_few_names_is_reported_as_no_view_rather_than_guessed():
    assert cm.prediction_dispersion(pd.Series([1.0, 2.0, 3.0])) == 0.0


def test_stage_8_gates_on_dispersion_without_closing_the_book():
    """A day with no view is a day to add nothing, not a day to liquidate."""
    import inspect

    from prosignal.stages import stage8_final_signal as s8

    src = inspect.getsource(s8.run)
    block = src[src.index("min_ratio = fv("):]
    block = block[:block.index("survivors")]
    assert "blocked_reason = (" in block, (
        "the dispersion gate must use the block-entries path, not return []"
    )


def test_the_dispersion_gate_is_a_ratio_not_a_level():
    """The level is a function of the ridge penalty, not of the market. Measured
    across 88 panel dates the entire range was 0.0355 to 0.0607, so an absolute
    floor of 0.15 -- which is where this started -- blocked 100% of days. That
    number was tried and measured before it was replaced."""
    from prosignal.config.loader import load_config

    scarcity = load_config().params.stage8_final_signal.scarcity
    assert hasattr(scarcity, "min_dispersion_ratio")
    assert not hasattr(scarcity, "min_prediction_dispersion")
    ratio = float(scarcity.min_dispersion_ratio.value)
    assert 0.0 < ratio < 1.0, "a ratio, so it lives strictly inside (0, 1)"


def test_a_normal_day_clears_the_measured_floor():
    """p0 of the measured distribution is 0.0355 against a median of 0.0460.
    The worst observed day is 77% of typical, comfortably above a 50% floor --
    the gate must catch degeneracy, not ordinary variation."""
    from prosignal.config.loader import load_config

    floor = float(load_config().params.stage8_final_signal
                  .scarcity.min_dispersion_ratio.value)
    worst_observed_ratio = 0.0355 / 0.0460
    assert worst_observed_ratio > floor


def test_a_store_without_sectors_ranks_against_the_universe_rather_than_failing():
    """Sectors are genuinely absent for part of this universe -- it reaches past
    any index constituent file -- so partial coverage is the normal state and
    total absence is the same state taken to its limit."""
    import inspect

    from prosignal.stages import stage4_core_score as s4

    src = inspect.getsource(s4._cross_sectional_model)
    block = src[src.index("store.read_sector_map()") - 200:]
    block = block[:block.index("cache = store.curated")]
    assert "except Exception" in block
    assert "sector_map = {}" in block


# ----------------------------------------------------- quality and controls
def test_the_quality_family_carries_the_six_it_should():
    assert set(cm.FAMILIES["quality"]) == {
        "gross_profitability_r", "cash_op_profitability_r", "roce_r",
        "accruals_r", "asset_growth_r", "net_issuance_r"}


def test_the_three_that_predict_badly_enter_the_family_negated():
    """The family is built so a HIGHER composite is a BETTER name. A member with
    the wrong sign cancels its neighbours instead of reinforcing them."""
    assert cm.NEGATED_IN_FAMILY == {
        "accruals_r", "asset_growth_r", "net_issuance_r"}
    frame = pd.DataFrame({"accruals_r": [1.0], "gross_profitability_r": [1.0]})
    cm.build_families(frame, list(frame.columns))
    assert frame["quality_f"].iloc[0] == pytest.approx(0.0), (
        "high accruals must cancel high profitability, not add to it"
    )


def test_size_is_computed_and_reported_but_not_scored():
    """Measured over 17 dates it reads IC -0.2297 at a hit rate of 0/17 -- three
    independent windows in which small caps happened to win. A family
    coefficient for that rebuilds by hand the small-cap tilt the point-in-time
    panel fix removed. The unintended-sector-bet problem size was raised against
    is solved by ranking within sector."""
    assert "log_mcap" in cm.FUNDAMENTAL_FEATURES
    scored = {m for members in cm.FAMILIES.values() for m in members}
    assert "log_mcap_r" not in scored
    assert "log_mcap_r" in cm.UNSCORED_CONTROLS


def test_a_bonus_is_not_counted_as_dilution():
    """A 1:1 bonus doubles the share count and dilutes nobody. The raw count
    cannot tell it from a placement."""
    actions = pd.DataFrame({
        "symbol": ["A", "B"],
        "ex_date": ["2026-03-01", "2026-03-01"],
        "action_type": ["bonus", "dividend"],
        "ratio": [0.5, 1.0],
    })
    adj = cm.share_count_adjustment(
        actions, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-01"))
    assert adj["A"] == pytest.approx(2.0)
    assert "B" not in adj.index, "a dividend does not change the share count"


def test_no_actions_means_no_adjustment_rather_than_a_guess():
    assert cm.share_count_adjustment(None, pd.Timestamp("2026-01-01"),
                                     pd.Timestamp("2026-06-01")) is None


# ------------------------------------------------ the 36-month reversal window
def test_residual_reversal_is_standardised_over_36_months():
    """Blitz, Huij, Lansdorp & Martens (2013) standardise by the trailing
    36-month residual standard deviation."""
    from prosignal.features.crosssec import REVERSAL_STD_WINDOW

    assert REVERSAL_STD_WINDOW == 756
    assert REVERSAL_STD_WINDOW / 21 == pytest.approx(36.0)


def test_the_reversal_window_degrades_rather_than_excluding_short_history():
    """Requiring all 756 sessions would exclude any name with under three years
    against a universe floor of 300."""
    import inspect

    from prosignal.features import crosssec

    src = inspect.getsource(crosssec._features_at)
    assert "resid.tail(REVERSAL_STD_WINDOW).std" in src, (
        "tail() takes what is there rather than requiring the full window"
    )


# --------------------------------------------------- the card must add up
def test_the_card_shows_the_coefficient_that_produced_the_contribution():
    """`contributions` stripped `_r` and not `_f`, so a family arrived
    downstream still called `mom_f`, and a lookup appending `_r` asked for
    `mom_f_r` and got nothing. The card printed a factor moving the score by
    0.0390 at a coefficient of +0.00000 -- self-contradictory on its face."""
    assert cm._bare("mom_f") == "mom"
    assert cm._bare("amihud_r") == "amihud"
    assert cm._bare("plain") == "plain"

    m = cm.CrossSectionalModel(
        coef={"mom_f": 0.025, "lottery_f": -0.019},
        n_train=1000, train_end=pd.Timestamp("2026-01-01").date(),
        features=["mom_f", "lottery_f"],
    )
    m.mu = np.array([0.0, 0.0]); m.sd = np.array([1.0, 1.0]); m.intercept = 0.0
    feats = pd.DataFrame({"symbol": ["A"], "mom_f": [1.0], "lottery_f": [1.0]})
    contrib = cm.contributions(m, feats)
    assert list(contrib.columns) == ["mom", "lottery"], (
        "the family suffix must be stripped so the coefficient can be found"
    )
    # And the lookup stage 4 performs finds a real number for that name.
    for name in contrib.columns:
        assert m.coef.get(name + "_f", m.coef.get(name + "_r", 0.0)) != 0.0


def test_every_scored_family_has_a_citation():
    """The card names the family, so the citation has to as well. It was
    printing `(None)` under each one."""
    from prosignal.stages.stage4_core_score import _MODEL_CITE as _CITATIONS

    for family in cm.FAMILIES:
        assert family in _CITATIONS, f"{family} would render as (None)"


def test_a_held_name_reads_as_one_sentence():
    import inspect

    from prosignal.stages import stage8_final_signal as s8

    src = inspect.getsource(s8.run)
    assert '" ".join(filter(None, [' in src, (
        "an empty Stage 6 reason left a double space mid-sentence"
    )
