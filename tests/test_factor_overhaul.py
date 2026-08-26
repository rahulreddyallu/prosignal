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

And every family beats its own best member on ICIR, which is what averaging
correlated members is supposed to do:

    family     ICIR      best member ICIR
    mom       +0.505     +0.462 (prox_52w)
    risk      +0.488     +0.327 (max_dd_120)
    delivery  +0.420     +0.388 (deliv_trend)
"""

from __future__ import annotations

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


def test_the_lottery_family_carries_every_lottery_moment():
    """MAX alone was -0.00137, second smallest in the live model. India's
    lottery effect is driven by retail flow and is stronger than the US
    literature suggests, so the moments are treated as one block."""
    assert set(cm.FAMILIES["lottery"]) == {
        "max5_21_r", "idio_vol_r", "idio_skew_r", "downside_vol_r"}


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
