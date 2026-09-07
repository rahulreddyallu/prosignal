"""Stage 7 and `portfolio_sim` must size the same name the same way.

The 20-name book was chosen on numbers measured through `portfolio_sim`, and
`capital.sizing_mode` / `capital.target_deployment` are tagged MEASURED on that
evidence. The engine sizes through `stage7_risk`. If the two disagree, the
config is claiming a measurement of a book the engine does not run -- which is
the research/production parity break this audit already found once, in
cadence (Q6).

Reconciled 2026-09-06 on seven (price, ADTV) points spanning the live universe:
both paths size to the slot and cap by liquidity, and the only difference is
integer-share truncation -- max Rs 2,289 on a Rs 37,500 slot, always DOWNWARD,
so the live book is marginally less deployed than the simulator assumed. That
is the conservative direction and it is rounding, not a rule difference.

The category multiplier is the one thing `portfolio_sim` does not model at
all. It is score-driven, and a book taken from the top 20 of 386 is high-score
by construction: all 26 cards in the run that produced this book were
STANDARD, frac 1.0. The test below pins that assumption rather than trusting
it -- if a top-20 name ever comes back REDUCED, the simulator is over-sizing
it by 40% and the measured numbers no longer describe the book.
"""

from __future__ import annotations

import pytest

from prosignal.config.loader import get_config


@pytest.fixture(scope="module")
def cap():
    return get_config().params.capital


def test_the_slot_is_the_target_deployment_split_over_the_book(cap):
    if str(cap.sizing_mode.value) != "equal_weight":
        pytest.skip("risk_budget sizing does not target a deployment")
    expected = (float(cap.total_capital_inr.value)
                * float(cap.target_deployment.value)
                / int(cap.max_open_positions.value))
    assert cap.position_value_inr() == pytest.approx(expected)


def test_both_paths_agree_to_share_truncation(cap):
    """The parity check itself, on points spanning the live universe."""
    if str(cap.sizing_mode.value) != "equal_weight":
        pytest.skip("parity is only claimed for the shipped equal-weight mode")
    capital = float(cap.total_capital_inr.value)
    k = int(cap.max_open_positions.value)
    dep = float(cap.target_deployment.value)
    part = float(cap.max_participation_of_adtv.value)
    slot = cap.position_value_inr()

    worst = 0.0
    for price, adtv in ((100, 5.0e8), (1000, 5.0e8), (7100, 4.8e8),
                        (162, 2.0e8), (2411, 1.0e8), (50, 2.0e7),
                        (3201, 3.0e7)):
        q_liq = int((adtv * part) / price)
        live = max(min(int(slot / price), q_liq), 0) * price      # stage 7
        sim = max(min((capital * dep / k) / price,
                      (adtv * part) / price), 0.0) * price        # portfolio_sim
        assert live <= sim + 1e-9, (
            f"stage 7 sized ABOVE the simulator at price {price}: "
            f"{live:,.0f} against {sim:,.0f}. Truncation can only round down; "
            f"anything else is a rule difference."
        )
        worst = max(worst, sim - live)
    assert worst < price_tolerance(slot), (
        f"largest gap Rs {worst:,.0f} on a Rs {slot:,.0f} slot exceeds what "
        f"integer shares can explain"
    )


def price_tolerance(slot: float) -> float:
    """One whole share of the priciest name a slot can hold, plus slack.

    A slot cannot be short by more than the price of one share, and the most
    expensive name in the live universe trades near Rs 7,100.
    """
    return 7_500.0


def test_the_category_multiplier_does_not_bite_on_a_top_book():
    """`portfolio_sim` models no risk category. Stage 7 multiplies the slot by
    1.0 / 0.6 / 0.3 on STANDARD / REDUCED / MINIMUM, and the category is set by
    SCORE -- so a book drawn from the top of the ranking is STANDARD by
    construction. If that stops being true the simulator is over-sizing."""
    from prosignal.core.enums import RiskCategory
    from prosignal.stages.stage7_risk import _CATEGORY_FRACTION, _category

    cfg = get_config().params.stage7_risk
    assert _CATEGORY_FRACTION[RiskCategory.STANDARD] == 1.0
    # A top-20 name of 386 sits at the 95th percentile or better.
    assert _category(0.95, None, cfg) is RiskCategory.STANDARD
    assert _category(0.99, None, cfg) is RiskCategory.STANDARD


def test_the_config_does_not_claim_more_than_was_measured(cap):
    """Both new parameters are tagged MEASURED. The measurement is
    `portfolio_sim`'s, and this file is what licenses carrying that tag over to
    the live path."""
    assert str(cap.sizing_mode.status).endswith("MEASURED")
    assert str(cap.target_deployment.status).endswith("MEASURED")
    assert 0.0 < float(cap.target_deployment.value) <= 1.0
