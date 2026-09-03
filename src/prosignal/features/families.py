"""The family registry the signal card reads.

One family is one bet. Several measured columns can express the same bet -- three
momentum ranks, two delivery ranks -- and averaging them before they are shown
keeps the card from implying more independent evidence than exists.

THIS IS DATA, NOT A MODEL. It lives on its own because it used to live in
`crossmodel.py`: rendering a card therefore imported the whole fitting stack --
crossmodel, famamacbeth, metalabel and linear -- to read one dictionary, and all
four loaded on every signal generation for it.
"""

from __future__ import annotations

from typing import Dict, Tuple

__all__ = ["FAMILIES", "UNSCORED_CONTROLS", "UNSCORED_DIAGNOSTICS"]


# =============================================================================
# Factor families
# =============================================================================
#
# Seventeen coefficients over a set this collinear is not estimable, and the
# near-uniform coefficient band was the model saying so. Measured on the live
# universe:
#
#     amihud / turnover_ratio   -0.869    one factor measured from two sides
#     resid_mom / mom_6_1       +0.770
#     resid_mom / prox_52w      +0.601
#
# Ridge does not pick a winner among collinear inputs, it spreads the penalty
# across the block, so three momentum coefficients that each look small carry an
# effective weight of roughly three times any one of them.
#
# The members are averaged as ranks FIRST and one coefficient is fitted per
# family. Five or six coefficients over several hundred names and a decade of
# cross-sections is estimable; seventeen is not.
#
# LIQUIDITY IS NOT HERE, deliberately. The illiquidity premium is real but it is
# compensation FOR trading costs, and a manual executor pays that cost rather
# than collecting it -- a positive amihud loading walks the book into names
# where realised slippage exceeds forecast alpha. It belongs in the universe
# screen as a floor, which is where `universe.pit_min_adtv_inr` already puts it.
FAMILIES: Dict[str, Tuple[str, ...]] = {
    # Three names for one bet. Averaged, not fitted separately.
    "mom": ("mom_6_1_r", "prox_52w_r", "resid_mom_r"),
    # Reversal is the opposite side of the same axis and stays on its own: it is
    # a different horizon, and folding it into `mom` would net out against it.
    "reversal": ("resid_reversal_r",),
    # Lottery demand. In India this is stronger than the US literature suggests,
    # because the marginal buyer is retail. Signs are aligned so that a HIGHER
    # composite means MORE lottery-like, and the fit is free to price it
    # negatively.
    #
    # THREE MEMBERS, not four. These three are volatility measures correlating
    # 0.48-0.68 within date, which is what makes them one family. `idio_skew`
    # was the fourth and correlates +0.04 with downside_vol and +0.28 with
    # max5_21 -- near-orthogonal to the family it was averaged into, while
    # carrying a quarter of its weight. That is not diversification inside a
    # family, it is a second factor hidden inside a first.
    #
    # The literature says the same: Bali, Cakici & Whitelaw (2011) find MAX
    # subsumes idiosyncratic volatility -- controlling for MAX kills the IVOL
    # effect -- so these three are one mechanism. Skewness preference is
    # controlled for SEPARATELY in that paper and is a different channel.
    "lottery": ("max5_21_r", "idio_vol_r", "downside_vol_r"),
    # Skewness preference, on its own rather than diluting `lottery`. Measured
    # against the real forward return it reads t -0.94, so the significance
    # gate will almost certainly zero it -- which is the point. A theme that is
    # visible and zeroed is more informative than one that is invisible and
    # quietly diluting its neighbours.
    "skew": ("idio_skew_r",),
    # SPLIT, not averaged. Measured within date these two correlate -0.42: a
    # high beta rank is RISKIER, a high max_dd rank is a SHALLOWER drawdown and
    # therefore SAFER. Averaging them under a common sign cancelled the axis --
    # beta alone t -3.67 and max_dd alone t +4.69 became a composite at t -0.93,
    # which the significance gate then discarded for being insignificant. Two
    # significant signals were averaged into one insignificant column.
    #
    # The families exist for ESTIMABILITY -- seventeen coefficients over a
    # collinear set is not estimable -- and two ANTICORRELATED members are not a
    # collinear block. They are two different bets. One coefficient each; the
    # fit prices either, both or neither on its own evidence.
    #
    # A second cost of the average: max_dd_120 correlates +0.63 with prox_52w,
    # so `risk` was partly a momentum factor wearing a risk label.
    "beta": ("beta_120_r",),
    "drawdown": ("max_dd_120_r",),
    # Delivered share of traded volume. No clean analogue outside India.
    "delivery": ("deliv_pct_r", "deliv_trend_r"),
    "value": ("earnings_yield_r", "book_to_price_r", "ebitda_to_ev_r",
              "fcf_yield_r", "sales_to_price_r"),
    # Quality is a SLOW factor: a modest gross edge that turns over slowly and
    # therefore sits far below breakeven turnover, which is where most of a
    # gross edge is otherwise lost.
    "quality": ("gross_profitability_r", "cash_op_profitability_r", "roce_r",
                "accruals_r", "asset_growth_r", "net_issuance_r"),
}

#: Computed, reported, and NOT scored. `log_mcap` is carried so the size of what
#: the model is ranking is visible, and so `research factors` can measure it --
#: but it does not get a coefficient.
#:
#: Measured over 17 dates it reads IC -0.2297 at a hit rate of 0/17, which is
#: not a factor, it is three independent 63-session windows in which small caps
#: happened to win. Giving that a family coefficient equal in weight to momentum
#: would rebuild by hand exactly the small-cap tilt the point-in-time panel fix
#: took out. The unintended-sector-bet problem it was raised against is solved
#: by ranking within sector, which is done.
UNSCORED_CONTROLS = ("log_mcap_r",)

#: Computed and reported, deliberately NOT scored, and not a control either.
#: These belong to no family, which made them invisible: they were ranked,
#: correlation-checked and discarded on every run with nothing saying so, and
#: `research factors` reported `amihud` at IC +0.0362 (t +2.19) and
#: `turnover_ratio` at -0.0319 (t -3.00) as though they were candidates.
#:
#: They are one factor measured from two sides -- they correlate -0.905 within
#: date -- and the side they measure is the ILLIQUIDITY PREMIUM. That premium is
#: real and it is compensation FOR trading costs, which a manual executor pays
#: rather than collects. A positive loading walks the book into names where
#: realised slippage exceeds forecast alpha. It belongs in the universe screen
#: as a floor, which is where `universe.pit_min_adtv_inr` already puts it.
#:
#: Declared here so the exclusion is a decision on the record rather than an
#: omission, and so the redundancy report can stop flagging a breach between two
#: factors neither of which is used.
UNSCORED_DIAGNOSTICS = ("amihud_r", "turnover_ratio_r")
