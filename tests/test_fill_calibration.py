"""Impact is an assumption, and the ledger cannot turn it into a measurement.

`CostModel.impact_bps` is `coefficient * participation ** exponent` plus an
assumed half-spread, and both constants come from the config. Nothing in this
engine has ever compared them to a price it traded at, so the 87 bps round trip
every backtest subtracts is an assumption wearing the clothes of a measurement.

The obvious calibration -- regress realised implementation shortfall on
participation -- cannot run, and the reason matters more than the result. In
126 of the 128 rows in `data/ledger/outcomes.jsonl` the recorded `entry_price`
equals the NEXT SESSION'S OPEN to the tick. They were written by the engine's
own entry rule, not by a broker. Fitting the impact model to them would fit it
to the assumption it was built from and report the circularity as agreement.

The remaining two are VEDL rows holding an unadjusted price against an adjusted
open -- 415.65 against 145.90, ratio 2.85, `price_basis_factor` recorded as 1.0
-- which is the price-basis defect, not a fill.

So the harness exists, it runs, and it returns SYNTHETIC_FILLS. That verdict is
a finding about the evidence, and these tests exist to stop it being quietly
rendered as "no impact" or as a passing calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.validation import fill_calibration as FC

DATES = pd.bdate_range("2024-01-01", periods=40)
SYMBOLS = [f"S{i:02d}" for i in range(30)]


def _prices() -> pd.DataFrame:
    """Flat per symbol, with turnover spanning three decades.

    FLAT ON PURPOSE. Shortfall is measured from the decision close to the fill
    price, so close-to-open drift lands in the dependent variable -- at 1%
    daily vol that is ~140bps of noise per fill, far larger than any impact
    curve at this book's participation. A fixture carrying it tests the noise.

    WIDE TURNOVER ON PURPOSE. The original fixture ran 2e8 to 4e8, so
    participation barely varied and there was no curve to fit in either
    direction. `calibrate` now fits a power law over participation buckets and
    correctly refuses a sample that has no participation range.
    """
    rng = np.random.default_rng(5)
    rows = []
    for s in SYMBOLS:
        base = 100.0 * (1 + rng.uniform(-0.2, 0.2))
        turnover = 10.0 ** rng.uniform(7.0, 10.0)
        for d in DATES:
            rows.append({"symbol": s, "date": d, "open": base,
                         "close": base, "vwap": base, "turnover": turnover})
    return pd.DataFrame(rows).set_index(["symbol", "date"]).sort_index()


def _ledger(prices: pd.DataFrame, n: int = 100, *, fill: str) -> pd.DataFrame:
    """`fill='open'` reproduces the shipped ledger; `fill='slipped'` writes a
    price carrying a real, participation-dependent impact curve.

    The slipped case used to be `o * (1 + |N(0, 0.004)|)` -- slippage
    uncorrelated with size. That is not impact, and `calibrate` now says so
    rather than fitting a coefficient to it, which is the whole point of
    fitting over participation buckets.
    """
    rng = np.random.default_rng(9)
    rows = []
    for i in range(n):
        s = SYMBOLS[i % len(SYMBOLS)]
        sd, ed = DATES[i % 30], DATES[(i % 30) + 1]
        o = float(prices.loc[(s, ed), "open"])
        if fill == "open":
            px = o
        else:
            part = 125_000.0 / float(prices.loc[(s, sd), "turnover"])
            slip = 0.10 * (part ** 0.5) * 1e4 + rng.normal(0, 2.0)
            px = o * (1 + slip / 1e4)
        rows.append({"ticker": s, "signal_date": sd, "entry_date": ed,
                     "entry_price": px})
    return pd.DataFrame(rows)


def test_a_ledger_of_next_opens_is_refused_as_synthetic():
    """The finding, as one assertion."""
    px = _prices()
    out = FC.calibrate(_ledger(px, fill="open"), px, position_value_inr=125_000.0)
    assert out.verdict == FC.SYNTHETIC
    assert out.synthetic_against == "entry_open"
    assert out.synthetic_share == pytest.approx(1.0)
    assert not out.usable
    assert np.isnan(out.fitted_coefficient), (
        "a refused calibration must not report a coefficient; a NaN is a "
        "missing measurement and a number is a claim"
    )


def test_the_refusal_says_why_rather_than_reporting_no_impact():
    out = FC.calibrate(_ledger(_prices(), fill="open"), _prices(),
                       position_value_inr=125_000.0)
    for bit in ("not executions", "circularity", "UNCALIBRATED"):
        assert bit in out.reason


def test_genuine_fills_are_calibrated():
    """The harness must actually work, or the refusal above proves nothing."""
    px = _prices()
    out = FC.calibrate(_ledger(px, n=120, fill="slipped"), px,
                       config_coefficient=0.1, config_exponent=0.5,
                       position_value_inr=125_000.0)
    assert out.verdict == FC.CALIBRATED, out.reason
    assert out.usable
    assert np.isfinite(out.fitted_coefficient)
    assert np.isfinite(out.fitted_exponent)
    assert out.config_coefficient == 0.1


def test_too_few_fills_is_its_own_verdict():
    px = _prices()
    out = FC.calibrate(_ledger(px, n=20, fill="slipped"), px,
                       position_value_inr=125_000.0)
    assert out.verdict == FC.INSUFFICIENT
    assert not out.usable


def test_an_unadjusted_price_basis_row_is_dropped_and_named():
    """A shortfall of +18,488 bps is a demerger, not slippage, and one such row
    moves a fitted coefficient by more than every genuine row combined."""
    px = _prices()
    led = _ledger(px, n=120, fill="slipped")
    led.loc[0, "entry_price"] = float(led.loc[0, "entry_price"]) * 2.85
    out = FC.calibrate(led, px, position_value_inr=125_000.0)
    assert len(out.rows_dropped) == 1
    assert "unadjusted price basis" in out.rows_dropped[0]
    assert out.verdict == FC.CALIBRATED


def test_an_empty_or_shapeless_ledger_is_no_data_not_a_calibration():
    assert FC.calibrate(pd.DataFrame(), _prices()).verdict == FC.NO_DATA
    bad = pd.DataFrame({"ticker": ["X"], "signal_date": [DATES[0]]})
    assert FC.calibrate(bad, _prices()).verdict == FC.NO_DATA


def test_the_shipped_ledger_is_still_synthetic():
    """The one that matters, run against the real ledger and the real store.

    If this ever fails with CALIBRATED, real fills have arrived and the impact
    coefficient can finally be checked against them -- which is the outcome
    this whole module is waiting for.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    led = FC.read_outcomes(root / "data/ledger/outcomes.jsonl")
    if led.empty:
        pytest.skip("no outcome ledger in this checkout")
    if not (root / "data/curated/prices").is_dir():
        pytest.skip("no price store in this checkout")

    # THROUGH `DataStore`, which is the engine's own read path and applies
    # corporate actions. It matters which basis this runs on: read straight off
    # the year parquets instead and the synthetic share falls to 92.4% and
    # TATAINVEST appears with a fill of 1066.00 against a raw open of 10660.00.
    # The ledger's `entry_price` is not on one consistent basis, which is a
    # separate defect and not a reason to calibrate impact on raw levels.
    from prosignal.config.loader import get_config
    from prosignal.data.store import DataStore

    cfg = get_config()
    px = DataStore(cfg.paths.curated, cfg.paths.snapshots).read_prices()
    px = px[px["symbol"].isin(set(led["ticker"].astype(str)))].copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.set_index(["symbol", "date"]).sort_index()

    out = FC.calibrate(led, px, position_value_inr=125_000.0)
    assert out.verdict == FC.SYNTHETIC, (
        f"expected the shipped ledger to be refused as synthetic; got "
        f"{out.verdict}: {out.reason}"
    )
    assert out.synthetic_against == "entry_open"
    assert out.synthetic_share >= 0.98
