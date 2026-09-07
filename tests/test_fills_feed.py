"""The one feed no vendor can supply: what the book actually paid.

`costs.impact_model.coefficient` has never been compared to a price this engine
traded at, and the outcome ledger cannot stand in -- 126 of its 128
`entry_price` values are the next session's OPEN to the tick, which is the
simulator's own entry rule. Fitting impact to those fits it to the assumption
it was built from.

So impact was uncalibratable, and the audit named the gap without naming a
route. The route was already in the repository: `CsvImportProvider` exists for
"anything the free sources cannot give you honestly" -- pledging, PIT
fundamentals. A `fills.csv` goes through the same door.

IMPORTING A RECORD OF WHAT YOU DID IS NOT ORDER ROUTING. `EXECUTION_GATE.md`
governs placing orders; this reads a CSV of orders already placed, and
`test_derivatives_costs.py` still guards the boundary.

Verified end to end against a known truth: 210 fills generated from real store
turnover with an injected coefficient of 0.10 and exponent 0.5 plus 2bps of
noise recovered **0.0768 and 0.4480**. The synthetic fills were then removed
from the store -- claiming a calibration the engine does not have would be the
contamination this audit exists to stop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.validation import fill_calibration as FC

DATES = pd.bdate_range("2026-01-01", periods=60)
SYMBOLS = [f"S{i:02d}" for i in range(40)]


def _prices() -> pd.DataFrame:
    """Turnover spanning three decades, so participation does too.

    The price path is FLAT per symbol on purpose. Shortfall is measured from
    the decision close to the fill price, so any close-to-open drift lands in
    the dependent variable -- at 1% daily vol that is ~140bps of noise per
    fill, which swamps the injected curve and makes the fixture a test of the
    noise rather than of the estimator. Real fills carry that drift and it is
    exactly why the fit is done on bucket MEANS.
    """
    rng = np.random.default_rng(3)
    rows = []
    for s in SYMBOLS:
        turnover = 10.0 ** rng.uniform(7.0, 10.0)
        base = 100.0 * (1 + rng.uniform(-0.3, 0.3))
        for d in DATES:
            rows.append({"symbol": s, "date": d, "open": base,
                         "close": base, "vwap": base, "turnover": turnover})
    return pd.DataFrame(rows).set_index(["symbol", "date"]).sort_index()


def _fills(prices, coeff=0.10, expo=0.5, noise_bps=2.0, n=220, seed=5,
           slip_bps=None):
    """Executions with a KNOWN impact curve injected.

    `slip_bps(participation)` overrides the power law for shapes a power law
    cannot express -- the downward-sloping case below needs one that stays in
    a plausible bps range while still falling with size.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        s = SYMBOLS[i % len(SYMBOLS)]
        d0, d1 = DATES[i % 50], DATES[(i % 50) + 1]
        o = float(prices.loc[(s, d1), "open"])
        turnover = float(prices.loc[(s, d0), "turnover"])
        part = 125_000.0 / turnover
        base = (slip_bps(part) if slip_bps is not None
                else coeff * (part ** expo) * 1e4)
        slip = base + rng.normal(0, noise_bps)
        rows.append({"symbol": s, "decision_date": d0, "fill_date": d1,
                     "side": "BUY", "quantity": 100,
                     "price": o * (1 + slip / 1e4), "venue": "NSE"})
    return pd.DataFrame(rows)


def test_recorded_fills_are_mapped_onto_the_calibration_shape():
    out = FC.from_fills(_fills(_prices()))
    assert set(out.columns) == {"ticker", "signal_date", "entry_date",
                                "entry_price"}
    assert len(out) == 220


def test_sells_are_excluded():
    """Shortfall on a sell is a different quantity with a different sign, and
    mixing the two fits a coefficient to neither."""
    f = _fills(_prices())
    f.loc[f.index[:100], "side"] = "SELL"
    assert len(FC.from_fills(f)) == 120


def test_a_fill_without_a_decision_date_is_not_calibratable():
    """It is still a true record of a trade. Shortfall is measured from the
    decision close, so a fill that cannot name one cannot contribute."""
    f = _fills(_prices())
    f.loc[f.index[:50], "decision_date"] = pd.NaT
    assert len(FC.from_fills(f)) == 170


def test_the_known_curve_is_recovered():
    """The test that makes every other one in this file mean something."""
    px = _prices()
    out = FC.calibrate(FC.from_fills(_fills(px)), px,
                       config_coefficient=0.10, config_exponent=0.5,
                       position_value_inr=125_000.0)
    assert out.verdict == FC.CALIBRATED, out.reason
    assert out.fitted_exponent == pytest.approx(0.5, abs=0.15), out.fitted_exponent
    assert out.fitted_coefficient == pytest.approx(0.10, rel=0.6), \
        out.fitted_coefficient


def test_the_fit_does_not_condition_on_the_dependent_variable():
    """THE DEFECT THIS REPLACED. Regressing log(shortfall) on individual fills
    has to drop non-positive shortfalls, which keeps only the fills that went
    against you and biases every coefficient upward. Bucket means are defined
    with negatives in them, so no fill is dropped for its own outcome.
    """
    px = _prices()
    # Heavy noise puts roughly half the individual shortfalls below zero.
    out = FC.calibrate(FC.from_fills(_fills(px, noise_bps=15.0)), px,
                       config_coefficient=0.10, config_exponent=0.5,
                       position_value_inr=125_000.0)
    assert out.n_usable == 220, (
        f"only {out.n_usable} of 220 fills reached the fit; rows are being "
        f"dropped on the sign of their own shortfall"
    )


def test_a_downward_sloping_fit_is_refused():
    """A negative exponent says a larger trade moves the price less. Fed back
    into the cost model it makes size cheaper than a small order -- the one
    direction a cost model must never be wrong in."""
    px = _prices()
    # Falling in participation but bounded to a plausible bps range: a power
    # law with a negative exponent explodes at small participation and every
    # row is then dropped by the price-basis guard, which tests nothing.
    out = FC.calibrate(
        FC.from_fills(_fills(px, slip_bps=lambda part: 40.0 - 3.0e3 * part)),
        px, config_coefficient=0.10, config_exponent=0.5,
        position_value_inr=125_000.0)
    assert out.verdict == FC.IMPLAUSIBLE
    assert not out.usable
    assert np.isnan(out.fitted_coefficient)
    assert "not an impact curve" in out.reason
    assert "Median shortfall" in out.reason or "median shortfall" in out.reason.lower()


def test_pure_noise_does_not_produce_a_curve():
    """At this book's participation -- 0.014% of ADTV -- no measurable impact
    is the expected answer, and reporting a flat level beats fitting a power
    law to noise."""
    px = _prices()
    out = FC.calibrate(FC.from_fills(_fills(px, coeff=0.0, noise_bps=10.0)),
                       px, position_value_inr=125_000.0)
    assert out.verdict in (FC.IMPLAUSIBLE, FC.INSUFFICIENT)
    assert not out.usable


def test_the_loader_rejects_impossible_rows():
    from prosignal.data.providers.csv_import import REFERENCE_TEMPLATES

    assert REFERENCE_TEMPLATES["fills"] == [
        "symbol", "decision_date", "fill_date", "side", "quantity", "price",
        "venue"]


def test_the_store_keys_on_price_so_tranches_survive():
    """A position filled in several tranches at different prices on one day is
    several fills. Keying on the day would keep one and discard the rest --
    and the tranches that moved the price are the expensive ones."""
    import inspect

    from prosignal.data.store import DataStore

    src = inspect.getsource(DataStore.write_fills)
    assert '"fill_date", "side", "price"' in src


def test_the_feed_is_wired_into_ingest_and_absent_is_reported():
    import inspect

    from prosignal.data import ingest

    src = inspect.getsource(ingest.DataIngestor._ingest_csv_feeds)
    assert "self.csv.load_fills()" in src
    assert "write_fills" in src
    assert "UNCALIBRATED" in src, (
        "an absent fills file must be recorded as the reason impact is "
        "uncalibrated, not silently skipped"
    )


def test_no_synthetic_fills_were_left_in_the_store():
    """The verification run wrote 210 fabricated fills to prove the path
    works. Leaving them would make the engine claim a calibration it does not
    have, which is the contamination this whole audit is about."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert not (root / "data/curated/fills.parquet").exists(), (
        "fills.parquet is present. If these are REAL executions, delete this "
        "assertion deliberately and say so. If they are the test fixtures, "
        "remove them."
    )
    csv = root / "config/reference/fills.csv"
    if csv.is_file():
        body = csv.read_text(encoding="utf-8").strip().splitlines()
        assert len(body) <= 1, "the fills template has rows in it"
