"""The short leg's cost stack: priced, dated, and not traded.

WHY IT EXISTS BEFORE ANY SHORT CODE DOES. `costs:` had no derivatives leg, so
the short side could not be priced -- and a leg with no cost model does not read
as UNKNOWN in a later comparison, it reads as FREE.

WHY THE DATES ARE MANDATORY. Futures STT tripled from 0.02% to 0.05% on
1 April 2026. Nothing in this repository could have noticed that, because a
STATUTORY value carried no record of when it was last checked. An undated
statutory rate reads as permanent, and it is not.

AND NOTHING HERE PLACES AN ORDER. `test_no_execution_capability_arrived_with_it`
is not decoration: a cost model is the kind of change that invites an order
object to follow it, and docs/EXECUTION_GATE.md governs.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from pydantic import ValidationError

from prosignal.config.loader import load_config
from prosignal.config.schema import ParamStatus, Tunable

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def deriv(cfg):
    d = cfg.params.costs.derivatives
    assert d is not None, (
        "costs.derivatives is absent. The short side cannot be priced without "
        "it, and an unpriceable leg reads as free.")
    return d


# ------------------------------------------------------------ statutory rates
def test_futures_stt_is_the_april_2026_rate(deriv):
    node = deriv.stt_futures_sell_pct
    assert node.value == pytest.approx(0.05), (
        f"futures STT is {node.value}. It was raised from 0.02% to 0.05% on "
        f"1 April 2026 (Budget 2026-27). A short leg priced at the old rate is "
        f"wrong by 150% of that leg, in the flattering direction.")
    assert node.status is ParamStatus.STATUTORY
    assert node.verified_on == dt.date(2026, 9, 3)


def test_options_premium_stt_is_carried_and_unused(deriv):
    assert deriv.stt_options_premium_sell_pct.value == pytest.approx(0.15)
    assert deriv.stt_options_premium_sell_pct.status is ParamStatus.STATUTORY
    # Carried for completeness. If something starts reading it, that is a
    # decision about options and it should not arrive by accident.
    src = (ROOT / "src").rglob("*.py")
    readers = [p.relative_to(ROOT) for p in src
               if "stt_options_premium_sell_pct" in p.read_text(encoding="utf-8")
               and p.name not in ("schema.py",)]
    assert not readers, (
        f"something now reads the options STT: {readers}. Options are rejected "
        f"on design grounds -- a 42-63 session cross-sectional forecast is a "
        f"directional view with no volatility view -- so a reader of this "
        f"parameter is a change of position, not a refactor.")


def test_every_statutory_rate_carries_a_verification_date(cfg):
    """The discipline, applied to the whole cost block and not only the new leg."""
    undated = [t["path"] for t in cfg.params.iter_tunables()
               if t["status"] == ParamStatus.STATUTORY.value
               and not t.get("verified_on")]
    assert not undated, (
        f"STATUTORY parameters with no `verified_on`: {undated}. Statutory "
        f"rates change on a budget cycle and an undated one is a rate nobody "
        f"can tell is stale.")


def test_the_loader_refuses_an_undated_statutory_rate():
    with pytest.raises(ValidationError) as exc:
        Tunable[float](value=0.05, status=ParamStatus.STATUTORY)
    assert "verified_on" in str(exc.value)


def test_a_dated_statutory_rate_is_accepted():
    t = Tunable[float](value=0.05, status=ParamStatus.STATUTORY,
                       verified_on=dt.date(2026, 9, 3))
    assert t.value == 0.05


# ------------------------------------------------------ the unvalidated ones
def test_borrow_and_roll_are_unvalidated_with_declared_ranges(deriv):
    """Swept, never claimed. Neither can be estimated from this store."""
    for node, lo, hi in ((deriv.borrow_fee_annual_pct, 0.5, 12.0),
                         (deriv.futures_roll_spread_bps, 2.0, 40.0)):
        assert node.status is ParamStatus.UNVALIDATED
        assert node.search_range == [lo, hi], node.search_range
        assert lo <= float(node.value) <= hi


def test_futures_margin_is_operational_and_says_it_is_a_proxy(deriv):
    """The real number is exchange-computed SPAN+ELM, per contract per day."""
    node = deriv.futures_margin_pct
    assert node.status is ParamStatus.OPERATIONAL
    note = (node.note or "").upper()
    assert "SPAN" in note and "ELM" in note, (
        "the futures margin note does not say the real figure is exchange-"
        "computed. A planning proxy presented as the margin that will be "
        "called is the kind of number that ends a paper trade at the clearing "
        "corporation.")


# ------------------------------------------------------------- the guard rail
def test_the_config_says_it_is_not_a_legal_source():
    text = (ROOT / "config" / "parameters.yaml").read_text(encoding="utf-8")
    block = text.split("derivatives:", 1)[0][-4000:]
    assert "NOT A LEGAL SOURCE" in block.upper() or \
           "not a legal source" in text[:text.find("derivatives:")][-4000:], (
        "the derivatives block does not warn that these rates are transcribed "
        "and must be checked against the live circular.")


def test_no_execution_capability_arrived_with_the_cost_model():
    """docs/EXECUTION_GATE.md governs. A cost model is not an order path.

    The gate file lists the triggers that require re-reading it; a broker
    client, an order/fill/ticket object, or a scheduler that acts on output.
    None may appear, and adding a derivatives cost stack is exactly the kind of
    change after which one quietly does.
    """
    banned = ("class Order", "class Fill", "class Ticket", "def place_order",
              "broker_client", "BrokerClient", "kiteconnect", "smartapi",
              "upstox_client")
    hits = []
    for p in (ROOT / "src").rglob("*.py"):
        body = p.read_text(encoding="utf-8")
        for token in banned:
            if token in body:
                hits.append(f"{p.relative_to(ROOT)}: {token}")
    assert not hits, (
        "order-routing surface appeared in src/: " + "; ".join(hits)
        + ". docs/EXECUTION_GATE.md governs; SEBI's retail algo framework has "
          "been mandatory since 1 April 2026 and any order-placing code makes "
          "this repository a different compliance object.")
