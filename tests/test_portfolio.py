"""Buffered portfolio construction.

The buffer is the point of this module, so the tests target it directly: a name
oscillating around the entry boundary must be HELD, not churned, and an
inverted band must be rejected rather than silently producing more turnover
than no buffer at all.
"""

from __future__ import annotations

import datetime as dt

import pytest

from prosignal.portfolio import BufferedPortfolio

D0, D1, D2 = dt.date(2026, 1, 1), dt.date(2026, 4, 1), dt.date(2026, 7, 1)
PRICES = {f"S{i:02d}": 100.0 for i in range(50)}
RANKED = [f"S{i:02d}" for i in range(50)]
SLOT = 100_000.0


def _pf(entry=5, exit_=10, cap=5):
    return BufferedPortfolio(entry_rank=entry, exit_rank=exit_, max_positions=cap)


def test_inverted_band_is_rejected():
    """exit tighter than entry inverts the hysteresis and INCREASES churn."""
    with pytest.raises(ValueError, match="inverts the buffer"):
        BufferedPortfolio(entry_rank=30, exit_rank=15)


def test_buys_only_inside_the_entry_band():
    pf = _pf()
    d = pf.rebalance(D0, RANKED, PRICES, SLOT)
    assert set(d.entries) == set(RANKED[:5])
    assert len(pf.positions) == 5


def test_a_name_drifting_past_entry_but_inside_exit_is_HELD():
    """The whole reason the buffer exists.

    Without it, a name oscillating around rank 5 is sold and re-bought every
    rebalance, paying a full round trip for no change in view.
    """
    pf = _pf(entry=5, exit_=10)
    pf.rebalance(D0, RANKED, PRICES, SLOT)
    assert "S00" in pf.held()

    # S00 slips to rank 8 -- outside entry, inside exit
    drifted = RANKED[1:8] + ["S00"] + RANKED[8:]
    d = pf.rebalance(D1, drifted, PRICES, SLOT)

    assert "S00" in pf.held(), "held name inside the exit band must not be sold"
    assert "S00" not in [t for t, _ in d.exits]
    assert any("without the buffer" in n.lower() for n in d.notes)


def test_a_name_outside_the_exit_band_is_sold():
    pf = _pf(entry=5, exit_=10)
    pf.rebalance(D0, RANKED, PRICES, SLOT)
    dropped = RANKED[1:20] + ["S00"] + RANKED[20:]      # S00 -> rank 20
    d = pf.rebalance(D1, dropped, PRICES, SLOT)
    assert "S00" not in pf.held()
    assert any(t == "S00" and "exit band" in r for t, r in d.exits)


def test_ineligible_names_are_sold_regardless_of_rank():
    """Eligibility is a tradability statement, not an attractiveness one."""
    pf = _pf()
    pf.rebalance(D0, RANKED, PRICES, SLOT)
    eligible = set(RANKED) - {"S00"}
    d = pf.rebalance(D1, RANKED, PRICES, SLOT, eligible=eligible)
    assert "S00" not in pf.held()
    assert any(t == "S00" and "eligible" in r for t, r in d.exits)


def test_buffer_produces_strictly_less_turnover_than_no_buffer():
    """The measured claim, asserted rather than assumed."""
    import random

    rng = random.Random(7)
    sequences = []
    order = list(RANKED)
    for _ in range(8):
        shuffled = order[:12]
        rng.shuffle(shuffled)
        sequences.append(shuffled + order[12:])

    buffered = BufferedPortfolio(entry_rank=5, exit_rank=12, max_positions=5)
    plain = BufferedPortfolio(entry_rank=5, exit_rank=5, max_positions=5)
    for i, seq in enumerate(sequences):
        day = dt.date(2026, 1, 1) + dt.timedelta(days=90 * i)
        buffered.rebalance(day, seq, PRICES, SLOT)
        plain.rebalance(day, seq, PRICES, SLOT)

    assert buffered.annualised_turnover(63) < plain.annualised_turnover(63)


def test_max_positions_is_respected():
    pf = _pf(entry=20, exit_=30, cap=5)
    pf.rebalance(D0, RANKED, PRICES, SLOT)
    assert len(pf.positions) == 5


def test_turnover_is_zero_when_nothing_changes():
    pf = _pf()
    pf.rebalance(D0, RANKED, PRICES, SLOT)
    d = pf.rebalance(D1, RANKED, PRICES, SLOT)
    assert d.entries == [] and d.exits == []
    assert d.turnover_fraction == 0.0


def test_blocked_regime_prevents_entries_but_reports_the_book():
    """An empty eligible set means the regime gate is shut."""
    pf = _pf()
    d = pf.rebalance(D0, RANKED, PRICES, SLOT, eligible=set())
    assert d.entries == []
    assert len(pf.positions) == 0


def test_summary_reports_annualised_turnover():
    pf = _pf()
    pf.rebalance(D0, RANKED, PRICES, SLOT)
    pf.rebalance(D1, RANKED[3:] + RANKED[:3], PRICES, SLOT)
    s = pf.summary(sessions_per_rebalance=63)
    assert s["buffer_width"] == 5
    assert s["rebalances"] == 2
    assert s["annualised_turnover"] is not None
