"""Measuring whether following the shortlist beat not following it.

The two failures this file exists to prevent both produce a NUMBER rather
than an error, which is why they need tests rather than a try/except.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal import performance as P


def _t(ticker="AAA", net=0.05, entry="2026-01-02", exit_="2026-03-02",
       reason="target_1", held=40):
    return {"ticker": ticker, "signal_date": "2026-01-01", "entry_date": entry,
            "exit_date": exit_, "sessions_held": held, "net_return": net,
            "gross_return": net + 0.01, "exit_reason": reason,
            "composite_score": 0.9}


class _Store:
    """A store holding several indices, which is what the real one holds."""
    def __init__(self, frame): self._f = frame
    def read_indices(self): return self._f


def _indices():
    dates = pd.date_range("2026-01-01", "2026-04-01", freq="D")
    rows = []
    for i, d in enumerate(dates):
        rows.append({"date": d, "index_name": "Nifty 200", "close": 100.0 + i * 0.1})
        # A second index on the SAME dates, at a wildly different level.
        rows.append({"date": d, "index_name": "Nifty Bank", "close": 45000.0 + i})
    return pd.DataFrame(rows)


def test_the_benchmark_is_one_index_not_whichever_row_came_first():
    """Failing to select on index_name divided one index's close by another's
    and reported a benchmark return of 537%. It shipped that way once."""
    perf = P.performance([_t()], _Store(_indices()), benchmark="Nifty 200")
    assert perf["benchmark_covered"] == 1
    # Nifty 200 moves ~0.1/day off a base of 100 -- single-digit percent.
    assert abs(perf["avg_benchmark_return"]) < 0.5, perf["avg_benchmark_return"]


def test_an_unknown_benchmark_refuses_rather_than_falling_back():
    perf = P.performance([_t()], _Store(_indices()), benchmark="Nifty Nonexistent")
    assert perf["benchmark_covered"] == 0
    assert "comparison_note" in perf
    assert "avg_excess" not in perf


def test_a_frame_without_a_name_column_is_not_a_benchmark():
    df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=30),
                       "close": np.linspace(100, 110, 30)})
    perf = P.performance([_t()], _Store(df))
    assert perf["benchmark_covered"] == 0


def test_returns_stand_alone_when_no_comparison_is_possible():
    perf = P.performance([_t(net=0.05), _t(net=-0.01)], None)
    assert perf["n"] == 2
    assert perf["win_rate"] == 0.5
    assert "comparison_note" in perf


def test_the_overlap_correction_is_applied_to_the_excess():
    """Trades held at the same time share most of their window. The naive t
    on that sample rejects a true null about a third of the time."""
    vals = np.array([0.02] * 10 + [-0.01] * 5, dtype=float)
    got = P._corrected_t(vals, horizon=63, step=21)
    assert got["t"] is not None and got["naive_t"] is not None
    assert abs(got["t"]) < abs(got["naive_t"]), "correction did not shrink t"
    assert got["effective_n"] < len(vals)


def test_no_t_is_offered_on_a_single_trade():
    assert P._corrected_t(np.array([0.05]), 63, 21)["t"] is None


def test_every_ratio_carries_its_count():
    perf = P.performance([_t(), _t(net=-0.02)], None)
    assert perf["n"] == 2
    assert "win_rate" in perf and "n" in perf


def test_nothing_resolved_is_said_plainly():
    perf = P.performance([], None)
    assert perf["n"] == 0
    assert "note" in perf


def test_the_curve_sums_rather_than_compounds():
    """The book holds several names at once. Compounding overlapping holds
    would draw a line the strategy never earned."""
    rows = [_t(ticker="A", net=0.10, exit_="2026-02-01"),
            _t(ticker="B", net=0.10, exit_="2026-03-01")]
    curve = P.equity_curve(rows, None)
    assert [round(c["cumulative"], 6) for c in curve] == [0.10, 0.20]  # not 0.21


def test_per_name_puts_the_losses_first():
    rows = [_t(ticker="WIN", net=0.20), _t(ticker="LOSS", net=-0.15)]
    assert [r["ticker"] for r in P.by_ticker(rows, None)] == ["LOSS", "WIN"]


# ===================================================================
# Cohort censoring -- the bias that nearly shipped
# ===================================================================

def _o(ticker, signal_date, net, held, reason="target_1", exit_date="2026-08-20"):
    return {"ticker": ticker, "signal_date": signal_date, "entry_date": signal_date,
            "exit_date": exit_date, "sessions_held": held, "net_return": net,
            "gross_return": net, "exit_reason": reason}


def test_partly_observed_signals_are_not_pooled_with_finished_ones():
    """A signal issued 63 sessions ago has had its whole window. One issued
    three days ago has not, and the only members of that cohort visible are
    the ones that finished fast -- which skews to winners, because target_1
    sits nearer than a 2.5-ATR stop.

    Measured on the real ledger: complete cohorts gave +1.31% at a 49.3% win
    rate; the incomplete ones gave +5.89% at 60.9% over a 1.79-session hold.
    Pooling them turned a corrected t of 0.41 into 8.47.
    """
    old = [_o("A", "2026-01-05", 0.01, 20), _o("B", "2026-01-06", -0.02, 25)]
    new = [_o("C", "2026-08-18", 0.09, 2), _o("D", "2026-08-19", 0.11, 1)]
    done, partial = P.split_cohorts(old + new, "2026-05-25")
    assert [x["ticker"] for x in done] == ["A", "B"]
    assert [x["ticker"] for x in partial] == ["C", "D"]

    headline = P.performance(done, None)
    assert headline["n"] == 2
    assert headline["avg_return"] < 0.01, "the fast winners must not be in here"


def test_without_a_cutoff_nothing_is_split():
    rows = [_o("A", "2026-01-05", 0.01, 20)]
    done, partial = P.split_cohorts(rows, None)
    assert len(done) == 1 and partial == []


def test_recent_activity_reports_but_flags_itself():
    rows = [_o("C", "2026-08-18", 0.09, 2), _o("D", "2026-08-19", -0.03, 1)]
    got = P.recent_activity(rows)
    assert got["n"] == 2
    assert got["win_rate"] == 0.5
    assert "kept out of the figures above" in got["note"]


def test_recent_activity_counts_one_close_once():
    """Several runs can issue the same name on the same day; that is one
    position closing, not three."""
    dup = [_o("C", "2026-08-18", 0.09, 2, exit_date="2026-08-20")] * 3
    assert P.recent_activity(dup)["n"] == 1


def test_open_positions_never_enter_the_realised_figures():
    """A mark is what a position happens to be worth today. Letting one into
    the win rate would let the record be improved by picking a good day."""
    src = (P.__file__)
    text = open(src, encoding="utf-8").read()
    body = text[text.index("def performance("):text.index("def by_ticker(")]
    assert "open_positions" not in body
    assert "unrealised" not in body
