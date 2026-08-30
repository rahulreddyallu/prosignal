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
    # Distinct calls: same name on one day is deduplicated to one.
    perf = P.performance([_t(net=0.05), _t(ticker="BBB", net=-0.01)], None)
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
    perf = P.performance([_t(), _t(ticker="BBB", net=-0.02)], None)
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


# ===================================================================
# One call is one call
# ===================================================================

def test_several_runs_issuing_a_name_on_one_day_is_one_call():
    """SUZLON read as five calls returning +70.71% when it was three calls
    returning +43.17% -- a day can produce several ledger runs and each
    wrote its own outcome row for the same signal."""
    # Non-overlapping windows, so this tests deduplication of duplicate
    # run_ids and not the separate re-entry merge.
    rows = [_o("SUZLON", "2024-01-09", 0.1023, 12, exit_date="2024-01-25"),
            _o("SUZLON", "2024-01-09", 0.1023, 12, exit_date="2024-01-25"),
            _o("SUZLON", "2024-03-16", 0.1731, 10, exit_date="2024-03-30"),
            _o("SUZLON", "2024-03-16", 0.1731, 10, exit_date="2024-03-30"),
            _o("SUZLON", "2024-08-02", 0.1563, 6, exit_date="2024-08-13")]
    assert len(P.dedupe(rows)) == 3
    got = P.by_ticker(rows, None)[0]
    assert got["n"] == 3, "three separate positions, not one held throughout"
    assert abs(got["total_return"] - 0.4317) < 1e-6


def test_calls_held_at_the_same_time_are_counted_as_overlapping():
    """Two positions in one name held together are two slots. Adding their
    returns is a return on twice the capital."""
    a = {"entry_date": "2024-01-15", "exit_date": "2024-02-02"}
    b = {"entry_date": "2024-01-17", "exit_date": "2024-02-02"}
    c = {"entry_date": "2024-08-05", "exit_date": "2024-08-13"}
    assert P.overlaps([a, b, c]) == 1
    assert P.overlaps([a, c]) == 0


def test_a_name_reports_both_ways_of_totalling_it():
    """They answer different questions and the screen shows both."""
    # Two separate positions -- the second opens long after the first closed.
    rows = [_o("X", "2024-01-09", 0.10, 12, exit_date="2024-01-25"),
            _o("X", "2024-08-02", 0.15, 6, exit_date="2024-08-13")]
    d = P.calls_for("X", rows, None)
    assert d["n_calls"] == 2 and d["n_paid_off"] == 2
    assert abs(d["taking_every_call"] - 0.25) < 1e-9
    assert d["first_call"] == "2024-01-09"
    assert d["holding_since_first"] is None     # no store, so no claim


def test_a_name_with_no_closed_call_says_so_without_inventing_one():
    d = P.calls_for("NOPE", [], None)
    assert d["n_calls"] == 0 and d["taking_every_call"] is None


def test_the_benchmark_frame_is_built_once_per_request():
    """performance(), by_ticker() and equity_curve() each want the same
    benchmark and each rebuilt it from a 224,000-row table."""
    calls = {"n": 0}

    class _Counting(_Store):
        def read_indices(self):
            calls["n"] += 1
            return super().read_indices()

    st = _Counting(_indices())
    rows = [_t(), _t(ticker="BBB", net=-0.02)]
    P.performance(rows, st, benchmark="Nifty 200")
    P.by_ticker(rows, st, benchmark="Nifty 200")
    P.equity_curve(rows, st, benchmark="Nifty 200")
    assert calls["n"] == 1, f"read the index table {calls['n']} times"


def test_two_different_stores_never_share_a_cached_benchmark():
    """The cache was keyed on id(store). CPython reuses an id once the object
    behind it is collected, so a freshly built store could be handed the
    previous store's benchmark frame -- which is how a store with no name
    column started returning a benchmark it could not possibly have."""
    import pandas as pd

    good = _Store(_indices())
    assert P.performance([_t()], good, benchmark="Nifty 200")["benchmark_covered"] == 1
    del good

    # A store with no way to identify the benchmark must refuse, whatever
    # object ids have been recycled in between.
    for _ in range(50):
        bare = _Store(pd.DataFrame({"date": pd.date_range("2026-01-01", periods=30),
                                    "close": range(30)}))
        assert P.performance([_t()], bare)["benchmark_covered"] == 0


# ===================================================================
# A name re-signalled while held is one position
# ===================================================================

def _oc(ticker, entry, exit_, net, held=10):
    return {"ticker": ticker, "signal_date": entry, "entry_date": entry,
            "exit_date": exit_, "entry_price": 100.0,
            "exit_price": 100.0 * (1 + net), "sessions_held": held,
            "net_return": net, "gross_return": net, "exit_reason": "target_1"}


def test_a_name_resignalled_while_held_is_one_position():
    """Stage 6 enters at rank 8 and holds while the name stays inside rank
    16, so a name reappearing tomorrow is the position being maintained. On a
    live slate two of twenty-one open positions were re-entries of names
    already held, and 137 closed trades collapsed to 86."""
    rows = [_oc("X", "2024-01-15", "2024-02-02", 0.10),
            _oc("X", "2024-01-17", "2024-02-02", 0.17)]
    merged = P.merge_reentries(rows)
    assert len(merged) == 1
    assert merged[0]["entry_date"] == "2024-01-15", "the first entry opens it"
    assert merged[0]["merged_legs"] == 2


def test_separate_positions_in_one_name_are_kept_apart():
    """Re-entering after the first has closed IS a second position."""
    rows = [_oc("X", "2024-01-15", "2024-02-02", 0.10),
            _oc("X", "2024-08-05", "2024-08-13", 0.15)]
    assert len(P.merge_reentries(rows)) == 2


def test_the_merged_return_is_measured_not_summed():
    """Summing two overlapping legs is the double-count this removes."""
    rows = [_oc("X", "2024-01-15", "2024-02-02", 0.10),
            _oc("X", "2024-01-17", "2024-02-02", 0.17)]
    got = P.merge_reentries(rows)[0]
    assert abs(got["net_return"] - 0.10) < 1e-9, \
        "one entry, one exit -- not 0.27"


def test_every_reader_of_closed_trades_goes_through_the_merge():
    src = open(P.__file__, encoding="utf-8").read()
    for fn in ("def performance(", "def by_ticker(", "def equity_curve("):
        body = src[src.index(fn):]
        body = body[:body.index("\n\n\ndef ")] if "\n\n\ndef " in body else body
        assert "merge_reentries(" in body[:1200], fn


# ===================================================================
# Several configurations are several populations
# ===================================================================

def _c(cv, ticker, sig, net, entry, exit_):
    return {"config_version": cv, "ticker": ticker, "signal_date": sig,
            "entry_date": entry, "exit_date": exit_, "net_return": net,
            "gross_return": None if net is None else net + 0.01,
            "sessions_held": 10,
            "exit_reason": "book_exit", "composite_score": 0.9}


def test_each_configuration_is_summarised_on_its_own():
    """A real deployment held 128 closed calls across EIGHT configurations and
    the page reduced all of it to "not counted here". Pooling them would
    report a strategy nobody ran; hiding them reported nothing at all. They
    are populations, so they are summarised as populations."""
    rows = [_c("v1@a", "AAA", "2024-01-02", 0.05, "2024-01-03", "2024-01-20"),
            _c("v1@a", "BBB", "2024-02-02", -0.03, "2024-02-05", "2024-02-20"),
            _c("v1@b", "CCC", "2025-01-02", 0.11, "2025-01-03", "2025-01-20")]
    got = P.by_configuration(rows)
    assert len(got) == 2, "two configurations, two records"
    by = {g["config_version"]: g for g in got}
    assert by["v1@a"]["n"] == 2 and by["v1@b"]["n"] == 1
    assert abs(by["v1@a"]["avg_return"] - 0.01) < 1e-9
    assert abs(by["v1@b"]["avg_return"] - 0.11) < 1e-9


def test_nothing_is_summed_across_configurations():
    """The whole point. If these were pooled the mean would be one number."""
    rows = [_c("v1@a", "AAA", "2024-01-02", 1.0, "2024-01-03", "2024-01-20"),
            _c("v1@b", "BBB", "2025-01-02", -1.0, "2025-01-03", "2025-01-20")]
    means = {g["config_version"]: g["avg_return"] for g in P.by_configuration(rows)}
    assert means == {"v1@a": 1.0, "v1@b": -1.0}, (
        "a pooled mean of 0.0 would describe an engine that never existed"
    )


def test_configurations_are_ordered_by_how_recently_they_ran():
    """Recency is the only ordering that means anything across populations
    this different: the engine that ran last is the one most like the engine
    running now."""
    rows = [_c("old", "AAA", "2024-01-02", 0.01, "2024-01-03", "2024-01-20"),
            _c("new", "BBB", "2026-01-02", 0.01, "2026-01-03", "2026-01-20"),
            _c("mid", "CCC", "2025-01-02", 0.01, "2025-01-03", "2025-01-20")]
    assert [g["config_version"] for g in P.by_configuration(rows)] == \
        ["new", "mid", "old"]


def test_every_configuration_row_carries_its_own_count():
    """A mean over four trades is not the same kind of object as a mean over
    sixty-three, and a row that shows one without the other invites them to
    be read as though they were."""
    rows = [_c("v1@a", f"S{i}", "2024-01-02", 0.01, "2024-01-03", "2024-01-20")
            for i in range(5)]
    got = P.by_configuration(rows)[0]
    assert got["n"] == 5
    assert got["first_signal"] and got["last_signal"]


def test_an_unresolved_row_is_not_a_configuration_record():
    """Open positions have no net_return. Counting them would inflate n with
    trades that have not happened."""
    rows = [_c("v1@a", "AAA", "2024-01-02", None, "2024-01-03", None)]
    assert P.by_configuration(rows) == []
