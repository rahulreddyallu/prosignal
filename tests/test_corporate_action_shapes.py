"""The corporate-action check, against every feed shape that reaches it.

The check reads `ratio` to tell a split from a dividend. That column is
canonical in CORPORATE_ACTION_COLUMNS and the NSE ingest fills it -- but the
reference-CSV override ships `symbol,ex_date,action_type,ratio_from,ratio_to,
details` and no `ratio` at all. Reading it unguarded raised inside Stage 5's
per-stock loop, which has no handler, and took the entire run down.

The original check never touched `ratio`; it rejected on ANY action. So this is
a regression introduced by making the check type-aware, and it is tested here
against the shapes a real feed can actually produce rather than the one the
first test happened to build.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from prosignal.core.enums import CheckOutcome
from prosignal.stages.stage5_false_signal import _corporate_action

EX = dt.date(2026, 8, 20)
AS_OF = dt.date(2026, 8, 25)


@pytest.fixture(scope="module")
def cfg():
    from prosignal.config.loader import load_config
    return load_config().params.stage5_false_signal.corporate_action_distortion


def check(rows, cfg):
    return _corporate_action(None, pd.DataFrame(rows), "X", AS_OF, cfg)


# ------------------------------------------------------- the canonical feed
def test_a_dividend_does_not_reject(cfg):
    """Dividends carry price factor 1.0 and are not adjusted for, so nothing is
    rescaled. Rejecting on them made this a seasonal cull of high-yield names:
    81 of 85 symbols in the window on 2026-08-25 were ordinary dividends."""
    r = check([{"symbol": "X", "ex_date": EX, "action_type": "dividend",
                "ratio": 1.0}], cfg)
    assert r.outcome is CheckOutcome.PASS


def test_a_split_rejects(cfg):
    r = check([{"symbol": "X", "ex_date": EX, "action_type": "split",
                "ratio": 0.2}], cfg)
    assert r.outcome is CheckOutcome.HARD_REJECT


def test_one_rescaling_action_among_dividends_still_rejects(cfg):
    r = check([{"symbol": "X", "ex_date": EX, "action_type": "dividend", "ratio": 1.0},
               {"symbol": "X", "ex_date": EX, "action_type": "split", "ratio": 0.5}], cfg)
    assert r.outcome is CheckOutcome.HARD_REJECT


# -------------------------------------------- the reference-CSV feed shape
def test_a_feed_with_no_ratio_column_does_not_crash(cfg):
    """config/reference/corporate_actions.csv has ratio_from/ratio_to and no
    `ratio`. This raised AttributeError inside the per-stock loop."""
    r = check([{"symbol": "X", "ex_date": EX, "action_type": "dividend"}], cfg)
    assert r.outcome is CheckOutcome.PASS
    assert r.observed["classified_by"] == "action_type"


def test_without_a_ratio_the_kind_still_classifies(cfg):
    r = check([{"symbol": "X", "ex_date": EX, "action_type": "split"}], cfg)
    assert r.outcome is CheckOutcome.HARD_REJECT
    assert r.observed["classified_by"] == "action_type"


def test_an_unrecognised_kind_is_treated_as_rescaling(cfg):
    """An action we cannot classify is not one we have cleared."""
    r = check([{"symbol": "X", "ex_date": EX, "action_type": "scheme_of_arrangement"}],
              cfg)
    assert r.outcome is CheckOutcome.HARD_REJECT


def test_an_all_nan_ratio_falls_back_to_the_kind(cfg):
    r = check([{"symbol": "X", "ex_date": EX, "action_type": "dividend",
                "ratio": np.nan}], cfg)
    assert r.outcome is CheckOutcome.PASS
    assert r.observed["classified_by"] == "action_type"


def test_a_feed_with_no_action_type_uses_the_ratio(cfg):
    r = check([{"symbol": "X", "ex_date": EX, "ratio": 0.2}], cfg)
    assert r.outcome is CheckOutcome.HARD_REJECT
    assert r.observed["classified_by"] == "ratio"


def test_neither_column_is_not_testable_rather_than_a_pass(cfg):
    """NOT_TESTABLE is never upgraded to PASS -- this stage's contract. An
    action that cannot be classified has not been cleared."""
    r = check([{"symbol": "X", "ex_date": EX}], cfg)
    assert r.outcome is CheckOutcome.NOT_TESTABLE


# ------------------------------------------------------------- other paths
def test_an_empty_feed_is_not_testable(cfg):
    assert _corporate_action(None, pd.DataFrame(), "X", AS_OF, cfg).outcome \
        is CheckOutcome.NOT_TESTABLE


def test_a_symbol_with_no_actions_passes(cfg):
    r = check([{"symbol": "OTHER", "ex_date": EX, "action_type": "split",
                "ratio": 0.2}], cfg)
    assert r.outcome is CheckOutcome.PASS
    assert r.observed["actions_in_window"] == 0


def test_an_action_outside_the_window_passes(cfg):
    r = check([{"symbol": "X", "ex_date": dt.date(2025, 1, 1),
                "action_type": "split", "ratio": 0.2}], cfg)
    assert r.outcome is CheckOutcome.PASS


def test_an_unparseable_ex_date_does_not_crash(cfg):
    r = check([{"symbol": "X", "ex_date": "not-a-date", "action_type": "split",
                "ratio": 0.2}], cfg)
    assert r.outcome in (CheckOutcome.PASS, CheckOutcome.NOT_TESTABLE)
