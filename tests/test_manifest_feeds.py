"""The manifest must report what the store actually holds.

A required feed that reports itself healthy by construction is not a check.
Two did: index_membership and equity_master were stamped OK with age 0
whatever the store contained, so missing_required() and stale_required() could
never see them. delivery_data had the opposite problem -- it was honest about
its state but declared optional, while carrying the largest coefficient in the
fitted model and being neutral-filled when absent, so an outage rewrote a third
of the watchlist without raising a flag.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from prosignal.core.enums import FeedStatus


def _manifest(config, store, as_of):
    from prosignal.pipeline import _manifest_from_store
    from prosignal.data.universe import UniverseSnapshot

    universe = UniverseSnapshot(
        index_name="TEST", as_of=as_of, symbols=["AAA"],
        sector_map={}, company_names={}, source="test",
    )
    return _manifest_from_store(store, config, "runid", as_of, universe)


def test_delivery_is_required(tmp_store_config):
    config, store, as_of = tmp_store_config
    m = _manifest(config, store, as_of)
    assert m.feeds["delivery_data"].required is True, (
        "deliv_pct is the model's largest coefficient and is neutral-filled when "
        "absent; an optional delivery feed lets an outage score every name as "
        "average without failing anything"
    )


def test_equity_master_reports_missing_when_the_store_is_empty(tmp_store_config):
    config, store, as_of = tmp_store_config
    m = _manifest(config, store, as_of)
    # the fixture store has no equity_master table
    assert m.feeds["equity_master"].status is FeedStatus.MISSING
    assert "equity_master" in m.missing_required()


def test_membership_is_only_required_when_the_universe_uses_it(tmp_store_config):
    config, store, as_of = tmp_store_config
    m = _manifest(config, store, as_of)
    source = str(config.params.universe.source.value).lower()
    if source == "liquidity_pit":
        assert m.feeds["index_membership"].required is False, (
            "nothing reads membership under liquidity_pit; requiring it would "
            "halt runs over a feed the decision never touches"
        )
    else:
        assert m.feeds["index_membership"].required is True


def test_an_empty_delivery_panel_stops_the_model_rather_than_zeroing_it():
    """deliv_pct ranks neutral when absent, so a swallowed read failure scores
    every name as exactly average on the model's largest coefficient and prints
    a normal-looking watchlist. Per-name gaps stay neutral; an absent panel is
    a failure."""
    import datetime as dt

    import pandas as pd
    import pytest

    from prosignal.core.errors import PipelineError
    from prosignal.stages.stage4_core_score import _cross_sectional_model

    class _Store:
        def __init__(self, tmp):
            self.curated = tmp

        def price_sessions(self):
            return [dt.date(2026, 1, d) for d in range(1, 20)]

        def read_prices(self, **kw):
            return pd.DataFrame({
                "date": pd.to_datetime(["2026-01-02"]), "symbol": ["AAA"],
                "close": [10.0], "turnover": [1000.0],
            })

        def read_statements(self):
            return pd.DataFrame()

        def read_delivery(self, **kw):
            return pd.DataFrame()          # the feed is gone

    class _Cfg:
        class max_fundamental_age_days: value = 450
        class model_horizon_sessions: value = 63
        class model_ridge_alpha: value = 20000.0
        class model_max_train_sessions: value = 3000
        class model_min_train_rows: value = 600
        class model_refit_every_sessions: value = 21
        class min_name_factor_coverage: value = 0.60

    # The outer handler converts this to a reason string; what matters is that a
    # PipelineError is what travels, not a silently zeroed factor.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        scores, model, reason, feats, verdict = _cross_sectional_model(
            _Store(Path(tmp)), ["AAA"], dt.date(2026, 1, 19), _Cfg()
        )
        assert scores is None
        assert reason is not None
        assert "delivery" in reason.lower(), reason
