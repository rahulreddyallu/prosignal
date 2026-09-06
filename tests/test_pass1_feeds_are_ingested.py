"""The Pass-1 point-in-time feeds must be reachable from `data ingest`.

`nse_shareholding` and `nse_surveillance` were built in commit 2922c30 under
that pass's own rule -- "DATA, NOT MODELS. No factor reads any of this yet" --
with named consumers waiting downstream: free float so `costs.impact_model` can
scale participation by float rather than traded value, and the F&O list because
only 207 of the 750-name universe (27.6%) is shortable, which is the number the
next pass needs to size a short leg.

Neither was ever connected to the ingest path. Both tables therefore sat at
whatever their one-off backfill left, and `test_no_dead_modules` had been
failing on them since before the 2026-09 audit branch.

Deleting them was the alternative and it was wrong here: the feeds are wanted,
the store already holds their data, and the surveillance table only becomes
point-in-time by ACCUMULATING snapshots forward -- every run that does not
store one leaves a hole that can never be filled, because NSE publishes neither
file at a dated URL.

These tests pin the connection and the two properties that make it safe to run
on every ingest: it never fails a run, and it does not touch the network
offline.
"""

from __future__ import annotations

import datetime as dt
import inspect

import pytest

from prosignal.data import ingest as ing


def test_both_feeds_are_called_from_the_ingest_path():
    """The connection itself. `test_no_dead_modules` proves the module is
    imported; this proves the ingest actually runs it."""
    src = inspect.getsource(ing.DataIngestor)
    assert "self._refresh_shareholding(as_of, opts)" in src
    assert "self._refresh_surveillance(as_of, opts)" in src


def test_each_feed_imports_its_provider():
    assert "from .providers.nse_shareholding import NseShareholdingProvider" in \
        inspect.getsource(ing.DataIngestor._refresh_shareholding)
    assert "from .providers.nse_surveillance import NseSurveillanceProvider" in \
        inspect.getsource(ing.DataIngestor._refresh_surveillance)


@pytest.mark.parametrize("method", ["_refresh_shareholding",
                                    "_refresh_surveillance"])
def test_neither_feed_can_fail_a_run(method):
    """A reference feed that raises takes down price ingestion with it. Both
    providers already degrade internally; the call sites must too."""
    src = inspect.getsource(getattr(ing.DataIngestor, method))
    assert "except Exception" in src
    assert "log.warning" in src
    assert "raise" not in src


@pytest.mark.parametrize("method", ["_refresh_shareholding",
                                    "_refresh_surveillance"])
def test_neither_feed_touches_the_network_offline(method, monkeypatch):
    """`--offline` must mean offline. A feed that ignores it turns an offline
    reproduction into a live fetch, and the run is then not reproducible."""
    from prosignal.config.loader import get_config

    reached = []

    class _Tripwire:
        """Anything the feed touches is recorded and then refused."""

        def __getattr__(self, name):
            reached.append(name)
            raise AssertionError(f"offline run reached {name}")

    obj = ing.DataIngestor.__new__(ing.DataIngestor)
    obj.config = get_config()
    obj.http = _Tripwire()
    obj.store = _Tripwire()

    getattr(ing.DataIngestor, method)(
        obj, dt.date(2026, 9, 6), ing.IngestOptions(offline=True))
    assert not reached, f"offline run reached {reached}"


def test_the_shareholding_refresh_is_gated_on_a_quarter():
    """One request per symbol against a quarterly feed. Refreshing more often
    spends several hundred requests re-reading numbers that have not changed."""
    src = inspect.getsource(ing.DataIngestor._refresh_shareholding)
    assert "112" in src, "the staleness window is not a quarter plus the lag"
    assert "read_shareholding" in src


def test_the_surveillance_refresh_stores_a_dated_snapshot():
    """Membership accumulates FORWARD and cannot be reconstructed backwards, so
    the snapshot date must be passed rather than left to default."""
    src = inspect.getsource(ing.DataIngestor._refresh_surveillance)
    assert "snapshot_date=as_of" in src
    assert "write_security_list" in src and "write_fo_lots" in src


def test_the_store_still_holds_both_tables():
    """If a future cleanup deletes the writers, this says what was lost."""
    from prosignal.data.store import DataStore

    for name in ("write_shareholding", "read_shareholding",
                 "write_security_list", "read_security_list",
                 "write_fo_lots", "read_fo_lots"):
        assert hasattr(DataStore, name), f"DataStore lost {name}"
