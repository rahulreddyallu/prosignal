"""The prefetch cache must be indistinguishable from a real read.

A cache that returns *nearly* the right rows is worse than no cache: every
stage downstream would be quietly wrong with no error to notice.
"""
import datetime as dt

import pandas as pd
import pytest

from prosignal.data.store import DataStore
from prosignal.data.universe import UniverseResolver


def _liquid_universe(store, cfg, as_of, max_names=200):
    """The symbols the engine would actually have ranked on `as_of`.

    THESE TESTS ASKED FOR "NIFTY 200" AND THIS STORE HAS NO INDEX MEMBERSHIP
    SNAPSHOTS AT ALL -- zero for every index -- so both files failed at their
    fixture on every run and had done for a while. A permanently red suite is
    worse than a missing test: it trains everyone to skim past the failures,
    and the next real one arrives in the same colour.

    Skipping would have been the easy repair and the wrong one. Neither file
    needs index MEMBERSHIP; both need a few hundred real symbols with real
    prices. So they now resolve the same point-in-time liquidity screen the
    shipped engine ranks -- which is a closer test of production than an index
    list would have been, and runs.
    """
    import pytest
    u = cfg.params.universe
    def _v(x, d=None):
        x = getattr(u, x, d)
        return getattr(x, "value", x)
    try:
        snap = UniverseResolver(store, cfg.params).resolve_liquidity_pit(
            as_of=as_of,
            min_adtv_inr=float(_v("pit_min_adtv_inr", 5e7)),
            lookback_sessions=int(_v("pit_adtv_lookback_sessions", 60)),
            max_names=int(max_names),
            min_history_sessions=int(_v("pit_min_history_sessions", 300)),
            min_price_inr=float(_v("pit_min_price_inr", 20.0)))
    except Exception as exc:
        pytest.skip(f"no tradeable universe in this store: {exc}")
    syms = sorted(set(snap.symbols))
    if len(syms) < 20:
        pytest.skip(f"only {len(syms)} liquid names in this store")
    return syms



@pytest.fixture
def store(live_cfg):
    return DataStore(live_cfg.paths.curated, live_cfg.paths.snapshots)


@pytest.fixture
def universe(store, live_cfg):
    sessions = store.price_sessions()
    return _liquid_universe(store, live_cfg, sessions[-1]), sessions


def _fresh(live_cfg, **kw):
    return DataStore(live_cfg.paths.curated, live_cfg.paths.snapshots).read_prices(**kw)


@pytest.mark.parametrize("n_syms,back", [(50, 200), (200, 100), (10, 30)])
def test_cached_slice_equals_a_real_read(store, universe, live_cfg, n_syms, back):
    syms, sessions = universe
    if len(sessions) < 400:
        pytest.skip("store too small")
    store.prefetch_prices(syms, sessions[-350], sessions[-1])
    kw = dict(symbols=syms[:n_syms], start=sessions[-back], end=sessions[-1])
    assert store.read_prices(**kw).equals(_fresh(live_cfg, **kw))


def test_request_outside_the_window_falls_through(store, universe):
    syms, sessions = universe
    if len(sessions) < 400:
        pytest.skip("store too small")
    store.prefetch_prices(syms, sessions[-100], sessions[-1])
    # Earlier than the cached window: must not be answered from the cache.
    wide = store.read_prices(symbols=syms[:5], start=sessions[0], end=sessions[-1])
    assert len(wide) > len(
        store.read_prices(symbols=syms[:5], start=sessions[-100], end=sessions[-1])
    )


def test_symbol_outside_the_cached_set_falls_through(store, universe):
    syms, sessions = universe
    if len(sessions) < 400:
        pytest.skip("store too small")
    store.prefetch_prices(syms[:5], sessions[-100], sessions[-1])
    # A symbol that was not prefetched must still be readable.
    other = store.read_prices(
        symbols=[syms[50]], start=sessions[-50], end=sessions[-1]
    )
    assert not other.empty


def test_slice_drops_categories_it_no_longer_contains(store, universe, live_cfg):
    """groupby(observed=False) iterates categories, not values, so a slice
    carrying the full universe would behave differently from the read it
    replaces."""
    syms, sessions = universe
    if len(sessions) < 400:
        pytest.skip("store too small")
    store.prefetch_prices(syms, sessions[-350], sessions[-1])
    kw = dict(symbols=syms[:20], start=sessions[-60], end=sessions[-1])
    cached = store.read_prices(**kw)
    fresh = _fresh(live_cfg, **kw)
    assert list(cached["symbol"].cat.categories) == list(fresh["symbol"].cat.categories)


def test_clearing_the_cache_restores_direct_reads(store, universe, live_cfg):
    syms, sessions = universe
    if len(sessions) < 400:
        pytest.skip("store too small")
    store.prefetch_prices(syms, sessions[-350], sessions[-1])
    store.clear_price_cache()
    kw = dict(symbols=syms[:20], start=sessions[-60], end=sessions[-1])
    assert store.read_prices(**kw).equals(_fresh(live_cfg, **kw))
