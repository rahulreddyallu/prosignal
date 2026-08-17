"""Storage policy: cache limits, eviction, and the budget interlocks.

Context for why this file exists: the first full backfill filled the machine's
disk. The measured cause was that raw payloads were 7.4x larger than the
curated data they produced, and 70% of that was F&O bhavcopy files -- 1.3 MB
each, from which the engine extracts about 5 KB of aggregated open interest.
These tests pin the policies that stop it happening again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prosignal.data.providers.http import HttpClient


@pytest.fixture
def client(tmp_path) -> HttpClient:
    return HttpClient(
        cache_dir=tmp_path / "cache",
        user_agent="test-agent",
        max_payload_bytes_to_cache=1_000,
        max_cache_bytes=5_000,
    )


def _seed(client: HttpClient, url: str, size: int) -> Path:
    """Write a cache entry directly, bypassing the network."""
    from prosignal.data.providers.http import FetchResult

    result = FetchResult(url=url, content=b"x" * size, status_code=200)
    client._write_cache(result)
    blob, _ = client._cache_paths(url)
    return blob


# =============================================================================
# write policy
# =============================================================================


def test_small_payload_is_cached(client):
    blob = _seed(client, "https://host/small.csv", 500)
    assert blob.is_file()
    assert client.cache_size_bytes() > 0


def test_oversized_payload_is_not_cached(client):
    """A payload bigger than the limit is parsed and discarded, not stored."""
    blob = _seed(client, "https://host/huge.zip", 5_000)
    assert not blob.is_file()
    assert client.stats["cache_skipped_large"] == 1


def test_never_cache_flag_skips_write(client):
    from prosignal.data.providers.http import FetchResult

    result = FetchResult(url="https://host/fo.zip", content=b"x" * 100, status_code=200)
    client._write_cache(result, cacheable=False)
    blob, _ = client._cache_paths(result.url)
    assert not blob.is_file()
    assert client.stats["cache_skipped_policy"] == 1


def test_cached_entry_is_served_back(client):
    _seed(client, "https://host/a.csv", 400)
    hit = client._read_cache("https://host/a.csv", ttl_seconds=-1)
    assert hit is not None
    assert hit.from_cache
    assert len(hit.content) == 400


# =============================================================================
# eviction
# =============================================================================


def test_evict_lru_respects_the_budget(client):
    for i in range(20):
        _seed(client, f"https://host/f{i}.csv", 900)
    assert client.cache_size_bytes() > 5_000

    result = client.evict_lru()
    assert result["evicted"] > 0
    assert client.cache_size_bytes() <= 5_000


def test_evict_lru_removes_oldest_first(client):
    import os
    import time

    old = _seed(client, "https://host/old.csv", 900)
    time.sleep(0.01)
    new = _seed(client, "https://host/new.csv", 900)
    # Make the age difference unambiguous regardless of filesystem resolution.
    os.utime(old, (1, 1))

    # Each entry costs its blob plus a ~200-byte sidecar, so a 1,500-byte
    # budget has room for exactly one of the two.
    client.max_cache_bytes = 1_500
    client.evict_lru()
    assert not old.is_file()
    assert new.is_file()


def test_eviction_removes_blob_and_sidecar_together(client):
    """A half-evicted entry must never be servable as a cache hit."""
    for i in range(10):
        _seed(client, f"https://host/g{i}.csv", 900)
    client.evict_lru()
    for meta in client.cache_dir.rglob("*.meta.json"):
        blob = meta.with_name(meta.name.replace(".meta.json", ".bin"))
        assert blob.is_file(), "sidecar survived without its payload"


def test_evict_is_a_noop_when_under_budget(client):
    _seed(client, "https://host/tiny.csv", 100)
    result = client.evict_lru()
    assert result["evicted"] == 0


# =============================================================================
# policy sweep (retroactive)
# =============================================================================


def test_policy_sweep_removes_entries_a_tightened_policy_forbids(client):
    """Policy applies on write, so entries cached under a looser policy linger.

    They are served from cache on later runs and therefore never re-evaluated.
    This is exactly how 128 stale F&O payloads survived the policy change that
    was supposed to stop caching them.
    """
    keep = _seed(client, "https://host/ind_close_all_01012026.csv", 400)

    # Simulate an entry written before the size limit existed.
    client.max_payload_bytes_to_cache = None
    big = _seed(client, "https://host/BhavCopy_NSE_FO_x.zip", 4_000)
    assert big.is_file()

    client.max_payload_bytes_to_cache = 1_000
    result = client.purge_violating_policy([])
    assert result["removed"] == 1
    assert not big.is_file()
    assert keep.is_file()


def test_policy_sweep_removes_never_cache_feeds_by_url_marker(client):
    keep = _seed(client, "https://host/ind_close_all_01012026.csv", 400)
    fo = _seed(client, "https://host/BhavCopy_NSE_FO_0_0_0_20260814_F_0000.zip", 400)

    result = client.purge_violating_policy(["BhavCopy_NSE_FO"])
    assert result["removed"] == 1
    assert not fo.is_file()
    assert keep.is_file()


def test_policy_sweep_is_safe_on_an_empty_cache(tmp_path):
    fresh = HttpClient(cache_dir=tmp_path / "nope", user_agent="t")
    assert fresh.purge_violating_policy(["anything"])["removed"] == 0
    assert fresh.cache_size_bytes() == 0


# =============================================================================
# config wiring
# =============================================================================


def test_shipped_storage_config_is_coherent(cfg):
    s = cfg.params.storage
    assert s.raw_cache.max_mb <= s.max_total_mb
    assert s.halt_free_disk_mb < s.warn_free_disk_mb
    assert "fo_bhavcopy" in s.raw_cache.never_cache_feeds, (
        "the F&O bhavcopy is the largest payload the engine downloads and "
        "yields the least data per byte; it must stay on the never-cache list"
    )
    assert s.write_batch_sessions > 1, (
        "per-session writes make a backfill O(n^2) in disk I/O"
    )


def test_never_cache_markers_are_derived_from_config(cfg):
    """Renaming an endpoint must not leave the purge sweep matching nothing."""
    from prosignal.data.providers.nse_archives import NseArchivesProvider

    provider = NseArchivesProvider(
        client=None,
        cfg=cfg.params.providers.nse_archives,
        ttl_historical_s=0,
        ttl_current_s=0,
        never_cache_feeds=cfg.params.storage.raw_cache.never_cache_feeds,
    )
    markers = provider.never_cache_url_markers()
    assert markers, "never_cache_feeds is set but produced no URL markers"
    for marker in markers:
        assert marker in cfg.params.providers.nse_archives.fo_bhavcopy_path
