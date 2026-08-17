"""Polite, cached, retrying HTTP client for NSE's static archives.

Three behaviours matter here and each was chosen after probing the live hosts:

1. **Host-aware politeness.** NSE's archive hosts are generous but not
   unlimited. Requests to the same host are spaced by
   ``providers.http.min_interval_seconds``.

2. **Immutable-file caching.** A bhavcopy for a past session never changes once
   published, so it is cached effectively forever. Files for *today* get a
   short TTL because NSE occasionally republishes them. This turns a 250-file
   history pull from a ten-minute job into a one-second one on re-runs, which
   matters enormously when iterating.

3. **404 is data, not an error.** A missing bhavcopy means "no session that
   day" (weekend, holiday, or not yet published). It is returned as ``None``,
   not raised, so the calendar-discovery logic can treat it as information.

The ``www.nseindia.com`` JSON API sits behind a bot shield that returns 403 to
many networks. :class:`NseJsonSession` implements the cookie warm-up it wants,
but every caller must treat its failure as soft -- the engine is designed so
that no *required* feed depends on it.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from ...core.errors import ProviderError
from ...core.logging import get_logger

__all__ = ["HttpClient", "FetchResult", "NseJsonSession"]

log = get_logger(__name__)

_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass
class FetchResult:
    """A successful fetch. ``from_cache`` tells the manifest whether we hit the network."""

    url: str
    content: bytes
    status_code: int
    from_cache: bool = False
    fetched_at: float = field(default_factory=time.time)
    content_type: str = ""

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def __len__(self) -> int:
        return len(self.content)


class HttpClient:
    """Shared session with disk cache, per-host throttling and retries."""

    def __init__(
        self,
        cache_dir: Path,
        user_agent: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.5,
        min_interval_seconds: float = 0.35,
        cache_enabled: bool = True,
        max_payload_bytes_to_cache: Optional[int] = None,
        max_cache_bytes: Optional[int] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.backoff_base = backoff_base_seconds
        self.min_interval = min_interval_seconds
        self.cache_enabled = cache_enabled
        #: Payloads larger than this are parsed and discarded rather than
        #: cached. Measured motivation: a 1.3 MB F&O zip yields 4.7 KB of
        #: aggregated open interest -- a ~280:1 ratio that is not worth disk
        #: to avoid re-downloading.
        self.max_payload_bytes_to_cache = max_payload_bytes_to_cache
        #: LRU ceiling for the whole cache directory.
        self.max_cache_bytes = max_cache_bytes
        self._last_request: Dict[str, float] = {}
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": " ".join(user_agent.split()),
                "Accept": "*/*",
                "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
        )
        self.stats = {
            "network": 0,
            "cache": 0,
            "miss_404": 0,
            "errors": 0,
            "cache_skipped_large": 0,
            "cache_skipped_policy": 0,
            "evicted": 0,
        }

    # -- cache ---------------------------------------------------------------
    def _cache_paths(self, url: str) -> "tuple[Path, Path]":
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        host = url.split("/")[2] if "://" in url else "misc"
        base = self.cache_dir / host / digest[:2]
        return base / f"{digest}.bin", base / f"{digest}.meta.json"

    def _read_cache(self, url: str, ttl_seconds: float) -> Optional[FetchResult]:
        if not self.cache_enabled:
            return None
        blob_path, meta_path = self._cache_paths(url)
        if not (blob_path.is_file() and meta_path.is_file()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        age = time.time() - float(meta.get("fetched_at", 0))
        if ttl_seconds >= 0 and age > ttl_seconds:
            return None
        try:
            content = blob_path.read_bytes()
        except OSError:
            return None
        self.stats["cache"] += 1
        return FetchResult(
            url=url,
            content=content,
            status_code=int(meta.get("status_code", 200)),
            from_cache=True,
            fetched_at=float(meta.get("fetched_at", 0)),
            content_type=str(meta.get("content_type", "")),
        )

    def _write_cache(self, result: FetchResult, cacheable: bool = True) -> None:
        if not self.cache_enabled:
            return
        if not cacheable:
            self.stats["cache_skipped_policy"] += 1
            return
        if (
            self.max_payload_bytes_to_cache is not None
            and len(result.content) > self.max_payload_bytes_to_cache
        ):
            self.stats["cache_skipped_large"] += 1
            log.debug(
                "payload too large to cache; parsed and discarded",
                extra={"url": result.url, "bytes": len(result.content)},
            )
            return
        blob_path, meta_path = self._cache_paths(result.url)
        try:
            blob_path.parent.mkdir(parents=True, exist_ok=True)
            blob_path.write_bytes(result.content)
            meta_path.write_text(
                json.dumps(
                    {
                        "url": result.url,
                        "status_code": result.status_code,
                        "fetched_at": result.fetched_at,
                        "content_type": result.content_type,
                        "bytes": len(result.content),
                    }
                ),
                encoding="utf-8",
            )
        except OSError as exc:  # cache failure must never break a run
            log.warning("cache write failed", extra={"url": result.url, "error": str(exc)})

    def purge_cache(self) -> int:
        """Delete every cached payload. Returns the number of files removed."""
        removed = 0
        if not self.cache_dir.is_dir():
            return 0
        for p in self.cache_dir.rglob("*"):
            if p.is_file():
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    pass
        return removed

    def purge_violating_policy(
        self, url_substrings: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """Remove cached entries that the CURRENT policy would not have written.

        Needed because policy is applied on write. Entries cached under an
        older, looser policy are still served on later runs and so are never
        re-evaluated -- they would sit there indefinitely. This sweeps them.
        """
        markers = [s for s in (url_substrings or []) if s]
        removed = 0
        freed = 0
        if not self.cache_dir.is_dir():
            return {"removed": 0, "freed_bytes": 0}

        for meta_path in list(self.cache_dir.rglob("*.meta.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            url = str(meta.get("url", ""))
            size = int(meta.get("bytes", 0))

            too_big = (
                self.max_payload_bytes_to_cache is not None
                and size > self.max_payload_bytes_to_cache
            )
            blocked = any(m in url for m in markers)
            if not (too_big or blocked):
                continue

            blob_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".bin"))
            try:
                if blob_path.is_file():
                    freed += blob_path.stat().st_size
                    blob_path.unlink()
                freed += meta_path.stat().st_size
                meta_path.unlink()
                removed += 1
            except OSError:  # pragma: no cover
                continue

        if removed:
            log.info(
                "purged cache entries violating current policy",
                extra={"removed": removed, "freed_mb": round(freed / 1e6, 1)},
            )
        return {"removed": removed, "freed_bytes": freed}

    def cache_size_bytes(self) -> int:
        if not self.cache_dir.is_dir():
            return 0
        return sum(p.stat().st_size for p in self.cache_dir.rglob("*") if p.is_file())

    def evict_lru(self, target_bytes: Optional[int] = None) -> Dict[str, int]:
        """Evict least-recently-used payloads until the cache fits its budget.

        Recency is read from the blob's mtime, which the OS updates on write.
        Eviction removes the blob and its sidecar together, so a half-evicted
        entry can never be served as a cache hit.

        Nothing durable is lost: the curated parquet store is the record, and
        the cache only ever saves a re-download.
        """
        limit = target_bytes if target_bytes is not None else self.max_cache_bytes
        if limit is None or not self.cache_dir.is_dir():
            return {"evicted": 0, "freed_bytes": 0, "size_bytes": self.cache_size_bytes()}

        blobs = [p for p in self.cache_dir.rglob("*.bin") if p.is_file()]
        total = self.cache_size_bytes()
        if total <= limit:
            return {"evicted": 0, "freed_bytes": 0, "size_bytes": total}

        blobs.sort(key=lambda p: p.stat().st_mtime)  # oldest first
        freed = 0
        evicted = 0
        for blob in blobs:
            if total - freed <= limit:
                break
            meta = blob.with_suffix(".meta.json")
            size = blob.stat().st_size
            meta_size = meta.stat().st_size if meta.is_file() else 0
            try:
                blob.unlink()
                if meta.is_file():
                    meta.unlink()
            except OSError:  # pragma: no cover
                continue
            freed += size + meta_size
            evicted += 1

        self.stats["evicted"] += evicted
        log.info(
            "cache evicted to budget",
            extra={"evicted": evicted, "freed_mb": round(freed / 1e6, 1)},
        )
        return {"evicted": evicted, "freed_bytes": freed, "size_bytes": total - freed}

    # -- throttle ------------------------------------------------------------
    def _throttle(self, url: str) -> None:
        if self.min_interval <= 0:
            return
        host = url.split("/")[2] if "://" in url else "misc"
        last = self._last_request.get(host)
        if last is not None:
            wait = self.min_interval - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
        self._last_request[host] = time.monotonic()

    # -- fetch ---------------------------------------------------------------
    def get(
        self,
        url: str,
        ttl_seconds: float = -1.0,
        headers: Optional[Dict[str, str]] = None,
        allow_404: bool = True,
        context: str = "",
        cacheable: bool = True,
    ) -> Optional[FetchResult]:
        """GET ``url``, returning ``None`` for a 404 when ``allow_404``.

        Parameters
        ----------
        ttl_seconds:
            Cache lifetime. ``-1`` means "never expire" (immutable historical
            files). ``0`` forces a network fetch.
        cacheable:
            ``False`` for parse-once feeds whose payload dwarfs the data
            extracted from it. The response is still returned in full; it is
            simply not written to disk.
        """
        cached = self._read_cache(url, ttl_seconds) if ttl_seconds != 0 else None
        if cached is not None:
            return cached

        last_error: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            self._throttle(url)
            try:
                resp = self._session.get(url, timeout=self.timeout, headers=headers)
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log.debug(
                    "http error, will retry",
                    extra={"url": url, "attempt": attempt, "error": last_error},
                )
            else:
                if resp.status_code == 200:
                    result = FetchResult(
                        url=url,
                        content=resp.content,
                        status_code=200,
                        content_type=resp.headers.get("Content-Type", ""),
                    )
                    self.stats["network"] += 1
                    self._write_cache(result, cacheable=cacheable)
                    return result

                if resp.status_code == 404:
                    self.stats["miss_404"] += 1
                    if allow_404:
                        # A missing archive file is information (no session /
                        # not yet published), not a failure.
                        return None
                    last_error = "HTTP 404"
                    break

                last_error = f"HTTP {resp.status_code}"
                if resp.status_code not in _RETRYABLE_STATUS:
                    break

            if attempt < self.max_retries:
                time.sleep(self.backoff_base ** (attempt + 1))

        self.stats["errors"] += 1
        raise ProviderError(
            provider=context or "http",
            message=f"GET failed after {self.max_retries + 1} attempt(s): {last_error}",
            url=url,
        )

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class NseJsonSession:
    """Best-effort client for ``www.nseindia.com``'s JSON API.

    NSE requires a browser-like cookie handshake, and even with it many
    networks (datacentres, some ISPs, non-Indian IPs) get a hard 403. Probing
    from the build machine returned 403 on the homepage while the archive hosts
    returned 200 -- which is exactly why no *required* feed in this engine
    depends on this class.

    Every method returns ``None`` on failure and records why in
    :attr:`last_error`, so the Stage 0 manifest can report the degradation
    honestly instead of the run dying.
    """

    def __init__(self, client: HttpClient, base: str, warmup_path: str = "/") -> None:
        self.client = client
        self.base = base.rstrip("/")
        self.warmup_path = warmup_path
        self._warm = False
        self.last_error: Optional[str] = None

    def _warmup(self) -> bool:
        if self._warm:
            return True
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
        }
        try:
            self.client.get(
                f"{self.base}{self.warmup_path}",
                ttl_seconds=0,
                headers=headers,
                allow_404=False,
                context="nse_json_api.warmup",
            )
        except ProviderError as exc:
            self.last_error = (
                f"cookie warm-up failed ({exc.context.get('url', '')}): {exc.message}. "
                "NSE's bot shield commonly blocks this host; archive-based feeds "
                "are unaffected."
            )
            log.info("NSE JSON API unavailable", extra={"reason": self.last_error})
            return False
        self._warm = True
        return True

    def get_json(self, path: str, ttl_seconds: float = 3600.0) -> Optional[Any]:
        if not self._warmup():
            return None
        url = f"{self.base}{path}"
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Referer": f"{self.base}/",
            "X-Requested-With": "XMLHttpRequest",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }
        try:
            res = self.client.get(
                url, ttl_seconds=ttl_seconds, headers=headers, context="nse_json_api"
            )
        except ProviderError as exc:
            self.last_error = exc.message
            return None
        if res is None:
            self.last_error = "HTTP 404"
            return None
        try:
            return json.loads(res.text)
        except ValueError:
            self.last_error = "response was not valid JSON (bot-shield HTML page?)"
            return None
