"""Stage 0 -- raw data ingestion and the run manifest.

The only module that talks to the network. It pulls every feed the pipeline
needs, normalises it, persists it, and emits a
:class:`~prosignal.core.contracts.RawDataManifest` recording what was obtained,
from which source, how fresh it is, and what is missing.

Stage 1 gates on that manifest, so a feed that silently failed must remain
distinguishable from one that succeeded.

Three behaviours to know before changing anything here:

* The decision date resolves backwards to a real session. NSE publishes the
  bhavcopy after the close, so a request for a date with no data returns the
  last session that has data, and reports that it did.
* The trading calendar is discovered rather than assumed: a 404 on the daily
  index file means no session, which is how holidays are learned.
* Fallbacks are recorded. If Yahoo served a feed because NSE failed, the
  manifest carries ``fallback_used=True`` and the primary error.
"""

from __future__ import annotations

import datetime as dt
import shutil
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd

from ..config.loader import AppConfig
from ..core.calendar import TradingCalendar, is_probably_closed
from ..core.contracts import FeedRecord, RawDataManifest
from ..core.enums import FeedStatus, SourceName
from ..core.errors import DataError, ProviderError
from ..core.logging import get_logger
from .corporate_actions import merge_action_sources
from .providers.csv_import import CsvImportProvider
from .providers.http import HttpClient, NseJsonSession
from .providers.nse_archives import INDIA_VIX_NAME, NseArchivesProvider
from .providers.yfinance_provider import YFinanceProvider
from .store import DataStore
from .types import DATE, SYMBOL
from .universe import UniverseResolver, UniverseSnapshot

__all__ = ["DataIngestor", "IngestOptions", "IngestResult"]

log = get_logger(__name__)

_DAYS_PER_SESSION = 1.45  # ~250 sessions per 365 calendar days, with slack


@dataclass
class IngestOptions:
    """Knobs the CLI exposes; defaults come from parameters.yaml."""

    #: How many trailing sessions of price history to guarantee. ``None`` uses
    #: ``universe.min_history_sessions`` plus a warm-up buffer.
    history_sessions: Optional[int] = None
    #: Skip the network entirely and work from what is already stored.
    offline: bool = False
    #: Refresh reference data (constituents, equity master) even if recent.
    force_reference_refresh: bool = False
    #: Pull optional feeds (delivery, open interest).
    include_delivery: bool = True
    #: Pull the secondary price source for cross-checking. Slow; worth it.
    include_secondary_prices: bool = True
    #: Corporate actions / earnings are refreshed at most this often.
    reference_refresh_sessions: int = 5
    #: Cap on how many calendar days back to probe when backfilling. ``None``
    #: takes ``storage.max_backfill_calendar_days``.
    max_backfill_calendar_days: Optional[int] = None
    #: Re-pull sessions already present in the store. Needed after a provider
    #: fix, since the normal path skips any session whose index file is stored.
    #: Cheap in practice: historical payloads come straight from the HTTP cache.
    refetch_stored_sessions: bool = False


@dataclass
class IngestResult:
    manifest: RawDataManifest
    calendar: TradingCalendar
    universe: UniverseSnapshot
    store: DataStore
    sessions_fetched: int = 0
    http_stats: Dict[str, int] = None  # type: ignore[assignment]


class DataIngestor:
    """Pull every Stage 0 feed and describe the result."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        p = config.params
        config.paths.create_all()

        http_cfg = p.providers.http
        storage = p.storage
        self.http = HttpClient(
            cache_dir=config.paths.cache,
            user_agent=http_cfg.user_agent,
            timeout_seconds=http_cfg.timeout_seconds,
            max_retries=http_cfg.max_retries,
            backoff_base_seconds=http_cfg.backoff_base_seconds,
            min_interval_seconds=http_cfg.min_interval_seconds,
            cache_enabled=http_cfg.cache_enabled and storage.raw_cache.enabled,
            max_payload_bytes_to_cache=int(
                storage.raw_cache.max_payload_mb_to_cache * 1_000_000
            ),
            max_cache_bytes=int(storage.raw_cache.max_mb * 1_000_000),
        )
        self.store = DataStore(config.paths.curated, config.paths.snapshots)
        self.nse = NseArchivesProvider(
            client=self.http,
            cfg=p.providers.nse_archives,
            ttl_historical_s=http_cfg.cache_ttl_days_historical * 86400.0,
            ttl_current_s=http_cfg.cache_ttl_hours_current * 3600.0,
            never_cache_feeds=storage.raw_cache.never_cache_feeds,
        )
        self.yf = YFinanceProvider(cfg=p.providers.yfinance)
        self.csv = CsvImportProvider(cfg=p.providers.csv_import, project_root=config.paths.root)
        self.json_api = NseJsonSession(
            client=self.http,
            base=p.providers.nse_json_api.base,
            warmup_path=p.providers.nse_json_api.warmup_path,
        )
        self.universe_resolver = UniverseResolver(self.store, p.universe)
        self._feeds: Dict[str, FeedRecord] = {}

    # =====================================================================
    # public entry point
    # =====================================================================
    def run(
        self,
        requested_date: Optional[dt.date] = None,
        options: Optional[IngestOptions] = None,
        run_id: Optional[str] = None,
    ) -> IngestResult:
        opts = options or IngestOptions()
        run_id = run_id or uuid.uuid4().hex[:12]
        p = self.config.params
        self._feeds = {}

        # Fail before the first byte rather than after filling the volume.
        if not opts.offline:
            self._assert_disk_headroom(stage="preflight")

        # -- reference data first: it defines the universe we then fetch for --
        if not opts.offline:
            self._ingest_equity_master(opts)

        as_of = self.resolve_decision_date(requested_date, offline=opts.offline)
        log.info(
            "decision date resolved",
            extra={"requested": str(requested_date), "resolved": as_of.isoformat()},
        )

        if not opts.offline:
            self._ingest_index_membership(as_of, opts)

        history = opts.history_sessions or self._default_history_sessions()
        sessions_fetched = 0
        if not opts.offline:
            sessions_fetched = self._backfill_sessions(as_of, history, opts)

        calendar = self._build_calendar(as_of)

        membership_csv = self.csv.load_index_membership() if p.providers.csv_import.enabled else None
        universe = self.universe_resolver.resolve(
            index_name=p.universe.index_name.value,
            as_of=as_of,
            membership_csv=membership_csv,
            manual_exclusions=p.universe.manual_exclusions.value,
            pre_snapshot_policy=p.universe.pre_snapshot_policy.value,
        )
        log.info(
            "universe resolved",
            extra={
                "index": universe.index_name,
                "size": len(universe),
                "source": universe.source,
                "survivorship_risk": universe.survivorship_risk,
            },
        )

        if not opts.offline:
            self._ingest_secondary_prices(universe.symbols, as_of, calendar, opts)
            self._ingest_corporate_actions(universe.symbols, as_of, calendar, opts)
            self._ingest_earnings(universe.symbols, as_of, calendar, opts)
        self._ingest_csv_feeds(as_of, calendar)

        # Keep the raw cache inside its budget after every run, so the ceiling
        # holds without anyone having to remember to run a cleanup command.
        if p.storage.raw_cache.enabled:
            self.http.evict_lru()

        manifest = self._build_manifest(
            run_id=run_id, as_of=as_of, calendar=calendar, universe=universe
        )
        return IngestResult(
            manifest=manifest,
            calendar=calendar,
            universe=universe,
            store=self.store,
            sessions_fetched=sessions_fetched,
            http_stats=dict(self.http.stats),
        )

    # =====================================================================
    # decision date
    # =====================================================================
    def resolve_decision_date(
        self, requested: Optional[dt.date] = None, offline: bool = False
    ) -> dt.date:
        """Snap a requested date back to the most recent session with data.

        Never silently rolls *forward*: returning a date later than requested
        would hand the caller information from the future.
        """
        p = self.config.params
        today = dt.date.today()
        target = requested or today

        if target > today and not p.runtime.date_resolution.allow_future_dates:
            raise DataError(
                f"requested decision date {target} is in the future; "
                f"runtime.date_resolution.allow_future_dates is false"
            )

        known = self.store.known_sessions()
        known_on_or_before = [d for d in known if d <= target]
        if known_on_or_before:
            candidate = known_on_or_before[-1]
            # If the stored calendar already reaches the target, trust it.
            if candidate == target or offline:
                return candidate
        elif offline:
            if known:
                raise DataError(
                    f"offline mode: no stored session on or before {target}; "
                    f"stored range is {known[0]} .. {known[-1]}"
                )
            raise DataError(
                "offline mode: the store is empty. Run `prosignal data ingest` "
                "with network access first."
            )

        # Probe backwards for a published session.
        max_back = p.runtime.date_resolution.max_lookback_calendar_days
        for back in range(0, max_back + 1):
            day = target - dt.timedelta(days=back)
            if is_probably_closed(day):
                continue
            if self.nse.session_exists(day):
                return day

        if known_on_or_before:
            log.warning(
                "no freshly published session found; falling back to stored calendar",
                extra={"target": target.isoformat()},
            )
            return known_on_or_before[-1]

        raise DataError(
            f"No NSE session found within {max_back} calendar days before "
            f"{target}. Either the archive host is unreachable or the lookback "
            f"window (runtime.date_resolution.max_lookback_calendar_days) is "
            f"too short."
        )

    def _default_history_sessions(self) -> int:
        p = self.config.params
        return int(p.universe.min_history_sessions.value) + 30

    # =====================================================================
    # reference feeds
    # =====================================================================
    def _should_refresh(self, feed: str, calendar_last: Optional[dt.date], every: int) -> bool:
        state = self.store.feed_state(feed)
        last = state.get("last_timestamp")
        if not last or calendar_last is None:
            return True
        try:
            last_date = dt.date.fromisoformat(str(last)[:10])
        except ValueError:
            return True
        return (calendar_last - last_date).days >= every

    def _ingest_equity_master(self, opts: IngestOptions) -> None:
        state = self.store.feed_state("equity_master")
        already = self.store.read_equity_master()
        if not opts.force_reference_refresh and not already.empty and state.get("last_timestamp"):
            try:
                last = dt.date.fromisoformat(str(state["last_timestamp"])[:10])
                if (dt.date.today() - last).days < 7:
                    self._record_feed(
                        "equity_master",
                        FeedStatus.OK,
                        SourceName.NSE_ARCHIVES,
                        last_timestamp=last,
                        row_count=len(already),
                        symbols_covered=int(already[SYMBOL].nunique()),
                        notes=["served from store (refreshed weekly)"],
                    )
                    return
            except ValueError:
                pass
        try:
            master = self.nse.fetch_equity_master()
        except ProviderError as exc:
            self._record_feed(
                "equity_master",
                FeedStatus.MISSING if already.empty else FeedStatus.STALE,
                None,
                notes=[f"fetch failed: {exc.message}"],
                row_count=len(already),
            )
            return
        self.store.write_equity_master(master)
        self.store.update_feed_state(
            "equity_master", dt.date.today(), SourceName.NSE_ARCHIVES.value, len(master)
        )
        self._record_feed(
            "equity_master",
            FeedStatus.OK,
            SourceName.NSE_ARCHIVES,
            last_timestamp=dt.date.today(),
            row_count=len(master),
            symbols_covered=int(master[SYMBOL].nunique()),
        )

    def _ingest_index_membership(self, as_of: dt.date, opts: IngestOptions) -> None:
        index_name = self.config.params.universe.index_name.value
        existing = self.store.universe_snapshot_dates(index_name)
        if existing and existing[-1] == as_of and not opts.force_reference_refresh:
            frame = self.store.read_universe_snapshot(index_name, as_of)
            self._record_feed(
                "index_membership",
                FeedStatus.OK,
                SourceName.NSE_ARCHIVES,
                last_timestamp=as_of,
                row_count=len(frame) if frame is not None else 0,
                notes=["snapshot for this session already stored"],
            )
            return
        try:
            constituents = self.nse.fetch_index_constituents(index_name)
        except ProviderError as exc:
            status = FeedStatus.STALE if existing else FeedStatus.MISSING
            self._record_feed(
                "index_membership",
                status,
                None,
                last_timestamp=existing[-1] if existing else None,
                notes=[f"constituent fetch failed: {exc.message}"],
            )
            return
        self.universe_resolver.snapshot_current(index_name, as_of, constituents)
        self.store.update_feed_state(
            "index_membership", as_of, SourceName.NSE_ARCHIVES.value, len(constituents)
        )
        self._record_feed(
            "index_membership",
            FeedStatus.OK,
            SourceName.NSE_ARCHIVES,
            last_timestamp=as_of,
            row_count=len(constituents),
            symbols_covered=len(constituents),
            notes=[f"dated snapshot written for {index_name}"],
        )

    # =====================================================================
    # session backfill
    # =====================================================================
    def _sessions_to_fetch(
        self, as_of: dt.date, history_sessions: int, opts: IngestOptions
    ) -> List[dt.date]:
        """Which calendar days still need pulling, newest first."""
        have_index: set = set() if opts.refetch_stored_sessions else set(self.store.known_sessions())
        cap = opts.max_backfill_calendar_days
        if cap is None:
            cap = int(self.config.params.storage.max_backfill_calendar_days)
        needed = int(history_sessions * _DAYS_PER_SESSION) + 20
        span_days = min(needed, cap)
        if needed > cap:
            # Silently returning fewer sessions than asked for is how a request
            # for ten years quietly becomes five, with every downstream sample
            # size wrong and nothing to show for it.
            log.warning(
                "backfill span capped; fewer sessions will be fetched than requested",
                extra={
                    "requested_sessions": history_sessions,
                    "days_needed": needed,
                    "cap_days": cap,
                    "reaches": str(as_of - dt.timedelta(days=cap)),
                    "raise": "storage.max_backfill_calendar_days",
                },
            )
        wanted: List[dt.date] = []
        day = as_of
        collected = 0
        while collected < history_sessions and (as_of - day).days <= span_days:
            if not is_probably_closed(day):
                if day in have_index:
                    collected += 1
                else:
                    wanted.append(day)
                    collected += 1  # optimistic; a 404 just costs one probe
            day -= dt.timedelta(days=1)
        return wanted

    def _backfill_sessions(
        self, as_of: dt.date, history_sessions: int, opts: IngestOptions
    ) -> int:
        candidates = self._sessions_to_fetch(as_of, history_sessions, opts)
        if not candidates:
            log.info("price history already complete", extra={"as_of": as_of.isoformat()})
            self._record_stored_price_feeds(as_of)
            return 0

        log.info(
            "backfilling sessions",
            extra={"count": len(candidates), "from": str(candidates[-1]), "to": str(candidates[0])},
        )

        fetched = 0
        index_errors: List[str] = []
        price_errors: List[str] = []

        # Sessions are buffered and flushed in batches. Writing one session at a
        # time forces a full read-modify-write of the year's parquet on every
        # iteration, which makes a backfill O(n^2) in disk I/O -- for 330
        # sessions against a growing 18 MB file that is gigabytes of pointless
        # rewriting, and it was the main reason the first full backfill crawled.
        batch_size = max(int(self.config.params.storage.write_batch_sessions), 1)
        buffers: Dict[str, List[pd.DataFrame]] = {
            "prices": [],
            "indices": [],
            "delivery": [],
            }

        def flush() -> None:
            if buffers["indices"]:
                self.store.write_indices(pd.concat(buffers["indices"], ignore_index=True))
            if buffers["prices"]:
                self.store.write_prices(pd.concat(buffers["prices"], ignore_index=True))
            if buffers["delivery"]:
                self.store.write_delivery(pd.concat(buffers["delivery"], ignore_index=True))
            for key in buffers:
                buffers[key] = []

        pending = 0
        try:
            for day in candidates:
                if pending and pending % batch_size == 0:
                    flush()
                    self._assert_disk_headroom(stage="backfill")

                index_frame = None
                try:
                    index_frame = self.nse.fetch_index_close_all(day)
                except ProviderError as exc:
                    index_errors.append(f"{day}: {exc.message}")
                if index_frame is None or index_frame.empty:
                    continue  # not a session

                buffers["indices"].append(index_frame)
                pending += 1

                try:
                    prices = self.nse.fetch_bhavcopy(day)
                except ProviderError as exc:
                    price_errors.append(f"{day}: {exc.message}")
                    prices = None
                if prices is not None and not prices.empty:
                    buffers["prices"].append(prices)
                    fetched += 1

                if opts.include_delivery:
                    try:
                        delivery = self.nse.fetch_delivery(day)
                        if delivery is not None and not delivery.empty:
                            buffers["delivery"].append(delivery)
                    except ProviderError as exc:
                        log.debug(
                            "delivery fetch failed",
                            extra={"day": str(day), "error": exc.message},
                        )

        finally:
            # Flush whatever was collected even if the loop aborted, so an
            # interrupted backfill resumes from real progress instead of
            # discarding a batch's worth of downloads.
            flush()

        self._record_stored_price_feeds(as_of, index_errors + price_errors)
        return fetched

    # =====================================================================
    # storage guards
    # =====================================================================
    def _free_disk_mb(self) -> float:
        usage = shutil.disk_usage(str(self.config.paths.data))
        return usage.free / 1_000_000

    def _data_dir_mb(self) -> float:
        root = self.config.paths.data
        if not root.is_dir():
            return 0.0
        return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 1_000_000

    def _assert_disk_headroom(self, stage: str) -> None:
        """Refuse to keep writing when the volume is nearly full.

        A data pipeline that fills the disk does not just fail itself -- it
        takes the rest of the machine with it, and the recovery is manual. It
        is strictly better to stop early with a clear message.
        """
        storage = self.config.params.storage
        free_mb = self._free_disk_mb()
        if free_mb < storage.halt_free_disk_mb:
            raise DataError(
                f"refusing to continue ({stage}): only {free_mb:,.0f} MB free on "
                f"the data volume, below the "
                f"storage.halt_free_disk_mb floor of {storage.halt_free_disk_mb:,.0f} MB. "
                f"Free space, or run `prosignal data gc`.",
                free_mb=round(free_mb),
                floor_mb=storage.halt_free_disk_mb,
            )
        if free_mb < storage.warn_free_disk_mb:
            log.warning(
                "low free disk", extra={"free_mb": round(free_mb), "stage": stage}
            )

        used_mb = self._data_dir_mb()
        if used_mb > storage.max_total_mb:
            evicted = self.http.evict_lru()
            used_mb = self._data_dir_mb()
            log.warning(
                "storage budget exceeded; evicted raw cache",
                extra={
                    "used_mb": round(used_mb),
                    "budget_mb": storage.max_total_mb,
                    "freed_mb": round(evicted["freed_bytes"] / 1e6, 1),
                },
            )
            if used_mb > storage.max_total_mb:
                raise DataError(
                    f"data/ is {used_mb:,.0f} MB, over the "
                    f"storage.max_total_mb budget of {storage.max_total_mb:,.0f} MB, "
                    f"and evicting the raw cache was not enough. The curated "
                    f"store itself is too large -- reduce the history you are "
                    f"keeping or raise the budget deliberately.",
                    used_mb=round(used_mb),
                    budget_mb=storage.max_total_mb,
                )

    def _record_stored_price_feeds(
        self, as_of: dt.date, errors: Optional[List[str]] = None
    ) -> None:
        notes = list(errors or [])[:5]

        price_last = self.store.prices.max_date()
        prices_today = self.store.read_prices(start=as_of, end=as_of)
        self._record_feed(
            "equity_ohlcv",
            FeedStatus.OK if price_last else FeedStatus.MISSING,
            SourceName.NSE_ARCHIVES,
            last_timestamp=price_last,
            row_count=len(prices_today),
            symbols_covered=int(prices_today[SYMBOL].nunique()) if not prices_today.empty else 0,
            notes=notes,
        )
        self.store.update_feed_state(
            "equity_ohlcv", price_last, SourceName.NSE_ARCHIVES.value, len(prices_today)
        )

        index_last = self.store.indices.max_date()
        self._record_feed(
            "index_ohlcv",
            FeedStatus.OK if index_last else FeedStatus.MISSING,
            SourceName.NSE_ARCHIVES,
            last_timestamp=index_last,
            row_count=len(self.store.available_index_names()),
        )
        self.store.update_feed_state("index_ohlcv", index_last, SourceName.NSE_ARCHIVES.value)

        vix = self.store.index_series(INDIA_VIX_NAME)
        vix_last = vix.index.max().date() if len(vix) else None
        self._record_feed(
            "india_vix",
            FeedStatus.OK if vix_last else FeedStatus.MISSING,
            SourceName.NSE_ARCHIVES,
            last_timestamp=vix_last,
            row_count=len(vix),
            notes=[] if vix_last else ["India VIX absent from ind_close_all files"],
        )
        self.store.update_feed_state("india_vix", vix_last, SourceName.NSE_ARCHIVES.value)

        deliv_last = self.store.delivery.max_date()
        deliv_today = self.store.read_delivery(start=as_of, end=as_of)
        self._record_feed(
            "delivery_data",
            FeedStatus.OK if deliv_last else FeedStatus.MISSING,
            SourceName.NSE_ARCHIVES,
            last_timestamp=deliv_last,
            row_count=len(deliv_today),
            symbols_covered=int(deliv_today[SYMBOL].nunique()) if not deliv_today.empty else 0,
            notes=[]
            if deliv_last
            else [
                "delivery percentage unavailable; the optional Stage 6 delivery "
                "confirmation will report NOT_TESTABLE"
            ],
        )


    # =====================================================================
    # secondary price source (cross-check)
    # =====================================================================
    def _ingest_secondary_prices(
        self,
        symbols: Sequence[str],
        as_of: dt.date,
        calendar: TradingCalendar,
        opts: IngestOptions,
    ) -> None:
        if not opts.include_secondary_prices or not self.config.params.providers.yfinance.enabled:
            self._record_feed(
                "equity_ohlcv_secondary",
                FeedStatus.MISSING,
                None,
                notes=["secondary price source disabled"],
            )
            return
        # Only a short recent window is needed: the cross-source agreement check
        # compares the latest close, not the whole history.
        start = calendar.previous_session(as_of, 5) or (as_of - dt.timedelta(days=10))
        try:
            frame = self.yf.fetch_ohlcv(symbols, start, as_of)
        except ProviderError as exc:
            self._record_feed(
                "equity_ohlcv_secondary", FeedStatus.MISSING, None, notes=[exc.message]
            )
            return
        if frame.empty:
            self._record_feed(
                "equity_ohlcv_secondary",
                FeedStatus.MISSING,
                SourceName.YFINANCE,
                notes=[self.yf.last_error or "yfinance returned no rows"],
            )
            return
        frame["source"] = SourceName.YFINANCE.value
        self.store.write_table(
            "prices_secondary", frame[[DATE, SYMBOL, "open", "high", "low", "close", "volume", "source"]],
            [SYMBOL, DATE],
        )
        last = pd.to_datetime(frame[DATE]).max().date()
        self._record_feed(
            "equity_ohlcv_secondary",
            FeedStatus.OK,
            SourceName.YFINANCE,
            last_timestamp=last,
            row_count=len(frame),
            symbols_covered=int(frame[SYMBOL].nunique()),
            notes=["used only for the Stage 1 cross-source agreement check"],
        )


    def _refresh_statements(self, symbols) -> int:
        """Income statement, balance sheet and cash flow per symbol.

        The NSE Ind-AS feed carries true filing dates but stops at the December
        2024 quarter and covers 186 names, and it is an income statement only --
        no equity, debt or cash flow, so return on equity, leverage and accruals
        are not derivable from it. This feed reaches the current quarter across
        the whole universe and carries the balance sheet, at the cost of period
        end without a filing date; the factor layer compensates by deriving
        availability from the SEBI LODR deadline.
        """
        if not self.config.params.providers.yfinance.enabled:
            return 0
        try:
            frame = self.yf.fetch_statements(symbols)
        except Exception as exc:
            log.warning("statement fetch failed", extra={"error": str(exc)})
            return 0
        if frame is None or frame.empty:
            return 0
        written = self.store.write_statements(frame)
        log.info("statements refreshed",
                 extra={"rows": written, "symbols": int(frame["symbol"].nunique())})
        return written

    def _refresh_sector_map(self) -> int:
        """Pool sectors from every configured index constituent file.

        NIFTY 500 subsumes the midcap and smallcap lists, so the union is about
        500 names. The point-in-time universe is wider than that and is never
        filtered to this list -- intersecting it would reintroduce exactly the
        survivorship bias the liquidity screen exists to remove. Names outside
        it simply have no sector, and Stage 8 says so.
        """
        files = dict(self.config.params.providers.nse_archives.index_constituent_files)
        frames = []
        for index_name in files:
            try:
                frame = self.nse.fetch_index_constituents(index_name)
            except Exception as exc:
                log.warning("sector source unavailable",
                            extra={"index": index_name, "error": str(exc)})
                continue
            if frame is not None and not frame.empty and "sector" in frame.columns:
                frames.append(frame[["symbol", "sector"]])
        if not frames:
            log.warning("no sector sources reachable; Stage 8 sector cap will report unknown")
            return 0
        pooled = (
            pd.concat(frames, ignore_index=True)
            .dropna(subset=["symbol", "sector"])
            .drop_duplicates(subset=["symbol"], keep="first")
            .reset_index(drop=True)
        )
        written = self.store.write_sector_map(pooled)
        log.info("sector map refreshed",
                 extra={"symbols": written, "sectors": int(pooled["sector"].nunique())})
        return written

    def _refresh_nse_fundamentals(self, as_of: dt.date, opts: "IngestOptions") -> None:
        """Pull quarterly results from NSE when the stored set has gone stale.

        This ran only from `prosignal data fundamentals`, so a normal ingest
        never refreshed it and the store sat at a filing date 525 days old while
        Stage 4 scored on it. The value and quality factors are the only inputs
        that are not derived from price and volume, so letting them decay
        removes the engine's only independent evidence.

        The per-symbol endpoint is slow and sits behind a bot shield, so this is
        rate-limited by the shared HTTP client, skipped when the store is fresh,
        and never fatal: a failure leaves the existing rows in place and the
        staleness guard in Stage 4 drops the factors as it already does.
        """
        if opts.offline:
            return
        stored = self.store.read_fundamentals()
        max_age = int(self.config.params.stage4_core_score.max_fundamental_age_days)
        if not stored.empty:
            newest = pd.to_datetime(stored["filing_date"], errors="coerce").max()
            if pd.notna(newest) and (pd.Timestamp(as_of) - newest).days <= max_age // 2:
                return  # still well inside tolerance; nothing to do

        try:
            from .providers.nse_fundamentals import NseFundamentalsProvider
            from .providers.http import NseJsonSession

            p = self.config.params.providers
            session = NseJsonSession(
                client=self.http, base=p.nse_json_api.base,
                warmup_path=p.nse_json_api.warmup_path,
            )
            provider = NseFundamentalsProvider(
                session=session, client=self.http,
                max_quarters=int(self.config.params.stage4_core_score.fundamental_quarters),
            )
            index = str(self.config.params.universe.index_name.value)
            dates = self.store.universe_snapshot_dates(index)
            if not dates:
                return
            symbols = self.store.read_universe_snapshot(index, dates[-1])[SYMBOL].tolist()
            frame = provider.fetch_universe(symbols)
            if frame is not None and not frame.empty:
                written = self.store.write_fundamentals(frame)
                log.info("fundamentals refreshed", extra={"rows": written,
                                                          "symbols": int(frame[SYMBOL].nunique())})
        except Exception as exc:
            log.warning("fundamentals refresh skipped", extra={"error": str(exc)})

    # =====================================================================
    # corporate actions & earnings
    # =====================================================================
    def _ingest_corporate_actions(
        self,
        symbols: Sequence[str],
        as_of: dt.date,
        calendar: TradingCalendar,
        opts: IngestOptions,
    ) -> None:
        csv_actions = self.csv.load_corporate_actions()
        need_refresh = opts.force_reference_refresh or self._should_refresh(
            "corporate_actions", as_of, opts.reference_refresh_sessions
        )

        yf_actions = pd.DataFrame()
        if need_refresh and self.config.params.providers.yfinance.enabled:
            try:
                yf_actions = self.yf.fetch_corporate_actions(symbols)
            except ProviderError as exc:
                log.warning("corporate action fetch failed", extra={"error": exc.message})

        # NSE is the issuer of record and the only source that reports each
        # action separately, so a date carrying both a split and a bonus keeps
        # both. yfinance stores one ratio per date and silently drops the
        # second, which is what left a residual -80% print in the adjusted
        # series for compound events.
        nse_actions = pd.DataFrame()
        if need_refresh:
            try:
                nse_actions = self.nse.fetch_corporate_actions(
                    as_of - dt.timedelta(days=3650), as_of
                )
            except Exception as exc:
                log.warning("NSE corporate action fetch failed",
                            extra={"error": str(exc)})

        # Order is precedence: yfinance is the fallback, NSE overrides it, and a
        # hand-curated CSV overrides both.
        merged = merge_action_sources(yf_actions, nse_actions, csv_actions)
        if not merged.empty:
            self.store.write_corporate_actions(merged)
            self.store.update_feed_state(
                "corporate_actions", as_of, "nse+yfinance+csv", len(merged)
            )

        stored = self.store.read_corporate_actions()
        last = (
            pd.to_datetime(stored["ex_date"]).max().date()
            if not stored.empty and stored["ex_date"].notna().any()
            else None
        )
        notes: List[str] = []
        if yf_actions.empty and csv_actions.empty:
            notes.append(
                "no corporate-action source available; the unexplained-jump "
                "detector is the only protection against an unadjusted split"
            )
        self._record_feed(
            "corporate_actions",
            FeedStatus.OK if not stored.empty else FeedStatus.MISSING,
            SourceName.YFINANCE if not yf_actions.empty else SourceName.CSV_IMPORT,
            last_timestamp=last,
            row_count=len(stored),
            notes=notes,
        )

    def _ingest_earnings(
        self,
        symbols: Sequence[str],
        as_of: dt.date,
        calendar: TradingCalendar,
        opts: IngestOptions,
    ) -> None:
        csv_cal = self.csv.load_earnings_calendar()
        need_refresh = opts.force_reference_refresh or self._should_refresh(
            "earnings_calendar", as_of, opts.reference_refresh_sessions
        )

        yf_cal = pd.DataFrame()
        if need_refresh and self.config.params.providers.yfinance.enabled:
            try:
                yf_cal = self.yf.fetch_earnings_dates(symbols)
            except ProviderError as exc:
                log.warning("earnings fetch failed", extra={"error": exc.message})

        # Company-filed board-meeting dates. yfinance projects the next print
        # from past quarters, which is an estimate; these are confirmed, so the
        # Stage 5 earnings check has something it can actually test against.
        nse_cal = pd.DataFrame()
        if need_refresh:
            try:
                nse_cal = self.nse.fetch_board_meetings(
                    as_of - dt.timedelta(days=180), as_of + dt.timedelta(days=180)
                )
            except Exception as exc:
                log.warning("board-meeting fetch failed", extra={"error": str(exc)})

        # NSE last so its confirmed rows win the de-duplication below.
        frames = [f for f in (yf_cal, csv_cal, nse_cal) if f is not None and not f.empty]
        if frames:
            # The stored frame dates as datetime64 and a freshly parsed one can
            # arrive as plain date objects; concatenating the two produces an
            # unordered categorical that drop_duplicates refuses to sort.
            frames = [
                f.assign(**{
                    SYMBOL: f[SYMBOL].astype(str),
                    "earnings_date": pd.to_datetime(f["earnings_date"], errors="coerce"),
                })
                for f in frames
            ]
            combined = pd.concat(frames, ignore_index=True)
            combined = combined.drop_duplicates(subset=[SYMBOL, "earnings_date"], keep="last")
            self.store.write_earnings_calendar(combined)
            self.store.update_feed_state("earnings_calendar", as_of, "yfinance+csv", len(combined))

        stored = self.store.read_earnings_calendar()
        future = (
            stored[pd.to_datetime(stored["earnings_date"]) >= pd.Timestamp(as_of)]
            if not stored.empty
            else stored
        )
        notes: List[str] = []
        if stored.empty:
            notes.append(
                "no earnings calendar available; the Stage 3 earnings-proximity "
                "gate will report NOT_TESTABLE rather than silently passing"
            )
        elif not yf_cal.empty and csv_cal.empty:
            notes.append(
                "dates are Yahoo estimates, not exchange-confirmed board-meeting "
                "notices; drop confirmed dates into "
                "config/reference/earnings_calendar.csv to upgrade this"
            )
        self._record_feed(
            "earnings_calendar",
            FeedStatus.OK if not stored.empty else FeedStatus.MISSING,
            SourceName.YFINANCE if not yf_cal.empty else SourceName.CSV_IMPORT,
            last_timestamp=as_of if not stored.empty else None,
            row_count=len(stored),
            symbols_covered=int(future[SYMBOL].nunique()) if not future.empty else 0,
            notes=notes,
        )

    # =====================================================================
    # CSV-only feeds
    # =====================================================================
    def _ingest_csv_feeds(self, as_of: dt.date, calendar: TradingCalendar) -> None:
        p = self.config.params

        pledging = self.csv.load_pledging()
        if not pledging.empty:
            self.store.write_pledging(pledging)
        stored_pledging = self.store.read_pledging()
        self._record_feed(
            "pledging",
            FeedStatus.OK if not stored_pledging.empty else FeedStatus.MISSING,
            SourceName.CSV_IMPORT,
            last_timestamp=(
                pd.to_datetime(stored_pledging["as_of_date"]).max().date()
                if not stored_pledging.empty
                else None
            ),
            row_count=len(stored_pledging),
            notes=[]
            if not stored_pledging.empty
            else [
                "no promoter-pledging data. No free NSE feed provides this "
                "reliably. The Stage 3 pledging gate reports NOT_TESTABLE -- it "
                "does NOT pass. Populate "
                f"{p.providers.csv_import.pledging_file} to enable it."
            ],
        )

        # Sectors feed the Stage 8 diversification cap. The point-in-time
        # universe reaches past any single index, so coverage is partial by
        # design and Stage 8 treats an unclassified name as unclassified rather
        # than pooling it with every other one.
        try:
            self._refresh_sector_map()
        except Exception as exc:
            log.warning("sector map refresh failed", extra={"error": str(exc)})

        if opts.force_reference_refresh or self._should_refresh(
            "statements", as_of, opts.reference_refresh_sessions
        ):
            if self._refresh_statements(symbols):
                self.store.update_feed_state("statements", as_of, "yfinance", 0)

        fundamentals = self.csv.load_fundamentals()
        if not fundamentals.empty:
            self.store.write_fundamentals(fundamentals)
        self._refresh_nse_fundamentals(as_of, opts)
        stored_fund = self.store.read_fundamentals()
        self._record_feed(
            "fundamentals",
            FeedStatus.OK if not stored_fund.empty else FeedStatus.MISSING,
            SourceName.CSV_IMPORT,
            last_timestamp=(
                pd.to_datetime(stored_fund["filing_date"]).max().date()
                if not stored_fund.empty
                else None
            ),
            row_count=len(stored_fund),
            symbols_covered=int(stored_fund[SYMBOL].nunique()) if not stored_fund.empty else 0,
            notes=[]
            if not stored_fund.empty
            else [
                "no point-in-time fundamentals. The Stage 4 quality factor will "
                "be dropped and the remaining factors renormalised -- stated on "
                "every card, not hidden."
            ],
        )

        events = self.csv.load_regulatory_events(
            int(p.stage3_eligibility.regulatory_cooldown.default_cooldown_sessions.value)
        )
        if not events.empty:
            self.store.write_regulatory_events(events)
        stored_events = self.store.read_regulatory_events()
        self._record_feed(
            "regulatory_events",
            FeedStatus.OK,
            SourceName.CSV_IMPORT,
            last_timestamp=as_of,
            row_count=len(stored_events),
            notes=[] if not stored_events.empty else ["no regulatory events logged"],
        )

    # =====================================================================
    # calendar & manifest
    # =====================================================================
    def _build_calendar(self, as_of: dt.date) -> TradingCalendar:
        sessions = self.store.known_sessions()
        if not sessions:
            log.warning("no stored sessions; falling back to an approximate calendar")
            return TradingCalendar.weekday_fallback(as_of - dt.timedelta(days=400), as_of)
        if as_of not in sessions:
            sessions = sorted(set(sessions) | {as_of})
        return TradingCalendar(sessions)

    def _record_feed(
        self,
        feed: str,
        status: FeedStatus,
        source: Optional[SourceName],
        last_timestamp: Optional[dt.date] = None,
        row_count: int = 0,
        symbols_covered: int = 0,
        notes: Optional[List[str]] = None,
        fallback_used: bool = False,
        primary_error: Optional[str] = None,
    ) -> None:
        self._feeds[feed] = FeedRecord(
            feed=feed,
            status=status,
            source=source,
            fallback_used=fallback_used,
            primary_source_error=primary_error,
            last_timestamp=last_timestamp,
            row_count=row_count,
            symbols_covered=symbols_covered,
            notes=list(notes or []),
        )

    def _build_manifest(
        self,
        run_id: str,
        as_of: dt.date,
        calendar: TradingCalendar,
        universe: UniverseSnapshot,
    ) -> RawDataManifest:
        policies = self.config.params.feeds
        for name, record in self._feeds.items():
            policy = policies.get(name)
            if policy is None:
                continue
            record.required = policy.required
            record.max_age_sessions = policy.max_age_sessions
            if record.last_timestamp is not None:
                record.age_sessions = calendar.age_in_sessions(record.last_timestamp, as_of)
                if record.age_sessions > policy.max_age_sessions and record.status is FeedStatus.OK:
                    record.status = FeedStatus.STALE
                    record.notes.append(
                        f"last update {record.last_timestamp} is {record.age_sessions} "
                        f"session(s) old; tolerance is {policy.max_age_sessions}"
                    )

        # Feeds declared in config but never touched this run are MISSING, not absent.
        for name, policy in policies.items():
            if name not in self._feeds:
                self._feeds[name] = FeedRecord(
                    feed=name,
                    status=FeedStatus.MISSING,
                    required=policy.required,
                    max_age_sessions=policy.max_age_sessions,
                    notes=["feed was not collected during this run"],
                )

        return RawDataManifest(
            run_id=run_id,
            as_of_date=as_of,
            generated_at=dt.datetime.now(),
            snapshot_id=f"{as_of.isoformat()}-{run_id}",
            feeds=self._feeds,
            universe_size_raw=len(universe),
            calendar_sessions_available=len(calendar),
            calendar_last_session=calendar.last if len(calendar) else None,
            calendar_is_approximate=calendar.is_approximate,
            survivorship_risk=universe.survivorship_risk,
            survivorship_note=universe.note,
        )

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> "DataIngestor":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
