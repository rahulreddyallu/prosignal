"""Stage 1 -- Data Quality and Leakage Gate.

The tests that matter here are the ones asserting the gate FAILS. A quality
gate that passes everything is indistinguishable from no gate at all, so each
defect this stage exists to catch is constructed explicitly: an unadjusted
split, a bad tick, a suspension, a broken feed.

The market-wide vs per-stock distinction gets particular attention. Getting it
backwards -- treating a broken feed as 50 individual bad stocks -- silently
shrinks the universe to whatever happened to survive, which is a far worse
failure than halting, and an invisible one.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from prosignal.core.calendar import TradingCalendar
from prosignal.core.contracts import FeedRecord, RawDataManifest
from prosignal.core.enums import FeedStatus, GateResult, SourceName
from prosignal.core.errors import MarketWideHalt
from prosignal.data.store import DataStore
from prosignal.data.types import DATE, SYMBOL
from prosignal.data.universe import UniverseSnapshot
from prosignal.stages import stage1_data_quality as s1

N_SESSIONS = 120


# =============================================================================
# fixtures / builders
# =============================================================================


def _sessions(n: int = N_SESSIONS) -> pd.DatetimeIndex:
    return pd.bdate_range("2025-01-01", periods=n)


def _clean_prices(symbols, dates, start=100.0, drift=0.0004, seed=0) -> pd.DataFrame:
    """A well-behaved price history: no gaps, no jumps, steady volume."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, symbol in enumerate(symbols):
        path = start * np.exp(
            np.cumsum(rng.normal(drift, 0.008, len(dates)))
        )
        for day, price in zip(dates, path):
            rows.append(
                {
                    DATE: day,
                    SYMBOL: symbol,
                    "series": "EQ",
                    "open": price,
                    "high": price * 1.005,
                    "low": price * 0.995,
                    "close": price,
                    "volume": 500_000.0,
                    "turnover": price * 500_000.0,
                    "trades": 2_000.0,
                    "isin": f"INE{i:04d}A01",
                    "source": "nse_archives",
                }
            )
    return pd.DataFrame(rows)


def _manifest(as_of, *, survivorship=False, overrides=None) -> RawDataManifest:
    """A manifest where every required feed is present and fresh."""
    required = {
        "equity_ohlcv": 1,
        "index_ohlcv": 1,
        "india_vix": 1,
        "index_membership": 25,
        "equity_master": 25,
    }
    feeds = {}
    for name, max_age in required.items():
        feeds[name] = FeedRecord(
            feed=name,
            status=FeedStatus.OK,
            source=SourceName.NSE_ARCHIVES,
            last_timestamp=as_of,
            age_sessions=0,
            max_age_sessions=max_age,
            required=True,
            row_count=100,
        )
    for name in ("fundamentals", "pledging"):
        feeds[name] = FeedRecord(
            feed=name, status=FeedStatus.MISSING, required=False, row_count=0
        )
    if overrides:
        feeds.update(overrides)

    return RawDataManifest(
        run_id="test-run",
        as_of_date=as_of,
        generated_at=dt.datetime(2025, 6, 1, 18, 0),
        snapshot_id="snap-1",
        feeds=feeds,
        universe_size_raw=len(feeds),
        survivorship_risk=survivorship,
        survivorship_note="reconstructed from a later snapshot" if survivorship else None,
    )


def _setup(tmp_path, prices: pd.DataFrame, dates, symbols, actions=None):
    store = DataStore(tmp_path / "curated", tmp_path / "snapshots")
    store.write_prices(prices)
    if actions is not None and not actions.empty:
        store.write_corporate_actions(actions)
    calendar = TradingCalendar([d.date() for d in dates])
    universe = UniverseSnapshot(
        index_name="NIFTY 200",
        as_of=dates[-1].date(),
        symbols=list(symbols),
        sector_map={s: "Test" for s in symbols},
    )
    return store, calendar, universe


# =============================================================================
# the happy path
# =============================================================================


def test_clean_data_passes_every_stock(tmp_path, cfg):
    dates = _sessions()
    symbols = ["AAA", "BBB", "CCC"]
    store, cal, uni = _setup(tmp_path, _clean_prices(symbols, dates), dates, symbols)

    report = s1.run(_manifest(dates[-1].date()), store, cal, uni, cfg)

    assert report.run_status is GateResult.PASS
    assert report.failed_symbols == 0
    assert report.checked_symbols == 3
    assert report.failed_tickers() == []
    assert all(report.is_clean(s) for s in symbols)


# =============================================================================
# market-wide halts
# =============================================================================


def test_missing_required_feed_halts_the_run(tmp_path, cfg):
    dates = _sessions()
    symbols = ["AAA"]
    store, cal, uni = _setup(tmp_path, _clean_prices(symbols, dates), dates, symbols)

    broken = {
        "index_ohlcv": FeedRecord(
            feed="index_ohlcv",
            status=FeedStatus.MISSING,
            required=True,
            max_age_sessions=1,
            primary_source_error="archive returned 404",
        )
    }
    manifest = _manifest(dates[-1].date(), overrides=broken)

    with pytest.raises(MarketWideHalt) as exc:
        s1.run(manifest, store, cal, uni, cfg)
    assert any("index_ohlcv" in r and "MISSING" in r for r in exc.value.reasons)


def test_stale_required_feed_halts_the_run(tmp_path, cfg):
    dates = _sessions()
    symbols = ["AAA"]
    store, cal, uni = _setup(tmp_path, _clean_prices(symbols, dates), dates, symbols)

    stale = {
        "india_vix": FeedRecord(
            feed="india_vix",
            status=FeedStatus.OK,
            required=True,
            last_timestamp=dates[-10].date(),
            age_sessions=9,
            max_age_sessions=1,
        )
    }
    with pytest.raises(MarketWideHalt) as exc:
        s1.run(_manifest(dates[-1].date(), overrides=stale), store, cal, uni, cfg)
    assert any("STALE" in r for r in exc.value.reasons)


def test_missing_optional_feed_never_halts(tmp_path, cfg):
    """Optional feeds degrade the output; they do not stop the run."""
    dates = _sessions()
    symbols = ["AAA"]
    store, cal, uni = _setup(tmp_path, _clean_prices(symbols, dates), dates, symbols)

    absent = {
        "delivery_data": FeedRecord(
            feed="delivery_data", status=FeedStatus.MISSING, required=False
        )
    }
    report = s1.run(_manifest(dates[-1].date(), overrides=absent), store, cal, uni, cfg)
    assert report.run_status is GateResult.PASS


def test_broken_feed_halts_rather_than_shrinking_the_universe(tmp_path, cfg):
    """The most important behaviour in this stage.

    When most of the universe fails at once, the feed is broken -- not the
    stocks. Excluding them individually would leave a handful of survivors and
    the engine would happily trade them, having quietly discarded 80% of its
    opportunity set for a reason that has nothing to do with those companies.
    """
    dates = _sessions()
    symbols = [f"S{i:02d}" for i in range(20)]
    prices = _clean_prices(symbols, dates)

    # Corrupt 80% of names with an unexplained 5:1 split on the last session.
    last = dates[-1]
    for symbol in symbols[:16]:
        mask = (prices[SYMBOL] == symbol) & (prices[DATE] == last)
        for _col in ("open", "high", "low", "close"):
            prices.loc[mask, _col] = prices.loc[mask, _col] / 5.0

    store, cal, uni = _setup(tmp_path, prices, dates, symbols)

    with pytest.raises(MarketWideHalt) as exc:
        s1.run(_manifest(last.date()), store, cal, uni, cfg)

    reason = exc.value.reasons[0]
    assert "of the universe failed" in reason
    assert "the feed is the likely fault" in reason


def test_failure_fraction_needs_a_population_to_be_meaningful(tmp_path, cfg):
    """A 3-name universe with one bad stock is not evidence of a broken feed.

    The fraction rule is a claim about a population. Applying it to a handful
    of names turns every single bad tick into a market-wide halt, which is how
    a safety check becomes an outage.
    """
    dates = _sessions()
    symbols = ["AAA", "BBB", "CCC"]
    prices = _clean_prices(symbols, dates)

    # Corrupt two of three names -- 67%, far above the 25% ceiling.
    for symbol in symbols[:2]:
        mask = (prices[SYMBOL] == symbol) & (prices[DATE] >= dates[-5])
        for _col in ("open", "high", "low", "close"):
            prices.loc[mask, _col] = prices.loc[mask, _col] / 5.0

    store, cal, uni = _setup(tmp_path, prices, dates, symbols)

    # No halt: the sample is below min_universe_for_failure_fraction.
    report = s1.run(_manifest(dates[-1].date()), store, cal, uni, cfg)
    assert report.run_status is GateResult.PASS
    assert report.failed_symbols == 2
    assert report.is_clean("CCC")


def test_no_price_data_at_all_halts(tmp_path, cfg):
    dates = _sessions()
    store = DataStore(tmp_path / "curated", tmp_path / "snapshots")
    cal = TradingCalendar([d.date() for d in dates])
    uni = UniverseSnapshot("NIFTY 200", dates[-1].date(), ["AAA"], {"AAA": "Test"})

    with pytest.raises(MarketWideHalt, match="no price rows"):
        s1.run(_manifest(dates[-1].date()), store, cal, uni, cfg)


# =============================================================================
# per-stock: unexplained corporate action
# =============================================================================


def test_unadjusted_split_hard_rejects_that_stock(tmp_path, cfg):
    """An unadjusted 5:1 split reads as a clean -80% session.

    It looks entirely normal on a chart and would poison a 12-1 momentum score
    for a year. This is the single most valuable check in the stage.
    """
    dates = _sessions()
    symbols = ["GOOD", "SPLIT"]
    prices = _clean_prices(symbols, dates)

    mask = (prices[SYMBOL] == "SPLIT") & (prices[DATE] >= dates[-5])
    for _col in ("open", "high", "low", "close"):
        prices.loc[mask, _col] = prices.loc[mask, _col] / 5.0

    store, cal, uni = _setup(tmp_path, prices, dates, symbols)
    report = s1.run(_manifest(dates[-1].date()), store, cal, uni, cfg)

    assert report.per_stock_flags["SPLIT"].status is GateResult.FAIL
    assert s1.CHECK_UNEXPLAINED_ACTION in report.per_stock_flags["SPLIT"].failed_checks
    assert report.is_clean("GOOD")


def test_a_declared_corporate_action_is_not_flagged(tmp_path, cfg):
    """The same price move, with the action on record, must pass cleanly."""
    dates = _sessions()
    symbols = ["SPLIT"]
    prices = _clean_prices(symbols, dates)
    split_date = dates[-5]
    mask = (prices[SYMBOL] == "SPLIT") & (prices[DATE] >= split_date)
    for _col in ("open", "high", "low", "close"):
        prices.loc[mask, _col] = prices.loc[mask, _col] / 5.0

    actions = pd.DataFrame(
        [
            {
                SYMBOL: "SPLIT",
                "ex_date": split_date.date(),
                "action_type": "split",
                "ratio": 5.0,
                "subject": "Face value split 10 to 2",
                "source": "test",
            }
        ]
    )

    store, cal, uni = _setup(tmp_path, prices, dates, symbols, actions=actions)
    report = s1.run(_manifest(dates[-1].date()), store, cal, uni, cfg)
    assert s1.CHECK_UNEXPLAINED_ACTION not in report.per_stock_flags["SPLIT"].failed_checks


# =============================================================================
# per-stock: outlier / bad tick
# =============================================================================


def test_uncorroborated_spike_is_rejected_as_a_bad_tick(tmp_path, cfg):
    """Extreme move, no volume behind it, no corporate action -> bad print."""
    dates = _sessions()
    symbols = ["TICK"]
    prices = _clean_prices(symbols, dates)

    mask = (prices[SYMBOL] == "TICK") & (prices[DATE] == dates[-1])
    for _col in ("open", "high", "low", "close"):
        prices.loc[mask, _col] = prices.loc[mask, _col] * 1.40  # +40%, normal volume

    store, cal, uni = _setup(tmp_path, prices, dates, symbols)
    report = s1.run(_manifest(dates[-1].date()), store, cal, uni, cfg)

    flags = report.per_stock_flags["TICK"]
    assert flags.status is GateResult.FAIL
    assert s1.CHECK_OUTLIER in flags.failed_checks
    assert flags.details["outlier"]["corroborated_by_volume"] is False


def test_the_same_move_on_heavy_volume_is_kept(tmp_path, cfg):
    """A real breakout looks extreme too. Volume is what tells them apart.

    Rejecting purely on size would discard exactly the moves this engine
    exists to find.
    """
    dates = _sessions()
    symbols = ["REAL"]
    prices = _clean_prices(symbols, dates)

    mask = (prices[SYMBOL] == "REAL") & (prices[DATE] == dates[-1])
    for _col in ("open", "high", "low", "close"):
        prices.loc[mask, _col] = prices.loc[mask, _col] * 1.40
    prices.loc[mask, "volume"] = 500_000.0 * 6.0  # unmistakable participation

    store, cal, uni = _setup(tmp_path, prices, dates, symbols)
    report = s1.run(_manifest(dates[-1].date()), store, cal, uni, cfg)

    flags = report.per_stock_flags["REAL"]
    assert s1.CHECK_OUTLIER not in flags.failed_checks
    assert flags.details["outlier"]["corroborated_by_volume"] is True


def test_ordinary_volatility_is_not_an_outlier(tmp_path, cfg):
    dates = _sessions()
    symbols = ["NORMAL"]
    store, cal, uni = _setup(tmp_path, _clean_prices(symbols, dates, seed=5), dates, symbols)
    report = s1.run(_manifest(dates[-1].date()), store, cal, uni, cfg)
    assert s1.CHECK_OUTLIER not in report.per_stock_flags["NORMAL"].failed_checks


# =============================================================================
# per-stock: continuity
# =============================================================================


def test_a_suspended_stock_is_excluded(tmp_path, cfg):
    """A long unbroken run of missing sessions is a suspension, not a glitch."""
    dates = _sessions()
    symbols = ["LIVE", "HALTED"]
    prices = _clean_prices(symbols, dates)

    limit = int(cfg.params.stage1_data_quality.max_consecutive_missing_sessions.value)
    gap_dates = dates[-(limit + 6) : -1]
    prices = prices[~((prices[SYMBOL] == "HALTED") & (prices[DATE].isin(gap_dates)))]

    store, cal, uni = _setup(tmp_path, prices, dates, symbols)
    report = s1.run(_manifest(dates[-1].date()), store, cal, uni, cfg)

    halted = report.per_stock_flags["HALTED"]
    assert halted.status is GateResult.FAIL
    assert s1.CHECK_CONTINUITY in halted.failed_checks
    assert halted.details["continuity"]["longest_consecutive_gap"] > limit
    assert report.is_clean("LIVE")


def test_scattered_missing_sessions_are_tolerated(tmp_path, cfg):
    """Counting CONSECUTIVE gaps, not total, is the point.

    A stock missing a handful of scattered sessions is a data annoyance; one
    missing the same number in a row was suspended, which is a different fact.
    """
    dates = _sessions()
    symbols = ["SCATTER"]
    prices = _clean_prices(symbols, dates)

    scattered = [dates[-30], dates[-24], dates[-18], dates[-12], dates[-6]]
    prices = prices[~((prices[SYMBOL] == "SCATTER") & (prices[DATE].isin(scattered)))]

    store, cal, uni = _setup(tmp_path, prices, dates, symbols)
    report = s1.run(_manifest(dates[-1].date()), store, cal, uni, cfg)

    flags = report.per_stock_flags["SCATTER"]
    assert flags.details["continuity"]["longest_consecutive_gap"] == 1
    assert s1.CHECK_CONTINUITY not in flags.failed_checks


def test_a_stock_with_no_rows_at_all_fails(tmp_path, cfg):
    dates = _sessions()
    symbols = ["PRESENT"]
    store, cal, _ = _setup(tmp_path, _clean_prices(symbols, dates), dates, symbols)
    uni = UniverseSnapshot(
        "NIFTY 200", dates[-1].date(), ["PRESENT", "GHOST"],
        {"PRESENT": "Test", "GHOST": "Test"},
    )
    report = s1.run(_manifest(dates[-1].date()), store, cal, uni, cfg)

    assert report.per_stock_flags["GHOST"].status is GateResult.FAIL
    assert s1.CHECK_NO_PRICE in report.per_stock_flags["GHOST"].failed_checks


# =============================================================================
# per-stock: cross-source agreement
# =============================================================================


def test_source_disagreement_is_flagged_never_silently_resolved(tmp_path, cfg):
    """Yahoo adjusts and rounds; a difference is as likely theirs as NSE's.

    So the engine reports the disagreement and keeps its own primary value. It
    must never quietly switch sources.
    """
    dates = _sessions()
    symbols = ["DIFF"]
    prices = _clean_prices(symbols, dates)
    store, cal, uni = _setup(tmp_path, prices, dates, symbols)

    primary_close = float(
        prices[(prices[SYMBOL] == "DIFF") & (prices[DATE] == dates[-1])]["close"].iloc[0]
    )
    secondary = pd.DataFrame(
        [
            {
                DATE: dates[-1],
                SYMBOL: "DIFF",
                "open": primary_close,
                "high": primary_close,
                "low": primary_close,
                "close": primary_close * 1.05,  # 500 bps apart
                "volume": 1000.0,
                "source": "yfinance",
            }
        ]
    )
    store.write_table("prices_secondary", secondary, [DATE, SYMBOL])

    report = s1.run(_manifest(dates[-1].date()), store, cal, uni, cfg)
    flags = report.per_stock_flags["DIFF"]

    assert s1.CHECK_SOURCE_AGREEMENT in flags.soft_flags
    assert flags.status is GateResult.PASS, "flag by default, not reject"
    assert flags.details["source_agreement"]["difference_bps"] > 400


def test_close_agreement_raises_no_flag(tmp_path, cfg):
    dates = _sessions()
    symbols = ["SAME"]
    prices = _clean_prices(symbols, dates)
    store, cal, uni = _setup(tmp_path, prices, dates, symbols)

    primary_close = float(
        prices[(prices[SYMBOL] == "SAME") & (prices[DATE] == dates[-1])]["close"].iloc[0]
    )
    store.write_table(
        "prices_secondary",
        pd.DataFrame(
            [
                {
                    DATE: dates[-1],
                    SYMBOL: "SAME",
                    "open": primary_close,
                    "high": primary_close,
                    "low": primary_close,
                    "close": primary_close * 1.0005,  # 5 bps
                    "volume": 1000.0,
                    "source": "yfinance",
                }
            ]
        ),
        [DATE, SYMBOL],
    )

    report = s1.run(_manifest(dates[-1].date()), store, cal, uni, cfg)
    assert s1.CHECK_SOURCE_AGREEMENT not in report.per_stock_flags["SAME"].soft_flags


def test_single_source_is_a_soft_flag_by_default(tmp_path, cfg):
    dates = _sessions()
    symbols = ["ONLY"]
    store, cal, uni = _setup(tmp_path, _clean_prices(symbols, dates), dates, symbols)

    report = s1.run(_manifest(dates[-1].date()), store, cal, uni, cfg)
    flags = report.per_stock_flags["ONLY"]

    assert s1.CHECK_SINGLE_SOURCE in flags.soft_flags
    assert flags.status is GateResult.PASS


# =============================================================================
# point-in-time audit
# =============================================================================


def test_survivorship_risk_is_recorded_not_hidden(tmp_path, cfg):
    dates = _sessions()
    symbols = ["AAA"]
    store, cal, uni = _setup(tmp_path, _clean_prices(symbols, dates), dates, symbols)

    report = s1.run(
        _manifest(dates[-1].date(), survivorship=True), store, cal, uni, cfg
    )

    assert report.pit_audit["historical_membership"] is False
    assert report.pit_audit["delisted_inclusion"] is False
    assert any("survivorship bias" in f for f in report.pit_audit_failures)


def test_clean_pit_run_records_the_guarantees_as_held(tmp_path, cfg):
    dates = _sessions()
    symbols = ["AAA"]
    store, cal, uni = _setup(tmp_path, _clean_prices(symbols, dates), dates, symbols)

    report = s1.run(_manifest(dates[-1].date()), store, cal, uni, cfg)

    assert report.pit_audit["historical_membership"] is True
    assert report.pit_audit["no_forward_fill"] is True


def test_absent_fundamentals_and_pledging_are_reported_as_unverifiable(tmp_path, cfg):
    """A check that cannot run reports so. It never quietly becomes a PASS."""
    dates = _sessions()
    symbols = ["AAA"]
    store, cal, uni = _setup(tmp_path, _clean_prices(symbols, dates), dates, symbols)

    report = s1.run(_manifest(dates[-1].date()), store, cal, uni, cfg)

    assert report.pit_audit["fundamentals_filing_date"] is False
    assert report.pit_audit["pledging_disclosure_date"] is False
    assert any("neither the NSE filings table nor the statement feed" in f
               for f in report.pit_audit_failures)
    assert any("NOT_TESTABLE" in f for f in report.pit_audit_failures)


def test_statements_without_filing_dates_pass_the_check_but_say_which(tmp_path, cfg):
    """Weaker evidence is not absent evidence.

    "fundamentals" is the NSE Ind-AS feed and carries true filing dates;
    "statements" carries period end only, so availability comes from the SEBI
    LODR deadline. Reporting the block missing while the model scores five
    fundamental factors off it was the misleading direction -- the deadline is
    later than a typical filing, so it understates what the market knew.
    """
    from prosignal.core.contracts import FeedRecord
    from prosignal.core.enums import FeedStatus

    dates = _sessions()
    symbols = ["AAA"]
    store, cal, uni = _setup(tmp_path, _clean_prices(symbols, dates), dates, symbols)
    manifest = _manifest(dates[-1].date())
    manifest.feeds["statements"] = FeedRecord(
        feed="statements", status=FeedStatus.OK, row_count=22256,
        symbols_covered=1284, required=False,
    )

    report = s1.run(manifest, store, cal, uni, cfg)

    assert report.pit_audit["fundamentals_filing_date"] is True
    assert not any("neither the NSE filings table" in f
                   for f in report.pit_audit_failures)
    assert any("SEBI LODR deadline" in f for f in report.market_wide_soft_flags), (
        "the run must say the value factors key off a deadline rather than a "
        "filing date, not stay silent about it"
    )


# =============================================================================
# contract
# =============================================================================


def test_report_is_serialisable(tmp_path, cfg):
    dates = _sessions()
    symbols = ["AAA", "BBB"]
    store, cal, uni = _setup(tmp_path, _clean_prices(symbols, dates), dates, symbols)

    report = s1.run(_manifest(dates[-1].date()), store, cal, uni, cfg)
    payload = report.model_dump(mode="json")

    assert payload["run_status"] == "PASS"
    assert set(payload["per_stock_flags"]) == {"AAA", "BBB"}


def test_checks_are_binary_and_independent(tmp_path, cfg):
    """Two defects on one stock produce two named failures, not a blended score.

    A score would let several moderate problems average into an acceptable
    number. Either the data can be trusted or it cannot.
    """
    dates = _sessions()
    symbols = ["BROKEN"]
    prices = _clean_prices(symbols, dates)

    limit = int(cfg.params.stage1_data_quality.max_consecutive_missing_sessions.value)
    gap_dates = dates[-(limit + 20) : -(limit + 20) + limit + 2]
    prices = prices[~((prices[SYMBOL] == "BROKEN") & (prices[DATE].isin(gap_dates)))]

    mask = (prices[SYMBOL] == "BROKEN") & (prices[DATE] == dates[-1])
    prices.loc[mask, "close"] = prices.loc[mask, "close"] * 1.45

    store, cal, uni = _setup(tmp_path, prices, dates, symbols)
    report = s1.run(_manifest(dates[-1].date()), store, cal, uni, cfg)

    failed = report.per_stock_flags["BROKEN"].failed_checks
    assert s1.CHECK_CONTINUITY in failed
    assert s1.CHECK_OUTLIER in failed
    assert len(failed) >= 2
