"""Stage 2 -- Market Regime Engine.

Built on synthetic index paths with a KNOWN regime, so each assertion is
"the engine identified the state I constructed", not "the engine returned
something". The crash-signature and transition tests reconstruct the market
shapes those rules exist to catch.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from prosignal.core.calendar import TradingCalendar
from prosignal.core.enums import (
    BreadthState,
    RegimeCompatibility,
    TrendRegime,
    VolContext,
    VolTercile,
)
from prosignal.core.errors import DataError
from prosignal.data.providers.nse_archives import INDIA_VIX_NAME
from prosignal.data.store import DataStore
from prosignal.data.types import DATE, SYMBOL
from prosignal.stages import stage2_regime

N_SESSIONS = 420


# =============================================================================
# synthetic market builders
# =============================================================================


def _sessions(n: int = N_SESSIONS) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


def _index_frame(paths: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for name, values in paths.items():
        for day, value in zip(dates, values):
            rows.append(
                {
                    DATE: day,
                    "index_name": name,
                    "open": value,
                    "high": value,
                    "low": value,
                    "close": value,
                    "volume": 0.0,
                    "source": "test",
                }
            )
    return pd.DataFrame(rows)


def _price_frame(paths: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for symbol, values in paths.items():
        for day, value in zip(dates, values):
            rows.append(
                {
                    DATE: day,
                    SYMBOL: symbol,
                    "series": "EQ",
                    "open": value,
                    "high": value * 1.01,
                    "low": value * 0.99,
                    "close": value,
                    "volume": 100_000.0,
                    "turnover": value * 100_000.0,
                    "trades": 500.0,
                    "isin": f"IN{symbol}",
                    "source": "test",
                }
            )
    return pd.DataFrame(rows)


def _build(tmp_path, index_path, vix_path, constituent_paths, benchmark="Nifty 200"):
    """Assemble a store containing a fully specified synthetic market."""
    dates = _sessions(len(index_path))
    store = DataStore(tmp_path / "curated", tmp_path / "snapshots")
    store.write_indices(
        _index_frame({benchmark: index_path, INDIA_VIX_NAME: vix_path}, dates)
    )
    store.write_prices(_price_frame(constituent_paths, dates))
    calendar = TradingCalendar([d.date() for d in dates])
    return store, calendar, list(constituent_paths), dates[-1].date()


def _rising(n=N_SESSIONS, rate=0.30):
    """Index compounding steadily at ``rate`` per year."""
    return 100 * np.exp((rate / 252) * np.arange(n))


def _falling(n=N_SESSIONS, rate=-0.30):
    return 100 * np.exp((rate / 252) * np.arange(n))


def _flat(n=N_SESSIONS, level=100.0):
    return np.full(n, level, dtype="float64")


def _vix(n=N_SESSIONS, level=15.0):
    return np.full(n, level, dtype="float64")


def _constituents(index_path, n_names=20, participation=1.0, seed=0):
    """Constituents that mirror the index for ``participation`` of the names.

    The rest are given a steadily falling path, which is how a narrowing
    advance is manufactured for the breadth tests.
    """
    rng = np.random.default_rng(seed)
    n = len(index_path)
    out = {}
    n_following = int(round(n_names * participation))
    for i in range(n_names):
        if i < n_following:
            noise = rng.normal(0, 0.001, n).cumsum()
            out[f"UP{i:02d}"] = index_path * np.exp(noise)
        else:
            out[f"DN{i:02d}"] = _falling(n, -0.40)
    return out


# =============================================================================
# trend
# =============================================================================


def test_sustained_rally_is_an_uptrend(tmp_path, cfg):
    path = _rising()
    store, cal, syms, as_of = _build(tmp_path, path, _vix(), _constituents(path))
    state = stage2_regime.run(store, cal, syms, cfg, as_of=as_of)
    assert state.trend_regime is TrendRegime.UPTREND
    assert state.trend_slope_annualised == pytest.approx(0.30, rel=0.15)
    assert state.index_vs_slow_ma_pct > 0


def test_sustained_decline_is_a_downtrend(tmp_path, cfg):
    path = _falling()
    store, cal, syms, as_of = _build(tmp_path, path, _vix(), _constituents(path))
    state = stage2_regime.run(store, cal, syms, cfg, as_of=as_of)
    assert state.trend_regime is TrendRegime.DOWNTREND
    assert state.index_vs_slow_ma_pct < 0


def test_flat_market_is_range_bound(tmp_path, cfg):
    path = _flat()
    store, cal, syms, as_of = _build(tmp_path, path, _vix(), _constituents(path))
    state = stage2_regime.run(store, cal, syms, cfg, as_of=as_of)
    assert state.trend_regime is TrendRegime.RANGE_BOUND


def test_bounce_inside_a_downtrend_is_not_an_uptrend(tmp_path, cfg):
    """The case the slope-alone test gets wrong.

    A long decline followed by a sharp rally: the recent slope is strongly
    positive, but price is still far below the 200-session average. That is a
    bounce, and calling it an uptrend is how a regime filter gets you long into
    a bear market rally.
    """
    decline = _falling(360, -0.50)
    bounce = decline[-1] * np.exp((0.9 / 252) * np.arange(1, 31))
    path = np.concatenate([decline, bounce])

    store, cal, syms, as_of = _build(
        tmp_path, path, _vix(len(path)), _constituents(path)
    )
    state = stage2_regime.run(store, cal, syms, cfg, as_of=as_of)

    assert state.trend_slope_annualised > 0, "the recent slope really is positive"
    assert state.index_vs_slow_ma_pct < 0, "but price is still under the 200-DMA"
    assert state.trend_regime is TrendRegime.RANGE_BOUND
    assert any("disagree" in n for n in state.notes)


def test_short_history_reports_range_bound_and_says_why(tmp_path, cfg):
    """No 200-session average means no trend claim, not a guessed one."""
    path = _rising(60)
    store, cal, syms, as_of = _build(tmp_path, path, _vix(60), _constituents(path))
    state = stage2_regime.run(store, cal, syms, cfg, as_of=as_of)
    assert state.trend_regime is TrendRegime.RANGE_BOUND
    assert any("Insufficient benchmark history" in n for n in state.notes)


# =============================================================================
# volatility
# =============================================================================


def test_vix_tercile_is_relative_not_absolute(tmp_path, cfg):
    """A VIX of 18 is 'high' in a calm year and 'low' in a violent one.

    This is the single most important property of the volatility read: the same
    absolute level maps to opposite terciles depending on its own history.
    """
    path = _rising()
    n = len(path)

    calm = np.concatenate([np.full(n - 1, 11.0), [18.0]])
    store, cal, syms, as_of = _build(tmp_path, path, calm, _constituents(path))
    assert stage2_regime.run(store, cal, syms, cfg, as_of=as_of).vol_tercile is VolTercile.HIGH


def test_same_vix_level_reads_low_in_a_violent_year(tmp_path, cfg):
    path = _rising()
    n = len(path)
    violent = np.concatenate([np.full(n - 1, 40.0), [18.0]])
    store, cal, syms, as_of = _build(tmp_path, path, violent, _constituents(path))
    assert stage2_regime.run(store, cal, syms, cfg, as_of=as_of).vol_tercile is VolTercile.LOW


def test_rising_vix_into_a_falling_market_is_rising_in_decline(tmp_path, cfg):
    n = N_SESSIONS
    path = np.concatenate([_rising(n - 5), _rising(n - 5)[-1] * np.linspace(1.0, 0.94, 5)])
    vix = np.concatenate([np.full(n - 5, 14.0), np.linspace(14.0, 25.0, 5)])

    store, cal, syms, as_of = _build(tmp_path, path, vix, _constituents(path))
    state = stage2_regime.run(store, cal, syms, cfg, as_of=as_of)
    assert state.vol_context is VolContext.RISING_IN_DECLINE
    assert state.vix_change_pct > 10


def test_rising_vix_into_a_rally_is_flagged_separately(tmp_path, cfg):
    """Volatility rising into strength is unusual and sits closer to tops."""
    n = N_SESSIONS
    path = np.concatenate([_rising(n - 5), _rising(n - 5)[-1] * np.linspace(1.0, 1.05, 5)])
    vix = np.concatenate([np.full(n - 5, 14.0), np.linspace(14.0, 25.0, 5)])

    store, cal, syms, as_of = _build(tmp_path, path, vix, _constituents(path))
    state = stage2_regime.run(store, cal, syms, cfg, as_of=as_of)
    assert state.vol_context is VolContext.RISING_IN_RALLY
    assert any("unusual" in n for n in state.notes)


def test_falling_vix_carries_reduced_confidence(tmp_path, cfg):
    """G.C. & Kothari: a falling-VIX all-clear is the least reliable read."""
    n = N_SESSIONS
    path = _rising()
    vix = np.concatenate([np.full(n - 5, 25.0), np.linspace(25.0, 14.0, 5)])

    store, cal, syms, as_of = _build(tmp_path, path, vix, _constituents(path))
    state = stage2_regime.run(store, cal, syms, cfg, as_of=as_of)

    assert state.vol_context is VolContext.FALLING
    assert state.vol_signal_confidence == pytest.approx(
        cfg.params.stage2_regime.volatility.asymmetric_confidence.falling_vix_weight
    )
    assert state.vol_signal_confidence < 1.0


def test_missing_vix_is_reported_not_assumed_calm(tmp_path, cfg):
    path = _rising()
    dates = _sessions(len(path))
    store = DataStore(tmp_path / "curated", tmp_path / "snapshots")
    store.write_indices(_index_frame({"Nifty 200": path}, dates))  # no VIX at all
    store.write_prices(_price_frame(_constituents(path), dates))
    cal = TradingCalendar([d.date() for d in dates])

    state = stage2_regime.run(
        store, cal, list(_constituents(path)), cfg, as_of=dates[-1].date()
    )
    assert state.vix_level is None
    assert state.vol_signal_confidence < 1.0
    assert any("absence of evidence" in n for n in state.notes)


# =============================================================================
# breadth
# =============================================================================


def test_broad_participation_is_strong_breadth(tmp_path, cfg):
    path = _rising()
    store, cal, syms, as_of = _build(
        tmp_path, path, _vix(), _constituents(path, participation=1.0)
    )
    state = stage2_regime.run(store, cal, syms, cfg, as_of=as_of)
    assert state.breadth_state is BreadthState.STRONG
    assert state.breadth_pct_above_ma > 60
    assert state.breadth_sample_size == 20


def test_narrow_advance_is_weak_breadth_and_cuts_momentum(tmp_path, cfg):
    """The 2021-22 NSE shape: index carried by a few names, most rolling over."""
    path = _rising()
    store, cal, syms, as_of = _build(
        tmp_path, path, _vix(), _constituents(path, participation=0.25)
    )
    state = stage2_regime.run(store, cal, syms, cfg, as_of=as_of)

    assert state.breadth_state is BreadthState.WEAK
    assert state.breadth_pct_above_ma < 40

    penalty = float(cfg.params.stage2_regime.multipliers.weak_breadth_momentum_penalty.value)
    table = cfg.params.stage2_regime.multipliers.table
    expected = table[state.regime_bucket][0] * penalty
    assert state.momentum_multiplier == pytest.approx(expected, rel=1e-6)
    assert any("Breadth is weak" in n for n in state.notes)


def test_breadth_is_never_measured_without_enough_history(tmp_path, cfg):
    path = _rising(120)
    store, cal, syms, as_of = _build(tmp_path, path, _vix(120), _constituents(path))
    state = stage2_regime.run(store, cal, syms, cfg, as_of=as_of)
    assert state.breadth_pct_above_ma is None
    assert state.breadth_state is BreadthState.NEUTRAL
    assert any("breadth could not be measured" in n for n in state.notes)


# =============================================================================
# transition
# =============================================================================


def test_stable_market_shows_no_transition(tmp_path, cfg):
    path = _rising()
    store, cal, syms, as_of = _build(tmp_path, path, _vix(), _constituents(path))
    state = stage2_regime.run(store, cal, syms, cfg, as_of=as_of)
    assert state.transition_flag is False
    assert state.dampener_applied == 1.0


def test_regime_turn_sets_the_transition_flag_and_dampens(tmp_path, cfg):
    """Volatility and breadth both flip inside the lookback window."""
    # A 12-session crash: at t-10 the market was still a calm broad uptrend,
    # by t both trend and breadth have flipped.
    n, crash_len = N_SESSIONS, 12
    base = _rising(n - crash_len)
    path = np.concatenate([base, base[-1] * np.linspace(1.0, 0.80, crash_len)])
    vix = np.concatenate([np.full(n - crash_len, 10.0), np.linspace(10.0, 55.0, crash_len)])

    store, cal, syms, as_of = _build(
        tmp_path, path, vix, _constituents(path, participation=1.0)
    )
    state = stage2_regime.run(store, cal, syms, cfg, as_of=as_of)

    assert state.transition_flag is True
    assert len(state.transition_components) >= int(
        cfg.params.stage2_regime.transition.min_components_disagreeing.value
    )
    assert state.dampener_applied == pytest.approx(
        float(cfg.params.stage2_regime.transition.dampener.value)
    )
    assert any("transition detected" in n for n in state.notes)


def test_dampener_spares_quality(tmp_path, cfg):
    """Quality is the crash stabiliser; dampening it in a turn is backwards."""
    # A 12-session crash: at t-10 the market was still a calm broad uptrend,
    # by t both trend and breadth have flipped. Two components disagree, which
    # is exactly the threshold the detector is configured for.
    n, crash_len = N_SESSIONS, 12
    base = _rising(n - crash_len)
    path = np.concatenate([base, base[-1] * np.linspace(1.0, 0.80, crash_len)])
    vix = np.concatenate([np.full(n - crash_len, 10.0), np.linspace(10.0, 55.0, crash_len)])

    store, cal, syms, as_of = _build(tmp_path, path, vix, _constituents(path))
    state = stage2_regime.run(store, cal, syms, cfg, as_of=as_of)

    assert state.transition_flag is True
    table = cfg.params.stage2_regime.multipliers.table
    assert state.quality_multiplier == pytest.approx(table[state.regime_bucket][1])


# =============================================================================
# momentum crash -- the Daniel & Moskowitz state
# =============================================================================


def test_crash_signature_fires_on_a_violent_rebound_after_a_decline(tmp_path, cfg):
    """March-June 2020 in shape: deep decline, high vol, sharp rebound.

    This is the state where momentum inverts hardest, so the engine must block
    new entries rather than merely trim the multiplier.
    """
    decline = _falling(340, -0.55)
    rebound = decline[-1] * np.exp((3.0 / 252) * np.arange(1, 22))
    path = np.concatenate([decline, rebound])
    n = len(path)

    # Volatility elevated throughout the back half so the tercile reads HIGH.
    vix = np.concatenate([np.full(n - 40, 12.0), np.full(40, 45.0)])

    store, cal, syms, as_of = _build(tmp_path, path, vix, _constituents(path))
    state = stage2_regime.run(store, cal, syms, cfg, as_of=as_of)

    assert state.vol_tercile is VolTercile.HIGH
    assert state.regime_bucket == stage2_regime.CRASH_BUCKET
    assert state.allow_new_entries is False
    assert state.compatibility() is RegimeCompatibility.UNFAVORABLE
    assert any("momentum-crash signature" in n for n in state.notes)


def test_crash_bucket_does_not_fire_in_calm_volatility(tmp_path, cfg):
    """Same price shape, ordinary volatility: not the crash state."""
    decline = _falling(340, -0.55)
    rebound = decline[-1] * np.exp((3.0 / 252) * np.arange(1, 22))
    path = np.concatenate([decline, rebound])

    store, cal, syms, as_of = _build(tmp_path, path, _vix(len(path), 12.0), _constituents(path))
    state = stage2_regime.run(store, cal, syms, cfg, as_of=as_of)
    assert state.regime_bucket != stage2_regime.CRASH_BUCKET


def test_no_new_entry_buckets_are_a_hard_gate(tmp_path, cfg):
    path = _falling(N_SESSIONS, -0.50)
    n = len(path)
    vix = np.concatenate([np.full(n - 40, 12.0), np.full(40, 45.0)])

    store, cal, syms, as_of = _build(tmp_path, path, vix, _constituents(path))
    state = stage2_regime.run(store, cal, syms, cfg, as_of=as_of)

    if state.regime_bucket in set(cfg.params.stage2_regime.no_new_entry_buckets.value):
        assert state.allow_new_entries is False
        assert state.block_reason and "hard market-wide gate" in state.block_reason


# =============================================================================
# multipliers, wiring, failure modes
# =============================================================================


def test_multipliers_come_from_the_config_table(tmp_path, cfg):
    """No multiplier is hardcoded; every one traces to parameters.yaml."""
    path = _rising()
    store, cal, syms, as_of = _build(tmp_path, path, _vix(), _constituents(path))
    state = stage2_regime.run(store, cal, syms, cfg, as_of=as_of)

    row = cfg.params.stage2_regime.multipliers.table[state.regime_bucket]
    assert state.quality_multiplier == pytest.approx(row[1])
    assert state.sector_rs_multiplier == pytest.approx(row[2])


def test_unknown_bucket_raises_rather_than_defaulting_to_neutral(cfg):
    """A silent 1.0 would mean the regime engine did nothing while looking fine."""
    with pytest.raises(DataError, match="no row in"):
        stage2_regime._lookup_multipliers("not_a_bucket", {"uptrend_lowvol": [1, 1, 1]})


def test_missing_benchmark_index_is_fatal(tmp_path, cfg):
    dates = _sessions(300)
    store = DataStore(tmp_path / "curated", tmp_path / "snapshots")
    store.write_indices(_index_frame({"Some Other Index": _rising(300)}, dates))
    cal = TradingCalendar([d.date() for d in dates])

    with pytest.raises(DataError, match="no data for benchmark index"):
        stage2_regime.run(store, cal, [], cfg, as_of=dates[-1].date())


def test_benchmark_name_resolves_case_insensitively(tmp_path, cfg):
    """NSE publishes 'Nifty 200'; the config says 'NIFTY 200'.

    This mismatch silently halted Stage 2 the first time it met real data.
    """
    path = _rising()
    dates = _sessions(len(path))
    store = DataStore(tmp_path / "curated", tmp_path / "snapshots")
    store.write_indices(_index_frame({"Nifty 200": path, INDIA_VIX_NAME: _vix()}, dates))

    assert store.resolve_index_name("NIFTY 200") == "Nifty 200"
    assert store.resolve_index_name("  nifty   200  ") == "Nifty 200"
    assert store.resolve_index_name("Nifty 9999") is None
    assert not store.index_series("NIFTY 200").empty


def test_empty_index_series_still_has_a_datetime_index(tmp_path, cfg):
    """Regression: an absent index must be date-sliceable, not a RangeIndex.

    Callers filter these series with ``series.index <= Timestamp``. A default
    RangeIndex makes that raise an opaque TypeError, which turned a missing
    India VIX -- a case Stage 2 is explicitly designed to survive -- into a
    crash.
    """
    store = DataStore(tmp_path / "curated", tmp_path / "snapshots")
    empty = store.index_series("Nothing At All")

    assert empty.empty
    assert isinstance(empty.index, pd.DatetimeIndex)
    assert empty[empty.index <= pd.Timestamp("2026-01-01")].empty


def test_regime_state_is_serialisable(tmp_path, cfg):
    """The contract is the API wire format; it must round-trip."""
    path = _rising()
    store, cal, syms, as_of = _build(tmp_path, path, _vix(), _constituents(path))
    state = stage2_regime.run(store, cal, syms, cfg, as_of=as_of)

    payload = state.model_dump(mode="json")
    assert payload["regime_bucket"] == state.regime_bucket
    assert payload["as_of_date"] == as_of.isoformat()


def test_compatibility_maps_multipliers_to_the_card_line(tmp_path, cfg):
    path = _rising()
    store, cal, syms, as_of = _build(tmp_path, path, _vix(), _constituents(path))
    state = stage2_regime.run(store, cal, syms, cfg, as_of=as_of)

    if state.momentum_multiplier >= 0.9 and not state.transition_flag:
        assert state.compatibility() is RegimeCompatibility.FAVORABLE
