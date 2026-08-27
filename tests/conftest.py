"""Shared pytest fixtures.

Every test here is offline and deterministic. Network-dependent behaviour is
exercised by `prosignal data ingest` against the live archives, not by the unit
suite -- a test that silently depends on NSE being up is a test that fails for
reasons unrelated to the code.
"""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
import yaml

from prosignal.config.loader import load_config, reset_config_cache
from prosignal.core.paths import CONFIG_RELPATH, find_project_root
from prosignal.data.types import DATE, SYMBOL, coerce_ohlcv


@pytest.fixture(scope="session")
def project_root() -> Path:
    return find_project_root()


@pytest.fixture(scope="session")
def baseline_yaml(project_root: Path) -> Dict[str, Any]:
    with (project_root / CONFIG_RELPATH).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture
def tmp_project(tmp_path: Path, project_root: Path) -> Path:
    """An isolated project root with the real parameters.yaml copied in."""
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    shutil.copy(project_root / CONFIG_RELPATH, tmp_path / CONFIG_RELPATH)
    return tmp_path


@pytest.fixture
def cfg(tmp_project: Path):
    reset_config_cache()
    config = load_config(project_root=tmp_project, use_cache=False)
    yield config
    reset_config_cache()


def write_config(root: Path, payload: Dict[str, Any]) -> Path:
    path = root / CONFIG_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    return path


# =============================================================================
# synthetic market data
# =============================================================================


def make_sessions(n: int, end: dt.date = dt.date(2026, 8, 14)) -> list:
    """`n` weekday sessions ending at `end` (weekends skipped)."""
    out = []
    day = end
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day -= dt.timedelta(days=1)
    return sorted(out)


@pytest.fixture
def sessions() -> list:
    return make_sessions(300)


def synthetic_prices(
    symbols,
    sessions,
    start_price: float = 100.0,
    drift: float = 0.0004,
    vol: float = 0.012,
    seed: int = 7,
) -> pd.DataFrame:
    """A deterministic geometric-random-walk panel in the canonical schema."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, sym in enumerate(symbols):
        px = start_price * (1.0 + 0.1 * i)
        prev = px
        for day in sessions:
            ret = rng.normal(drift, vol)
            close = max(prev * (1.0 + ret), 1.0)
            high = max(close, prev) * (1.0 + abs(rng.normal(0, 0.003)))
            low = min(close, prev) * (1.0 - abs(rng.normal(0, 0.003)))
            open_ = float(np.clip(prev * (1.0 + rng.normal(0, 0.004)), low, high))
            volume = float(rng.integers(80_000, 900_000))
            rows.append(
                {
                    DATE: pd.Timestamp(day),
                    SYMBOL: sym,
                    "series": "EQ",
                    "isin": f"INE{i:06d}01010",
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "prev_close": prev,
                    "last": close,
                    "volume": volume,
                    "turnover": volume * close,
                    "trades": float(rng.integers(500, 20_000)),
                }
            )
            prev = close
    return coerce_ohlcv(pd.DataFrame(rows), source="synthetic")


@pytest.fixture
def prices(sessions) -> pd.DataFrame:
    return synthetic_prices(["AAA", "BBB", "CCC"], sessions)


@pytest.fixture(scope="session")
def live_cfg(tmp_path_factory):
    """The REAL project store, with every WRITE path redirected to a temp dir.

    The store is real because that is the point: these tests exercise the
    pipeline against actual market data. The ledger is NOT, and it used to be.

    `run_analysis` appends a permanent row to `paths.ledger` on every call, so
    each pipeline test wrote into the production research record. That record is
    not a log:

      * `Ledger.trial_count()` is the multiple-testing input to the Deflated
        Sharpe Ratio, so a test run raised the penalty applied to every real
        result;
      * `validation.forward.progress()` counts rows by market date, so test
        runs were being counted as forward-test observations;
      * the next real run reads the newest row back as its open book, so a
        test's book could seed production hysteresis.

    Measured on this repository: 41 rows were written into the research ledger
    by test runs in a single afternoon.

    Redirecting the write paths keeps the tests end-to-end and keeps the record
    clean. `curated` and `snapshots` stay real and are only read.
    """
    from pathlib import Path as _P

    from prosignal.config import load_config as _load
    from prosignal.data.store import DataStore as _DS

    root = _P(__file__).resolve().parents[1]
    cfg = _load(project_root=root, use_cache=False)
    store = _DS(cfg.paths.curated, cfg.paths.snapshots)
    if not store.price_sessions():
        pytest.skip("no ingested data; run `prosignal data ingest --full`")

    sandbox = tmp_path_factory.mktemp("live_cfg_writes")
    for attr in ("ledger", "logs", "raw", "cache"):
        target = sandbox / attr
        target.mkdir(parents=True, exist_ok=True)
        setattr(cfg.paths, attr, target)
    return cfg


@pytest.fixture
def tmp_store_config(tmp_path):
    """A config pointed at an empty store, for manifest checks."""
    import datetime as dt

    import pandas as pd

    from prosignal.config.loader import load_config
    from prosignal.data.store import DataStore

    config = load_config()
    curated = tmp_path / "curated"
    snapshots = tmp_path / "snapshots"
    curated.mkdir(parents=True, exist_ok=True)
    snapshots.mkdir(parents=True, exist_ok=True)
    store = DataStore(curated, snapshots)
    # one session so TradingCalendar can be built
    sessions = [dt.date(2026, 1, 5), dt.date(2026, 1, 6)]
    frame = pd.DataFrame({
        "date": pd.to_datetime(sessions), "symbol": ["AAA", "AAA"],
        "series": ["EQ", "EQ"], "open": [10.0, 11.0], "high": [11.0, 12.0],
        "low": [9.0, 10.0], "close": [10.5, 11.5], "volume": [1000, 1000],
        "turnover": [10500.0, 11500.0], "deliv_pct": [40.0, 41.0],
    })
    store.prices.write(frame)
    return config, store, sessions[-1]


@pytest.fixture
def runnable_cfg(live_cfg):
    """`live_cfg`, and the store is fresh enough for the pipeline to run.

    Stage 1 halts market-wide on a required feed older than
    `max_age_sessions`, so every test that calls `run_analysis` against the
    real store goes red the moment the local store falls a session behind --
    which it does by itself, every day nobody ingests. Nine did exactly that
    on 2026-08-27 against a store ending 2026-08-25, all with the identical
    MarketWideHalt.

    A suite that turns red for reasons unrelated to the code stops being a
    signal about the code, and the habit it teaches -- ignoring the failures --
    is the one you least want. The refusal itself is correct and is asserted
    directly, on purpose, in `test_ready_freshness` and
    `test_staleness_and_ingest`. These tests are about stage discipline and
    need a store they can actually run on.

    Only for tests that execute the pipeline. Tests that merely read the real
    store keep using `live_cfg` and keep running on a stale one.
    """
    from prosignal.core.clock import market_today
    from prosignal.data.store import DataStore
    from prosignal.pipeline import _sessions_behind

    sessions = DataStore(live_cfg.paths.curated, live_cfg.paths.snapshots).price_sessions()
    behind = _sessions_behind(sessions[-1], market_today(live_cfg))
    limit = int(live_cfg.params.feeds["equity_ohlcv"].max_age_sessions)
    if behind > limit:
        pytest.skip(
            f"the local store ends {sessions[-1]}, {behind} weekday(s) behind "
            f"{market_today(live_cfg)} against a tolerance of {limit} -- the "
            f"pipeline halts at Stage 1. Run `prosignal data ingest`. This is "
            f"the engine refusing stale data, not a defect."
        )
    return live_cfg
