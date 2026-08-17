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
