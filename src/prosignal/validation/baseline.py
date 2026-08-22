"""BASELINE_V1 -- the frozen reference every future experiment is measured against.

Recording a baseline is not bookkeeping. Without one, "the new model is better"
is unfalsifiable: the comparison drifts as the code, the config, the universe
and the data all move underneath it. This captures enough to reconstruct the
comparison later, and refuses to record one from a dirty working tree.

The most important field is not a performance number. It is
``independent_observations``: how many non-overlapping forward-return windows
each factor family actually has. Every Sharpe, DSR, PBO and calibration claim
this engine makes is bounded by it, and the three families do not have the same
number -- the value block has roughly a third of the history the price block
has, because its vendor coverage begins later.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

__all__ = ["Baseline", "capture", "load", "BASELINE_PATH"]

BASELINE_PATH = Path("research/BASELINE_V1.json")


@dataclass
class Baseline:
    label: str
    captured_at: str
    git_commit: str
    git_dirty: bool
    engine_version: str
    config_version: str
    config_hash: str
    #: sessions, first/last date, and per-feed coverage
    data: Dict[str, object] = field(default_factory=dict)
    #: family -> {first_live, live_dates, independent_obs}
    independent_observations: Dict[str, object] = field(default_factory=dict)
    universe: Dict[str, object] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    coefficients: Dict[str, float] = field(default_factory=dict)
    stage_logic: Dict[str, object] = field(default_factory=dict)
    costs: Dict[str, object] = field(default_factory=dict)
    validation: Dict[str, object] = field(default_factory=dict)
    metrics: Dict[str, object] = field(default_factory=dict)
    known_limits: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return ""


def _config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def capture(config, *, label: str = "BASELINE_V1",
            allow_dirty: bool = False) -> Baseline:
    """Record the current system. Refuses a dirty tree unless told otherwise.

    A baseline captured from uncommitted code cannot be reproduced, and a
    comparison against something unreproducible is not a comparison.
    """
    dirty = bool(_git("status", "--porcelain"))
    if dirty and not allow_dirty:
        raise ValueError(
            "the working tree has uncommitted changes, so this baseline could "
            "not be reconstructed later. Commit first, or pass allow_dirty=True "
            "and accept that the record is approximate."
        )
    return Baseline(
        label=label,
        captured_at=dt.datetime.now().isoformat(timespec="seconds"),
        git_commit=_git("rev-parse", "HEAD"),
        git_dirty=dirty,
        engine_version="",
        config_version=str(config.version),
        config_hash=_config_hash(Path("config/parameters.yaml")),
    )


def load(path: Path = BASELINE_PATH) -> Optional[Baseline]:
    if not Path(path).is_file():
        return None
    return Baseline(**json.loads(Path(path).read_text(encoding="utf-8")))
