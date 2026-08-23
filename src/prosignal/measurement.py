"""Measurement periods: start one, stop it, measure only inside it.

The pre-registered forward test answers one question over a fixed 18 months
and refuses to be reopened, which is what makes its p-value worth anything.
It is also the wrong instrument for someone who wants to watch for three
months, change something, and watch again -- that person needs a clock they
can start and stop, and a guarantee that the evidence from before a change
never gets pooled with the evidence from after it.

That guarantee is the whole point of this module. A period pins the config
hash and the commit it started under. Performance is filtered to the period,
so a run scored under different coefficients cannot silently join the same
average. If the config moves while a period is open the period is marked
DRIFTED and says so, rather than continuing to report a number that now
describes two different models.

Starting a period after a change IS the re-registration: new id, new
fingerprint, new baseline, and the old period preserved beside it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["Period", "PERIODS_FILE", "start", "stop", "active", "periods",
           "annotate", "period_of"]

PERIODS_FILE = "measurement.jsonl"


@dataclass
class Period:
    id: str
    started: str                     # ISO date of the first day it counts
    config_version: str
    engine_version: str
    git_commit: str
    fingerprint: str
    label: str = ""
    ended: Optional[str] = None
    #: Set when the config changed underneath an open period.
    drifted_to: Optional[str] = None

    @property
    def open(self) -> bool:
        return self.ended is None

    @property
    def status(self) -> str:
        if self.drifted_to:
            return "DRIFTED"
        return "RUNNING" if self.open else "CLOSED"

    def covers(self, iso_date: str) -> bool:
        if not iso_date:
            return False
        d = str(iso_date)[:10]
        if d < self.started[:10]:
            return False
        return self.ended is None or d <= self.ended[:10]

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "status": self.status, "open": self.open}


def _path(root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return root / PERIODS_FILE


def _fingerprint(config_version: str, git_commit: str, started: str) -> str:
    """Identifies WHAT was being measured, so two periods under the same
    config and commit are distinguishable only by when they began."""
    blob = "|".join([config_version, git_commit, started])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _read(root: Path) -> List[Period]:
    path = _path(root)
    if not path.exists():
        return []
    out: List[Period] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue                  # a torn line is not a reason to fail
        d.pop("status", None)
        d.pop("open", None)
        try:
            out.append(Period(**d))
        except TypeError:
            continue
    return out


def _write(root: Path, rows: List[Period]) -> None:
    path = _path(root)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r), sort_keys=True) + "\n")
    tmp.replace(path)


def periods(root: Path) -> List[Period]:
    """Newest first."""
    return sorted(_read(root), key=lambda p: p.started, reverse=True)


def active(root: Path, *, config_version: str = "") -> Optional[Period]:
    """The open period, if there is one.

    If a config_version is supplied and differs from the one the period was
    opened under, the period is marked DRIFTED here rather than at write
    time -- the change can happen while nothing is running, so the check has
    to be on read.
    """
    rows = _read(root)
    open_rows = [p for p in rows if p.open]
    if not open_rows:
        return None
    current = sorted(open_rows, key=lambda p: p.started)[-1]
    if (config_version and current.config_version
            and config_version != current.config_version
            and not current.drifted_to):
        current.drifted_to = config_version
        _write(root, [current if r.id == current.id else r for r in rows])
    return current


def start(root: Path, *, config_version: str, engine_version: str = "",
          git_commit: str = "", label: str = "",
          today: Optional[dt.date] = None) -> Period:
    """Open a period. Any period still open is closed first.

    Two open periods would make "which one does this run belong to"
    ambiguous, and an ambiguous answer to that question is the one thing a
    measurement period exists to prevent.
    """
    now = (today or dt.date.today()).isoformat()
    rows = _read(root)
    for r in rows:
        if r.open:
            r.ended = now
    p = Period(
        id=uuid.uuid4().hex[:12],
        started=now,
        config_version=config_version,
        engine_version=engine_version,
        git_commit=git_commit,
        fingerprint=_fingerprint(config_version, git_commit, now),
        label=label,
    )
    rows.append(p)
    _write(root, rows)
    return p


def stop(root: Path, *, today: Optional[dt.date] = None) -> Optional[Period]:
    now = (today or dt.date.today()).isoformat()
    rows = _read(root)
    closed = None
    for r in rows:
        if r.open:
            r.ended = now
            closed = r
    if closed is not None:
        _write(root, rows)
    return closed


def annotate(root: Path, period_id: str, label: str) -> Optional[Period]:
    rows = _read(root)
    hit = None
    for r in rows:
        if r.id == period_id:
            r.label = label
            hit = r
    if hit is not None:
        _write(root, rows)
    return hit


def period_of(root: Path, iso_date: str) -> Optional[Period]:
    for p in periods(root):
        if p.covers(iso_date):
            return p
    return None
