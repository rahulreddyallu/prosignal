"""Mutual exclusion between a writing ingest and a reading analysis.

Individual files are already safe: every write goes to ``.tmp`` and lands via
``os.replace``, which is atomic on POSIX, and a reader holding an open
descriptor keeps reading the old inode. Corruption is not the exposure.

The exposure is ACROSS files. The store is roughly twenty-two of them --
year-partitioned prices, indices, delivery, statements, corporate actions,
sector map -- and an ingest rewrites them one at a time over several minutes.
An analysis starting halfway through sees prices through Friday and delivery
through Tuesday: each file internally valid, the set of them describing no
moment that ever existed. Every feature computed across that boundary is
computed on two different days.

The API cannot hit this because bootstrap and analysis share one job slot. Two
terminals can: `prosignal data ingest` and `prosignal analyse run` have no
relationship at all.

An advisory lock rather than a mandatory one, because the failure it prevents
is worth naming rather than blocking silently. The reader is told what holds
the lock and what to do about it.
"""

from __future__ import annotations

import errno
import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from ..core.errors import DataError

__all__ = ["store_lock", "LOCK_NAME", "StoreBusy"]

LOCK_NAME = ".store.lock"


class StoreBusy(DataError):
    """Another process holds the store lock."""

    code = "STORE_BUSY"


def _lock_path(root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return root / LOCK_NAME


@contextmanager
def store_lock(root: Path, *, exclusive: bool, what: str = "operation",
               blocking: bool = False) -> Iterator[None]:
    """Hold the store lock for the duration of the block.

    ``exclusive`` for a writer (ingest), shared for a reader (analysis). Many
    readers may hold it at once; a writer waits for all of them and excludes
    everything while it holds it.

    Non-blocking by default: an analysis that finds an ingest in progress
    should say so and stop, not sit for the several minutes a full ingest
    takes. Pass ``blocking=True`` where waiting is the right behaviour.
    """
    path = _lock_path(root)
    mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    if not blocking:
        mode |= fcntl.LOCK_NB
    handle = open(path, "a+")
    try:
        try:
            fcntl.flock(handle.fileno(), mode)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            holder = _describe_holder(path)
            raise StoreBusy(
                f"the market-data store is locked by another process{holder}, so "
                f"this {what} would read a store being rewritten underneath it. "
                f"Prices and delivery could land on different days and every "
                f"feature spanning them would be computed across two. Wait for "
                f"the ingest to finish and run this again."
            ) from exc
        if exclusive:
            handle.seek(0)
            handle.truncate()
            handle.write(f"{os.getpid()} {what}\n")
            handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _describe_holder(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return f" ({text})" if text else ""
