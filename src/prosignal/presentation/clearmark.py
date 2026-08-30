"""Hiding past runs from the screen without destroying the research record.

The ledger is not a display cache. It is the permanent record every run is
written to, `fail_run_if_unwritable` is true by design, and the comment on that
setting says why: an unlogged run corrupts the deflated-Sharpe trial count.
Deleting rows to clear a screen would silently invalidate the statistic that
decides whether this strategy is distinguishable from luck.

So clearing sets a watermark instead. The screen shows only runs logged after
it, which is what "start capturing from here" means in practice, and the
record underneath stays whole. It is also reversible, which deletion is not.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Optional

MARK_NAME = ".history-cleared"


def _path(root: Path) -> Path:
    return Path(root) / MARK_NAME


def read_mark(root: Path) -> Optional[str]:
    """The ISO timestamp runs must beat to be shown, or None."""
    path = _path(root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # An unreadable watermark must not hide everything -- failing open
        # shows more than intended, failing closed shows nothing at all.
        return None
    value = data.get("cleared_at")
    return str(value) if value else None


def set_mark(root: Path, when: Optional[dt.datetime] = None) -> str:
    """Hide everything logged up to now. Returns the watermark."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = (when or dt.datetime.now()).isoformat(timespec="seconds")
    path = _path(root)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"cleared_at": stamp}), encoding="utf-8")
    os.replace(str(tmp), str(path))
    return stamp


def clear_mark(root: Path) -> None:
    """Show everything again."""
    path = _path(root)
    if path.is_file():
        path.unlink()


def logged_after(row: dict, stamp: Optional[str]) -> bool:
    """Was this run logged after the watermark?

    THE COMPARISON THAT MATTERS. A run carries two times and only one of them
    is comparable to a watermark: `logged_at` is the wall clock, `date` is the
    market session it scored, and those differ by a weekend every Monday.

    Filtering on `date` emptied the open book permanently. A clear on Sunday
    the 30th wrote `2026-08-30`; the newest run that could exist scored Friday
    the 28th; `"2026-08-28" >= "2026-08-30"` is false, so every run ever
    written was hidden -- including the ones logged minutes after the clear,
    which are the exact runs a clear is supposed to KEEP.

    No `logged_at` means keep. Failing open shows more than intended; failing
    closed shows nothing at all, and nothing at all is the failure this
    function exists to prevent.
    """
    if not stamp:
        return True
    logged = str(row.get("logged_at") or "")
    return (not logged) or logged > str(stamp)


def kept(rows, stamp: Optional[str]) -> list:
    """The rows a clear did not hide, by the rule above."""
    if not stamp:
        return list(rows)
    return [r for r in rows if logged_after(r, stamp)]
