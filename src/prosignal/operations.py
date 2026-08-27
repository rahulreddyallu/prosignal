"""Operator actions the interface can take, and the guards on each.

Three of these are destructive and one changes what the forward test is
measuring, so each records WHY it happened and what it cost. The record is
the point: a gap in the observation record that nobody can explain is
indistinguishable from a gap that was hidden.

On pausing. The cron entry lives in /etc/cron.d/prosignal, owned by root,
and the service runs as a non-root user -- so the API cannot edit it and
should not be given the privilege to. Pausing therefore writes a flag the
run script checks and exits on. That is weaker than stopping cron (the job
still wakes) and it is also honest: the schedule is untouched, the
observation is declined, and the decline is written down.

On resetting. "Clear the market data" and "erase the record" are different
actions with different blast radius, and collapsing them into one button is
how a person loses evidence they meant to keep. The store can be rebuilt
from NSE in an afternoon. The ledger cannot be rebuilt at all -- it is the
only record of what the engine said on a date that has passed.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "PAUSE_FILE", "OPS_LOG", "PRESERVE_ON_RESET",
    "pause_state", "pause", "resume",
    "reset_market_data", "erase_everything",
    "operations_log",
]

PAUSE_FILE = "cron.paused"
OPS_LOG = "operations.jsonl"

def _ledger(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root


def _log(ledger_root: Path, action: str, detail: Dict[str, Any]) -> None:
    """Append-only. Never rewritten, so a reset cannot erase the note saying
    a reset happened -- the note is written after the deletion completes."""
    path = _ledger(ledger_root) / OPS_LOG
    row = {"at": dt.datetime.now().isoformat(timespec="seconds"),
           "action": action, **detail}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def operations_log(ledger_root: Path, limit: int = 50) -> List[Dict[str, Any]]:
    path = Path(ledger_root) / OPS_LOG
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # a torn final line is not a reason to fail
    return rows[-limit:][::-1]


# ---------------------------------------------------------------------------
# Pausing the scheduled observation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PauseState:
    paused: bool
    since: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def pause_state(ledger_root: Path) -> PauseState:
    path = Path(ledger_root) / PAUSE_FILE
    if not path.exists():
        return PauseState(paused=False)
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # The file existing is the signal. An unreadable one still pauses:
        # failing open here would silently resume a run the operator stopped.
        return PauseState(paused=True)
    return PauseState(paused=True, since=d.get("since"), reason=d.get("reason"))


def pause(ledger_root: Path, reason: str = "") -> PauseState:
    root = _ledger(Path(ledger_root))
    since = dt.datetime.now().isoformat(timespec="seconds")
    payload = {"since": since, "reason": reason or "paused from the interface"}
    tmp = root / (PAUSE_FILE + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tmp.replace(root / PAUSE_FILE)
    _log(root, "pause", payload)
    return PauseState(paused=True, since=since, reason=payload["reason"])


def resume(ledger_root: Path) -> PauseState:
    root = _ledger(Path(ledger_root))
    prior = pause_state(root)
    (root / PAUSE_FILE).unlink(missing_ok=True)
    detail: Dict[str, Any] = {"was_paused_since": prior.since}
    if prior.since:
        try:
            began = dt.datetime.fromisoformat(prior.since)
            detail["paused_days"] = (dt.datetime.now() - began).days
        except ValueError:
            pass
    _log(root, "resume", detail)
    return PauseState(paused=False)


# ---------------------------------------------------------------------------
# Resetting
# ---------------------------------------------------------------------------

#: Things that live in the curated directory and are NOT market data.
#:
#: `curated` holds the price store, which NSE will serve again on request, and
#: it also holds two files that no amount of re-ingesting reconstructs:
#:
#:   trial_registry.jsonl      every research configuration ever tried. It is
#:                             the Deflated Sharpe Ratio's multiple-testing
#:                             input, so losing it does not merely forget the
#:                             count -- it silently LOWERS the bar the strategy
#:                             has to clear, which is the direction that
#:                             flatters.
#:   crosssec_model_versions/  the archive the refit gate keeps so a bad refit
#:                             is recoverable. Deleting it removes the recovery
#:                             path at the moment it is most needed.
#:
#: The reset's own docstring promised "the record of what the engine SAID" was
#: kept, and it was -- the ledger is elsewhere. These two were counted as
#: market data because of where they happened to sit.
PRESERVE_ON_RESET = ("trial_registry.jsonl", "crosssec_model_versions")


def _wipe(path: Path, preserve: tuple = ()) -> int:
    """Remove a directory's contents and report how many files went.

    Anything named in ``preserve`` is carried across the wipe. Copied out to a
    temporary directory and back rather than deleted around, so a failure
    part-way leaves the originals on disk rather than half of them.
    """
    if not path.exists():
        return 0
    n = sum(1 for p in path.rglob("*") if p.is_file())
    keep = [name for name in preserve if (path / name).exists()]
    if not keep:
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        return n

    with tempfile.TemporaryDirectory() as hold:
        holding = Path(hold)
        for name in keep:
            src = path / name
            if src.is_dir():
                shutil.copytree(src, holding / name)
            else:
                shutil.copy2(src, holding / name)
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        for name in keep:
            src = holding / name
            if src.is_dir():
                shutil.copytree(src, path / name)
            else:
                shutil.copy2(src, path / name)
    n -= sum(1 for p in path.rglob("*") if p.is_file())
    return max(n, 0)


def reset_market_data(paths: Any) -> Dict[str, Any]:
    """Clear the price store so the build can run again.

    Keeps the ledger, the resolved outcomes and the forward registration.
    Those describe what the engine SAID, which no amount of re-ingesting
    reconstructs; the store describes what the market DID, which NSE will
    serve again on request.

    Also keeps the two things inside `curated` that are not market data: the
    trial registry, which is the Deflated Sharpe's multiple-testing input, and
    the model version archive, which is the refit gate's recovery path. Both
    were being destroyed by a button whose label promises the record is kept.
    """
    removed = {
        "curated": _wipe(Path(paths.curated), PRESERVE_ON_RESET),
        "snapshots": _wipe(Path(paths.snapshots)),
        "cache": _wipe(Path(paths.cache)),
        "raw": _wipe(Path(paths.raw)),
    }
    detail = {"scope": "market_data", "files_removed": removed,
              "kept": ["ledger", "outcomes", "forward_registration",
                       *PRESERVE_ON_RESET]}
    _log(Path(paths.ledger), "reset_market_data", detail)
    return detail


def erase_everything(paths: Any) -> Dict[str, Any]:
    """Market data AND the entire record. Irreversible in the way that
    matters: the run history cannot be rebuilt from any external source."""
    ledger_root = Path(paths.ledger)
    # Count before deleting, and keep the operations log itself -- the note
    # that an erase happened must survive the erase.
    kept_log = operations_log(ledger_root, limit=10_000)
    removed = {
        "curated": _wipe(Path(paths.curated)),
        "snapshots": _wipe(Path(paths.snapshots)),
        "cache": _wipe(Path(paths.cache)),
        "raw": _wipe(Path(paths.raw)),
        "ledger": _wipe(ledger_root),
    }
    with (ledger_root / OPS_LOG).open("w", encoding="utf-8") as fh:
        for row in reversed(kept_log):
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    detail = {"scope": "everything", "files_removed": removed,
              "kept": ["operations_log"]}
    _log(ledger_root, "erase_everything", detail)
    return detail
