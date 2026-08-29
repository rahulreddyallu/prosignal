"""Research epochs: the identity a result belongs to, and when it ends.

WHY THIS EXISTS. The September remediation changed the training label, the
family construction, a feature's lookback and the panel's population, and then
the forward test opened on 2026-08-27 was still described as running. It was
not: the engine's own `progress()` reports it INVALID because the config moved
underneath it. The repair log even predicted that, and the re-registration was
never done.

The deeper problem is that "the engine" was being treated as a continuing thing
that gets better, when a forward test is a test of one specific model on one
specific data state under one specific configuration. Change any of them and the
observations before and after are not the same experiment. The engine had no
way to SAY that, so it kept a single forward ledger across a moving target.

An epoch is that missing noun. It binds, in one immutable record:

    code          the sources that decide a ranking (`modelprint`)
    config        parameters.yaml, by its own hash
    data          the curated store, by its committed manifest digest
    features      the feature set and its family construction
    universe      the admission policy, by name and version
    execution     the cost and liquidity policy, by name and version

An epoch is OPEN, or it is closed with a reason and a successor. Closing one is
the honest alternative to quietly continuing a broken window: the observations
already collected keep their meaning inside the epoch that produced them, and
the next experiment starts with a clean identity instead of inheriting a
contaminated one.

WHAT THIS DOES NOT DO. It does not decide when to open a new epoch, and it does
not open one automatically on drift. Detecting drift and deciding to start a new
eighteen-month clock are different acts, and the second belongs to whoever holds
the capital. `drifted_from` reports; a person opens.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core.logging import get_logger

__all__ = [
    "LEDGER_NAME", "UNIVERSE_POLICY", "EXECUTION_MODEL",
    "Epoch", "Identity", "current_identity", "load_all", "open_epoch",
    "close_epoch", "active", "drifted_from", "STATUS_OPEN", "STATUS_VOID",
    "STATUS_SUPERSEDED",
]

log = get_logger(__name__)

LEDGER_NAME = "epochs.jsonl"

STATUS_OPEN = "OPEN"
STATUS_VOID = "VOID"
STATUS_SUPERSEDED = "SUPERSEDED"

#: Named, versioned policies. These are STRINGS ON PURPOSE. A hash of the code
#: would change when a comment moved; a version a person bumps changes when the
#: POLICY changes, which is the thing a reader of an old result needs to know.
#: Bumping one without changing behaviour is a lie, and leaving one alone after
#: changing behaviour is the defect this whole exercise is about -- so each is
#: pinned by a test that describes what the current version means.
UNIVERSE_POLICY = "pit-liquidity-v1+admissible-r9"
EXECUTION_MODEL = "statutory-india-v1+liquidity-gate-r13"


def _git_sha(root: Path) -> str:
    """HEAD, or `"unversioned"`. Never a guess."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                             capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        return sha[:12] if out.returncode == 0 and len(sha) >= 12 else "unversioned"
    except (OSError, subprocess.SubprocessError):
        return "unversioned"


#: Tracked paths the epoch machinery WRITES WHILE OPENING AN EPOCH. They are
#: provenance, not code, and counting them made the reproducibility gate
#: unsatisfiable by construction: `open_production_epoch.sh` re-manifests the
#: store (step 3) and appends the epoch row (step 4) before the gate is checked
#: (step 5), so the tree was always dirty and the forward-test restart was
#: always refused -- on a tree whose CODE was committed and clean. A gate that
#: can never pass is a gate somebody eventually routes around, which is worse
#: than not having one.
#:
#: `MANIFEST.json` in particular carries a `built_at` timestamp, so re-running
#: the same command on unchanged data produces a diff every time.
_PROVENANCE_PATHS = ("data/ledger/", "data/curated/MANIFEST.json")


def _dirty(root: Path) -> Optional[bool]:
    """Whether the working tree has uncommitted CODE changes, or None if unknown.

    An epoch opened from a dirty tree names a commit that does not contain the
    code that ran. That is not automatically wrong -- it is how research is
    done -- but it must be recorded, because `code_sha` otherwise reads as a
    promise it cannot keep.

    Provenance files the epoch machinery writes as part of opening are excluded
    (see `_PROVENANCE_PATHS`); `provenance_uncommitted` reports them separately
    so nothing is hidden, it is just not confused with a code change.
    """
    d = _status(root)
    return None if d is None else d["code_dirty"]


def _status(root: Path) -> Optional[dict]:
    """Split `git status --porcelain` into code changes and provenance writes."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=str(root),
                             capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
    except (OSError, subprocess.SubprocessError):
        return None
    code, prov = [], []
    for line in out.stdout.splitlines():
        path = line[3:].strip().strip('"')
        if "->" in path:                      # a rename: judge the destination
            path = path.split("->")[-1].strip().strip('"')
        (prov if path.startswith(_PROVENANCE_PATHS) else code).append(path)
    return {"code_dirty": bool(code), "code_paths": code,
            "provenance_uncommitted": bool(prov), "provenance_paths": prov}


def _feature_schema_sha() -> str:
    """The feature set and its family construction, as one hash.

    Adding a factor, retiring one, or moving a member between families changes
    what the model can express. None of that touches `parameters.yaml`.
    """
    try:
        from ..features.crosssec import FEATURES, NEUTRAL_WHEN_MISSING
        from ..features import crossmodel as cm

        families = getattr(cm, "FAMILIES", None) or getattr(cm, "THEMES", None) or {}
        # WHAT ACTUALLY RANKS THE BOOK belongs in the fingerprint. Under
        # `ranking.source: v2_composite` that is `features/v2.py` -- its factor
        # names, signs, weights and lookbacks -- and hashing only `crosssec`
        # would let the shipped factor set change without the epoch noticing,
        # which is the one thing this hash exists to prevent.
        from ..features.v2 import V2_FACTORS

        payload = {
            "features": {k: v[0] for k, v in sorted(FEATURES.items())},
            "v2": [[f.name, f.sign, f.weight, f.lookback] for f in V2_FACTORS],
            "neutral_when_missing": sorted(NEUTRAL_WHEN_MISSING),
            "families": {k: sorted(v) for k, v in sorted(
                (families or {}).items())} if isinstance(families, dict) else str(families),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    except Exception:                                     # pragma: no cover
        return "unknown"


@dataclass(frozen=True)
class Identity:
    """Everything that decides what the engine produces, hashed together."""

    code_sha: str
    code_dirty: Optional[bool]
    model_sources_sha: str
    config_version: str
    data_manifest_sha: str
    feature_schema_sha: str
    universe_policy: str
    execution_model: str

    def fingerprint(self) -> str:
        payload = json.dumps({
            # `code_dirty` is deliberately OUT of the fingerprint: it describes
            # the tree an epoch was opened from, not the model. Two clean runs
            # of the same commit must fingerprint identically.
            "code_sha": self.code_sha,
            "model_sources_sha": self.model_sources_sha,
            "config_version": self.config_version,
            "data_manifest_sha": self.data_manifest_sha,
            "feature_schema_sha": self.feature_schema_sha,
            "universe_policy": self.universe_policy,
            "execution_model": self.execution_model,
        }, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def differences(self, other: "Identity") -> List[str]:
        out = []
        for f in ("code_sha", "model_sources_sha", "config_version",
                  "data_manifest_sha", "feature_schema_sha", "universe_policy",
                  "execution_model"):
            a, b = getattr(self, f), getattr(other, f)
            if a != b:
                out.append(f"{f}: {b} -> {a}")
        return out


def current_identity(cfg) -> Identity:
    """What the engine is right now."""
    from ..data.manifest import digest_of
    from ..modelprint import source_digest

    root = Path(cfg.paths.root)
    return Identity(
        code_sha=_git_sha(root),
        code_dirty=_dirty(root),
        model_sources_sha=source_digest(),
        config_version=str(getattr(cfg, "version", "")),
        data_manifest_sha=digest_of(cfg.paths.curated),
        feature_schema_sha=_feature_schema_sha(),
        universe_policy=UNIVERSE_POLICY,
        execution_model=EXECUTION_MODEL,
    )


@dataclass(frozen=True)
class Epoch:
    epoch_id: str
    label: str
    opened_on: str
    status: str
    identity: Dict[str, object]
    note: str = ""
    closed_on: str = ""
    close_reason: str = ""
    superseded_by: str = ""

    @property
    def is_open(self) -> bool:
        return self.status == STATUS_OPEN

    def ident(self) -> Identity:
        keep = {k: self.identity.get(k) for k in
                ("code_sha", "code_dirty", "model_sources_sha", "config_version",
                 "data_manifest_sha", "feature_schema_sha", "universe_policy",
                 "execution_model")}
        keep.setdefault("code_dirty", None)
        return Identity(**keep)                       # type: ignore[arg-type]

    def summary(self) -> str:
        state = self.status
        if self.close_reason:
            state += f" ({self.close_reason})"
        return (f"{self.epoch_id}  {self.label}  opened {self.opened_on}  "
                f"{state}")


def _path(ledger_root: Path) -> Path:
    return Path(ledger_root) / LEDGER_NAME


def load_all(ledger_root: Path) -> List[Epoch]:
    """Every epoch, oldest first. Later lines supersede earlier ones by id.

    Append-only on disk: closing an epoch writes a new line rather than
    rewriting the old one, so the history of what was believed and when is
    itself recoverable.
    """
    path = _path(ledger_root)
    if not path.is_file():
        return []
    order: List[str] = []
    latest: Dict[str, Epoch] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            blob = json.loads(line)
            e = Epoch(
                epoch_id=str(blob["epoch_id"]), label=str(blob.get("label", "")),
                opened_on=str(blob.get("opened_on", "")),
                status=str(blob.get("status", STATUS_OPEN)),
                identity=dict(blob.get("identity", {})),
                note=str(blob.get("note", "")),
                closed_on=str(blob.get("closed_on", "")),
                close_reason=str(blob.get("close_reason", "")),
                superseded_by=str(blob.get("superseded_by", "")))
        except (ValueError, KeyError, TypeError):
            log.warning("unreadable epoch line; skipped",
                        extra={"path": str(path)})
            continue
        if e.epoch_id not in latest:
            order.append(e.epoch_id)
        latest[e.epoch_id] = e
    return [latest[k] for k in order]


def _append(ledger_root: Path, epoch: Epoch) -> None:
    path = _path(ledger_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(epoch), sort_keys=True) + "\n")


def active(ledger_root: Path) -> Optional[Epoch]:
    for e in reversed(load_all(ledger_root)):
        if e.is_open:
            return e
    return None


def open_epoch(ledger_root: Path, cfg, *, label: str, note: str = "",
               identity: Optional[Identity] = None,
               allow_while_open: bool = False) -> Epoch:
    """Start a new epoch against the engine as it stands.

    Refuses while another is open unless told otherwise, because two open
    epochs mean no ledger row can say which experiment it belongs to.
    """
    if not allow_while_open:
        live = active(ledger_root)
        if live is not None:
            raise ValueError(
                f"epoch {live.epoch_id} is still OPEN. Close it with a reason "
                f"before opening another -- two open epochs leave every ledger "
                f"row ambiguous about which experiment it belongs to."
            )
    ident = identity or current_identity(cfg)
    today = dt.date.today().isoformat()
    e = Epoch(epoch_id=f"{today}-{ident.fingerprint()}", label=label,
              opened_on=today, status=STATUS_OPEN,
              identity=asdict(ident), note=note)
    _append(ledger_root, e)
    log.info("epoch opened", extra={"epoch_id": e.epoch_id, "label": label})
    return e


def close_epoch(ledger_root: Path, epoch_id: str, *, reason: str,
                status: str = STATUS_SUPERSEDED,
                superseded_by: str = "") -> Epoch:
    """Close an epoch with a stated reason. Never deletes it."""
    if status not in (STATUS_VOID, STATUS_SUPERSEDED):
        raise ValueError(f"an epoch closes VOID or SUPERSEDED, not {status!r}")
    if not str(reason).strip():
        raise ValueError(
            "an epoch cannot be closed without a reason. The reason is the "
            "whole value of the record: an epoch that ends without one is "
            "indistinguishable from an experiment that was abandoned because "
            "it was going badly."
        )
    for e in load_all(ledger_root):
        if e.epoch_id == epoch_id:
            closed = Epoch(
                epoch_id=e.epoch_id, label=e.label, opened_on=e.opened_on,
                status=status, identity=e.identity, note=e.note,
                closed_on=dt.date.today().isoformat(), close_reason=reason,
                superseded_by=superseded_by)
            _append(ledger_root, closed)
            log.info("epoch closed", extra={"epoch_id": epoch_id,
                                            "status": status})
            return closed
    raise KeyError(f"no epoch {epoch_id!r} in {_path(ledger_root)}")


def drifted_from(ledger_root: Path, cfg) -> Tuple[Optional[Epoch], List[str]]:
    """The open epoch and how the engine differs from it. Reports; never acts.

    An empty list means the engine is running the experiment it says it is.
    Anything else means the forward test in flight is measuring two things.
    """
    live = active(ledger_root)
    if live is None:
        return None, ["no epoch is open; nothing identifies what is running"]
    return live, current_identity(cfg).differences(live.ident())
