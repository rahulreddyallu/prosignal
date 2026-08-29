"""A content-addressed description of the store, so a result can name its data.

WHY THIS EXISTS. `data/` is not in version control and cannot sensibly be put
there: the curated store is a quarter of a gigabyte of parquet and it grows
every session. But the September dossier's own closing finding was that every
panel-derived figure in it -- the recovered corporate-action cells, the sector
coverage, the whole of its section F -- is "internally consistent and externally
unchecked without the store". A number nobody can tie to a dataset is not a
result, it is an anecdote.

The fix is not to commit the data. It is to commit a MANIFEST: for every file,
its SHA-256, its size, its row count, the dates it spans and a hash of its
schema. The manifest is small, it goes in git, and its own digest is a name for
the exact store that produced a figure. Two runs quoting the same
`data_manifest_sha` read the same bytes; two quoting different ones did not,
whatever else they agree about.

WHAT IS DELIBERATELY EXCLUDED. Lock files, caches and `_state.json` change on
every read and describe the store's housekeeping rather than its content. A
digest that moved when nothing had been ingested would be ignored within a week,
which is how `cumulative_trials_logged` ended up shipping at zero.

`trial_registry.jsonl` is excluded for the same reason and it is the harder
call, because it is not housekeeping -- it is the record of every configuration
this engine has been compared at, and it feeds the Deflated Sharpe directly. It
sits under `curated/` only because that is where the engine keeps append-only
ledgers, and it is written by the RESEARCH commands, not by ingestion. Leaving
it in meant the data identity changed every time a hypothesis was recorded: the
readiness gate reported the market data as drifted when nothing about the market
data had moved. It has its own append-only discipline, its own merge-by-id rule
and its own tests. A manifest that flags a research note as data corruption
teaches its reader to ignore it.

The fitted model cache is excluded on the same principle: `crosssec_model.json`
is an OUTPUT of the store, refitted on a schedule, and a digest that moved every
refit would say the data had changed when the model had.

THE DIGEST IS OVER CONTENT, NOT OVER THE FILESYSTEM. `built_at` and each file's
mtime are recorded because they are useful to a person, and are excluded from
the digest because they are not facts about the data. Re-running the manifest on
an untouched store returns the same digest.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.logging import get_logger

__all__ = [
    "MANIFEST_NAME", "EXCLUDED_SUFFIXES", "EXCLUDED_NAMES",
    "FileRecord", "Manifest", "build", "load", "write", "verify", "digest_of",
]

log = get_logger(__name__)

MANIFEST_NAME = "MANIFEST.json"

#: Housekeeping, not content. See the module docstring.
EXCLUDED_SUFFIXES = (".lock", ".tmp", ".log")
EXCLUDED_NAMES = frozenset({MANIFEST_NAME, "_state.json", ".store.lock",
                            ".DS_Store",
                            # Research ledger and model output, not data. See
                            # the module docstring -- both change without the
                            # market data changing, and a manifest that cries
                            # wolf is a manifest nobody reads.
                            "trial_registry.jsonl", "crosssec_model.json"})

#: Read in 8 MiB blocks so a 30 MB parquet does not become a 30 MB string.
_CHUNK = 8 * 1024 * 1024


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(_CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class FileRecord:
    path: str
    bytes: int
    sha256: str
    #: Parquet only, and absent when the file cannot be read as a table. A
    #: manifest that silently recorded zero rows for an unreadable file would
    #: verify happily against a corrupt store.
    rows: Optional[int] = None
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    schema_sha: Optional[str] = None
    mtime_ms: Optional[int] = None

    def content(self) -> Dict[str, object]:
        """The fields the digest is taken over -- everything except mtime."""
        d = {k: v for k, v in asdict(self).items() if k != "mtime_ms"}
        return {k: v for k, v in d.items() if v is not None}


def _describe_parquet(path: Path) -> Dict[str, object]:
    """Rows, span and schema, or nothing if it is not a readable table."""
    try:
        import pyarrow.parquet as pq
    except ImportError:                                   # pragma: no cover
        return {}
    try:
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        out: Dict[str, object] = {
            "rows": int(pf.metadata.num_rows),
            "schema_sha": hashlib.sha256(
                json.dumps([[f.name, str(f.type)] for f in schema],
                           sort_keys=False).encode("utf-8")
            ).hexdigest()[:16],
        }
        for candidate in ("date", "DATE", "session_date"):
            if candidate in schema.names:
                col = pq.read_table(path, columns=[candidate])[candidate]
                if len(col):
                    import pandas as pd

                    s = pd.to_datetime(col.to_pandas(), errors="coerce").dropna()
                    if len(s):
                        out["min_date"] = str(s.min().date())
                        out["max_date"] = str(s.max().date())
                break
        return out
    except Exception as exc:                              # pragma: no cover
        log.warning("manifest: unreadable parquet",
                    extra={"path": str(path), "error": str(exc)})
        return {}


@dataclass
class Manifest:
    root: str
    built_at: str
    files: List[FileRecord] = field(default_factory=list)
    #: Set on load. Recomputed rather than trusted, so a hand-edited manifest
    #: cannot claim a digest it does not have.
    digest: str = ""

    def compute_digest(self) -> str:
        payload = json.dumps([f.content() for f in sorted(self.files,
                                                          key=lambda r: r.path)],
                             sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, object]:
        return {
            "root": self.root,
            "built_at": self.built_at,
            "digest": self.compute_digest(),
            "n_files": len(self.files),
            "total_bytes": sum(f.bytes for f in self.files),
            "files": [asdict(f) for f in sorted(self.files, key=lambda r: r.path)],
        }

    def by_path(self) -> Dict[str, FileRecord]:
        return {f.path: f for f in self.files}

    def summary(self) -> str:
        gb = sum(f.bytes for f in self.files) / 1e9
        rows = sum(f.rows or 0 for f in self.files)
        return (f"{len(self.files)} files, {gb:.2f} GB, {rows:,} rows, "
                f"digest {self.compute_digest()}")


def _walk(root: Path) -> List[Path]:
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name in EXCLUDED_NAMES or p.suffix in EXCLUDED_SUFFIXES:
            continue
        out.append(p)
    return out


def build(root: Path, *, describe: bool = True,
          progress=None) -> Manifest:
    """Hash every content file under ``root``.

    ``describe`` reads parquet metadata for rows, span and schema. It is the
    slow part and it is what makes the manifest useful to a person rather than
    only to a comparison, so it is on by default.
    """
    root = Path(root)
    files: List[FileRecord] = []
    paths = _walk(root)
    for n, p in enumerate(paths, start=1):
        rel = p.relative_to(root).as_posix()
        extra = _describe_parquet(p) if (describe and p.suffix == ".parquet") else {}
        files.append(FileRecord(
            path=rel, bytes=p.stat().st_size, sha256=_sha256(p),
            mtime_ms=int(p.stat().st_mtime * 1000), **extra))
        if progress:
            progress(n, len(paths), rel)
    m = Manifest(root=str(root),
                 built_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                 files=files)
    m.digest = m.compute_digest()
    return m


def write(manifest: Manifest, root: Path) -> Path:
    path = Path(root) / MANIFEST_NAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest.to_dict(), indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


def load(root: Path) -> Optional[Manifest]:
    path = Path(root) / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    files = []
    for row in blob.get("files", []):
        try:
            files.append(FileRecord(**row))
        except TypeError:
            continue
    m = Manifest(root=str(blob.get("root", root)),
                 built_at=str(blob.get("built_at", "")), files=files)
    # RECOMPUTED, never read from the file. A digest a manifest asserts about
    # itself is not evidence; a digest derived from its contents is.
    m.digest = m.compute_digest()
    return m


def digest_of(root: Path) -> str:
    """The store's recorded digest, or `"unmanifested"`.

    Deliberately does not build one on the fly: a research run must record the
    digest of a manifest somebody committed, not of whatever happened to be on
    disk at the moment it ran.
    """
    m = load(root)
    return m.compute_digest() if m is not None else "unmanifested"


@dataclass(frozen=True)
class Drift:
    path: str
    kind: str            # "changed" | "missing" | "untracked"
    detail: str = ""


def verify(root: Path, *, quick: bool = False) -> Tuple[bool, List[Drift]]:
    """Does the store on disk still match the manifest?

    ``quick`` compares size only, which catches truncation and replacement but
    not an in-place edit that preserves length. The full check re-hashes, which
    on this store is a few seconds and is what "reproducible" has to mean.
    """
    root = Path(root)
    m = load(root)
    if m is None:
        return False, [Drift(MANIFEST_NAME, "missing",
                             "no manifest; the store cannot be identified")]
    recorded = m.by_path()
    seen = set()
    drift: List[Drift] = []
    for p in _walk(root):
        rel = p.relative_to(root).as_posix()
        seen.add(rel)
        rec = recorded.get(rel)
        if rec is None:
            drift.append(Drift(rel, "untracked",
                               "present on disk and absent from the manifest"))
            continue
        size = p.stat().st_size
        if size != rec.bytes:
            drift.append(Drift(rel, "changed",
                               f"{rec.bytes:,} bytes recorded, {size:,} on disk"))
            continue
        if not quick:
            actual = _sha256(p)
            if actual != rec.sha256:
                drift.append(Drift(rel, "changed",
                                   f"sha256 {rec.sha256[:12]} recorded, "
                                   f"{actual[:12]} on disk"))
    for rel in recorded:
        if rel not in seen:
            drift.append(Drift(rel, "missing",
                               "in the manifest and absent from the store"))
    return (not drift), drift
