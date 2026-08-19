"""Project-root discovery and directory management.

Paths in config/parameters.yaml are relative to the project root, defined as the
directory containing ``config/parameters.yaml``. Discovery order:

1. ``$PROSIGNAL_HOME``, if set.
2. Walk up from the current working directory.
3. Walk up from this source file, covering ``pip install -e .`` layouts.

Keeping this in one place means the engine behaves identically from the CLI,
from pytest, or from uvicorn with a different cwd.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

from .errors import ConfigError

__all__ = [
    "CONFIG_RELPATH",
    "find_project_root",
    "resolve",
    "ensure_dir",
    "ProjectPaths",
]

CONFIG_RELPATH = Path("config") / "parameters.yaml"

_MAX_WALK_UP = 8


def _walk_up(start: Path) -> Iterable[Path]:
    cur = start.resolve()
    for _ in range(_MAX_WALK_UP):
        yield cur
        if cur.parent == cur:
            break
        cur = cur.parent


def find_project_root(explicit: Optional[Path] = None) -> Path:
    """Return the directory that contains ``config/parameters.yaml``."""
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
        if not (root / CONFIG_RELPATH).is_file():
            raise ConfigError(
                f"No {CONFIG_RELPATH} under the supplied project root",
                project_root=str(root),
            )
        return root

    env = os.environ.get("PROSIGNAL_HOME")
    if env:
        root = Path(env).expanduser().resolve()
        if not (root / CONFIG_RELPATH).is_file():
            raise ConfigError(
                f"PROSIGNAL_HOME is set but contains no {CONFIG_RELPATH}",
                PROSIGNAL_HOME=str(root),
            )
        return root

    for candidate in _walk_up(Path.cwd()):
        if (candidate / CONFIG_RELPATH).is_file():
            return candidate

    # src/prosignal/core/paths.py -> src/prosignal/core -> src/prosignal -> src -> root
    for candidate in _walk_up(Path(__file__).parent):
        if (candidate / CONFIG_RELPATH).is_file():
            return candidate

    raise ConfigError(
        "Could not locate the project root. Expected to find "
        f"{CONFIG_RELPATH} by walking up from the working directory. "
        "Set PROSIGNAL_HOME to the project folder, or run the command from "
        "inside it.",
        cwd=str(Path.cwd()),
    )


def resolve(root: Path, relative: str) -> Path:
    """Resolve a config-declared path against the project root."""
    p = Path(relative).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class ProjectPaths:
    """Concrete, absolute paths derived from ``runtime.paths``.

    Directories are created lazily on first access of the owning attribute so
    that merely importing the config never litters the filesystem.
    """

    __slots__ = (
        "root",
        "data",
        "raw",
        "curated",
        "snapshots",
        "cache",
        "ledger",
        "reference",
        "logs",
        "config_file",
    )

    def __init__(self, root: Path, paths_cfg: "object") -> None:
        self.root = root
        self.config_file = root / CONFIG_RELPATH
        self.data = resolve(root, getattr(paths_cfg, "data_dir"))
        self.raw = resolve(root, getattr(paths_cfg, "raw_dir"))
        self.curated = resolve(root, getattr(paths_cfg, "curated_dir"))
        self.snapshots = resolve(root, getattr(paths_cfg, "snapshot_dir"))
        self.cache = resolve(root, getattr(paths_cfg, "cache_dir"))
        self.ledger = resolve(root, getattr(paths_cfg, "ledger_dir"))
        self.reference = resolve(root, getattr(paths_cfg, "reference_dir"))
        self.logs = resolve(root, getattr(paths_cfg, "log_dir"))

    def create_all(self) -> "ProjectPaths":
        for p in (
            self.data,
            self.raw,
            self.curated,
            self.snapshots,
            self.cache,
            self.ledger,
            self.reference,
            self.logs,
        ):
            ensure_dir(p)
        return self

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"ProjectPaths(root={self.root})"
