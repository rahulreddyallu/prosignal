"""Load, validate, hash and cache config/parameters.yaml.

The loader's job is to make a bad edit impossible to miss:

* pydantic validation errors are re-rendered as ``section.key: message`` lines
  that map straight onto the YAML the user just edited;
* the resulting object is frozen behind a module-level cache so that two
  stages can never see two different parameter sets within one run;
* a SHA-256 over every *value* (ignoring prose like ``note``/``description``)
  produces the ``config_version`` stamped onto every research-ledger row, which
  is what eventually lets live performance be attributed to an exact dated
  parameter set.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import ValidationError

from ..core.errors import ConfigError
from ..core.paths import CONFIG_RELPATH, ProjectPaths, find_project_root
from .schema import ParamStatus, RootConfig, Tunable

__all__ = [
    "AppConfig",
    "load_config",
    "get_config",
    "reset_config_cache",
    "config_hash",
]

#: Keys that are prose, not settings. Editing them must not change the hash.
_HASH_IGNORED_KEYS = frozenset({"note", "description", "validated_by", "validated_on"})

_lock = threading.Lock()
_cache: Dict[Tuple[str, Optional[str]], "AppConfig"] = {}


# =============================================================================
# hashing
# =============================================================================


def _strip_prose(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            k: _strip_prose(v) for k, v in sorted(node.items()) if k not in _HASH_IGNORED_KEYS
        }
    if isinstance(node, list):
        return [_strip_prose(x) for x in node]
    return node


def config_hash(cfg: RootConfig) -> str:
    """Deterministic 16-hex-char digest of every effective parameter value."""
    payload = _strip_prose(cfg.model_dump(mode="json"))
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# =============================================================================
# error rendering
# =============================================================================


def _render_validation_error(exc: ValidationError, config_path: Path) -> str:
    lines: List[str] = [
        f"{config_path} failed validation ({exc.error_count()} problem(s)):",
        "",
    ]
    for err in exc.errors():
        loc_parts: List[str] = []
        for part in err["loc"]:
            # pydantic inserts the generic parameterisation for Tunable[...]
            if isinstance(part, str) and part.startswith("Tunable["):
                continue
            loc_parts.append(str(part))
        loc = ".".join(loc_parts) or "<root>"
        msg = err["msg"]
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, ") :]
        lines.append(f"  {loc}")
        lines.append(f"      -> {msg}")
        if err["type"] == "extra_forbidden":
            lines.append(
                "      -> This key is not part of the schema. Check the spelling "
                "against config/parameters.yaml's documented structure; the "
                "engine refuses unknown keys so a typo can never silently "
                "fall back to a hidden default."
            )
    return "\n".join(lines)


# =============================================================================
# AppConfig
# =============================================================================


class AppConfig:
    """The validated parameter set plus everything derived from it."""

    __slots__ = ("params", "paths", "hash", "source_file", "_tunable_index")

    def __init__(self, params: RootConfig, root: Path, source_file: Path) -> None:
        self.params = params
        self.source_file = source_file
        self.paths = ProjectPaths(root, params.runtime.paths)
        self.hash = config_hash(params)
        self._tunable_index: Optional[Dict[str, Dict[str, Any]]] = None

    # -- identity -----------------------------------------------------------
    @property
    def version(self) -> str:
        """The string written to every ledger row, e.g. ``baseline-v1@3f9c...``."""
        return f"{self.params.meta.config_label}@{self.hash}"

    # -- convenience accessors ---------------------------------------------
    def __getattr__(self, item: str) -> Any:
        # Delegate unknown attributes to the parameter tree so call sites can
        # write cfg.stage4_core_score instead of cfg.params.stage4_core_score.
        try:
            return getattr(self.params, item)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc

    def tunable(self, dotted_path: str) -> Dict[str, Any]:
        """Look up one tunable's full metadata by dotted path."""
        if self._tunable_index is None:
            self._tunable_index = {t["path"]: t for t in self.params.iter_tunables()}
        try:
            return self._tunable_index[dotted_path]
        except KeyError:
            raise ConfigError(f"no such tunable: {dotted_path!r}") from None

    def transparency_report(self) -> Dict[str, Any]:
        """Payload for GET /config -- the webapp's config transparency panel."""
        tunables = self.params.iter_tunables()
        by_status: Dict[str, int] = {}
        for t in tunables:
            by_status[t["status"]] = by_status.get(t["status"], 0) + 1
        return {
            "config_label": self.params.meta.config_label,
            "config_hash": self.hash,
            "config_version": self.version,
            "source_file": str(self.source_file),
            "total_parameters": len(tunables),
            "counts_by_status": by_status,
            "unvalidated_count": by_status.get(ParamStatus.UNVALIDATED.value, 0),
            "parameters": tunables,
            "honesty_note": (
                "Every parameter marked UNVALIDATED is a hypothesis that has not "
                "been through CPCV on point-in-time India data. The engine uses "
                "them because it must use something; it reports them because "
                "pretending they are validated would be the actual error."
            ),
        }

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"AppConfig({self.version}, root={self.paths.root})"


# =============================================================================
# loading
# =============================================================================


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ConfigError(
            f"Parameter file not found: {path}. This is the one file the engine "
            f"cannot run without.",
            expected=str(path),
        )
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise ConfigError(
            f"{path} is not valid YAML{where}: {getattr(exc, 'problem', exc)}"
        ) from exc
    if raw is None:
        raise ConfigError(f"{path} is empty")
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return raw


def _apply_overlay(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge an optional local overlay file over the committed baseline."""
    out = dict(base)
    for key, val in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _apply_overlay(out[key], val)
        else:
            out[key] = val
    return out


def load_config(
    config_path: Optional[Path] = None,
    project_root: Optional[Path] = None,
    overlay_path: Optional[Path] = None,
    use_cache: bool = True,
) -> AppConfig:
    """Load and validate the parameter file.

    Parameters
    ----------
    config_path:
        Explicit path to a parameters.yaml. Defaults to
        ``<project_root>/config/parameters.yaml``.
    overlay_path:
        Optional second YAML deep-merged over the first. Defaults to
        ``config/parameters.local.yaml`` when that file exists, which lets you
        keep machine-specific settings (capital, broker fees) out of git
        without touching the committed baseline.
    """
    root = find_project_root(project_root)
    cfg_file = Path(config_path).expanduser().resolve() if config_path else (root / CONFIG_RELPATH)

    if overlay_path is None:
        default_overlay = cfg_file.parent / "parameters.local.yaml"
        overlay_path = default_overlay if default_overlay.is_file() else None
    overlay_key = str(overlay_path) if overlay_path else None

    cache_key = (str(cfg_file), overlay_key)
    if use_cache:
        with _lock:
            hit = _cache.get(cache_key)
            if hit is not None:
                return hit

    raw = _read_yaml(cfg_file)
    if overlay_path is not None:
        raw = _apply_overlay(raw, _read_yaml(Path(overlay_path)))

    try:
        params = RootConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_render_validation_error(exc, cfg_file)) from exc

    app = AppConfig(params=params, root=root, source_file=cfg_file)

    if use_cache:
        with _lock:
            _cache[cache_key] = app
    return app


def get_config(**kwargs: Any) -> AppConfig:
    """Process-wide accessor. Honours ``$PROSIGNAL_CONFIG`` if set."""
    if "config_path" not in kwargs:
        env = os.environ.get("PROSIGNAL_CONFIG")
        if env:
            kwargs["config_path"] = Path(env)
    return load_config(**kwargs)


def reset_config_cache() -> None:
    """Drop the cache. Used by tests and by the CLI's ``config reload``."""
    with _lock:
        _cache.clear()


def describe_tunable(t: Tunable) -> str:  # pragma: no cover - display only
    bits = [f"{t.value!r}", t.status.value]
    if t.search_range:
        bits.append(f"range={t.search_range}")
    return " | ".join(bits)
