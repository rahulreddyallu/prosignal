"""Load, validate, hash and cache config/parameters.yaml.

* pydantic validation errors are re-rendered as ``section.key: message`` lines
  matching the YAML that was edited;
* the result is cached at module level so two stages cannot see different
  parameter sets within one run;
* a SHA-256 over every value, ignoring prose fields such as ``note``, produces
  the ``config_version`` stamped onto each ledger row, which is what lets live
  performance be attributed to an exact dated parameter set.
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
#:
#: `verified_on` joins them because re-checking a statutory rate that has NOT
#: moved is not a change to the model. If the rate itself moves, `value` moves
#: and the hash moves with it -- which is the behaviour that matters.
_HASH_IGNORED_KEYS = frozenset({"note", "description", "validated_by",
                                "validated_on", "verified_on"})

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

    __slots__ = ("params", "paths", "hash", "source_file", "_tunable_index",
                 "_identity", "_bound_store")

    def __init__(self, params: RootConfig, root: Path, source_file: Path) -> None:
        self.params = params
        self.source_file = source_file
        self.paths = ProjectPaths(root, params.runtime.paths)
        self.hash = config_hash(params)
        self._tunable_index: Optional[Dict[str, Dict[str, Any]]] = None
        self._identity: Optional[Any] = None
        self._bound_store: Optional[Any] = None

    # -- identity -----------------------------------------------------------
    def bind_store(self, store: Any) -> "AppConfig":
        """Resolve the FULL identity -- parameters AND data AND training window.

        WHY THIS EXISTS. A hash over parameters.yaml cannot identify a model.
        The fit reads history from the store on every run, so a store that grew
        produces different coefficients from identical code and identical knobs
        -- and the README has carried exactly that as a known limitation, while
        the forward test's whole integrity check is "did `config_version`
        change". `config/identity.py` sets out the three components.

        IT IS A SEPARATE STEP because loading a config must not require a store.
        `config show`, `config validate` and most of the schema tests have no
        data at all, and a loader that opened parquet to tell you whether your
        YAML parses would be the wrong shape entirely.

        UNBOUND IS HONEST, NOT WRONG. Until a store is bound, `version` is the
        parameters-only string it has always been, and `identity` is None so a
        caller can tell which one it is holding. What changes is that a run
        WITH a store now stamps the fuller identity, so two runs trained on
        different data can no longer quote the same version.

        RESOLVED LAZILY. Binding only records the store; the fingerprint is
        computed on first access to `version` or `identity`. Reading the
        coverage of every feed costs a couple of seconds even now that it reads
        parquet columns directly rather than adjusted frames, and the CLI binds
        on every invocation -- so `config show`, `config validate` and
        `data status`, none of which ask what the version is, should not pay
        for an answer they never read.

        Returns self, so it can be chained at the call site.
        """
        # RE-BINDING THE SAME STORE MUST NOT RE-RESOLVE. `load_config` caches
        # the AppConfig process-wide, and `run_analysis` binds on every call --
        # so clearing the cached identity unconditionally made every analysis in
        # a process pay the fingerprint again. Identity by object, because two
        # DataStore instances over the same directory are the same store for
        # this purpose and a test that constructs one per call should not be
        # charged for it.
        if store is self._bound_store:
            return self
        same_dir = (self._bound_store is not None
                    and getattr(store, "curated", None) is not None
                    and getattr(store, "curated", None)
                    == getattr(self._bound_store, "curated", object()))
        self._bound_store = store
        if not same_dir:
            self._identity = None
        return self

    @property
    def identity(self) -> Optional[Any]:
        """The resolved `ConfigIdentity`, or None when no store is bound.

        Computes it on first access and caches it. A store does not change
        underneath a running process -- and if it did, two stages within one
        run seeing different identities would be worse than either answer.
        """
        if self._identity is None and self._bound_store is not None:
            from .identity import identify

            self._identity = identify(self, self._bound_store)
        return self._identity

    @property
    def version(self) -> str:
        """The string written to every ledger row, e.g. ``baseline-v1@3f9c...``.

        With a store bound this is
        ``label@ H(params) XOR H(store_fingerprint) XOR H(train_window)``;
        without one it is ``label@H(params)``.
        """
        ident = self.identity                    # resolves lazily, then caches
        if ident is not None:
            return ident.version
        return f"{self.params.meta.config_label}@{self.hash}"

    @property
    def params_version(self) -> str:
        """``label@H(params)`` -- parameters only, whatever store is bound.

        Kept reachable because "did the knobs move" is a real question with a
        real answer, and it is not the same question as "is this the same
        model".
        """
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
