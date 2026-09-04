"""What a `config_version` has to cover before it can identify a model.

THE HOLE THIS CLOSES, stated as the repository already recorded it (README,
"Known limitations"): a store-depth change moves the fitted coefficients and
leaves the config hash identical. Two runs then quote the same
`baseline-v2@9ffe2b1b` and are not the same model. Everything that trusts the
hash inherits the error -- and the thing that trusts it hardest is the forward
test, whose entire integrity check is "did `config_version` change".

`modelprint.model_fingerprint` already noticed half of this and hashes the
model's SOURCE alongside a coarse depth bucket. It is a separate field on the
ledger row, and `forward.progress` reports it separately, which is why the last
window flagged five fingerprints against one config version. That is the right
diagnosis reported in the wrong place: a reader comparing two results compares
`config_version`, so the identity has to be IN it.

    config_version = label @ ( H(params) XOR H(store_fingerprint) XOR H(train_window) )

THREE COMPONENTS, EACH ANSWERING A DIFFERENT QUESTION.

  H(params)             which knobs. Already existed, unchanged, and still
                        available on its own as `AppConfig.hash` -- plenty of
                        code legitimately wants to know whether the PARAMETERS
                        moved, separately from whether the data did.
  H(store_fingerprint)  which data. Per feed: first session, last session,
                        session count, symbol count. Not file bytes -- that is
                        `data/manifest.py`'s job and it is a stricter question
                        than this one. Two stores that span the same sessions
                        over the same symbols will train the same model even if
                        one of them was rewritten by a re-ingest, and a version
                        that churned on re-ingest would be ignored within a
                        week, which is exactly how `cumulative_trials_logged`
                        came to ship at zero.
  H(train_window)       which slice of it the model actually fits on. The store
                        can grow at the front without the training window
                        moving, because the fit is capped at
                        `model_max_train_sessions`; and the window can move
                        while the store does not, if the cap changes. Neither
                        implies the other, so both are hashed.

ON XOR. It is the operator the build plan specifies and it has one property
worth stating rather than discovering: XOR is commutative, so this identity
cannot say WHICH component moved, only that one did. That is acceptable because
`describe()` returns the three parts separately and the CLI prints them; the
combined value is an identity, not a diagnosis. Two components colliding would
cancel, which for independent SHA-256 digests is not a risk anybody needs to
manage.

NOTHING HERE READS A PRICE. The fingerprint is metadata about coverage -- dates
and symbol counts -- so it costs a session index rather than a panel.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "StoreFingerprint", "TrainWindow", "ConfigIdentity",
    "store_fingerprint", "train_window", "combine", "identify",
]

#: Feeds whose depth can move a coefficient. `prices` is the training set;
#: `delivery` feeds three shipped factors; `indices` sets the benchmark and the
#: regime; `fundamentals` gates the quality theme's coverage.
#:
#: Deliberately NOT everything under `curated/`. The sector map, the equity
#: master and the corporate-action table change what is ELIGIBLE and what a
#: price MEANS, and they are covered by the store manifest and by the epoch's
#: `data_manifest_sha`. Putting them here too would move the version on every
#: weekly master refresh, and a version that moves for harmless reasons is one
#: nobody reads.
FINGERPRINTED_FEEDS: Tuple[str, ...] = ("prices", "delivery", "indices",
                                        "fundamentals")


def _h(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _xor_hex(*digests: str) -> str:
    """XOR full-width hex digests, returned as the leading 16 hex chars.

    Sixteen to match `config_hash`, so a version string keeps the shape every
    existing ledger row, test and document already uses.
    """
    width = max(len(d) for d in digests)
    acc = 0
    for d in digests:
        acc ^= int(d.ljust(width, "0"), 16)
    return f"{acc:0{width}x}"[:16]


@dataclass(frozen=True)
class FeedCoverage:
    feed: str
    first_session: Optional[str]
    last_session: Optional[str]
    n_sessions: int
    n_symbols: int
    #: Set when the feed could not be read at all. UNKNOWN is not zero: a feed
    #: that failed to load and a feed that is genuinely empty produce different
    #: models, and collapsing them would let a broken read masquerade as an
    #: empty one -- the failure the NOT_TESTABLE convention exists to prevent.
    unavailable: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "feed": self.feed, "first_session": self.first_session,
            "last_session": self.last_session, "n_sessions": self.n_sessions,
            "n_symbols": self.n_symbols}
        if self.unavailable:
            out["unavailable"] = self.unavailable
        return out


@dataclass(frozen=True)
class StoreFingerprint:
    feeds: List[FeedCoverage] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"feeds": [f.as_dict() for f in
                          sorted(self.feeds, key=lambda x: x.feed)]}

    def digest(self) -> str:
        return _h(self.as_dict())

    def summary(self) -> str:
        bits = []
        for f in sorted(self.feeds, key=lambda x: x.feed):
            if f.unavailable:
                bits.append(f"{f.feed} UNAVAILABLE")
            else:
                bits.append(f"{f.feed} {f.n_sessions}s/{f.n_symbols}n "
                            f"{f.first_session}..{f.last_session}")
        return "; ".join(bits) or "empty store"


@dataclass(frozen=True)
class TrainWindow:
    """The slice of the store the cross-sectional model would fit on."""

    first_session: Optional[str]
    last_session: Optional[str]
    n_sessions: int
    max_train_sessions: int
    horizon_sessions: int
    purge_sessions: int
    embargo_sessions: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "first_session": self.first_session,
            "last_session": self.last_session,
            "n_sessions": self.n_sessions,
            "max_train_sessions": self.max_train_sessions,
            "horizon_sessions": self.horizon_sessions,
            "purge_sessions": self.purge_sessions,
            "embargo_sessions": self.embargo_sessions,
        }

    def digest(self) -> str:
        return _h(self.as_dict())

    def summary(self) -> str:
        return (f"{self.n_sessions} sessions "
                f"{self.first_session}..{self.last_session} "
                f"(cap {self.max_train_sessions}, h={self.horizon_sessions}, "
                f"purge={self.purge_sessions}, embargo={self.embargo_sessions})")


@dataclass(frozen=True)
class ConfigIdentity:
    """The three components and the value they combine to."""

    label: str
    params_hash: str
    store_hash: str
    train_hash: str
    combined: str
    store: StoreFingerprint
    train: TrainWindow

    @property
    def version(self) -> str:
        return f"{self.label}@{self.combined}"

    def describe(self) -> Dict[str, Any]:
        return {
            "config_version": self.version,
            "label": self.label,
            "params_hash": self.params_hash,
            "store_hash": self.store_hash[:16],
            "train_hash": self.train_hash[:16],
            "combined": self.combined,
            "store_fingerprint": self.store.as_dict(),
            "train_window": self.train.as_dict(),
        }


def _iso(day: Any) -> Optional[str]:
    if day is None:
        return None
    if isinstance(day, dt.datetime):
        day = day.date()
    if isinstance(day, dt.date):
        return day.isoformat()
    try:
        return dt.date.fromisoformat(str(day)[:10]).isoformat()
    except (TypeError, ValueError):
        return None


def _coverage(store, feed: str) -> FeedCoverage:
    """Sessions and symbols for one feed, or an explicit UNAVAILABLE."""
    readers = {
        "prices": ("price_sessions", "read_prices"),
        "delivery": (None, "read_delivery"),
        "indices": (None, "read_indices"),
        "fundamentals": (None, "read_fundamentals"),
    }
    sess_attr, read_attr = readers.get(feed, (None, None))
    if read_attr is None or not hasattr(store, read_attr):
        return FeedCoverage(feed, None, None, 0, 0,
                            unavailable=f"the store exposes no {read_attr!r}")
    try:
        frame = getattr(store, read_attr)()
    except Exception as exc:                       # noqa: BLE001 -- see below
        # An unreadable feed is UNKNOWN, never empty. Returning a zero-coverage
        # record with no marker would give a broken read the same digest as a
        # genuinely absent feed.
        return FeedCoverage(feed, None, None, 0, 0, unavailable=str(exc)[:200])
    if frame is None or getattr(frame, "empty", True):
        return FeedCoverage(feed, None, None, 0, 0)

    date_col = next((c for c in ("date", "DATE", "session_date", "period_end",
                                 "filing_date") if c in frame.columns), None)
    # `index_name` is the indices feed's series key, and it has to be here: NSE
    # publishes India VIX in the same frame as the price indices, so a feed that
    # gained or lost a series would otherwise leave the fingerprint untouched
    # while changing both the benchmark and the regime read.
    sym_col = next((c for c in ("symbol", "SYMBOL", "ticker", "index_name",
                                "index") if c in frame.columns), None)
    first = last = None
    n_sessions = 0
    if date_col is not None:
        import pandas as pd
        s = pd.to_datetime(frame[date_col], errors="coerce").dropna()
        if len(s):
            first, last = _iso(s.min()), _iso(s.max())
            n_sessions = int(s.dt.normalize().nunique())
    if sess_attr and hasattr(store, sess_attr):
        try:
            n_sessions = len(getattr(store, sess_attr)()) or n_sessions
        except Exception:                          # noqa: BLE001
            pass
    n_symbols = int(frame[sym_col].nunique()) if sym_col is not None else 0
    return FeedCoverage(feed, first, last, n_sessions, n_symbols)


def store_fingerprint(store, feeds: Optional[Tuple[str, ...]] = None
                      ) -> StoreFingerprint:
    """Per-feed coverage: first session, last session, session and symbol count."""
    return StoreFingerprint([_coverage(store, f)
                             for f in (feeds or FINGERPRINTED_FEEDS)])


def train_window(cfg, store) -> TrainWindow:
    """The window the cross-sectional model would fit on, as of the last session.

    Mirrors `crossmodel`: history strictly before the decision date, capped at
    `model_max_train_sessions`. The cap is read from config rather than from the
    module constant so that changing it moves the version, which is the whole
    point -- a smaller cap is a different training set and therefore a different
    model.
    """
    c4 = cfg.params.stage4_core_score
    val = cfg.params.validation
    cap = int(getattr(c4, "model_max_train_sessions", 3000))
    horizon = int(getattr(c4.model_horizon_sessions, "value",
                          c4.model_horizon_sessions))
    purge = int(getattr(val.cpcv.purge_sessions, "value", val.cpcv.purge_sessions))
    embargo = int(getattr(val.cpcv.embargo_sessions, "value",
                          val.cpcv.embargo_sessions))
    try:
        sessions = list(store.price_sessions() or ())
    except Exception:                              # noqa: BLE001
        sessions = []
    used = sessions[-cap:] if cap and len(sessions) > cap else sessions
    return TrainWindow(
        first_session=_iso(used[0]) if used else None,
        last_session=_iso(used[-1]) if used else None,
        n_sessions=len(used),
        max_train_sessions=cap,
        horizon_sessions=horizon,
        purge_sessions=purge,
        embargo_sessions=embargo,
    )


def combine(params_hash: str, store_hash: str, train_hash: str) -> str:
    """`H(params) XOR H(store) XOR H(train)`, truncated to 16 hex chars."""
    return _xor_hex(params_hash, store_hash, train_hash)


def identify(cfg, store) -> ConfigIdentity:
    """The full identity of the model this config and this store describe."""
    sf = store_fingerprint(store)
    tw = train_window(cfg, store)
    sh, th = sf.digest(), tw.digest()
    return ConfigIdentity(
        label=str(cfg.params.meta.config_label),
        params_hash=cfg.hash,
        store_hash=sh,
        train_hash=th,
        combined=combine(cfg.hash, sh, th),
        store=sf,
        train=tw,
    )
