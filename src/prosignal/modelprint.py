"""What actually decides the ranking, in one hash.

The forward test's integrity check watches `config_version`, which is a hash
of parameters.yaml. That covers the knobs and nothing else, and two things
that change the model leave it identical.

The first is code. Editing the ridge fit, a feature definition or a stage
rule changes what the engine does and not one byte of the config.
`engine_version` does not help: it is a literal in version.py that nobody
bumps, and it has read 0.1.0 through every change this project has made.

The second is data depth. The model refits from stored history on every run,
capped at MAX_TRAIN_SESSIONS, so the store IS the training set. A store that
grew from 1,900 sessions to 2,200 between two runs produces different
coefficients from identical code and identical config. That is the case the
coverage module was written about, and it is invisible to a hash of the
knobs.

So this hashes the source that defines the model, alongside the depth it was
fitted on. Two runs with the same fingerprint were the same model. Two with
different fingerprints were not, whatever the config hash says, and a
forward test that pools them is measuring two things at once.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List, Optional

__all__ = ["MODEL_SOURCES", "source_digest", "model_fingerprint", "train_bucket"]

#: The files whose contents decide a ranking. Deliberately narrow: adding the
#: API or the interface here would churn the fingerprint on changes that
#: cannot alter a single coefficient, and a fingerprint that changes for
#: harmless reasons gets ignored.
MODEL_SOURCES: List[str] = [
    "features/crossmodel.py",
    "features/crosssec.py",
    "features/linear.py",
    "features/fundamental_factors.py",
    "stages/stage3_eligibility.py",
    "stages/stage4_core_score.py",
    "stages/stage5_false_signal.py",
    "stages/stage6_entry.py",
    "stages/stage7_risk.py",
    "stages/stage8_final_signal.py",
    # These two decide what the engine HOLDS and what it SHOWS, which the
    # forward test's secondary criterion measures directly -- it is a pooled
    # rank IC "of the daily shortlist". They sit outside the stages, so a
    # change to either altered the tested object while leaving both the config
    # hash and this fingerprint untouched.
    #
    # This is not the interface. `presentation/viewmodel.py` and the static
    # page are deliberately still absent: they render the decision and cannot
    # change it. `selection.py` chooses which names are on the list and
    # `positions.py` decides whether a position survives an event, and both of
    # those are decisions.
    "presentation/selection.py",
    "positions.py",
]

#: Training depth is bucketed rather than exact. A store gains a session a
#: day, and a fingerprint that changed every night would report drift on
#: every run and mean nothing. 250 sessions is about a year -- enough of a
#: change in training span to be a different model.
TRAIN_BUCKET_SESSIONS = 250


def train_bucket(sessions: Optional[int]) -> Optional[int]:
    if sessions is None or sessions < 0:
        return None
    return int(sessions) // TRAIN_BUCKET_SESSIONS


def source_digest(root: Optional[Path] = None,
                  files: Optional[Iterable[str]] = None) -> str:
    """Hash of the model's source, or "unknown" if it cannot be read.

    An unreadable source is reported as unknown rather than as some other
    hash: a fingerprint that silently changes meaning is worse than one that
    admits it does not know.
    """
    base = Path(root) if root is not None else Path(__file__).resolve().parent
    h = hashlib.sha256()
    seen = 0
    for rel in sorted(files if files is not None else MODEL_SOURCES):
        path = base / rel
        try:
            h.update(rel.encode("utf-8"))
            h.update(path.read_bytes())
            seen += 1
        except OSError:
            continue
    if not seen:
        return "unknown"
    return h.hexdigest()[:12]


def model_fingerprint(train_sessions: Optional[int] = None,
                      root: Optional[Path] = None) -> str:
    """`<source>/<depth-bucket>`, or `<source>/?` when the depth is unknown."""
    bucket = train_bucket(train_sessions)
    return f"{source_digest(root)}/{'?' if bucket is None else bucket}"
