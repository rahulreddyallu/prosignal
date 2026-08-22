"""Business-ready shapes for the interface.

The engine speaks in factor loadings, standardised deviations and stage
numbers. That vocabulary is correct for the model and wrong for the person
deciding what to do with their money. This package is the translation layer,
and it is deliberately one-directional: it reads what the engine produced and
never feeds anything back into it.
"""

from .selection import SLOTS, Slate, select_slate
from .evidence import EVIDENCE_CATEGORIES, build_evidence
from .narrative import build_narrative
from .viewmodel import build_view

__all__ = [
    "SLOTS", "Slate", "select_slate",
    "EVIDENCE_CATEGORIES", "build_evidence",
    "build_narrative", "build_view",
]
