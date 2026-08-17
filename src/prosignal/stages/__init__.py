"""Pipeline stages. Each is a pure function with a declared contract.

No stage calls the next; `prosignal.pipeline` composes them. That is what keeps
each independently testable and independently re-validatable, which matters
because the parameters inside them are hypotheses CPCV will promote or reject.

Ordering constraint that must never be relaxed: hard rejections fire before
score penalties, and eligibility runs before scoring. A stock excluded for bad
data must never reach a stage that could score it well enough to overcome the
exclusion.
"""

from __future__ import annotations

from . import (
    stage1_data_quality,
    stage2_regime,
    stage3_eligibility,
    stage4_core_score,
    stage5_false_signal,
    stage6_entry,
    stage7_risk,
    stage8_final_signal,
)

__all__ = [
    "stage1_data_quality",
    "stage2_regime",
    "stage3_eligibility",
    "stage4_core_score",
    "stage5_false_signal",
    "stage6_entry",
    "stage7_risk",
    "stage8_final_signal",
]
