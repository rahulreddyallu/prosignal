"""Pipeline stages.

Each stage is a pure function with a declared input/output contract from
``core.contracts``. No stage reads global state, no stage mutates its inputs,
and no stage calls the next one -- the orchestrator (Chunk 7) composes them.
That is what makes each independently testable and independently re-validatable,
which matters because the parameters inside them are hypotheses that CPCV will
either promote or reject.

    Stage 0  data.ingest                RawDataManifest
    Stage 1  stage1_data_quality        DataQualityReport
    Stage 2  stage2_regime              RegimeState
    Stage 3  eligibility                (chunk 3)
    Stage 4  core score                 (chunk 3)
    Stage 5  false-signal defense       (chunk 4)
    Stage 6  entry confirmation         (chunk 5)
    Stage 7  risk / position            (chunk 5)
    Stage 8  final signal               (chunk 6)

The ordering constraint that must never be relaxed: hard rejections fire before
score penalties, and eligibility runs before scoring. A stock excluded for bad
data must never reach a stage that could score it well enough to overcome the
exclusion.
"""

from __future__ import annotations

from . import stage1_data_quality, stage2_regime

__all__ = ["stage1_data_quality", "stage2_regime"]
