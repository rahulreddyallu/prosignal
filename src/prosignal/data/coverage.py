"""How much stored history is enough, answered in one place.

Three parts of this project used to answer that question differently and none
of them agreed:

  /ready              called the store usable at min_history_sessions (300)
  the bootstrap       built to min_history_sessions + 30 (330) and stopped
  the ranking model   abstains below MIN_LOOKBACK + horizon + 60 (376)

A fresh deployment therefore built to 330, reported itself ready, hid the
build button, and then produced no ranking at all -- because the model it
exists to run needs 376 and had 330. Every number was defensible on its own
and the combination was a trap.

Worse, passing 376 is only the point at which the model CONSENTS to fit. It
refits from stored history on every run, so the store IS the training set: at
400 sessions it fits on sixteen months, and the coefficients that were
validated came from nine years. Same code, same config hash, materially
different model -- and the forward test's integrity check would not notice,
because that hash covers parameters.yaml and not the data underneath it.

So this module reports three thresholds rather than one, and they mean
different things: what makes a stock scoreable, what makes the model run at
all, and what makes it the model that was actually validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

__all__ = ["Coverage", "assess", "MINIMUM_NOTE"]

MINIMUM_NOTE = (
    "Passing the model minimum only means the fit will run. The model refits "
    "from stored history on every analysis, so a short store yields a model "
    "trained on a short history -- not the one the validation measured."
)


@dataclass(frozen=True)
class Coverage:
    stored: int
    #: A stock needs this much history before any feature can be computed.
    eligibility_minimum: int
    #: Below this the cross-sectional model abstains rather than fitting noise.
    model_minimum: int
    #: The training span the shipped coefficients were validated against.
    validated_target: int

    @property
    def can_score_stocks(self) -> bool:
        return self.stored > self.eligibility_minimum

    @property
    def model_will_fit(self) -> bool:
        return self.stored >= self.model_minimum

    @property
    def matches_validation(self) -> bool:
        return self.stored >= self.validated_target

    @property
    def ready(self) -> bool:
        """Usable at all. Deliberately the MODEL's bar, not eligibility's."""
        return self.model_will_fit

    @property
    def shortfall(self) -> int:
        return max(self.model_minimum - self.stored, 0)

    def status(self) -> str:
        if not self.model_will_fit:
            return (
                f"{self.stored} sessions stored; the ranking model needs "
                f"{self.model_minimum} before it will fit at all "
                f"({self.shortfall} short)."
            )
        if not self.matches_validation:
            return (
                f"{self.stored} sessions stored. The model will fit, but it "
                f"refits from stored history, so it is training on "
                f"{self.stored / 250:.1f} years rather than the "
                f"{self.validated_target / 250:.1f} the shipped coefficients "
                f"were validated on. Keep building to "
                f"{self.validated_target}."
            )
        return f"{self.stored} sessions stored; full validated depth."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "price_sessions": self.stored,
            "eligibility_minimum": self.eligibility_minimum,
            "model_minimum": self.model_minimum,
            "validated_target": self.validated_target,
            "model_will_fit": self.model_will_fit,
            "matches_validation": self.matches_validation,
            "status": self.status(),
        }


def assess(config, stored: int) -> Coverage:
    """Read every threshold from the code that enforces it, never a copy."""
    from ..features import crossmodel as cm
    from ..features.crosssec import MIN_LOOKBACK
    from ..stages._cfg import iv

    horizon = int(iv(config.params.stage4_core_score.model_horizon_sessions))
    # The exact expression crossmodel.fit_predict guards on. Duplicating the
    # NUMBER here would let the two drift; duplicating the EXPRESSION means a
    # change to either input moves both together.
    model_minimum = MIN_LOOKBACK + horizon + 60
    return Coverage(
        stored=int(stored),
        eligibility_minimum=int(config.params.universe.min_history_sessions.value),
        model_minimum=model_minimum,
        validated_target=int(config.params.storage.validated_training_sessions),
    )
