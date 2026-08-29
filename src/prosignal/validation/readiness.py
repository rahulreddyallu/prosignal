"""READY as a gate over eight dimensions, not as "the tests passed".

WHAT WAS WRONG WITH THE OLD MEANING. Readiness was a green suite and a
document. A green suite says the code does what its tests say; it says nothing
about whether the model was fitted on a population the book can buy, whether
the execution assumptions are monotone, whether anyone can reconstruct the data
a result came from, or whether the forward window measuring all of it is still
valid. Every one of those was wrong at some point in this engine's life while
the suite was green -- R9, R13, D1 and R1 respectively.

So READY is eight gates and the engine is as ready as its weakest one:

    DATA            the store is manifested and the manifest verifies
    UNIVERSE        the training population is the one the book can buy
    FEATURE         the feature schema is pinned and matches the open epoch
    MODEL           the fit is current for this universe and config
    EXECUTION       liquidity is a gate, and the cost model is monotone
    VALIDATION      no open finding, and the multiple-testing defence can fail
    REPRODUCIBILITY code, config, data and policy are one identity
    FORWARD         a pre-registered, uncontaminated window is running

Seven of the eight are decidable here, from files. The eighth -- FORWARD -- is
decidable but not satisfiable: it is a clock, and the clock has not run.

THIS MODULE REPORTS. It refuses exactly one thing, the restart, because
restarting a window while a restart-blocking finding is open reproduces the
condition that voided the last one. Everything else it states and leaves to a
person, for the same reason `epoch.drifted_from` reports rather than acts: a
gate that silently fixes what it finds is a gate nobody reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from . import findings as _find

__all__ = [
    "DIMENSIONS", "Gate", "Readiness", "assess", "restart_refusals",
    "RestartRefused", "check_may_restart",
]

#: Order matters: it is the dependency chain. Data provenance decides the
#: universe, the universe decides the fit, the fit decides what execution is
#: applied to, and the forward test is the consumer of all of them. Reporting
#: them in any other order invites fixing the last one first, which is how a
#: forward window gets restarted onto an uncorrected engine.
DIMENSIONS = ("DATA", "UNIVERSE", "FEATURE", "MODEL", "EXECUTION",
              "VALIDATION", "REPRODUCIBILITY", "FORWARD")


@dataclass(frozen=True)
class Gate:
    name: str
    passed: bool
    detail: str
    #: What to do about it, when it is not passed. Empty when it is.
    remedy: str = ""
    #: True when the gate could not be decided rather than decided negatively.
    #: An undecidable gate is NOT a pass -- but saying so is different from
    #: saying the thing failed, and conflating them is how "no data" becomes
    #: "no problem".
    unknown: bool = False

    def line(self) -> str:
        mark = "PASS" if self.passed else ("UNKNOWN" if self.unknown else "FAIL")
        return f"{self.name:<16} {mark:<8} {self.detail}"


@dataclass(frozen=True)
class Readiness:
    gates: List[Gate]

    @property
    def ready(self) -> bool:
        return all(g.passed for g in self.gates)

    @property
    def failed(self) -> List[Gate]:
        return [g for g in self.gates if not g.passed]

    def gate(self, name: str) -> Optional[Gate]:
        return next((g for g in self.gates if g.name == name), None)

    def verdict(self) -> str:
        if self.ready:
            return "READY"
        n = len(self.failed)
        return (f"NOT READY -- {n} of {len(self.gates)} gates "
                f"{'is' if n == 1 else 'are'} not met: "
                + ", ".join(g.name for g in self.failed))


def _data_gate(cfg) -> Gate:
    from ..data.manifest import load, verify

    root = Path(cfg.paths.curated)
    man = load(root)
    if man is None:
        return Gate("DATA", False,
                    f"no manifest at {root}; no result can name its inputs",
                    "prosignal data manifest --write")
    ok, drift = verify(root, quick=True)
    if not ok:
        top = "; ".join(f"{d.path}: {d.detail}" for d in drift[:3])
        more = f" (+{len(drift) - 3} more)" if len(drift) > 3 else ""
        return Gate("DATA", False,
                    f"manifest {man.digest} does not describe the store: "
                    f"{top}{more}",
                    "re-ingest to the manifested state, or re-manifest and "
                    "open a new epoch -- the data changed, so the results did")
    return Gate("DATA", True, f"{man.summary()} verifies against {man.digest}")


def _universe_gate(cfg) -> Gate:
    from ..stages._cfg import bv

    try:
        on = bool(bv(cfg.params.universe.train_on_admissible_only))
    except Exception as exc:                              # pragma: no cover
        return Gate("UNIVERSE", False, f"cannot read the policy: {exc}",
                    "", unknown=True)
    if not on:
        return Gate("UNIVERSE", False,
                    "the training panel includes names the book cannot fill "
                    "(R9); the fit describes a population the engine does not "
                    "trade",
                    "universe.train_on_admissible_only: true, then refit")
    return Gate("UNIVERSE", True,
                "training panel restricted to the admissible population (R9)")


def _feature_gate(cfg, epoch) -> Gate:
    from .epoch import _feature_schema_sha

    sha = _feature_schema_sha()
    if not sha or sha == "unknown":
        return Gate("FEATURE", False, "the feature schema does not hash",
                    "", unknown=True)
    if epoch is None:
        return Gate("FEATURE", False,
                    f"schema {sha} is pinned but no epoch claims it",
                    "prosignal research epoch open")
    was = str(epoch.identity.get("feature_schema_sha") or "")
    if was != sha:
        return Gate("FEATURE", False,
                    f"features changed since the epoch opened ({was} -> {sha})",
                    "open a new epoch: different features are a different model")
    return Gate("FEATURE", True, f"schema {sha}, matching the open epoch")


def _model_gate(cfg, epoch) -> Gate:
    """Is the shipped fit the fit for THIS universe and config?

    Not "is there a fit" -- there is always a fit. The question is whether the
    coefficients on disk were produced by the engine now described by the open
    epoch, which is the thing R9 broke: the fit was real, tested and current,
    and it had been estimated on a population the book cannot buy.
    """
    from ..modelprint import source_digest

    if epoch is None:
        return Gate("MODEL", False, "no epoch, so no fit can be attributed",
                    "prosignal research epoch open")
    was = str(epoch.identity.get("model_sources_sha") or "")
    now = source_digest()
    if was != now:
        return Gate("MODEL", False,
                    f"the scoring path changed since the epoch opened "
                    f"({was} -> {now}); the shipped coefficients were fitted "
                    f"by different code",
                    "refit and open a new epoch")
    cfg_was = str(epoch.identity.get("config_version") or "")
    cfg_now = str(getattr(cfg, "version", ""))
    if cfg_was != cfg_now:
        return Gate("MODEL", False,
                    f"configuration changed since the epoch opened "
                    f"({cfg_was} -> {cfg_now})",
                    "open a new epoch, or revert the configuration")
    return Gate("MODEL", True,
                f"fit attributable to {now} under {cfg_now}")


def _execution_gate(cfg) -> Gate:
    """R13, checked by running it rather than by reading the config.

    The three properties are the ones a missing branch inverts: unknown
    liquidity must not be the cheapest fill in the model, a bigger order must
    not get cheaper, and deeper liquidity must not cost more.
    """
    from ..costs import CostModel

    try:
        m = CostModel(cfg)
        unknown = m.impact_bps(1.2e5, None)
        deep = m.impact_bps(1.2e5, 2e9)
        thin = m.impact_bps(1.2e5, 5e6)
        small = m.impact_bps(1e4, 5e7)
        large = m.impact_bps(2e6, 5e7)
    except Exception as exc:                              # pragma: no cover
        return Gate("EXECUTION", False, f"cost model unusable: {exc}",
                    "", unknown=True)

    broken = []
    if unknown <= deep:
        broken.append(f"unknown liquidity ({unknown:.1f} bps) is cheaper than "
                      f"a deep name ({deep:.1f} bps)")
    if large < small:
        broken.append(f"a larger order is cheaper ({large:.1f} < {small:.1f})")
    if thin < deep:
        broken.append(f"a thinner name is cheaper ({thin:.1f} < {deep:.1f})")
    if broken:
        return Gate("EXECUTION", False, "; ".join(broken),
                    "the execution model is not monotone -- see R13")

    try:
        from ..validation.portfolio_sim import PortfolioParams
        refuses = bool(PortfolioParams.__dataclass_fields__[
            "refuse_unknown_liquidity"].default)
    except Exception:
        refuses = False
    if not refuses:
        return Gate("EXECUTION", False,
                    "the simulator sizes names whose liquidity is unknown",
                    "PortfolioParams.refuse_unknown_liquidity = True")
    return Gate("EXECUTION", True,
                f"monotone, and unknown liquidity prices at {unknown:.1f} bps "
                f"against {deep:.1f} for a deep name")


def _validation_gate(cfg) -> Gate:
    # A finding that only the forward test can settle cannot block the forward
    # test. Read off the finding rather than matched by id: the id match was
    # correct while R1 was the only such finding and silently wrong the moment
    # T5 arrived, which is the failure mode of every hardcoded exemption.
    deferred = [f for f in _find.open_findings() if f.resolved_by_forward_test]
    still_open = [f for f in _find.open_findings()
                  if not f.resolved_by_forward_test]
    if still_open:
        return Gate("VALIDATION", False,
                    "open findings: " + ", ".join(f.fid for f in still_open),
                    "resolve or consciously defer each; 'reviewed' is not a "
                    "disposition")
    tail = (f" ({', '.join(f.fid for f in deferred)} excepted -- the forward "
            f"test is what resolves them)" if deferred else "")
    return Gate("VALIDATION", True,
                f"{len(_find.REGISTER)} findings registered, none open{tail}")


def _repro_gate(cfg) -> Gate:
    from .epoch import active, drifted_from

    root = Path(cfg.paths.ledger)
    ep = active(root)
    if ep is None:
        return Gate("REPRODUCIBILITY", False,
                    "no research epoch is open, so nothing produced now can be "
                    "attributed to a described engine",
                    "prosignal research epoch open --label ...")
    _, diffs = drifted_from(root, cfg)
    if diffs:
        return Gate("REPRODUCIBILITY", False,
                    f"the tree has drifted from epoch {ep.epoch_id}: "
                    + "; ".join(diffs),
                    "open a new epoch -- a material change is a new "
                    "out-of-sample question")
    ident = ep.ident()
    if ident.data_manifest_sha in ("", "unmanifested"):
        return Gate("REPRODUCIBILITY", False,
                    f"epoch {ep.epoch_id} does not name its data",
                    "prosignal data manifest --write, then reopen the epoch")
    if ident.code_dirty:
        return Gate("REPRODUCIBILITY", False,
                    f"epoch {ep.epoch_id} was opened from a dirty tree at "
                    f"{ident.code_sha}; the commit does not describe the code "
                    f"that ran",
                    "commit, then reopen the epoch")
    return Gate("REPRODUCIBILITY", True,
                f"epoch {ep.epoch_id} fingerprint {ident.fingerprint()}")


def _forward_gate(cfg) -> Gate:
    from .forward import load_registration, verify

    root = Path(cfg.paths.ledger)
    reg = load_registration(root)
    if reg is None:
        return Gate("FORWARD", False,
                    "no forward test is registered; the only stated route to "
                    "READY has not started",
                    "prosignal research forward --start")
    if not verify(root):
        return Gate("FORWARD", False,
                    "the registration file has been edited since it was "
                    "written; the criteria are no longer pre-registered",
                    "the window is void -- restart it")
    live = str(getattr(cfg, "version", ""))
    if reg.config_version != live:
        return Gate("FORWARD", False,
                    f"registered against {reg.config_version}, the engine now "
                    f"runs {live}; the window is void, not failed",
                    "prosignal research forward --restart, once the gates "
                    "above pass")
    return Gate("FORWARD", True,
                f"registered {reg.started_on} against {reg.config_version}, "
                f"fingerprint {reg.fingerprint()}")


def assess(cfg) -> Readiness:
    """Every gate, in dependency order. Nothing here changes anything."""
    from .epoch import active

    try:
        ep = active(Path(cfg.paths.ledger))
    except Exception:                                     # pragma: no cover
        ep = None
    return Readiness([
        _data_gate(cfg),
        _universe_gate(cfg),
        _feature_gate(cfg, ep),
        _model_gate(cfg, ep),
        _execution_gate(cfg),
        _validation_gate(cfg),
        _repro_gate(cfg),
        _forward_gate(cfg),
    ])


# =============================================================================
# The one thing this module refuses
# =============================================================================


class RestartRefused(RuntimeError):
    """Raised when a forward test is asked to restart onto an engine that is
    not ready to be measured."""

    def __init__(self, reasons: Sequence[str]):
        self.reasons = list(reasons)
        body = "\n".join(f"  - {r}" for r in self.reasons)
        super().__init__(
            "the forward test cannot restart yet:\n" + body + "\n\n"
            "Restarting now would open a window on an engine that is still "
            "changing, which is exactly why the current window is void. Each "
            "line above is a precondition, not a warning."
        )


#: The gates a restart depends on. EXECUTION, MODEL, UNIVERSE, FEATURE, DATA
#: and REPRODUCIBILITY all describe the engine being measured. FORWARD is
#: excluded because it is the thing being restarted -- requiring it would make
#: the gate unopenable -- and VALIDATION is covered more precisely by the
#: restart-blocking findings themselves.
RESTART_GATES = ("DATA", "UNIVERSE", "FEATURE", "MODEL", "EXECUTION",
                 "REPRODUCIBILITY")


def restart_refusals(cfg) -> List[str]:
    """Why a restart is refused right now. Empty means it may proceed.

    Two independent sources, deliberately. The findings register is the
    human judgement -- somebody decided R9 must be closed before a window
    opens. The gates are the machine check -- the config actually says so
    today. Either can catch what the other misses: a finding can be marked
    resolved while the switch that resolves it is off, and a switch can be on
    while the finding it belongs to was never really settled.
    """
    out: List[str] = []
    for f in _find.unresolved_restart_blockers():
        out.append(f"{f.fid} is open and blocks a restart: {f.title}")

    r = assess(cfg)
    for name in RESTART_GATES:
        g = r.gate(name)
        if g is not None and not g.passed:
            tail = f" -> {g.remedy}" if g.remedy else ""
            out.append(f"{name} gate not met: {g.detail}{tail}")
    return out


def check_may_restart(cfg) -> None:
    """Raise unless every precondition for opening a new window holds."""
    reasons = restart_refusals(cfg)
    if reasons:
        raise RestartRefused(reasons)
