# Re-registering the forward test

The forward test is the only clean out-of-sample answer this engine has left.
The holdout was spent once and its figures are withdrawn; re-running it would
buy a nicer number at the cost of the last honest test. This document is what
to do instead, and it is written to be followed without interpretation.

## Register against this configuration

    config_version   baseline-v1@127d8a314ec49aa2
    config_hash      127d8a314ec49aa2

Command:

    prosignal research forward --start

It refuses to overwrite an existing registration. `--restart` exists and
discards every observation collected so far; if you are reaching for it, stop
and read "When re-registration is legitimate" below.

The registration writes a hash of everything that must not change afterwards —
the start date, the config version, the horizon, and all three hypotheses — and
`verify()` fails if any of it is edited later. That is the point.

## The three hypotheses

All are stated as PASS conditions so that a result cannot be reinterpreted as a
success after it lands.

**PRIMARY.** Regress the paper portfolio's monthly excess return on the six
long-short factors from `validation.attribution`, over 18 forward months. Passes
if the intercept is positive with an overlap-corrected t ≥ 2.0.
*Selection-period reference: alpha −1.01% at t −0.38 on the holdout, with 15
observations against 6 factors. Eighteen monthly observations leave 11 degrees
of freedom rather than 8.*

**SECONDARY.** Pooled rank IC of the daily shortlist against the 63-session
forward return. Passes if positive with an overlap-corrected t ≥ 2.0. Recorded
so a failure here is visible rather than assumed away — not the question at
issue.

**TERTIARY — and the one that decides whether any of this is worth running.**
Mean excess return of the paper portfolio over the **equal-weight eligible
universe**, on the same holding windows. Passes if the mean excess is positive
with an overlap-corrected t ≥ 2.0.

> This hypothesis did not exist in the first registration, and neither did any
> benchmark-relative code path anywhere in the repository. Every economic
> conclusion the engine had ever produced was stated against zero. On the
> selection period the book returns **+1.04%** per 63-session period against
> **+5.27%** for the universe it selects from — mean excess **−4.23%**,
> information ratio **−0.83**, alpha **−0.67%**, and 32.9% of periods beating
> the benchmark.
>
> **The engine is expected to fail this test.** It is registered for that
> reason. A forward test whose outcome is not in doubt is not a test.

## Invalidation — the window is void, not merely disappointing

* `config_version` changes during the window.
* Any factor, gate or parameter is retuned using data from inside the window.
* The shortlist is acted on with real capital.
* Fewer than 60% of expected sessions produce a recorded run.
* The benchmark panel is unavailable for any part of the window.

## When re-registration is legitimate

Adding or changing a hypothesis is legitimate **only before the first forward
session lands**. The tertiary test above was added under exactly that condition:
the window had not opened. After the first observation, the registration is
frozen and `--restart` is an admission that the experiment was abandoned, which
must be recorded rather than quietly performed.

If the model changes, the honest move is a NEW registration with a new start
date, and the old window reported as abandoned with its partial result intact.
Do not delete it.

## What is already in place

`Progress.model_fingerprints` records the distinct model fingerprints seen since
the start, and `broken` reports a config drift or a hash mismatch on every
status call.

> **Correcting the audit.** The original dossier listed M5 — "no model
> fingerprint travels with a recorded run" — as an open finding. That was
> wrong: `model_fingerprint` is written on each recorded run and
> `progress()` already collects the distinct values and flags more than one.
> The finding is withdrawn. It is recorded here rather than deleted, because
> an audit that quietly drops its own mistakes is not an audit.

## Reading the result

`prosignal research forward` prints elapsed sessions against target, the
fingerprints seen, and anything in `broken`. It deliberately will not grade the
test before the window closes. Wait for it.
