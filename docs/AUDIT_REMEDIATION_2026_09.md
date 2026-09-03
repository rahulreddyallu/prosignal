# Quant-audit remediation — 2026-09

End-to-end tracker for the adversarial quant review. Every item maps a finding to
a concrete action, a type, and a status. **The governing rule:** the shipped
model's signs and weights are frozen in `features/v3.py`, hashed into
`config_sha256`, matched by `tests/test_v3_score.py`, and tied to two sealed
holdouts. Fixes therefore split into:

- **SAFE (merge now):** presentation, docs, and experiment harnesses. Never
  changes what the engine recommends BUY.
- **EPOCH-GATED:** anything that moves a sign, weight, factor set, or the traded
  book. These do not ship until an experiment justifies them AND the frozen
  pipeline is re-run and re-sealed (a new epoch). Hand-editing a frozen number
  to "fix" it silently ships an unvalidated model — the exact failure the repo
  and `MEMORY.md` ("production is not the validated model") warn against.

## Priority ladder

| # | Finding (audit §) | Action | Type | Status |
|---|---|---|---|---|
| P0-1 | "quality" shown as *Business quality* but fitted anti-quality; "ownership" ≠ ownership (§D) | Honest card labels + config sign-warning | SAFE | ✅ merged (`fae96e2`) |
| P0-2 | Book selected on holdout-overlapping data; only clean OOS book test lost (§I, K-1) | `recheck_status.py` (forward-gated status) + document that clean K-1 is forward-only | SAFE harness | ✅ harness; ⏳ verdict forward-gated |
| P1-3 | Blend may not beat momentum-alone (§K-2) | `audit_2026_09.py` EXP-C (IC-level precursor) | SAFE harness | ✅ built; run ↓ |
| P1-4 | Perf-proportional weights overfit / anti-OOS (§H, #4) | EXP-D equal- vs frozen-weight, same sub-scores | SAFE harness | ✅ built; run ↓ |
| P1-5 | "quality" sign may be a 2020–21 artifact (§K-4) | EXP-A sign-stability across halves/thirds | SAFE harness | ✅ built; run ↓ |
| P1-6 | Uncalibrated impact coeff gates net-of-cost (§H, K-6) | Cost break-even driver (needs book sim) | EPOCH-adjacent | ⏳ next round |
| P2-7 | Delivery may be a liquidity/vol proxy (§K-3) | EXP-B incremental IC after controls | SAFE harness | ✅ built; run ↓ |
| P2-8 | Residual panel survivorship inflates OOS IC (§E, K-5) | Survivorship-bounding run (needs delisted names) | DATA-gated | ⏳ data required |
| P2-9 | Momentum factor theatre — prox/voladj dupes (§D, F) | Prune experiment → new epoch if it holds | EPOCH-gated | ⏳ after EXP results |
| P3-10 | Doc contradictions / withdrawn-number consistency | Config note done; README/CHANGELOG sweep | SAFE | 🔄 partial |

## What ships in this round (safe, merged)

1. **Honest theme labels** (`presentation/viewmodel.py`): `quality → "Low-margin
   tilt"`, `ownership → "Delivery strength"`. Keys/weights/signs untouched;
   `test_v3_score.py` still green.
2. **Config sign-warning** on the `quality` factor block.
3. **Runnable experiment harnesses** under `research/v3/experiments/`:
   - `audit_2026_09.py` — EXP-A (sign stability), EXP-B (delivery incremental),
     EXP-C (blend vs single theme), EXP-D (equal vs frozen weights).
   - `recheck_status.py` — K-1 forward-gated ranking status via the shipped
     `recheck()`.

## Results — `audit_2026_09.py`

_Filled from `research/v3/experiments/results_2026_09.json` after the run._

| Experiment | Question | Result | Verdict |
|---|---|---|---|
| EXP-A | Does "quality" hold its (negative) sign across sub-periods? | _pending_ | — |
| EXP-B | Does delivery add IC after other themes + liquidity/vol? | _pending_ | — |
| EXP-C | Does the 5-theme blend beat the best single theme (IC)? | _pending_ | — |
| EXP-D | Does equal weight match/beat frozen perf-weights? | _pending_ | — |

## Epoch-gated decisions (do NOT hand-edit; require a re-fit + re-seal)

- **Drop / re-sign "quality"** — only if EXP-A shows the negative sign is a
  sub-period artifact (sign flips across thirds).
- **Move to equal theme weights** — only if EXP-D shows equal ≥ frozen OOS.
- **Prune momentum to ~4 representatives** — cosmetic to weight (two-level cap
  already contains it); do it in the same epoch as any of the above, never alone.
- **6-name book concentration** — do not re-baseline off the 378-cell surface;
  let the forward test / `recheck` grade it on non-overlapping data.

## Definition of done (per item)

- SAFE items: merged to `main`, tests green.
- EPOCH-gated items: experiment result recorded here → if it justifies a change,
  open a new epoch (re-run `freeze`, re-seal a holdout, re-register the forward
  test) → only then update `features/v3.py`.
