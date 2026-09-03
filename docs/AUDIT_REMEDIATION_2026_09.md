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
| P1-3 | Blend may not beat momentum-alone (§K-2) | `audit_2026_09.py` EXP-C (IC-level precursor) | SAFE harness | ✅ run — blend wins at IC level; book K-2 open |
| P1-4 | Perf-proportional weights overfit / anti-OOS (§H, #4) | EXP-D equal- vs frozen-weight, same sub-scores | SAFE harness | ✅ run — frozen wins; not supported |
| P1-5 | "quality" sign may be a 2020–21 artifact (§K-4) | EXP-A sign-stability across halves/thirds | SAFE harness | ✅ run — **stable, REFUTED**; keep+relabel |
| P1-6 | Uncalibrated impact coeff gates net-of-cost (§H, K-6) | `cost_sensitivity.py` — exact cost model on real selected names, swept | SAFE harness | ✅ cost side done; full book P&L still gated |
| P2-7 | Delivery may be a liquidity/vol proxy (§K-3) | EXP-B incremental IC after controls | SAFE harness | ✅ run — **not a proxy, REFUTED** |
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

## Results — `audit_2026_09.py`  (run 2026-09-03)

Panel: 199,823 rows, **364 signal dates, 2019-03-19 .. 2026-07-30**, label `y21`.
**Caveat that governs every number here:** signs/weights were fit on 2018-2024,
so full-window IC is **partly in-sample**. Sign STABILITY across sub-periods and
the liquidity/vol control test are the trustworthy reads; IC *levels* are not
out-of-sample. The window does extend ~21 months past the 2024-10 fit end.

**These results OVERTURN three of the audit's own hypotheses. Recorded as such —
a reviewer updates on evidence, and the point of gating model edits behind
experiments is to catch exactly this before deleting a working factor.**

| Exp | Question | Result | Verdict vs audit |
|---|---|---|---|
| EXP-A | Is the inverted "quality" sign a 2020-21 artifact? | quality IC **+0.0374 (t 6.34)**, no sign flip across halves or thirds; net_margin +0.040/+0.029 (no flip), margin_stability +0.009/+0.021 (no flip) | **REFUTED.** Sign is inverted vs the namesake but empirically **stable**. Do NOT drop it. Relabel only (done). |
| EXP-B | Is delivery just a liquidity/vol proxy? | delivery IC raw **t 9.14**; residualised on ADTV+downside-vol **t 9.83** (unchanged) | **REFUTED.** Delivery is real and price-orthogonal, not a liquidity proxy. |
| EXP-B′ | Does delivery add in a JOINT all-theme FM? | partial slope **t 0.36** (momentum t 3.38, quality t 6.12 are the only jointly-significant themes; risk −0.55, reversal 0.80) | Nuance: univariately strong, but partly spanned by momentum+quality at the return level. Worth a cleaner incremental test; not a reason to cut it. |
| EXP-C | Does the 5-theme blend beat the best single theme? | composite spread **+0.0151 (t 6.85)** > best single (quality +0.0116; momentum +0.0112) | **REFUTED (IC level).** The blend adds ~35% over the best single theme. (Net-of-cost book K-2 still open.) |
| EXP-D | Does equal weight beat the frozen perf-weights? | frozen spread **+0.0151** vs equal **+0.0122**; equal has marginally higher IC (0.0609 vs 0.0580) | **Not supported.** Frozen weights win on the headline spread. Finding #4 does not hold at the ranking level. |

### What this means for the model-level items

- **"Quality" / low-margin theme:** downgraded from *INVALID UNTIL FIXED* to
  **KEEP + relabel + monitor**. The sign is stable; the honesty fix (label) is
  the correct and sufficient action. `v3_monitor.review_themes` already flags an
  inversion if it ever starts.
- **Delivery theme:** **KEEP.** Genuine, price-orthogonal, non-liquidity signal.
- **Equal-weight switch:** **do not make it.** No OOS-justified benefit.
- **Blend vs momentum-alone:** the ensemble earns its place at the ranking
  level; the remaining open question is net-of-cost at 6 names (K-2 book).

### K-1 status (`recheck_status.py`, run 2026-09-03)

Verdict **HOLDS** on the last 18 months (rank IC +0.0542 t 4.13, quintile spread
+1.25% t 3.56, null p 0.000), no theme inverted. **But the runner reports the
window is 94% inside sealed holdout A** — so this is a re-reading of the dates
the deploy was already judged on, not independent evidence. The clean K-1
(traded book on dates that never touched the 378-cell surface) is **forward-only**
and accrues as the live window moves past 2026-08. This is the honest status, not
a pass.

### K-6 cost sensitivity (`cost_sensitivity.py`, run 2026-09-03)

Prices the EXACT shipped cost model on the real names the v3 score ranks top-6
(median ADTV ₹34cr; ₹1.67L/position at ₹10L, 6 slots). Round-trip cost and the
annual drag it implies at 34 trades/yr:

| impact coeff | round-trip (median) | annual drag | vs window-A gross (+2.2%) | vs window-B gross (+15.6%) |
|---|---|---|---|---|
| 0.10 (shipped) | 80 bps | 4.5% | **net −2.3%** | net +11.1% |
| 0.25 (2.5×) | 146 bps | 8.3% | net −6.1% | net +7.3% |
| 0.50 (5×) | 256 bps | 14.5% | net −12.3% | net +1.1% |

**The cost question is really a gross-edge question.** At the shipped coefficient
the 6-name book's drag is ~4.5%/yr — below the 10-name/weekly holdout's 9.5%
(slower cadence, as `BOOK_NOTE` predicts), but still **more than window A's entire
gross edge**. Break-even vs window A needs a coefficient of ~0.006 (1/16th the
shipped 0.10); window B survives even at 5×. **The two sealed windows disagree ~7×
on the gross edge, so the net-of-cost verdict is undetermined until the forward
test settles which gross number is real.** (Caveat: uses the 10-name book's gross
as a proxy for the 6-name book's; it is a cost-side bound, not a re-simulated P&L.)

### What remains genuinely unresolved (all BOOK / validation level)

- **K-1** clean-window book test — forward-gated (`recheck_status.py`).
- **K-6 full book P&L** — the *cost side* is now bounded (above); a faithful
  re-simulation of the 6-name book's *gross* edge (floor + cadence, validated
  against the sealed book) is the remaining piece.
- **DSR failure** (0.030 / 0.97) and **holdout-overlap book selection** — these
  are about the traded book and multiple testing, not the ranking, and stand.
- **Gross-edge instability across windows** (A +2.2% vs B +15.6%) — the single
  biggest open question; only forward data resolves it.

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
