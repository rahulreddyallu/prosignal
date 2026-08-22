# Phase 14 — Momentum concentration

**Question (Gap 5).** Momentum is 41% of the model's IC across three factors.
Is that three sources of information or one?

**Answer. One.**

## The three shipped momentum factors are one dimension

Selection period, 45 dates, rank IC against the 63-session forward return:

| construction | IC | t |
|---|---|---|
| `mom_6_1` alone | +0.0649 | +4.22 |
| `prox_52w` alone | +0.0583 | +3.27 |
| `resid_mom` alone | +0.0705 | +4.36 |
| `mom_6_1` + `prox_52w` | +0.0684 | +4.09 |
| all three, equal weight | +0.0752 | +4.48 |
| **first principal component of the three** | **+0.0755** | **+4.48** |

A single principal component reproduces the equal-weighted block to three
decimal places. The three factors add +0.005 of IC over the best single one.
**The 41% share is one latent factor wearing three names.**

## Nine variants, 2.57 effective dimensions

Pre-registered from the literature before any result was seen: `mom_12_1`
(Jegadeesh & Titman 1993), `mom_6_1`, `mom_3_1`, `prox_52w` (George & Hwang
2004), `resid_mom` (Blitz, Huij & Martens 2011), `ts_mom` (Moskowitz, Ooi &
Pedersen 2012), `vol_adj_mom` (Barroso & Santa-Clara 2015), `mom_consist`
(Grinblatt & Moskowitz 2004), `accel` (Chan, Jegadeesh & Lakonishok 1996).

- PC1 carries **59%** of the variance, PC2 a further 16%
- eigenvalues above 1: **2**
- participation ratio: **2.57** of 9

Two pairs are near-duplicates:

- `resid_mom` vs `mom_12_1`: **ρ = +0.96**. Blitz et al. strip market beta so
  the residual carries the premium with less crash exposure. At 0.96 the
  *ranking* is barely different. The crash characteristics do differ, slightly
  and in the predicted direction — worst period −6.95% against −7.56%, standard
  deviation 5.20% against 5.40% — which is a Gap 6 result, not a Gap 5 one.
- `mom_6_1` vs `vol_adj_mom`: **ρ = +0.97**. Scaling momentum by its own
  volatility does essentially nothing to the cross-sectional ordering.

`accel` is the one genuinely orthogonal variant (ρ ≈ 0.01 against everything)
and it carries no signal: IC −0.0090, t −0.79. **Independence without
information.**

## Two candidates passed the screen and failed the holdout

Marginal IC over the shipped three, residualised within each date:

| variant | marginal IC | t | |
|---|---|---|---|
| `ts_mom` | +0.0189 | +2.85 | screened in |
| `mom_consist` | +0.0186 | +2.67 | screened in |
| `mom_12_1` | +0.0129 | +1.00 | spanned |
| `mom_3_1` | −0.0026 | −0.25 | spanned |
| `vol_adj_mom` | −0.0025 | −0.24 | spanned |
| `accel` | −0.0092 | −0.86 | spanned |

Nine variants were tested. Bonferroni at α = 0.05 across nine wants |t| ≥ 2.77,
so `ts_mom` barely clears and `mom_consist` does not — before accounting for the
fact that these are the maximum of nine correlated draws. The selection-period
result therefore decided nothing, and the holdout was read once.

| model | k | selection IC | holdout IC | ΔIC vs shipped | Δexcess |
|---|---|---|---|---|---|
| shipped 17 | 17 | +0.0821 | +0.0907 | — | — |
| + `ts_mom` | 18 | +0.0834 | +0.0922 | +0.0015 | +0.09% |
| + `mom_consist` | 18 | +0.0835 | +0.0881 | −0.0027 | +0.20% |
| + both | 19 | +0.0844 | +0.0893 | −0.0014 | +0.01% |

Every variant improved the selection period. **None improved the holdout.**

**Both rejected. Nothing shipped.**

This is PBO 44.3% reproducing under controlled conditions: candidates found at
marginal t ≈ 2.7–2.9 out of nine trials look better in-sample and deliver
nothing out-of-sample. The correct response to a 41% concentration is not to
manufacture a fourth momentum factor.

## What this leaves

The momentum concentration is real and is **not** reducible by adding momentum
variants — they are all the same factor. Genuine diversification has to come
from outside the family. On this store that means delivery (26 independent
observations) and value (11), which is a data-supply problem, not a modelling
one.

## Method note

An earlier draft of this measurement reported PC1 with IC −0.0107 (t −0.56) and
would have concluded that no single dimension carries the block. That was a bug:
SVD sign is arbitrary, and unpinned it flips from date to date so the pooled IC
collapses for a reason unrelated to the factor. With the loading oriented toward
the mean of its inputs, PC1 gives +0.0755. The conclusion inverted once the sign
was fixed.
