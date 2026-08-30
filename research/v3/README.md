# v3 search artefacts

Everything the two-level thematic composite was selected on, and the four
sealed-holdout evaluations. Narrative in `research/V3_SEARCH.md`; search code in
`work/v3/`.

| file | what it is |
|---|---|
| `SEAL2.json` | the two windows, sealed before any v3 model was fitted, with row counts and sha256 |
| `FROZEN_V3.json` | the frozen configuration, sha `2b48aee0…8e2be33ea`, fitted 2018-11-27 → 2024-10-25 |
| `FROZEN_V3_PRE_B.json` | the SAME PIPELINE re-run on data ending 2021-02-17, sha `eb2bdb69…03cee4e8`, so window B tests the method rather than a fitted model |
| `HOLDOUT_V3_A_EQUITY.json` | **window A, the result that stands** |
| `HOLDOUT_V3_B_EQUITY.json` | **window B, the result that stands** |
| `HOLDOUT_V3_A.json` | window A's FIRST evaluation, on a universe that still contained ETFs |
| `HOLDOUT_V3_B.json` | window B's first evaluation, same defect |
| `power.json` | the permuted-label test: the whole pipeline re-run on shuffled labels, 40 draws |
| `integrity.json` | the ten-name book against the same null |
| `logs_screen2b.txt` | the placebo screen, 93 factors × 4 horizons |
| `logs_stab2.txt` | both-halves stability for the 33 that cleared |
| `logs_redun2.txt` | the correlation matrix, including the brief's three suspected duplicates |
| `logs_surv2.txt` | the order-independent redundancy survivor pass: 22 admitted, 11 cut with reasons |
| `logs_s3b3.txt` | level-1 combination methods: equal vs IC-weighted vs ridge vs XGBoost |
| `logs_s3e.txt` | the coverage cap, on vs off, on an identical grid |
| `logs_s3d.txt` | the absolute floor: does it ever fire, and what each variant costs |
| `logs_lt.txt` | the cost curve across 704 book settings |
| `logs_eo.txt` | entry-only floor vs population filter |

**Both evaluations of each window are here on purpose.** The first pair scored a
universe that still contained ETFs and gold funds — NSE publishes them in the
same EQ-series bhavcopy — and they took 26.25% of window A's top-ten slots. The
universe was defective, not the configuration, so both windows were re-run with
every parameter untouched. Keeping only the better pair would hide that window A
has now been opened three times, and its t-statistics carry that multiplicity.
