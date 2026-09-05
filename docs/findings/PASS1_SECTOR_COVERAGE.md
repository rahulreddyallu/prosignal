# Pass 1 — sector rank coverage, and why 80% is not reachable from free sources

**Measured 2026-09-04** on the v3 score panel: 204,425 rows, 380 signal dates,
2018-11-27 → 2026-08-03, 1,414 distinct symbols. Store manifest
`024097d7280e36cc`. Nothing here computes a factor; this is a data finding.

## What the gate asks, and what was achieved

v10 Pass 1's gate G1 asks for **sector rank coverage ≥ 80% of rows**, from a
recorded baseline of 41.2%.

| | symbols in map | pooled coverage |
|---|---|---|
| Before (NIFTY 500 + MIDCAP 150 + SMALLCAP 250) | 500 | **46.9%** |
| After (+ NIFTY TOTAL MARKET + MICROCAP 250) | 754 | **67.7%** |
| Gate | — | 80% |

**The gate is NOT met. 67.7% is the ceiling from free NSE sources**, and the
remainder is not a matter of trying harder — see below.

"Coverage" here is the engine's own rule, not a looser one: a row counts as
covered when `features/v3.py:sector_neutral_rank` ranks it *within its sector*,
which requires the sector to hold at least `MIN_SECTOR_NAMES = 12` names on that
date. Everything else falls into one `__RESID__` pool that is not a sector.

## Why the last 12 points are unreachable

729 of the panel's 1,414 symbols are absent from every current NSE index file.
They account for **23.9% of panel rows**. These are names that were liquid
enough to enter the universe and have since left the indices or delisted —
`ABAN`, `ADANITRANS`, `63MOONS`, `5PAISA` and so on.

Three sources were probed for their classification and all three fail:

| Source | Result |
|---|---|
| NSE index constituent CSVs | Current membership only. No dated history is published. |
| `EQUITY_L.csv` (full equity master, 2,568 rows) | **No industry column at all.** |
| `corporates-financial-results` `industry` field | **Empty on every row** of all ten missing symbols probed, despite 27–148 result rows each. |

There is no free NSE source for the industry classification of a delisted or
dropped Indian equity. Assigning a name its *current* sector for past dates
would also be a lookahead where a company has been reclassified, and inventing
one would be worse. Per the repository's `NOT_TESTABLE` convention and Pass 1's
own stop rule — *"mark that field NOT_TESTABLE and move on. Do not interpolate."*
— the gap is recorded rather than filled.

## The finding worth more than the gate

Coverage is **not stationary**. Measured by year on the same panel:

| year | coverage |
|---|---|
| 2018 | 52.2% |
| 2019 | 51.8% |
| 2020 | 58.2% |
| 2021 | 64.6% |
| 2022 | 66.0% |
| 2023 | 68.4% |
| 2024 | 66.6% |
| 2025 | 74.8% |
| 2026 | **78.7%** |

It rises almost monotonically toward the present, because today's index
membership describes today's market and progressively less of each earlier one.

**This is a survivorship artefact inside the neutralisation step itself, and it
had not been quantified.** Its consequence is specific: the composite is
*less* sector-neutral the further back you look, so any factor with a sector
tilt is progressively less neutralised in the older half of the panel — the
half that carries most of the independent observations. That is a bias in the
direction of flattering long-window results, and it applies to every number
this engine has computed over a multi-year window.

It does not invalidate anything on its own. It is a term nobody had put a size
on, and it is now sized.

## What would actually close it

In rough order of cost:

1. **A paid classification vendor** with delisted coverage (CMIE Prowess,
   Refinitiv). Prowess is what the IIMA factor library uses, which is a point
   in its favour for Pass 2's external attribution as well.
2. **Archived NSE index files.** If dated constituent CSVs can be recovered
   from a mirror the way the 2012-2017 bhavcopy was, membership becomes
   point-in-time and the classification arrives with it. The bhavcopy
   reconciliation at 0.0000 bp over 43,359 closes is the template.
3. **Accept 67.7% and neutralise differently.** If the sector map cannot be
   made point-in-time, a sector-neutral rank is the wrong instrument for the
   older panel, and a statistical neutralisation — residualising against the
   first few principal components of the return covariance — needs no
   classification at all. That is a Pass 3 decision, not a Pass 1 one, and it
   is noted here so the option is on the record.

## Reproducing

```bash
prosignal data ingest          # refreshes the sector map from all 8 index files
```

The two files added to `providers.nse_archives.index_constituent_files` are
`ind_niftytotalmarket_list.csv` (750 names, matching the universe cap) and
`ind_niftymicrocap250_list.csv`. Both returned HTTP 200 on 2026-09-04.
