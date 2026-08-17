# Production Readiness Audit — 2026-08-17

I built this codebase. This audit tests my own prior claims rather than
restating them. Every number below came from running something.

---

## 1. EXECUTIVE VERDICT

### MVP / PARTIALLY WORKING

The analysis engine is real: it runs end-to-end on live NSE data in 3.8s and
produces a traceable decision. But measured against **your own stated
definition of done** —

> "a non-technical user can click one button and reliably receive a correctly
> computed, explainable stock-analysis result"

— the system is **NOT FUNCTIONAL**, because there is no button, no app to open,
and no path for a non-technical user. Verified by count, not by reading:

| Component | Count found |
|---|---|
| Frontend files (html/jsx/tsx/vue/js) | **0** |
| FastAPI/Flask app | **0** |
| HTTP route decorators | **0** |
| Telegram / notification / webhook | **0** |
| Job queue (celery/rq/apscheduler) | **0** |
| Database ORM | **0** |
| Files in `data/ledger/` | **0** |

The only entry point is `prosignal.cli:main`. Phase 5 of your brief — "the most
important functional requirement" — is 0% built.

Nothing is mocked or faked. The gap is absence, not deception.

---

## 2. WHAT ACTUALLY WORKS (verified this session)

| Claim | Evidence |
|---|---|
| Eight-stage pipeline runs end to end | `prosignal analyse run` → 200 considered → 19 eligible → 13 survived → 1 triggered → NO TRADE, 3.8s |
| **No pipeline-level look-ahead** | Instrumented `DataStore.read_prices/read_indices` to fail on any row dated after `as_of`, ran with **278 sessions of future data present**. Zero violations. |
| Indicator maths correct | Cross-checked against reference implementations written from the formula: `true_range` diff 0.00e+00, `sma` 0.00e+00, `ema` 2.8e-14, `momentum_12_1` 0.00e+00, Wilder ATR tail 0.03% (seeding convention, converges) |
| Reproducible | Two runs → identical funnel and identical ticker list (asserted by test) |
| No silent failures | `except:` → 0. `except Exception: pass` → 0. The 5 `except…continue` sites all record the error first |
| No secrets committed | 0 hardcoded keys/tokens, no `.env` tracked |
| Emits no probability | A test walks the serialised output and fails on any field name implying likelihood |
| 275 tests pass | Full suite, 31s |

---

## 3. WHAT DOES NOT WORK / DOES NOT EXIST

**P0 — the product does not exist as specified**
- No UI, no API, no button. CLI only.
- **Runs are not persisted.** `LedgerRow` is a defined contract and
  `data/ledger` is configured — and contains **0 files**. Every run is lost.
  There is no audit trail, so Phase 64 (signal audit trail) is unimplemented
  despite the contract implying it.
- No job management: no job IDs, no states, no duplicate-click protection, no
  cancellation, no timeout. Clicking twice would run twice.
- No notifications.

**P0 — no evidence of edge**
- No backtest driver exists. The CPCV/PBO/DSR code in `validation/` has **never
  been called by anything**.
- 179 parameters, **127 still UNVALIDATED**.
- **The system may produce signals, but there is insufficient evidence to claim
  those signals have predictive edge.** That sentence is the honest status.

**P0 — insufficient data to ever validate**
- Store holds **318 sessions**. `min_history_sessions` is 300.
- Therefore the engine can analyse **18 dates — 5.7% of held data**.
- Running `as_of` 40 sessions back returned `passed_eligibility: 0` because no
  stock had 300 sessions of prior history.
- A 5-year CPCV needs ~1,550 sessions. **A backtest is arithmetically
  impossible today.**

**P1**
- `yfinance_provider.py` has **zero tests** and logs per-symbol failures at
  `debug`, so partial corporate-action coverage degrades silently at default
  INFO level.
- Sequential execution only; no parallelism, no rate-limit backpressure beyond
  a fixed inter-request delay.

---

## 4. DEAD CODE

Aggressive sweep found very little, because earlier audits removed it.

| Item | Status | Evidence | Action |
|---|---|---|---|
| `validation/` (CPCV, PBO, DSR) | **UNUSED** — 28 tests, but no production caller | `grep` shows no import from pipeline or stages | **KEEP.** It is the harness Phase 1 of the build plan needs. Not dead, unwired. |
| `data/raw/` audit tier | **UNUSED** | `audit_raw.enabled: false` | Keep — cheap, off by default |
| pyflakes findings | 1 (was 24) | `python -m pyflakes src tests` | Cosmetic |
| Unconsumed config | 22 of 179 (12%, was 57%) | leaf-name scan of runtime code | Mostly ledger/API sections for unbuilt parts |

---

## 5. FAKE / PLACEHOLDER / MOCK FUNCTIONALITY

**None found.** No hardcoded outputs, no mocked responses in production paths,
no fabricated probabilities, no invented performance numbers. Checks that
cannot run return `NOT_TESTABLE` and say why — verified on a live card:

```
NOT TESTABLE WITH CURRENT DATA
  ? insider_activity: SEBI PIT disclosures not ingested
  ? earnings_distortion: only estimated results dates available
  ? regulatory_shock: regulatory events feed empty
```

The one thing that could be *mistaken* for fake sophistication is `validation/`
— real, correct, tested statistics that nothing calls yet.

---

## 6. CRITICAL BUGS

**P0-1 — Runs are never persisted.**
Impact: no audit trail, no reproducibility in practice, no forward-performance
dataset. Root cause: ledger writer never implemented. Fix: write `LedgerRow`
per run at the end of `pipeline.run_analysis`.

**P0-2 — No backtest driver.**
Impact: zero evidence of edge; every threshold is a hypothesis.
Fix: walk-forward driver calling the existing CPCV splitter.

**P1-1 — Earnings gate rejects 85% of the universe.**
171 of 200 rejected for `earnings_conflict`. The 45-session blackout is the
expected *holding* window; against quarterly reporters it excludes most names
most of the time. Compounded by the dates being **yfinance estimates, not
exchange-confirmed**. Fix: separate "pre-results blackout" (~5-10 sessions)
from "results fall inside my hold" (a position-management concern, not
eligibility).

**P1-2 — Participation gate is unreachable.**
`min_adtv_inr` (₹5 Cr) is 4× stricter than `position_value /
max_participation_of_adtv` (₹1.25 Cr), so the gate can never bind at ₹10L
capital — despite being Tier-A search parameter #1. Pinned by a test.

**P2-1 — yfinance failures logged at debug.** Raise to warning and surface in
the manifest.

---

## 7. TRADING LOGIC INTEGRITY

- **Indicators: correct.** Independently verified above.
- **Signals: deterministic.** Same inputs → same funnel, asserted.
- **Scoring: honest.** A cross-sectional rank in [0,1], labelled as a rank.
- **Probability claims: none made.** Correct, since nothing is calibrated.
- **Look-ahead: none found**, at indicator level (truncation invariance) and
  pipeline level (instrumented, 278 sessions of future data present).
- **Survivorship: detected, not solved.** `survivorship_risk` is set when the
  snapshot post-dates the decision, and `pre_snapshot_policy: halt` can refuse
  to run. But membership history genuinely does not exist, so any long backtest
  will carry bias of unknown size.
- **Backtesting: does not exist.** Not "untrustworthy" — absent.
- **Risk calculations: valid and sensibly constructed.** ATR-derived stops with
  floor/ceiling, thesis invalidation kept distinct from the stop, position size
  = min(risk budget, capital slot, liquidity cap) with the binding constraint
  named. All UNVALIDATED.

---

## 8. DATA INTEGRITY

**Trustworthy for what it covers.** NSE archives are the authoritative primary
source, no auth, stable. 1.03M price rows, 318 sessions, calendar discovered
from 404-vs-200 rather than hardcoded. Store is atomic and idempotent, no
forward-fill. Corporate actions detected including one true-positive unadjusted
10:1 split.

**Limits:** only ~16 months of history; sector labels are current-vintage; no
point-in-time fundamentals; earnings dates are estimates.

---

## 9. SINGLE-BUTTON END-TO-END TEST

```
CLICK        -> DOES NOT EXIST (no UI, no API, no endpoint)
   |
   +-- substitute: `prosignal analyse run` (CLI, technical user only)
EXECUTION    -> OK   9 stages, 3.8s
DATA         -> OK   318 sessions, real NSE archives
ANALYSIS     -> OK   200 -> 19 -> 13 -> 1 -> 0
SIGNAL       -> OK   NO TRADE + funnel + 5 closest candidates + gate failed
PERSIST      -> FAILS  nothing written to data/ledger
UI           -> DOES NOT EXIST
```

**The flow succeeds from execution to signal, and fails at both ends:** no
entry point a non-technical user can reach, and no persistence of the result.

---

## 10. PRODUCTION READINESS SCORES

| Dimension | Score | Why |
|---|---|---|
| Code quality | 8/10 | 1 pyflakes finding, consistent contracts, no silent failures |
| Architecture | 7/10 | Clean stage separation; no job layer, no persistence |
| Data integrity | 7/10 | Authoritative source, verified no look-ahead; only 16 months, no PIT fundamentals |
| Quant logic | 6/10 | Maths verified correct; **zero validation of edge** |
| Testing | 7/10 | 275 tests, property-based look-ahead tests; yfinance untested |
| Reliability | 4/10 | Fails loudly, but no jobs/retries/recovery/persistence |
| Security | 8/10 | No secrets, no network exposure (nothing to expose) |
| Performance | 7/10 | 3.8s for 200 names; sequential, untested at scale |
| UX | **1/10** | CLI only. No app, no button |
| Observability | 4/10 | Structured logs; no run history, no metrics, no drift |
| **Production readiness** | **3/10** | Engine real; product absent; edge unproven |

---

## 11. MISSING PIECES

**Must build (P0)**
1. Ledger writer — persist every run. ~100 lines, unblocks everything else.
2. Backtest driver — walk-forward over the CPCV splitter. Without it there is
   no edge evidence and no calibration.
3. More history — 1,500+ sessions. Currently only 18 analysable dates.
4. HTTP API + one-page UI with a Run Analysis button and job polling.

**Should build (P1)**
5. Job manager (IDs, states, duplicate-click guard, timeout).
6. Fix the earnings gate.
7. Tests for `yfinance_provider`.
8. Calibration — only after (2) and (3), and only if it earns the word.

**Could build later (P3)**
9. Notifications, parallel fetch, drift monitoring, paper trading.

---

## 12. RECOMMENDED ARCHITECTURE (deliberately small)

```
Browser (one page, one button, polls job status)
      |
   FastAPI  POST /runs -> job_id      GET /runs/{id} -> status + result
      |
  JobManager (in-process, single-flight lock, SQLite job table)
      |
  pipeline.run_analysis()      <-- EXISTS AND WORKS
      |
  DataStore (parquet)  +  Ledger (append-only JSONL)
```

No celery, no redis, no microservices. One process, SQLite for jobs, JSONL for
the ledger. The engine already works; it needs a front door and a memory.

---

## 13. EXACT USER JOURNEY (target)

1. Open `localhost:8000`. See market regime and the date of the last analysis.
2. Click **RUN ANALYSIS**. Button disables immediately (single-flight).
3. Watch 9 stage labels tick over (~5s).
4. Receive either NO TRADE with the funnel, or 1-2 cards with entry / stop /
   targets / R:R / evidence / contrarian evidence / what would invalidate it.
5. Every card states: probability unavailable, thresholds unvalidated.
6. User decides and places any order themselves. The system never trades.

---

## 14. FINAL BUILD PLAN

**Phase 1 — Make it correct**
Ledger writer; fix the earnings gate; decide on the participation-gate finding.

**Phase 2 — Make it reliable**
Backtest driver over the CPCV splitter; ingest 1,500+ sessions; run PBO/DSR and
report the result *even if it is bad*.

**Phase 3 — Make it testable**
Tests for yfinance; failure-path tests (API down, partial data, duplicate job).

**Phase 4 — Make it production-ready**
FastAPI + JobManager + SQLite; single-flight lock; timeouts.

**Phase 5 — Improve UX**
One-page UI, job polling, regime strip, cards, NO-TRADE funnel.

---

## UNVERIFIED

- Behaviour under API failure, rate limiting, or partial provider responses:
  **not tested**. Requires fault injection.
- Concurrency: **not tested**. No job layer exists to test.
- Performance beyond 200 symbols: **not measured**.
- Any claim of predictive edge: **unverifiable today** — 18 analysable dates.
