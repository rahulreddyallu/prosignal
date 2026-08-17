# Production Readiness — Truth Report, 2026-08-17

Every number here came from running something. Nothing is estimated.

---

## EXECUTIVE VERDICT

### PRODUCTION READY WITH LIMITATIONS (engineering) — NO DEMONSTRATED EDGE (strategy)

These are two separate verdicts and conflating them would be the most dangerous
thing this report could do.

**The software is ready.** One button, real data, real analysis, persisted,
recoverable, tested.

**The strategy is not.** A real walk-forward backtest now exists and its answer
is negative. Do not trade this with money.

---

## QUANT VERDICT — the strategy does NOT demonstrate an edge

A genuine walk-forward backtest was executed: 104 decision dates from
2024-07-11 to 2026-08-13, entry at the **next session's open**, full Indian
transaction costs, pessimistic stop/target resolution.

| Metric | Value |
|---|---|
| Decision dates | 104 |
| Produced a signal | 26 |
| **NO TRADE** | **78 (75%)** |
| Trades | 37 |
| Win rate | 48.6% |
| Mean net return / trade | **+1.19%** |
| **Median net return / trade** | **−5.01%** |
| Profit factor | 1.245 |
| Max drawdown | **−49.8%** |
| Sharpe per trade | 0.103 |
| Cost drag per trade | 0.38% |
| Avg holding | 19.4 sessions |
| Exits | 19 stop / 17 target / 1 time |

### The statistical answer

| Test | Result |
|---|---|
| PSR vs zero | 73.2% |
| Honest trial count | 540 |
| **Deflated Sharpe Ratio** | **0.7%** (bar: 95%) |
| **Passes** | **NO** |

> *"Given 540 trials, an observed Sharpe of 0.102 is not distinguishable from
> the best of that many coin flips. Simplify the model or gather more data — do
> not search further."*

**This is not an artefact of the trial count.** Even pretending only ONE
configuration was ever tried, PSR is 73.2% — still far below the 95% bar.

**PBO: NOT COMPUTED.** It requires returns across many candidate
configurations; only one has been run. Reporting a PBO from a single column
would be fabrication.

### Reading the numbers as a trader

The mean is positive while the **median is −5%**. A handful of winners carry the
whole result; most trades lose. Combined with a −49.8% drawdown and 37
observations, this is the signature of a strategy that has not been shown to
work. Profit factor 1.245 is inside the range noise produces at this sample size.

**Weaknesses:** tiny sample (37 trades); 2.9 years covering one broadly rising
market, so bear-regime behaviour is untested; NIFTY 200 membership is
current-vintage, so survivorship bias of unknown size is present.

---

## ENGINEERING VERDICT

| Area | State |
|---|---|
| Tests | **306 passing** (was 275) |
| Ledger | Every run persisted to append-only JSONL, fsync'd, fatal on failure |
| Jobs | SQLite, 5 states, **single-flight verified**, stale-job reaping verified |
| API | FastAPI, `/health` + `/ready` separated, thin handlers |
| UI | One page, one button, progress, results, NO-TRADE funnel, error state |
| Failure handling | Zero bare excepts; failures recorded with traceback; a failed run returns 409, never a signal |
| Look-ahead | **Verified absent** at indicator and pipeline level |
| Data | **718 sessions, 2.9 years**, 418 analysable dates (was 318 / 18) |
| Security | No secrets committed; binds to 127.0.0.1 |
| Runtime | ~4s for 200 names |

---

## PRODUCT VERDICT — can a non-technical person use it?

| Question | Answer |
|---|---|
| Open the application? | **Yes** — `http://127.0.0.1:8000` |
| Click one button? | **Yes** — RUN ANALYSIS |
| Understand what happened? | **Yes** — funnel table shows every gate |
| Understand the signal? | **Yes** — why / why-wrong / not-testable / exits |
| Know when data was generated? | **Yes** — data date and analysis time shown |
| Know when no trade exists? | **Yes** — explicit, framed as expected behaviour |
| Understand errors? | **Yes** — named failure + run ID + "no signals were issued" |

---

## SCORES

| Dimension | Before | Now |
|---|---|---|
| Code quality | 8 | 8 |
| Architecture | 7 | 8 |
| Data integrity | 7 | 8 |
| Quant logic | 6 | **6** — maths correct, edge disproven |
| Testing | 7 | 8 |
| Reliability | 4 | 8 |
| Security | 8 | 8 |
| Performance | 7 | 8 |
| UX | 1 | **8** |
| Observability | 4 | 7 |
| **Production readiness** | **3** | **7** |

Quant logic stays at 6 deliberately. The engineering improved; the evidence got
worse, because for the first time there is some.

---

## WHAT REMAINS UNVERIFIED

- **PBO** — needs a multi-configuration sweep.
- **Bear-regime behaviour** — the 2.9-year window has no sustained bear market.
- **Point-in-time universe** — current NIFTY 200 membership applied
  historically. Survivorship bias present, magnitude unknown.
- **Telegram** — not built (no credentials, and it is not on the critical path).
- **Fault injection** — provider outage/timeout paths untested against a real
  failing API.
- **Fresh-machine install** — documented, not executed.
- **Multi-user concurrency** — single-process design; untested beyond one user.

---

## HOW TO START

```bash
.venv/bin/uvicorn prosignal.api:create_app --factory --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` and click **RUN ANALYSIS**.

CLI equivalent: `prosignal analyse run`

---

## THE HONEST BOTTOM LINE

The system does what was asked: one button, real data, explainable output, and
it says NO TRADE 75% of the time rather than manufacturing activity.

It also now tells you something you could not know before: **on 2.9 years of
NSE data, this strategy's edge is not statistically distinguishable from
chance.** That is a finding, not a failure of the software — it is precisely
what the validation machinery was built to detect, and it detected it on the
first honest run.

Trading this with real capital is not justified by the current evidence.
