# Execution gate

**Status: this system does not place orders, and nothing in it may.**

Prosignal is decision support. It emits BUY / WATCHLIST / NO TRADE and a risk
plan. It has no broker connection, no order object, no execution venue, and no
code path that could acquire one incidentally.

That is not an accident of scope. It is what keeps the system outside SEBI's
retail algorithmic trading framework, in force since **1 April 2026**. Under
that framework:

- an algorithmic order must carry an **exchange-assigned Algo-ID**;
- the **broker is the principal** responsible for the algo's behaviour;
- an algo provider must operate **through a registered broker**, not by
  connecting to an exchange directly.

A system that outputs a recommendation a person reads and acts on is not
covered. A system that sends the order is, entirely, and retrofitting
compliance onto an engine designed without it is not a small change.

## Required before any execution capability is discussed

Every box below must be ticked, in writing, before a design document proposing
order placement is accepted — and this file must be cited by that document.

- [ ] Registered broker identified, and the relationship documented
- [ ] Exchange Algo-ID obtained for each distinct strategy to be automated
- [ ] Broker has reviewed and accepted principal responsibility for the algo
- [ ] Static and dynamic order-rate limits agreed with the broker
- [ ] Kill-switch tested end to end, including a broker-side path that works
      when this system is unreachable
- [ ] Order-audit trail specified: every order traceable to the run_id and the
      exact coefficient version that produced it
- [ ] Legal review of NSE archive terms and Yahoo/yfinance terms for
      **commercial** use, not research use (see `DATA_LICENSING.md`)
- [ ] Independent review of the statistical case, given that the current
      verdict is RESEARCH ONLY
- [ ] Documented rollback: how to revert to decision-support-only within one
      trading session

## Re-check trigger

Re-read this file whenever any of the following is proposed:

- a broker API client, SDK, or credential of any kind
- an "order", "trade ticket", "execution" or "fill" object in the contracts
- a scheduler that acts on output rather than producing it
- any integration described as "closing the loop"

If a change would let the system place, modify, or cancel an order, it does not
belong in this repository until every box above is ticked.
