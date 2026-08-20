# Data licensing — open question for the operator

**This is a legal question. It is flagged here, not resolved here.**

The engine currently depends on two feeds whose terms were read for research
use and have not been reviewed for real-money or commercial deployment.

| Feed | Used for | Question |
|---|---|---|
| NSE archives (bhavcopy, `sec_bhavdata_full`, `ind_close_all`, constituent lists, corporate actions, board meetings) | every price, delivery figure, index level, sector label and corporate action | Do NSE's website terms permit systematic download and commercial use of archive files, or is a data licence required at some usage tier? |
| Yahoo Finance via `yfinance` | statements, earnings surprises, corporate-action cross-check, secondary price agreement | Yahoo's terms restrict redistribution and commercial use. `yfinance` is an unofficial client and its use does not itself grant any right. Is this permissible for a commercially deployed system? |

## Why it matters before capital, not after

Both dependencies are load-bearing. NSE is the only price source. Yahoo is the
only statement source, and after the point-in-time universe change it is also
the only source of share counts, which market capitalisation and therefore the
entire value family depend on.

If either turns out to require a licence for commercial use, the options are to
obtain one or to replace the feed — and replacing the statement feed would mean
rebuilding the fundamental layer against a different vintage of history.

## Not a blocker for research

Nothing here affects the current posture. The system is RESEARCH ONLY and its
output is read by its author. This becomes live the moment output informs
third-party capital or the system is offered as a service.
