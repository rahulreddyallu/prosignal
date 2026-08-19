"""Pro Stock Signal BOT -- India solo-quant decision-support signal engine.

Ranks and screens NSE-listed equities and returns either a small set of
high-conviction candidates or an explicit ``NO TRADE`` state. It places no
orders and contains no order-routing code.

Output is expressed in bands, gates and stated hypotheses rather than
fabricated precision. Every tunable parameter carries its evidence status in
``config/parameters.yaml`` and is surfaced on every run.

    Decision-support tool. Not financial advice. No trades are placed
    automatically.
"""

from __future__ import annotations

from .version import ENGINE_NAME, ENGINE_VERSION, SCHEMA_VERSION

__all__ = ["ENGINE_NAME", "ENGINE_VERSION", "SCHEMA_VERSION", "__version__"]

__version__ = ENGINE_VERSION
