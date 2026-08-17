"""Pro Stock Signal BOT -- India solo-quant decision-support signal engine.

This package ranks and screens NSE-listed equities and returns either a small
set of high-conviction candidates or an explicit ``NO TRADE`` state. It does
not place orders, and it contains no order-routing code.

It reasons in bands, gates and explicit hypotheses -- never in fabricated
precision. Every tunable parameter is tagged with its evidence status in
``config/parameters.yaml`` and surfaced to the user on every run.

    Decision-support tool. Not financial advice. No trades are placed
    automatically.
"""

from __future__ import annotations

from .version import ENGINE_NAME, ENGINE_VERSION, SCHEMA_VERSION

__all__ = ["ENGINE_NAME", "ENGINE_VERSION", "SCHEMA_VERSION", "__version__"]

__version__ = ENGINE_VERSION
