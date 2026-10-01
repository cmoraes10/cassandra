"""
Risk management for Cassandra.

A good signal is worthless without position sizing and exit rules. This module
decides how much of the available capital goes into each trade and where the
stop and target levels are.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskRules:
    """Parameters that govern the size and limits of each trade.

    trade_fraction is the share of total capital allocated to a single trade.
    stop_loss and take_profit are price moves, relative to entry, that close
    the position. max_positions caps concentration.
    """

    trade_fraction: float = 0.10
    stop_loss: float = 0.35
    take_profit: float = 0.80
    max_positions: int = 4


@dataclass
class Order:
    """A sized trade ready to enter the portfolio."""

    allocated_capital: float
    entry_price: float
    stop_price: float
    target_price: float


def size_order(
    base_capital: float,
    price: float,
    is_long: bool,
    rules: RiskRules | None = None,
) -> Order:
    """Builds the order from available capital and the current price.

    Computes how much money goes into the trade and projects the stop and target
    prices. For a long trade the stop is below entry and the target is above.
    For a short trade the logic is reversed.
    """
    if price <= 0:
        raise ValueError("price must be positive to size the order.")

    rules = rules or RiskRules()
    allocated_capital = base_capital * rules.trade_fraction

    if is_long:
        stop_price = price * (1 - rules.stop_loss)
        target_price = price * (1 + rules.take_profit)
    else:
        stop_price = price * (1 + rules.stop_loss)
        target_price = price * (1 - rules.take_profit)

    return Order(
        allocated_capital=allocated_capital,
        entry_price=price,
        stop_price=stop_price,
        target_price=target_price,
    )
