"""
Position and portfolio management for Cassandra.

This module is the financial memory of the system. It tracks free cash, open
positions, and the result of each closed trade, both during a backtest and in a
live simulation.

Each trade reserves a cash amount on entry. On exit, that amount is returned to
the portfolio adjusted by the price move in the direction of the bet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .risk import Order


@dataclass
class Position:
    """An open trade in the portfolio."""

    ticker: str
    capital: float
    entry_price: float
    is_long: bool
    stop_price: float
    target_price: float


@dataclass
class Portfolio:
    """Free cash, open positions, and closed trade history."""

    capital: float
    positions: dict[str, Position] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def open_position(self, ticker: str, order: Order, is_long: bool) -> bool:
        """Opens a position if there is enough cash and the ticker is not already held.

        Returns True when the position is created and False when the trade is
        rejected because of insufficient cash or a duplicate ticker.
        """
        if ticker in self.positions or order.allocated_capital <= 0:
            return False
        if order.allocated_capital > self.capital + 1e-9:
            return False

        self.capital -= order.allocated_capital
        self.positions[ticker] = Position(
            ticker=ticker,
            capital=order.allocated_capital,
            entry_price=order.entry_price,
            is_long=is_long,
            stop_price=order.stop_price,
            target_price=order.target_price,
        )
        return True

    def close_position(self, ticker: str, price: float) -> float:
        """Closes the position and returns the financial result of the trade."""
        pos = self.positions.pop(ticker, None)
        if pos is None:
            return 0.0

        move = price / pos.entry_price - 1
        if not pos.is_long:
            # for a short trade, gain comes from a price drop; loss is capped at allocated capital
            move = max(-move, -1.0)

        result = pos.capital * move
        self.capital += pos.capital + result
        self.history.append(
            {
                "ticker": ticker,
                "return": move,
                "result": result,
                "is_long": pos.is_long,
            }
        )
        return result

    def net_worth(self, prices: dict[str, float]) -> float:
        """Sums free cash with the current value of all open positions."""
        total = self.capital
        for ticker, pos in self.positions.items():
            price = prices.get(ticker, pos.entry_price)
            move = price / pos.entry_price - 1
            if not pos.is_long:
                move = -move
            total += pos.capital * (1 + move)
        return total
