"""
Backtest engine for Cassandra.

A signal that has never been tested is just a guess. The backtest reconstructs
the past day by day, generates signals as if trading in real time, respects the
risk rules, and measures the result. This is how the strategy proves its value
before risking real money.

An important discipline to avoid lookahead bias: at each point in time the model
only sees prices up to that day. Nothing from the future enters the decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .position import Portfolio
from .risk import RiskRules, size_order
from .simulation import fit_model, simulate
from .signal import BUY, NEUTRAL, SELL, generate_signal


@dataclass
class BacktestResult:
    """Equity curve, closed trades, and performance metrics."""

    equity_curve: pd.Series
    trades: list[dict]
    metrics: dict


def run_backtest(
    prices: pd.Series,
    initial_capital: float = 10000.0,
    train_window: int = 252,
    horizon: int = 21,
    n_paths: int = 2000,
    confidence_threshold: float = 60.0,
    step: int = 5,
    rules: RiskRules | None = None,
    seed: int | None = 42,
) -> BacktestResult:
    """Runs the strategy over the price history of a single ticker.

    At each trading interval the routine updates open positions, checking for
    stops and targets, then generates a new signal using only past prices. When
    the signal points a direction with enough confidence and there is room in the
    portfolio, a trade is opened. At the end everything is liquidated and metrics
    are computed.
    """
    rules = rules or RiskRules()

    minimum = train_window + 2 * step
    if len(prices.dropna()) < minimum:
        raise ValueError(
            f"price history too short for backtest: {len(prices.dropna())} prices, "
            f"at least {minimum} are needed for the chosen window and step."
        )

    portfolio = Portfolio(capital=initial_capital)
    ticker = prices.name or "ASSET"
    dates = prices.index

    records: list[tuple] = []

    for i in range(train_window, len(prices) - 1, step):
        history = prices.iloc[:i]
        today_price = float(prices.iloc[i])
        date = dates[i]

        _update_positions(portfolio, {ticker: today_price})

        model = fit_model(history)
        result = simulate(model, horizon=horizon, n_paths=n_paths, seed=seed)
        signal = generate_signal(result, confidence_threshold=confidence_threshold)

        can_open = (
            signal.direction in (BUY, SELL)
            and ticker not in portfolio.positions
            and len(portfolio.positions) < rules.max_positions
        )
        if can_open:
            base = portfolio.net_worth({ticker: today_price})
            order = size_order(base, today_price, signal.direction == BUY, rules)
            order.allocated_capital = min(order.allocated_capital, portfolio.capital)
            portfolio.open_position(ticker, order, signal.direction == BUY)

        records.append((date, portfolio.net_worth({ticker: today_price})))

    # liquidate anything remaining at the last price in the series
    final_price = float(prices.iloc[-1])
    for open_ticker in list(portfolio.positions):
        portfolio.close_position(open_ticker, final_price)

    if records:
        curve = pd.Series(dict(records))
    else:
        curve = pd.Series(dtype=float)

    periods_per_year = 252 / step
    metrics = _compute_metrics(curve, portfolio.history, initial_capital, periods_per_year)

    return BacktestResult(equity_curve=curve, trades=portfolio.history, metrics=metrics)


def _update_positions(portfolio: Portfolio, prices: dict[str, float]) -> None:
    """Closes positions that hit their stop or target."""
    for ticker in list(portfolio.positions):
        pos = portfolio.positions[ticker]
        price = prices.get(ticker)
        if price is None:
            continue

        if pos.is_long:
            hit_stop = price <= pos.stop_price
            hit_target = price >= pos.target_price
        else:
            hit_stop = price >= pos.stop_price
            hit_target = price <= pos.target_price

        if hit_stop or hit_target:
            portfolio.close_position(ticker, price)


def _compute_metrics(
    curve: pd.Series,
    trades: list[dict],
    initial_capital: float,
    periods_per_year: float,
) -> dict:
    """Computes total return, Sharpe ratio, max drawdown, and win rate."""
    if curve.empty:
        return {
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "num_trades": 0,
            "win_rate": 0.0,
        }

    total_return = float(curve.iloc[-1] / initial_capital - 1)

    changes = curve.pct_change().dropna()
    if len(changes) > 1 and changes.std() > 0:
        sharpe = float(changes.mean() / changes.std() * np.sqrt(periods_per_year))
    else:
        sharpe = 0.0

    peak = curve.cummax()
    drawdown = (curve - peak) / peak
    max_drawdown = float(drawdown.min())

    if trades:
        wins = sum(1 for t in trades if t["result"] > 0)
        win_rate = wins / len(trades)
    else:
        win_rate = 0.0

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "num_trades": len(trades),
        "win_rate": win_rate,
    }
