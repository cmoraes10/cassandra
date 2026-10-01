"""
Parameter optimizer for Cassandra.

Strategy parameters such as the simulation horizon and confidence threshold
affect results significantly. Instead of guessing, the optimizer sweeps
combinations of values, runs a backtest for each one, and returns the
combination that performed best on the chosen metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pandas as pd

from .backtest import run_backtest


@dataclass
class OptimizationResult:
    """Best combination found and the full list of trials."""

    best_params: dict
    best_value: float
    best_metrics: dict
    trials: list[dict]


def optimize(
    prices: pd.Series,
    grid: dict[str, list],
    initial_capital: float = 10000.0,
    metric: str = "sharpe",
    **fixed,
) -> OptimizationResult:
    """Searches the grid for the parameter combination with the best performance.

    grid is a dict mapping parameter names to lists of values to try. Fixed
    keyword arguments apply to every backtest run. metric can be any key
    returned by run_backtest, such as sharpe or total_return.
    """
    trials: list[dict] = []
    best: dict | None = None

    for combination in _combinations(grid):
        params = {**fixed, **combination}
        result = run_backtest(prices, initial_capital=initial_capital, **params)
        value = result.metrics.get(metric, float("-inf"))

        trial = {"params": combination, "value": value, "metrics": result.metrics}
        trials.append(trial)

        if best is None or value > best["value"]:
            best = trial

    if best is None:
        return OptimizationResult({}, float("-inf"), {}, [])

    return OptimizationResult(
        best_params=best["params"],
        best_value=best["value"],
        best_metrics=best["metrics"],
        trials=trials,
    )


def _combinations(grid: dict[str, list]):
    """Yields every possible combination of values from the grid."""
    keys = list(grid)
    for values in product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))
