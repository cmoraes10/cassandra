"""
Cassandra, a system that estimates which direction a stock price tends to move.

The package brings together the pieces of the strategy. The data layer fetches
prices, the simulation engine generates future scenarios, the signal generator
turns those scenarios into a decision, and the risk, position, backtest, and
optimizer modules handle the trading mechanics and validation.
"""

from .simulation import fit_model, simulate, RegimeModel, SimulationResult
from .signal import generate_signal, Signal, BUY, SELL, NEUTRAL
from .risk import RiskRules, size_order
from .position import Portfolio
from .backtest import run_backtest, BacktestResult
from .optimizer import optimize

__all__ = [
    "fit_model",
    "simulate",
    "RegimeModel",
    "SimulationResult",
    "generate_signal",
    "Signal",
    "BUY",
    "SELL",
    "NEUTRAL",
    "RiskRules",
    "size_order",
    "Portfolio",
    "run_backtest",
    "BacktestResult",
    "optimize",
]
