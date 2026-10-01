"""
Simulation engine for Cassandra.

The core idea is to simulate thousands of possible futures for a stock price
and, from that cloud of scenarios, measure the probability of the price going
up or down.

Instead of assuming the market always behaves the same way, the engine works
with two regimes, bull and bear, each with its own tendency and volatility. The
switch between regimes follows a Markov chain: tomorrow's mood depends on
today's mood. Each simulated trajectory is a price path built day by day,
drawing returns from whichever regime that path is currently in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegimeParams:
    """Price behavior within a single regime.

    mean is the typical daily log return for this regime and std is the daily
    volatility.
    """

    mean: float
    std: float


@dataclass
class RegimeModel:
    """Everything the simulator needs to generate scenarios.

    Holds the parameters for each regime, the transition matrix between them,
    the most recently observed regime, and the last known price, which is the
    starting point for all paths.
    """

    regimes: list[RegimeParams]
    transition: np.ndarray
    current_regime: int
    initial_price: float


def fit_model(prices: pd.Series, trend_window: int = 20) -> RegimeModel:
    """Estimates regimes from the price history.

    Bull and bear separation is done by looking at the rolling mean of returns.
    When that mean is positive the day is labelled bull; when negative, bear.
    With those labels in hand the function computes the tendency and volatility
    of each regime and counts how often the market switches from one to the other.
    """
    prices = prices.dropna()
    minimum = trend_window + 5
    if len(prices) < minimum:
        raise ValueError(
            f"price history too short to estimate regimes: {len(prices)} prices, "
            f"at least {minimum} are needed."
        )

    log_ret = np.log(prices / prices.shift(1)).dropna()
    trend = log_ret.rolling(trend_window).mean()
    labels = (trend > 0).astype(int).dropna()
    log_ret = log_ret.loc[labels.index]

    global_std = float(log_ret.std())
    if not np.isfinite(global_std) or global_std <= 0:
        # series with almost no variation; use a small floor to avoid division errors
        global_std = 1e-6

    regimes: list[RegimeParams] = []
    for r in (0, 1):
        sample = log_ret[labels == r]
        if len(sample) < 2:
            # not enough data in this regime; fall back to global statistics
            regimes.append(RegimeParams(float(log_ret.mean()), global_std))
        else:
            s = float(sample.std())
            if not np.isfinite(s) or s <= 0:
                s = global_std
            regimes.append(RegimeParams(float(sample.mean()), s))

    transition = _transition_matrix(labels.to_numpy())
    current_regime = int(labels.iloc[-1]) if len(labels) else 1

    return RegimeModel(
        regimes=regimes,
        transition=transition,
        current_regime=current_regime,
        initial_price=float(prices.iloc[-1]),
    )


def _transition_matrix(labels: np.ndarray) -> np.ndarray:
    """Counts regime switches and normalizes to probabilities.

    Starts with a count of one in every cell (Laplace smoothing) so that no
    probability is zero, which would trap a simulation in a single regime.
    """
    matrix = np.ones((2, 2))
    for current, next_r in zip(labels[:-1], labels[1:]):
        matrix[current, next_r] += 1
    return matrix / matrix.sum(axis=1, keepdims=True)


@dataclass
class SimulationResult:
    """Output of the simulator, with all trajectories and a few useful shortcuts."""

    trajectories: np.ndarray
    initial_price: float
    horizon: int

    @property
    def final_prices(self) -> np.ndarray:
        """Price at the end of each simulated trajectory."""
        return self.trajectories[:, -1]

    def bull_probability(self) -> float:
        """Fraction of trajectories that end above the starting price."""
        return float((self.final_prices > self.initial_price).mean())

    def expected_return(self) -> float:
        """Return of the median scenario relative to the starting price.

        The median is more honest than the mean here because it is not pulled by
        a handful of extreme scenarios.
        """
        return float(np.median(self.final_prices) / self.initial_price - 1)

    def interval(self, confidence: float = 0.9) -> tuple[float, float]:
        """Price range that contains most of the simulated scenarios."""
        margin = (1 - confidence) / 2
        low = float(np.quantile(self.final_prices, margin))
        high = float(np.quantile(self.final_prices, 1 - margin))
        return low, high


def simulate(
    model: RegimeModel,
    horizon: int = 21,
    n_paths: int = 25000,
    seed: int | None = None,
) -> SimulationResult:
    """Generates projected price trajectories for the next trading days.

    Each path starts at the last known price and advances one day at a time. At
    each step the return is drawn from the current regime of that path, the price
    is updated, and the regime may switch according to the transition matrix. The
    default horizon of 21 days corresponds to roughly one trading month.
    """
    rng = np.random.default_rng(seed)
    n_regimes = len(model.regimes)
    means = np.array([r.mean for r in model.regimes])
    stds = np.array([r.std for r in model.regimes])

    trajectories = np.empty((n_paths, horizon + 1))
    trajectories[:, 0] = model.initial_price
    regime = np.full(n_paths, model.current_regime, dtype=int)

    # cumulative thresholds for drawing the next regime with a single comparison
    cumulative = np.cumsum(model.transition, axis=1)

    for step in range(1, horizon + 1):
        shock = rng.standard_normal(n_paths)
        log_ret = means[regime] + stds[regime] * shock
        trajectories[:, step] = trajectories[:, step - 1] * np.exp(log_ret)

        draw = rng.random(n_paths)
        new_regime = regime.copy()
        for r in range(n_regimes):
            mask = regime == r
            if not mask.any():
                continue
            thresholds = cumulative[r][:-1]
            new_regime[mask] = (draw[mask][:, None] > thresholds).sum(axis=1)
        regime = new_regime

    return SimulationResult(trajectories, model.initial_price, horizon)
