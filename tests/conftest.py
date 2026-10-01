"""
Shared test fixtures.

Tests do not access the network. Instead of fetching real prices, they use
synthetic series with controlled trend and noise, which keeps tests fast and
deterministic.
"""

import numpy as np
import pandas as pd
import pytest


def synthetic_series(n_days=600, drift=0.0005, volatility=0.02, seed=7, initial_price=100.0):
    """Creates a price series simulating a random walk with trend."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, volatility, n_days)
    prices = initial_price * np.exp(np.cumsum(returns))
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    return pd.Series(prices, index=dates, name="TEST")


@pytest.fixture
def prices_bull():
    """Series with a strong enough upward trend to dominate noise."""
    return synthetic_series(drift=0.003, volatility=0.010)


@pytest.fixture
def prices_bear():
    """Series with a strong enough downward trend to dominate noise."""
    return synthetic_series(drift=-0.003, volatility=0.010)
