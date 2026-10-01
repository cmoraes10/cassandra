"""Tests for the simulation engine and regime estimation."""

import numpy as np
import pandas as pd
import pytest

from cassandra.simulation import fit_model, simulate


def test_short_series_raises():
    short = pd.Series([100.0, 101.0, 99.0], name="SHORT")
    with pytest.raises(ValueError):
        fit_model(short)


def test_model_has_two_regimes_and_valid_transition(prices_bull):
    model = fit_model(prices_bull)
    assert len(model.regimes) == 2
    # each row of the transition matrix is a probability distribution
    row_sums = model.transition.sum(axis=1)
    assert np.allclose(row_sums, 1.0)
    assert model.initial_price > 0


def test_simulation_output_shape(prices_bull):
    model = fit_model(prices_bull)
    result = simulate(model, horizon=10, n_paths=500, seed=1)
    assert result.trajectories.shape == (500, 11)
    # every path starts at the initial price
    assert np.allclose(result.trajectories[:, 0], model.initial_price)


def test_same_seed_gives_same_result(prices_bull):
    model = fit_model(prices_bull)
    a = simulate(model, horizon=10, n_paths=300, seed=99)
    b = simulate(model, horizon=10, n_paths=300, seed=99)
    assert np.array_equal(a.trajectories, b.trajectories)


def test_bull_trend_pushes_probability_above_half(prices_bull):
    model = fit_model(prices_bull)
    result = simulate(model, horizon=21, n_paths=5000, seed=3)
    assert result.bull_probability() > 0.5


def test_interval_is_ordered(prices_bull):
    model = fit_model(prices_bull)
    result = simulate(model, horizon=21, n_paths=5000, seed=3)
    low, high = result.interval(0.9)
    assert low < high
