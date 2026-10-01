"""Tests for signal generation."""

from cassandra.simulation import fit_model, simulate
from cassandra.signal import BUY, NEUTRAL, SELL, generate_signal


def test_bull_trend_produces_buy_signal(prices_bull):
    model = fit_model(prices_bull)
    result = simulate(model, horizon=21, n_paths=5000, seed=3)
    signal = generate_signal(result, confidence_threshold=10.0)
    assert signal.direction == BUY


def test_bear_trend_produces_sell_signal(prices_bear):
    model = fit_model(prices_bear)
    result = simulate(model, horizon=21, n_paths=5000, seed=3)
    signal = generate_signal(result, confidence_threshold=10.0)
    assert signal.direction == SELL


def test_high_threshold_forces_neutral(prices_bull):
    model = fit_model(prices_bull)
    result = simulate(model, horizon=21, n_paths=5000, seed=3)
    signal = generate_signal(result, confidence_threshold=100.0)
    assert signal.direction == NEUTRAL


def test_confidence_is_between_zero_and_hundred(prices_bull):
    model = fit_model(prices_bull)
    result = simulate(model, horizon=21, n_paths=3000, seed=5)
    signal = generate_signal(result)
    assert 0.0 <= signal.confidence <= 100.0
