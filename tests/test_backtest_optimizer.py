"""Tests for the backtest engine and parameter optimizer."""

from cassandra.backtest import run_backtest
from cassandra.optimizer import optimize


def test_backtest_runs_and_returns_metrics(prices_bull):
    result = run_backtest(
        prices_bull,
        initial_capital=10000,
        train_window=252,
        n_paths=300,
        step=10,
        seed=1,
    )
    assert not result.equity_curve.empty
    for key in ("total_return", "sharpe", "max_drawdown", "num_trades", "win_rate"):
        assert key in result.metrics


def test_backtest_is_reproducible(prices_bull):
    a = run_backtest(prices_bull, n_paths=300, step=10, seed=1)
    b = run_backtest(prices_bull, n_paths=300, step=10, seed=1)
    assert a.metrics == b.metrics


def test_drawdown_is_never_positive(prices_bear):
    result = run_backtest(prices_bear, n_paths=300, step=10, seed=1)
    assert result.metrics["max_drawdown"] <= 0


def test_optimizer_finds_best_combination(prices_bull):
    grid = {"confidence_threshold": [40.0, 70.0], "horizon": [10, 21]}
    result = optimize(
        prices_bull,
        grid,
        metric="sharpe",
        n_paths=200,
        step=15,
        seed=1,
    )
    assert result.best_params
    assert len(result.trials) == 4
