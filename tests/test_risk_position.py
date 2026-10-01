"""Tests for risk management and portfolio tracking."""

import pytest

from cassandra.position import Portfolio
from cassandra.risk import RiskRules, size_order


def test_long_order_has_stop_below_and_target_above():
    order = size_order(10000, 100, is_long=True, rules=RiskRules())
    assert order.allocated_capital == 1000
    assert order.stop_price < 100 < order.target_price


def test_short_order_inverts_stop_and_target():
    order = size_order(10000, 100, is_long=False, rules=RiskRules())
    assert order.target_price < 100 < order.stop_price


def test_portfolio_opens_and_closes_with_profit():
    portfolio = Portfolio(capital=10000)
    order = size_order(10000, 100, is_long=True)
    assert portfolio.open_position("TEST", order, is_long=True)
    assert portfolio.capital == 9000
    # price rises 10 percent; reserved capital was 1000, so profit is 100
    result = portfolio.close_position("TEST", 110)
    assert round(result, 2) == 100.0
    assert round(portfolio.capital, 2) == 10100.0


def test_short_profits_when_price_falls():
    portfolio = Portfolio(capital=10000)
    order = size_order(10000, 100, is_long=False)
    assert portfolio.open_position("TEST", order, is_long=False)
    # price falls 10 percent; short reserved 1000, so profit is 100
    result = portfolio.close_position("TEST", 90)
    assert round(result, 2) == 100.0
    assert round(portfolio.capital, 2) == 10100.0


def test_short_loses_when_price_rises():
    portfolio = Portfolio(capital=10000)
    order = size_order(10000, 100, is_long=False)
    portfolio.open_position("TEST", order, is_long=False)
    # price rises 10 percent; short loses 100
    result = portfolio.close_position("TEST", 110)
    assert round(result, 2) == -100.0


def test_short_loss_capped_at_allocated_capital():
    portfolio = Portfolio(capital=10000)
    order = size_order(10000, 100, is_long=False)
    portfolio.open_position("TEST", order, is_long=False)
    # price triples; without the floor the loss would exceed 100 percent of allocated
    result = portfolio.close_position("TEST", 300)
    assert round(result, 2) == -1000.0


def test_size_order_rejects_zero_price():
    with pytest.raises(ValueError):
        size_order(10000, 0, is_long=True)


def test_portfolio_rejects_duplicate_ticker():
    portfolio = Portfolio(capital=10000)
    order = size_order(10000, 100, is_long=True)
    assert portfolio.open_position("TEST", order, is_long=True)
    assert not portfolio.open_position("TEST", order, is_long=True)


def test_portfolio_rejects_insufficient_cash():
    portfolio = Portfolio(capital=500)
    order = size_order(10000, 100, is_long=True)
    assert not portfolio.open_position("TEST", order, is_long=True)


def test_net_worth_adds_cash_and_open_positions():
    portfolio = Portfolio(capital=10000)
    order = size_order(10000, 100, is_long=True)
    portfolio.open_position("TEST", order, is_long=True)
    # price unchanged; net worth equals initial capital
    assert round(portfolio.net_worth({"TEST": 100}), 2) == 10000.0
