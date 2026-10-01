"""Tests for the data layer. No network access."""

import pandas as pd
import pytest

from cassandra import data as data_module
from cassandra.data import BR_MARKET, US_MARKET, normalize_ticker, daily_returns


def test_br_ticker_gets_suffix():
    assert normalize_ticker("petr4", BR_MARKET) == "PETR4.SA"
    # does not duplicate the suffix if already present
    assert normalize_ticker("PETR4.SA", BR_MARKET) == "PETR4.SA"


def test_us_ticker_stays_plain():
    assert normalize_ticker(" aapl ", US_MARKET) == "AAPL"


def test_daily_returns():
    prices = pd.Series([100.0, 110.0, 99.0])
    ret = daily_returns(prices)
    assert round(ret.iloc[0], 2) == 0.10
    assert round(ret.iloc[1], 2) == -0.10


def test_fetch_prices_handles_single_column_dataframe(monkeypatch):
    # simulates yfinance returning Close as a single-column DataFrame
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    fake = pd.DataFrame({"Close": [10.0, 11.0, 12.0]}, index=dates)
    fake.columns = pd.MultiIndex.from_tuples([("Close", "XPTO")])

    monkeypatch.setattr(data_module.yf, "download", lambda *a, **k: fake)
    series = data_module.fetch_prices("XPTO", market=US_MARKET)
    assert list(series) == [10.0, 11.0, 12.0]


def test_fetch_prices_raises_on_empty(monkeypatch):
    monkeypatch.setattr(data_module.yf, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(ValueError):
        data_module.fetch_prices("NONE", market=US_MARKET)
