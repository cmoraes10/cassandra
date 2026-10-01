"""
Data layer for Cassandra.

Fetches historical price data using yfinance. The Brazilian exchange requires
the .SA suffix on ticker symbols, so that detail is handled here and the rest
of the system does not need to know about it.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

BR_MARKET = "B3"
US_MARKET = "US"


def normalize_ticker(ticker: str, market: str) -> str:
    """Adjusts the ticker symbol to the format yfinance expects.

    Brazilian tickers need the .SA suffix, like PETR4.SA. US tickers go in as-is.
    """
    ticker = ticker.strip().upper()
    if market == BR_MARKET and not ticker.endswith(".SA"):
        return f"{ticker}.SA"
    return ticker


def fetch_prices(
    ticker: str,
    market: str = US_MARKET,
    period: str = "2y",
    interval: str = "1d",
) -> pd.Series:
    """Downloads the adjusted close price series for a ticker.

    Returns a pandas Series indexed by date. Raises ValueError if the ticker
    does not exist or the period returns no data.
    """
    symbol = normalize_ticker(ticker, market)
    raw = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"No data found for {symbol}.")

    close = raw["Close"]
    # some yfinance versions return Close as a single-column DataFrame
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    series = close.dropna()
    if series.empty:
        raise ValueError(f"Price series for {symbol} is empty after cleaning.")

    series.name = symbol
    return series


def daily_returns(prices: pd.Series) -> pd.Series:
    """Percentage return from one day to the next."""
    return prices.pct_change().dropna()
