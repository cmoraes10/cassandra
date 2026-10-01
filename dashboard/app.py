"""
Cassandra dashboard.

Web interface where you pick a market and ticker, run the simulation in real
time, and see which direction Cassandra thinks the price tends to move. A second
tab tests the strategy against historical data.

To run from the project root:

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ensure the package root is on the path when running via streamlit
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cassandra.backtest import run_backtest
from cassandra.data import BR_MARKET, US_MARKET, fetch_prices
from cassandra.simulation import fit_model, simulate
from cassandra.signal import BUY, NEUTRAL, SELL, generate_signal


st.set_page_config(page_title="Cassandra", page_icon=":material/query_stats:", layout="wide")
st.title("Cassandra")
st.caption(
    "Simulates thousands of possible futures for a stock and measures the probability "
    "of the price going up or down. Study tool, not investment advice."
)

with st.expander("How to use", expanded=True):
    st.markdown(
        """
        1. In the sidebar, choose the **market** and type the **ticker** you want to analyze, such as PETR4 or AAPL.
        2. Adjust, if you want, the **horizon** (how many days ahead to project) and the **minimum confidence** for Cassandra to commit to a direction.
        3. Click **Run simulation**. It fetches the real price, simulates thousands of scenarios, and shows the result.

        The result comes in three parts. The **signal** says buy, sell, or stay out. The **confidence**, from 0 to 100,
        says how far the projection is from a pure coin flip. And the **fan chart** shows the most likely price range
        over time. The Backtest tab tests the strategy against historical data.
        """
    )

with st.sidebar:
    st.header("Configuration")
    market_name = st.radio(
        "Market",
        ["Brazilian exchange", "US exchange"],
        help="Brazilian exchange uses tickers like PETR4 and VALE3. US exchange uses AAPL, TSLA, etc.",
    )
    market = BR_MARKET if market_name == "Brazilian exchange" else US_MARKET
    example = "PETR4" if market == BR_MARKET else "AAPL"
    ticker = st.text_input(
        "Ticker",
        value=example,
        help="The stock symbol. For the Brazilian exchange you do not need the .SA suffix; Cassandra adds it.",
    )
    period = st.selectbox(
        "Historical period",
        ["1y", "2y", "5y"],
        index=1,
        help="How much history Cassandra uses to learn the behavior of the ticker. 2y means two years.",
    )
    horizon = st.slider(
        "Simulation horizon in days",
        5,
        60,
        21,
        help="How many trading days ahead to project. 21 is roughly one month.",
    )
    n_paths = st.select_slider(
        "Number of scenarios",
        options=[5000, 10000, 25000, 50000],
        value=25000,
        help="How many different futures to simulate. More paths give a more stable result, and are slightly slower.",
    )
    threshold = st.slider(
        "Minimum confidence to act",
        0,
        100,
        60,
        help="How convinced the simulation needs to be before suggesting a direction. Below this, the signal is neutral.",
    )


@st.cache_data(show_spinner=False)
def load(ticker: str, market: str, period: str):
    return fetch_prices(ticker, market=market, period=period)


def fan_chart(result) -> go.Figure:
    """Builds the fan chart with median and probability bands."""
    traj = result.trajectories
    days = list(range(traj.shape[1]))
    p05, p25, p50, p75, p95 = (np.percentile(traj, q, axis=0) for q in (5, 25, 50, 75, 95))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=days + days[::-1],
            y=list(p95) + list(p05[::-1]),
            fill="toself",
            fillcolor="rgba(70,130,180,0.15)",
            line=dict(width=0),
            name="90% of scenarios",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days + days[::-1],
            y=list(p75) + list(p25[::-1]),
            fill="toself",
            fillcolor="rgba(70,130,180,0.30)",
            line=dict(width=0),
            name="50% of scenarios",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(x=days, y=p50, line=dict(color="#e8a838", width=2.5), name="Median scenario")
    )
    fig.add_hline(
        y=result.initial_price,
        line_dash="dot",
        line_color="#888",
        annotation_text="today's price",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Where the price tends to go",
        xaxis_title="Days ahead",
        yaxis_title="Projected price",
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


tab_sim, tab_backtest = st.tabs(["Simulation", "Backtest"])

with tab_sim:
    st.write("Set the parameters in the sidebar and run the simulation for the chosen ticker.")
    if st.button("Run simulation", type="primary"):
        try:
            with st.spinner(f"Fetching history for {ticker.upper()}..."):
                prices = load(ticker, market, period)
        except Exception as e:
            st.error(f"Could not load {ticker}. Check the ticker symbol. Detail: {e}")
            st.stop()

        try:
            model = fit_model(prices)
        except ValueError as e:
            st.warning(f"Insufficient data for {ticker}. {e}")
            st.stop()

        with st.spinner(f"Simulating {n_paths:,} scenarios..."):
            result = simulate(model, horizon=horizon, n_paths=n_paths, seed=42)
        signal = generate_signal(result, confidence_threshold=float(threshold))

        low, high = result.interval(0.9)
        change = signal.expected_return * 100

        if signal.direction == BUY:
            st.success(
                f"**Buy signal** for {ticker.upper()}. Over {horizon} days, most scenarios end "
                f"above today's price, with a median return of {change:+.1f} percent."
            )
        elif signal.direction == SELL:
            st.error(
                f"**Sell signal** for {ticker.upper()}. Over {horizon} days, most scenarios end "
                f"below today's price, with a median return of {change:+.1f} percent."
            )
        else:
            st.info(
                f"**Neutral signal** for {ticker.upper()}. The projection stayed close to a coin flip, "
                f"below the minimum confidence of {threshold}. Cassandra prefers to stay out."
            )

        col1, col2, col3 = st.columns(3)
        col1.metric("Signal", signal.direction, help="Buy, sell, or stay out.")
        col2.metric(
            "Confidence",
            f"{signal.confidence:.0f}/100",
            help="How far from a pure coin flip the projection is. Higher means more conviction.",
        )
        col3.metric(
            "Probability of rally",
            f"{signal.bull_probability * 100:.0f}%",
            help="Share of simulated scenarios that end above today's price.",
        )

        st.plotly_chart(fan_chart(result), use_container_width=True)

        with st.expander("How to read the chart"):
            st.markdown(
                f"""
                The center line is the **median scenario**, the most typical path among all simulated ones.
                The darker band contains **half** of the scenarios and the lighter one contains **90 percent** of them.
                The wider the fan, the more uncertain the future of that ticker.

                In 90 percent of scenarios the price of {ticker.upper()} ends between
                **{low:.2f}** and **{high:.2f}** in {horizon} days.
                """
            )

with tab_backtest:
    st.write(
        "The backtest reconstructs the past, generates signals using only information available on each day, "
        "applies the risk rules, and measures the result. This is the proof that the strategy holds up."
    )
    capital = st.number_input(
        "Starting capital",
        value=10000,
        step=1000,
        help="How much money the strategy starts with in the test.",
    )
    if st.button("Run backtest"):
        try:
            with st.spinner(f"Fetching history for {ticker.upper()}..."):
                prices = load(ticker, market, period)
        except Exception as e:
            st.error(f"Could not load {ticker}. Detail: {e}")
            st.stop()

        try:
            with st.spinner("Reconstructing the past and testing the strategy..."):
                result = run_backtest(
                    prices,
                    initial_capital=float(capital),
                    horizon=horizon,
                    n_paths=2000,
                    confidence_threshold=float(threshold),
                    seed=42,
                )
        except ValueError as e:
            st.warning(f"Cannot run backtest for {ticker}. {e}")
            st.stop()

        m = result.metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Total return",
            f"{m['total_return'] * 100:.1f}%",
            help="How much the portfolio changed from start to end.",
        )
        c2.metric(
            "Sharpe",
            f"{m['sharpe']:.2f}",
            help="Risk-adjusted return. Above 1 is generally considered good.",
        )
        c3.metric(
            "Max drawdown",
            f"{m['max_drawdown'] * 100:.1f}%",
            help="Worst peak-to-trough drop in portfolio value during the test.",
        )
        c4.metric(
            "Win rate",
            f"{m['win_rate'] * 100:.0f}%",
            help="Share of trades that closed in profit.",
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=result.equity_curve.index,
                y=result.equity_curve.values,
                mode="lines",
                line=dict(color="#1a9850", width=2),
                name="Portfolio",
            )
        )
        fig.add_hline(y=float(capital), line_dash="dot", line_color="#888", annotation_text="starting capital")
        fig.update_layout(
            title="Portfolio value over the test period",
            xaxis_title="Date",
            yaxis_title="Portfolio value",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Trades executed during the period: {m['num_trades']}.")

st.divider()
st.caption("Built by Cauã Moraes · [mowaveone.com](https://mowaveone.com)")
