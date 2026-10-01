# Cassandra

A system that estimates which way a stock price tends to move by simulating
thousands of possible futures and measuring in how many of them the price ends
up higher or lower.

## Where the name comes from

Cassandra is a figure from Greek mythology, a priestess who received the gift
of seeing the future. The name was chosen deliberately. This project lives by
looking ahead and pointing to tendencies, but without promising certainty. Like
the mythological Cassandra, it delivers a reading of what is likely coming, and
it is up to the listener to decide what to do with that.

## What it does

You give it a stock, from the Brazilian or US exchange, and Cassandra does three
things.

First it studies the recent price history to understand the behavior of that
asset. Then it projects thousands of possible paths for the next few days.
Finally it summarizes everything into a clear signal, buy, sell, or stay out,
accompanied by a confidence score from zero to a hundred.

All of this appears in a web dashboard, with the chart of simulated trajectories
and a backtest tab that tests the strategy against the past.

## How it works internally

The idea behind the project is that the market does not always behave the same
way. Sometimes it is in a bull run, with positive momentum, and sometimes in a
bear run. Cassandra separates those two regimes from the price history and
measures the tendency and volatility of each one.

The switch between bull and bear is treated as a Markov chain, which in
practice means that tomorrow's mood depends on today's mood. With those
ingredients the engine runs a Monte Carlo simulation, generating thousands of
price trajectories, day by day, each one following the regime it finds itself
in at that moment.

At the end, the proportion of trajectories that finish above the starting price
becomes the probability of a rally, and the distance of that probability from a
pure coin flip becomes the signal's confidence.

## Structure

```
cassandra/
  data.py          fetches prices from yfinance
  simulation.py    estimates regimes and runs Monte Carlo
  signal.py        turns the simulation into a trade signal
  risk.py          sets position size and stop/target levels
  position.py      tracks cash and open positions
  backtest.py      tests the strategy against historical data
  optimizer.py     sweeps parameters to find the best configuration
dashboard/
  app.py           Streamlit web interface
tests/             automated tests
```

## How to run

The project was tested with Python 3.12. Install dependencies and open the
dashboard.

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

To run the tests, also install the development dependencies.

```bash
pip install -r requirements-dev.txt
pytest
```

## Deploy

The dashboard runs for free on Streamlit Community Cloud. Push the repo to
GitHub, go to share.streamlit.io, point it at this repository, and set
`dashboard/app.py` as the main file. Dependencies come from `requirements.txt`
and the rest is automatic.

## Quick example

```python
from cassandra.data import fetch_prices
from cassandra.simulation import fit_model, simulate
from cassandra.signal import generate_signal

prices = fetch_prices("PETR4", market="B3")
model = fit_model(prices)
result = simulate(model, horizon=21, n_paths=25000)
signal = generate_signal(result)
print(signal.summary())
```

## Disclaimer

This is a study project built to explore Monte Carlo simulation, Markov chains,
and strategy testing. Nothing here is investment advice. Markets involve risk
and past performance does not guarantee future results.

## Author

Built by Cauã Moraes. Other projects at [mowaveone.com](https://mowaveone.com).
