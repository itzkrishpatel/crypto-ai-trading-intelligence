src/data.py"""
data.py
-------
Data loading interface for the signal engine and backtester.

This module defines the contract the rest of the pipeline expects:

    load_ohlcv(coin, timeframe, start, end)  -> DataFrame[open, high, low, close, volume]
    load_news_sentiment(coin, start, end)    -> DataFrame[timestamp, sentiment_score]

`CsvDataSource` reads from local CSV files (point it at your own exchange export / API dump).
`SyntheticDataSource` generates reproducible, geometric-Brownian-motion-style OHLCV data with a
FinBERT-style sentiment series, purely so the rest of the pipeline (features, model, backtest)
can be run and inspected end-to-end without a live exchange or news API connection.

Swap in your own subclass of `DataSource` to point this at a real exchange (e.g. via `ccxt`)
or a news API — nothing else in the repo needs to change.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DataSource(abc.ABC):
    @abc.abstractmethod
    def load_ohlcv(self, coin: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        ...

    @abc.abstractmethod
    def load_news_sentiment(self, coin: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        ...


class CsvDataSource(DataSource):
    """Reads OHLCV / sentiment data from local CSVs.

    Expected layout:
        data/ohlcv/{coin}_{timeframe}.csv   columns: timestamp, open, high, low, close, volume
        data/sentiment/{coin}.csv           columns: timestamp, sentiment_score
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir

    def load_ohlcv(self, coin: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        path = f"{self.data_dir}/ohlcv/{coin}_{timeframe}.csv"
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].reset_index(drop=True)

    def load_news_sentiment(self, coin: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        path = f"{self.data_dir}/sentiment/{coin}.csv"
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].reset_index(drop=True)


class SyntheticDataSource(DataSource):
    """Deterministic synthetic OHLCV + sentiment generator, for running the pipeline
    end-to-end without a live data connection. NOT real market data."""

    _TIMEFRAME_MINUTES = {"1h": 60, "4h": 240, "1d": 1440}

    def __init__(self, seed: int = 42):
        self.seed = seed

    def _rng(self, coin: str) -> np.random.Generator:
        # Per-coin deterministic seed so repeated calls for the same coin are reproducible.
        return np.random.default_rng(self.seed + abs(hash(coin)) % 10_000)

    def load_ohlcv(self, coin: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        minutes = self._TIMEFRAME_MINUTES[timeframe]
        timestamps = pd.date_range(start, end, freq=f"{minutes}min")
        n = len(timestamps)
        rng = self._rng(coin)

        # Geometric Brownian motion for close prices, with mild autocorrelated volatility
        # clustering (GARCH-lite) so features like rolling volatility have something to find.
        vol = np.abs(rng.normal(0.01, 0.002, size=n))
        vol = pd.Series(vol).rolling(5, min_periods=1).mean().to_numpy()
        drift = rng.normal(0.0, 1.0) * 1e-4
        log_returns = rng.normal(drift, vol)
        close = 100.0 * np.exp(np.cumsum(log_returns))

        high = close * (1 + np.abs(rng.normal(0, 0.003, size=n)))
        low = close * (1 - np.abs(rng.normal(0, 0.003, size=n)))
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        volume = np.abs(rng.normal(1_000_000, 200_000, size=n))

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    def load_news_sentiment(self, coin: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        timestamps = pd.date_range(start, end, freq="1h")
        rng = self._rng(coin + "_sentiment")
        # FinBERT scores are typically in [-1, 1]; simulate a mildly mean-reverting series.
        n = len(timestamps)
        scores = np.zeros(n)
        for i in range(1, n):
            scores[i] = 0.9 * scores[i - 1] + rng.normal(0, 0.15)
        scores = np.clip(scores, -1, 1)
        return pd.DataFrame({"timestamp": timestamps, "sentiment_score": scores})
