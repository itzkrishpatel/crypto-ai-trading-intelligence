src/backtest.py"""
backtest.py
-----------
Walk-forward, zero-look-ahead backtest engine.

The model is retrained on a rolling `train_window_days` window and evaluated only on the
`test_window_days` immediately after it, then the window slides forward. A prediction made
for time t only ever uses a model trained on data from before that fold's test period, so
there is no leakage of future information into past predictions.

Usage:
    python -m src.backtest --config configs/coins.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from .data import SyntheticDataSource
from .features import build_feature_matrix
from .model import CoinTimeframeModel, SignalThresholds, make_labels


@dataclass
class BacktestMetrics:
    coin: str
    timeframe: str
    n_trades: int
    hit_rate: float
    sharpe: float
    max_drawdown: float
    total_return: float


def _sharpe_ratio(returns: pd.Series, periods_per_year: float, risk_free_rate: float = 0.0) -> float:
    excess = returns - risk_free_rate / periods_per_year
    if excess.std(ddof=0) == 0 or excess.empty:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=0))


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return float(drawdown.min())


_PERIODS_PER_YEAR = {"1h": 24 * 365, "4h": 6 * 365, "1d": 365}


def walk_forward_backtest(
    coin: str,
    timeframe: str,
    ohlcv: pd.DataFrame,
    btc_ohlcv: pd.DataFrame | None,
    sentiment: pd.DataFrame,
    thresholds: SignalThresholds,
    train_window_days: int,
    test_window_days: int,
    fee_bps: float,
    risk_free_rate: float,
) -> tuple[BacktestMetrics, pd.DataFrame]:
    features = build_feature_matrix(coin, ohlcv, btc_ohlcv, sentiment)
    close = ohlcv.set_index("timestamp")["close"]
    labels = make_labels(close, horizon=1)
    forward_returns = close.pct_change().shift(-1)  # realized return of holding from t to t+1

    minutes_per_bar = {"1h": 60, "4h": 240, "1d": 1440}[timeframe]
    bars_per_day = 1440 // minutes_per_bar
    train_bars = train_window_days * bars_per_day
    test_bars = test_window_days * bars_per_day

    all_signals = []
    idx = features.index

    start = train_bars
    while start + test_bars <= len(idx):
        train_slice = slice(start - train_bars, start)
        test_slice = slice(start, start + test_bars)

        X_train, y_train = features.iloc[train_slice], labels.iloc[train_slice]
        X_test = features.iloc[test_slice]

        # Skip folds that don't have enough clean rows to fit a stable model.
        if X_train.notna().all(axis=1).sum() < 30:
            start += test_bars
            continue

        model = CoinTimeframeModel(coin, timeframe, thresholds)
        model.fit(X_train, y_train)
        signals = model.predict_signal(X_test)
        all_signals.append(signals)

        start += test_bars

    if not all_signals:
        empty = BacktestMetrics(coin, timeframe, 0, 0.0, 0.0, 0.0, 0.0)
        return empty, pd.DataFrame()

    signal_series = pd.concat(all_signals)
    position = signal_series.map({"BUY": 1, "SELL": -1, "HOLD": 0}).reindex(forward_returns.index).fillna(0)

    fee = fee_bps / 1e4
    trade_occurred = position.diff().fillna(position).abs() > 0
    strategy_returns = position * forward_returns - trade_occurred * fee
    strategy_returns = strategy_returns.dropna()

    equity_curve = (1 + strategy_returns).cumprod()
    n_trades = int(trade_occurred.sum())
    hit_rate = float((np.sign(strategy_returns) > 0).mean()) if n_trades else 0.0

    metrics = BacktestMetrics(
        coin=coin,
        timeframe=timeframe,
        n_trades=n_trades,
        hit_rate=hit_rate,
        sharpe=_sharpe_ratio(strategy_returns, _PERIODS_PER_YEAR[timeframe], risk_free_rate),
        max_drawdown=_max_drawdown(equity_curve) if not equity_curve.empty else 0.0,
        total_return=float(equity_curve.iloc[-1] - 1) if not equity_curve.empty else 0.0,
    )
    return metrics, equity_curve.to_frame("equity")


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest for the crypto signal models")
    parser.add_argument("--config", type=str, default="configs/coins.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    thresholds = SignalThresholds(**cfg["signal_thresholds"])
    bt_cfg = cfg["backtest"]

    source = SyntheticDataSource()  # swap for a real DataSource (see src/data.py) for live data
    end = pd.Timestamp.utcnow().tz_localize(None).floor("h")
    start = end - pd.Timedelta(days=bt_cfg["lookback_days"])

    btc_ohlcv_by_tf = {tf: source.load_ohlcv("BTC", tf, start, end) for tf in cfg["timeframes"]}

    results = []
    for coin in cfg["coins"]:
        for tf in cfg["timeframes"]:
            ohlcv = source.load_ohlcv(coin, tf, start, end)
            sentiment = source.load_news_sentiment(coin, start, end)
            btc_ohlcv = None if coin == "BTC" else btc_ohlcv_by_tf[tf]

            metrics, _ = walk_forward_backtest(
                coin,
                tf,
                ohlcv,
                btc_ohlcv,
                sentiment,
                thresholds,
                bt_cfg["train_window_days"],
                bt_cfg["test_window_days"],
                bt_cfg["fee_bps"],
                bt_cfg["risk_free_rate"],
            )
            results.append(metrics)
            print(
                f"{coin:>5} {tf:>3}  sharpe={metrics.sharpe:6.2f}  "
                f"hit_rate={metrics.hit_rate:5.2%}  trades={metrics.n_trades:4d}  "
                f"max_dd={metrics.max_drawdown:6.2%}"
            )

    return results


if __name__ == "__main__":
    main()
