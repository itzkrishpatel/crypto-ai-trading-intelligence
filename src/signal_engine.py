src/signal_engine.py"""
signal_engine.py
-----------------
Ties the pipeline together for "live-style" use: load recent data, build features, load (or
train) a model per coin/timeframe, and print the current BUY/HOLD/SELL signal for each.

This is the inference-side counterpart to backtest.py's historical evaluation.

Usage:
    python -m src.signal_engine --config configs/coins.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd
import yaml

from .data import SyntheticDataSource
from .features import build_feature_matrix
from .model import CoinTimeframeModel, SignalThresholds, make_labels


def generate_signals(cfg: dict) -> pd.DataFrame:
    thresholds = SignalThresholds(**cfg["signal_thresholds"])
    source = SyntheticDataSource()  # swap for a real DataSource (see src/data.py) for live data

    end = pd.Timestamp.utcnow().tz_localize(None).floor("h")
    start = end - pd.Timedelta(days=cfg["backtest"]["train_window_days"] + 5)

    btc_ohlcv_by_tf = {tf: source.load_ohlcv("BTC", tf, start, end) for tf in cfg["timeframes"]}

    rows = []
    for coin in cfg["coins"]:
        for tf in cfg["timeframes"]:
            ohlcv = source.load_ohlcv(coin, tf, start, end)
            sentiment = source.load_news_sentiment(coin, start, end)
            btc_ohlcv = None if coin == "BTC" else btc_ohlcv_by_tf[tf]

            features = build_feature_matrix(coin, ohlcv, btc_ohlcv, sentiment)
            close = ohlcv.set_index("timestamp")["close"]
            labels = make_labels(close, horizon=1)

            model = CoinTimeframeModel(coin, tf, thresholds)
            # Train on everything except the final (label-less) row, then score that last row.
            model.fit(features.iloc[:-1], labels.iloc[:-1])

            latest = features.iloc[[-1]]
            proba_up = model.predict_proba_up(latest).iloc[0]
            signal = thresholds.classify(proba_up) if pd.notna(proba_up) else "HOLD"

            rows.append(
                {
                    "coin": coin,
                    "timeframe": tf,
                    "timestamp": latest.index[0],
                    "prob_up": proba_up,
                    "signal": signal,
                }
            )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate current BUY/HOLD/SELL signals")
    parser.add_argument("--config", type=str, default="configs/coins.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    signals = generate_signals(cfg)
    print(signals.to_string(index=False))


if __name__ == "__main__":
    main()
