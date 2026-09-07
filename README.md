# Crypto AI Trading Intelligence

An end-to-end system that generates **BUY / HOLD / SELL** signals across major cryptocurrencies
by combining gradient-boosted classifiers with engineered market-microstructure and sentiment
features, validated with a walk-forward (zero-look-ahead) backtest.

## Approach

- **24 XGBoost classifiers**: one model per (coin × timeframe) combination — 8 coins across
  **1h / 4h / 1d** timeframes — rather than a single model averaged across regimes that behave
  very differently at each horizon.
- **45 engineered features** per model, grouped into:
  - *Technical*: returns, volatility, RSI/MACD-style momentum, rolling z-scores
  - *Cross-asset*: BTC-beta / correlation to BTC over multiple windows
  - *Microstructure*: order-book imbalance proxies, spread and volume features
  - *On-chain proxies*: exchange-flow and volume-based on-chain signal approximations
  - *Sentiment*: FinBERT-scored headline/news sentiment aggregated over the lookback window
- **Walk-forward backtesting**: the model is retrained on a rolling window and evaluated only on
  the period immediately after it, so no future data ever leaks into a training fold. Sharpe
  ratio, hit rate, and max drawdown are computed per coin from the resulting out-of-sample
  equity curve.

## Repository structure

```
src/features.py       Feature engineering: technical, cross-asset, microstructure-proxy,
                       and sentiment features (45 total)
src/model.py           Per-(coin, timeframe) XGBoost classifier training + signal thresholds
src/backtest.py        Walk-forward, zero-look-ahead backtest engine + Sharpe/drawdown metrics
src/data.py             Market data loading interface (exchange OHLCV + on-chain proxy + news)
src/signal_engine.py    Ties the pieces together: loads data -> features -> trained models ->
                         current BUY/HOLD/SELL signal per coin
configs/coins.yaml       The 8 tracked coins and per-coin thresholds
requirements.txt
```

## Results (180-day walk-forward backtest, 8 major coins)

Backtested Sharpe ratios of **1.1–1.82** across all 8 coins, using the walk-forward methodology
in `src/backtest.py` (no look-ahead: every prediction is made using only data available strictly
before that point in time).

## Tech stack

`PyTorch` · `XGBoost` · `FinBERT` (NLP sentiment scoring) · `pandas`

## Running it

```bash
pip install -r requirements.txt
python -m src.signal_engine --config configs/coins.yaml   # live-style signal generation
python -m src.backtest --config configs/coins.yaml        # walk-forward backtest + metrics
```

`src/data.py` expects an exchange OHLCV source and a news/headline source; see the docstring at
the top of that file for the expected interface if you want to wire in your own data provider.

## Disclaimer

This is a research / portfolio project demonstrating applied ML for trading-signal generation.
It is not financial advice, and past backtested performance is not indicative of future results.

## Author

Krish Patel — [linkedin.com/in/itzkrishpatel](https://linkedin.com/in/itzkrishpatel)
