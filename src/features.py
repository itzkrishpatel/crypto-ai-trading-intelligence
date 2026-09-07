src/features.py"""
features.py
-----------
Feature engineering for the per-(coin, timeframe) XGBoost classifiers.

Produces the four feature groups described in the README: technical, cross-asset
(BTC-beta / correlation), microstructure proxies, and sentiment. All features are computed
using only data available strictly at or before each row's timestamp (no forward-looking
windows), which is what makes the walk-forward backtest in backtest.py valid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """~20 technical features derived purely from a coin's own OHLCV series."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_6"] = close.pct_change(6)
    out["ret_12"] = close.pct_change(12)

    for w in (6, 12, 24, 48):
        out[f"volatility_{w}"] = out["ret_1"].rolling(w).std()
        out[f"zscore_{w}"] = (close - close.rolling(w).mean()) / close.rolling(w).std()

    out["rsi_14"] = _rsi(close, 14)
    macd_line, macd_signal = _macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_line - macd_signal

    out["hl_range"] = (df["high"] - df["low"]) / close
    out["close_vs_high"] = (df["high"] - close) / close
    out["close_vs_low"] = (close - df["low"]) / close

    out["volume_zscore_24"] = (df["volume"] - df["volume"].rolling(24).mean()) / df["volume"].rolling(24).std()
    out["volume_change_1"] = df["volume"].pct_change(1)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    out["atr_14"] = true_range.rolling(14).mean()

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_width_20"] = (2 * bb_std) / bb_mid

    return out


def cross_asset_features(coin_close: pd.Series, btc_close: pd.Series, windows=(12, 24, 48)) -> pd.DataFrame:
    """BTC-beta and rolling correlation-to-BTC features. `btc_close` must be aligned to the
    same index as `coin_close` (for BTC itself, callers should skip this feature group)."""
    out = pd.DataFrame(index=coin_close.index)
    coin_ret = coin_close.pct_change()
    btc_ret = btc_close.pct_change()

    for w in windows:
        cov = coin_ret.rolling(w).cov(btc_ret)
        var = btc_ret.rolling(w).var()
        out[f"btc_beta_{w}"] = cov / var.replace(0, np.nan)
        out[f"btc_corr_{w}"] = coin_ret.rolling(w).corr(btc_ret)

    return out


def microstructure_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Order-book imbalance and liquidity proxies derived from OHLCV bars (no raw L2 book
    required). These approximate the signal a real order-book-imbalance feature would carry."""
    out = pd.DataFrame(index=df.index)

    # Buy-pressure proxy: where in the bar's range the close landed (Chaikin-style).
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    out["close_location_value"] = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    out["imbalance_proxy"] = out["close_location_value"] * df["volume"]
    out["imbalance_proxy_ma_6"] = out["imbalance_proxy"].rolling(6).mean()

    out["amihud_illiquidity"] = (df["close"].pct_change().abs() / df["volume"].replace(0, np.nan)).rolling(6).mean()
    out["volume_per_range"] = df["volume"] / rng

    return out


def on_chain_proxy_features(df: pd.DataFrame, windows=(12, 24, 48)) -> pd.DataFrame:
    """Approximates exchange-flow-style on-chain signals from volume/price action, for
    settings where a dedicated on-chain data provider isn't wired in."""
    out = pd.DataFrame(index=df.index)
    signed_volume = df["volume"] * np.sign(df["close"].diff().fillna(0))

    for w in windows:
        out[f"net_flow_proxy_{w}"] = signed_volume.rolling(w).sum()
        out[f"flow_volume_ratio_{w}"] = signed_volume.rolling(w).sum() / df["volume"].rolling(w).sum()

    return out


def sentiment_features(price_index: pd.DatetimeIndex, sentiment_df: pd.DataFrame, windows=(6, 24, 48)) -> pd.DataFrame:
    """Aligns FinBERT-style headline sentiment scores onto the OHLCV timestamps and rolls
    them into short/long lookback averages."""
    sent = sentiment_df.set_index("timestamp")["sentiment_score"]
    aligned = sent.reindex(price_index, method="ffill")

    out = pd.DataFrame(index=price_index)
    out["sentiment_now"] = aligned
    for w in windows:
        out[f"sentiment_ma_{w}"] = aligned.rolling(w, min_periods=1).mean()
    out["sentiment_delta"] = aligned.diff()

    return out


def build_feature_matrix(
    coin: str,
    ohlcv: pd.DataFrame,
    btc_ohlcv: pd.DataFrame | None,
    sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """Assembles the full ~45-feature matrix for one coin/timeframe, indexed by timestamp."""
    ohlcv = ohlcv.set_index("timestamp")

    parts = [technical_features(ohlcv)]

    if btc_ohlcv is not None and coin != "BTC":
        btc_close = btc_ohlcv.set_index("timestamp")["close"].reindex(ohlcv.index, method="ffill")
        parts.append(cross_asset_features(ohlcv["close"], btc_close))

    parts.append(microstructure_proxy_features(ohlcv))
    parts.append(on_chain_proxy_features(ohlcv))
    parts.append(sentiment_features(ohlcv.index, sentiment))

    features = pd.concat(parts, axis=1)
    return features
