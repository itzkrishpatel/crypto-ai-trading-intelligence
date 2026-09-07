src/model.py"""
model.py
--------
Per-(coin, timeframe) XGBoost classifier: predicts P(price up over the next bar) and
converts that probability into a BUY / HOLD / SELL signal using the thresholds in
configs/coins.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb


@dataclass
class SignalThresholds:
    buy: float = 0.60
    sell: float = 0.40

    def classify(self, prob_up: float) -> str:
        if prob_up >= self.buy:
            return "BUY"
        if prob_up <= self.sell:
            return "SELL"
        return "HOLD"


def make_labels(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Binary label: 1 if price is higher `horizon` bars ahead, else 0. Shifted so the label
    at time t only uses information from t+horizon — callers must drop the resulting NaN tail
    before training, and must never let a training fold see rows whose label depends on data
    past the fold's cutoff (see backtest.py)."""
    future_return = close.shift(-horizon) / close - 1
    return (future_return > 0).astype(int)


class CoinTimeframeModel:
    """Wraps one XGBoost classifier for a single (coin, timeframe) pair."""

    def __init__(self, coin: str, timeframe: str, thresholds: SignalThresholds, params: dict | None = None):
        self.coin = coin
        self.timeframe = timeframe
        self.thresholds = thresholds
        self.params = params or {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_jobs": -1,
        }
        self.booster: xgb.XGBClassifier | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CoinTimeframeModel":
        mask = X.notna().all(axis=1) & y.notna()
        X_clean, y_clean = X.loc[mask], y.loc[mask]

        self.booster = xgb.XGBClassifier(**self.params)
        self.booster.fit(X_clean, y_clean)
        self.feature_names_ = list(X.columns)
        return self

    def predict_proba_up(self, X: pd.DataFrame) -> pd.Series:
        if self.booster is None:
            raise RuntimeError("Model must be fit() before predicting.")
        X_aligned = X[self.feature_names_]
        mask = X_aligned.notna().all(axis=1)
        proba = pd.Series(np.nan, index=X.index)
        if mask.any():
            proba.loc[mask] = self.booster.predict_proba(X_aligned.loc[mask])[:, 1]
        return proba

    def predict_signal(self, X: pd.DataFrame) -> pd.Series:
        proba = self.predict_proba_up(X)
        return proba.apply(lambda p: self.thresholds.classify(p) if pd.notna(p) else "HOLD")

    def feature_importances(self) -> pd.Series:
        if self.booster is None:
            raise RuntimeError("Model must be fit() before inspecting importances.")
        return pd.Series(self.booster.feature_importances_, index=self.feature_names_).sort_values(ascending=False)
