"""Naive reference models (B0, B1)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class PriorBaseline(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.p_ = float(np.mean(y)); self.classes_ = np.array([0, 1]); return self

    def predict_proba(self, X):
        return np.c_[np.full(len(X), 1 - self.p_), np.full(len(X), self.p_)]


class CyclePositionBaseline(BaseEstimator, ClassifierMixin):
    """P(y = 1 | position in cycle), Laplace-smoothed toward the prior (alpha pseudo-rows)."""

    def __init__(self, alpha=20.0, cap=40):
        self.alpha, self.cap = alpha, cap

    def fit(self, X, y):
        pos = X["pos"].clip(upper=self.cap).astype(int)
        y = np.asarray(y)
        self.prior_ = y.mean()
        g = pd.DataFrame({"pos": pos.to_numpy(), "y": y}).groupby("pos")["y"].agg(["sum", "count"])
        self.rate_ = (g["sum"] + self.alpha * self.prior_) / (g["count"] + self.alpha)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = X["pos"].clip(upper=self.cap).astype(int).map(self.rate_).fillna(self.prior_).to_numpy()
        return np.c_[1 - p, p]
