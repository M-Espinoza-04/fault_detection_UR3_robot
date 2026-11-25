"""Feature engineering.

Two levels:
1. `build_base_features(df)` computes, on the whole timeline, every *causal* candidate
   feature (only rows <= t are used, windows never cross a segment break). Being
   causal and label-free, it can be computed once before the split without leakage.
   The only label-based block ('burst') uses grip-loss labels strictly before t,
   which an operator / the gripper would know in real time.
2. `FeatureMaker` is a scikit-learn transformer placed *inside* the pipeline. Anything
   that is learned from data (normal cycle profile, spline basis) is fitted in
   `fit`, i.e. on the training fold only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import SplineTransformer

from . import config as C

DYN = C.CUR + C.TOOL + ["abs_speed"]           # signals that get lag / window features
RESID = C.CUR + C.SPD + C.TOOL + ["abs_speed"]  # signals compared with the normal cycle profile


def build_base_features(df: pd.DataFrame, window: int = C.WINDOW) -> pd.DataFrame:
    seg = df["segment"]
    X = pd.DataFrame(index=df.index)

    # causal forward fill of empty rows (at most FFILL_LIMIT s, never across segments)
    s = df[C.SENSORS].groupby(seg).ffill(limit=C.FFILL_LIMIT)
    X[C.SENSORS] = s
    X["temp_mean"] = s[C.TEM].mean(axis=1)                       # 6 temps: r >= 0.99
    X["abs_speed"] = s[C.SPD].abs().sum(axis=1)
    X["pos"] = df["pos"].astype(float)

    g = X[DYN].groupby(seg)
    for col in DYN:
        gc = g[col]
        X[f"{col}_lag1"] = gc.shift(1)
        X[f"{col}_lag2"] = gc.shift(2)
        X[f"{col}_diff1"] = X[col] - X[f"{col}_lag1"]
        X[f"{col}_rmean{window}"] = gc.transform(lambda v: v.rolling(window, min_periods=window).mean())
        X[f"{col}_rstd{window}"] = gc.transform(lambda v: v.rolling(window, min_periods=window).std())
    X["temp_mean_diff60"] = X["temp_mean"] - X.groupby(seg)["temp_mean"].shift(60)  # warm-up rate

    # burst features: past grip losses only (strictly before t, same session)
    X = X.join(_burst_features(df))
    return X


def _burst_features(df: pd.DataFrame, cap: float = 600.0) -> pd.DataFrame:
    t = (df["ts"] - df["ts"].iloc[0]).dt.total_seconds().to_numpy()
    gl = df["GL"].to_numpy()
    onset = df["onset"].to_numpy()
    sess = df["session"].to_numpy()
    since = np.full(len(df), cap)
    n_recent = np.zeros(len(df))
    last_gl, onsets = None, []
    for i in range(len(df)):
        if i > 0 and sess[i] != sess[i - 1]:
            last_gl, onsets = None, []                 # reset after the 5 h break
        if last_gl is not None:
            since[i] = min(t[i] - last_gl, cap)
        onsets = [o for o in onsets if t[i] - o <= 120]
        n_recent[i] = len(onsets)
        # update AFTER computing row i: row i itself is never used for row i
        if gl[i] == 1:
            last_gl = t[i]
        if onset[i]:
            onsets.append(t[i])
    return pd.DataFrame({"secs_since_last_gl": since, "n_gl_last_120s": n_recent}, index=df.index)


# ----------------------------------------------------------------------------- blocks
def feature_blocks(columns) -> dict[str, list[str]]:
    cols = list(columns)
    return {
        "raw": C.CUR + C.SPD + C.TOOL + ["abs_speed"],
        "temp_all": C.TEM,
        "temp_mean": ["temp_mean", "temp_mean_diff60"],
        "window": [c for c in cols if any(c.endswith(s) for s in
                                          ("_lag1", "_lag2", "_diff1", f"_rmean{C.WINDOW}", f"_rstd{C.WINDOW}"))],
        "burst": ["secs_since_last_gl", "n_gl_last_120s"],
    }


class FeatureMaker(BaseEstimator, TransformerMixin):
    """Select feature blocks and build the train-fitted features.

    Parameters
    ----------
    window   : add lag / diff / rolling features
    residual : add x - median(x | position in cycle), the profile being learned in fit
    burst    : add time since the last grip loss and number of recent losses
    temperature : 'mean' (1 reduced variable + warm-up rate), 'all' (6 raw) or 'none'
    pos_encoding: 'raw' (trees), 'spline' (linear models: non-monotonic effect), 'none'
    """

    def __init__(self, window=True, residual=True, burst=False, temperature="mean",
                 pos_encoding="raw", n_knots=8):
        self.window = window
        self.residual = residual
        self.burst = burst
        self.temperature = temperature
        self.pos_encoding = pos_encoding
        self.n_knots = n_knots

    def fit(self, X: pd.DataFrame, y=None):
        blocks = feature_blocks(X.columns)
        cols = list(blocks["raw"])
        if self.temperature == "mean":
            cols += blocks["temp_mean"]
        elif self.temperature == "all":
            cols += blocks["temp_all"]
        if self.window:
            cols += blocks["window"]
        if self.burst:
            cols += blocks["burst"]
        self.cols_ = cols
        if self.residual:
            pos = X["pos"].clip(upper=self._pos_cap(X))
            self.profile_ = X[RESID].groupby(pos).median()
            self.pos_cap_ = self._pos_cap(X)
        if self.pos_encoding == "spline":
            self.spline_ = SplineTransformer(n_knots=self.n_knots, degree=3, extrapolation="constant")
            self.spline_.fit(X[["pos"]].clip(upper=40))
        self.feature_names_out_ = self._names()
        return self

    @staticmethod
    def _pos_cap(X):
        counts = X["pos"].value_counts()
        ok = counts[counts >= 20].index
        return float(ok.max()) if len(ok) else float(X["pos"].max())

    def _names(self):
        names = list(self.cols_)
        if self.residual:
            names += [f"{c}_res" for c in RESID]
        if self.pos_encoding == "raw":
            names += ["pos"]
        elif self.pos_encoding == "spline":
            names += [f"pos_spline{i}" for i in range(self.spline_.n_features_out_)]
        return names

    def transform(self, X: pd.DataFrame):
        parts = [X[self.cols_].to_numpy(dtype=float)]
        if self.residual:
            pos = X["pos"].clip(upper=self.pos_cap_)
            ref = self.profile_.reindex(pos.to_numpy()).to_numpy()
            parts.append(X[RESID].to_numpy(dtype=float) - ref)
        if self.pos_encoding == "raw":
            parts.append(X[["pos"]].to_numpy(dtype=float))
        elif self.pos_encoding == "spline":
            parts.append(self.spline_.transform(X[["pos"]].clip(upper=40)))
        return np.hstack(parts)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)
