"""One call that every step uses: clean data, label, causal features and split."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config as C, data, features, splits


@dataclass
class Prepared:
    df: pd.DataFrame          # full time line with labels and structure columns
    X: pd.DataFrame           # causal candidate features (whole time line)
    part: pd.Series           # 'train' / 'val' / 'test' / 'embargo' / None

    def get(self, name):
        m = self.part == name
        return self.X.loc[m], self.df.loc[m, "y"], self.df.loc[m]


def prepare(horizon: int = C.HORIZON, window: int = C.WINDOW) -> Prepared:
    df = data.make_target(data.clean(data.load_raw()), horizon=horizon, window=window)
    X = features.build_base_features(df, window=window)
    part = splits.chronological_split(df, gap=horizon + window)
    return Prepared(df, X, part)
