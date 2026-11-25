"""Chronological train / validation / test split and time-series cross-validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from . import config as C


def chronological_split(df: pd.DataFrame, fractions=C.SPLIT, gap: int = C.GAP) -> pd.Series:
    """Label every row 'train' / 'val' / 'test' / 'embargo' / None (not predictable).

    * cut points are placed at the 70 % and 85 % quantiles of the *predictable* rows,
      then moved forward to the next cycle start, so that no cycle is cut in two;
    * the last `gap` rows before each cut are put in an embargo: their label window
      (t+1 ... t+N) or their past window would otherwise overlap the next set.
    """
    pred_idx = df.index[df["predictable"]]
    cuts = []
    for q in np.cumsum(fractions)[:-1]:
        k = pred_idx[int(round(q * len(pred_idx)))]
        starts = df.index[(df["pos"] == 0) & (df.index >= k)]
        cuts.append(int(starts[0]) if len(starts) else int(k))

    part = pd.Series(None, index=df.index, dtype=object)
    part[df.index < cuts[0]] = "train"
    part[(df.index >= cuts[0]) & (df.index < cuts[1])] = "val"
    part[df.index >= cuts[1]] = "test"
    for c in cuts:
        part[(df.index >= c - gap) & (df.index < c)] = "embargo"
    part[~df["predictable"]] = None
    return part


def ts_cv(n_splits: int = C.N_SPLITS_CV, gap: int = C.GAP) -> TimeSeriesSplit:
    """Expanding-window CV inside the training period, with a gap of `gap` rows."""
    return TimeSeriesSplit(n_splits=n_splits, gap=gap)


def split_summary(df: pd.DataFrame, part: pd.Series) -> pd.DataFrame:
    rows = []
    for name in ["train", "embargo", "val", "test"]:
        d = df[part == name]
        if d.empty:
            continue
        rows.append({
            "set": name,
            "rows": len(d),
            "positives (y=1)": int(d["y"].sum()),
            "prevalence %": round(100 * d["y"].mean(), 2),
            "events announced": int(d["target_event"].nunique()),
            "cycles": int(d["cycle_id"].nunique()),
            "start (UTC)": d["ts"].min().strftime("%H:%M:%S"),
            "end (UTC)": d["ts"].max().strftime("%H:%M:%S"),
            "mean temp J0 (°C)": round(d["Temperature_J0"].mean(), 1),
        })
    return pd.DataFrame(rows)


def event_balanced_cv(target_event: pd.Series, n_splits: int = C.N_SPLITS_CV, gap: int = C.GAP):
    """Expanding-window CV whose validation folds contain the same number of events.

    A plain TimeSeriesSplit cuts the training period into folds of equal *rows*; here one
    fold (around the lunch break) would contain a single grip-loss event, which makes its
    PR-AUC meaningless. The fold boundaries are therefore placed at the event quantiles
    k / (n_splits + 1); training always uses only rows before the fold (minus `gap`).
    Returns a list of (train positions, validation positions), usable as `cv=` in sklearn.
    """
    ev = target_event.to_numpy()
    n = len(ev)
    first_row = {}
    for i, e in enumerate(ev):
        if not np.isnan(e) and e not in first_row:
            first_row[e] = i
    starts = np.sort(np.array(list(first_row.values())))
    n_ev = len(starts)
    bounds = [int(starts[int(np.floor(k * n_ev / (n_splits + 1)))]) for k in range(1, n_splits + 1)]
    # move each boundary back to just after the previous event's announcing rows
    bounds = [max(b - 3, 0) for b in bounds] + [n]
    out = []
    for k in range(n_splits):
        tr = np.arange(0, max(bounds[k] - gap, 0))
        va = np.arange(bounds[k], bounds[k + 1])
        out.append((tr, va))
    return out
