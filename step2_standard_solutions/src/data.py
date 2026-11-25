"""Loading, cleaning and time-series structure of UR3 CobotOps.

Design rule: nothing in this file looks into the future of a row except
`make_target`, which builds the label (the label is *allowed* to look ahead,
the features are not).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# ----------------------------------------------------------------------------- load
def load_raw(path=C.DATA_CSV) -> pd.DataFrame:
    """Read the CSV exactly as stored (French format: ';' separator, ',' decimal)."""
    return pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")


# ----------------------------------------------------------------------------- clean
def clean(raw: pd.DataFrame, merge_gap: int = C.MERGE_GAP) -> pd.DataFrame:
    """Return one row per recorded second with clean names, types and structure columns.

    All 7,409 rows are kept: the empty rows stay in the timeline (flagged with
    `sensor_missing`) so that durations, lags and events are not distorted.
    """
    df = raw.loc[:, ~raw.columns.str.startswith("Unnamed")].copy()   # 11 empty trailing columns
    df.columns = df.columns.str.strip()                                # 'cycle ' -> 'cycle'
    df = df.rename(columns={"Temperature_T0": "Temperature_J0"})       # T0 is joint J0

    df["ts"] = pd.to_datetime(df["Timestamp"].str.strip('"'), utc=True)  # 863 quoted stamps
    df["GL_raw"] = df["grip_lost"].eq("VRAI").astype(int)
    df["PS"] = df["Robot_ProtectiveStop"].map({"VRAI": 1, "FAUX": 0})   # NaN on empty rows

    # Empty rows: no sensor at all except a placeholder Current_J0 == 1.0 on 8 of them.
    df["sensor_missing"] = df["Current_J1"].isna()
    df.loc[df["sensor_missing"], "Current_J0"] = np.nan

    # Exact duplicates of the measurements (row number and timestamp excluded)
    df["duplicate"] = df.duplicated(subset=C.SENSORS + ["cycle"], keep="first") & ~df["sensor_missing"]

    # ---- time-series structure
    dt = df["ts"].diff().dt.total_seconds()
    df["dt"] = dt
    df["segment"] = (dt > C.SEGMENT_BREAK_S).cumsum()                 # recording breaks
    df["session"] = (dt > 3600).cumsum().map({0: "morning", 1: "afternoon"})
    df["cycle_id"] = (df["cycle"] != df["cycle"].shift()).cumsum()     # counter is not unique
    df["pos"] = df.groupby("cycle_id").cumcount()                      # seconds since cycle start

    # ---- label debouncing: a 1-2 s "hole" inside a grip loss is treated as the same event
    df["GL"] = debounce(df["GL_raw"].to_numpy(), df["segment"].to_numpy(), merge_gap)
    df["onset"] = (df["GL"] == 1) & (df.groupby("segment")["GL"].shift(fill_value=0) == 0)
    df["event_id"] = df["onset"].cumsum().where(df["GL"] == 1)
    return df


def debounce(gl: np.ndarray, segment: np.ndarray, merge_gap: int) -> np.ndarray:
    """Fill runs of <= merge_gap zeros that sit between two ones in the same segment."""
    out = gl.copy()
    if merge_gap <= 0:
        return out
    n = len(gl)
    i = 0
    while i < n:
        if out[i] == 0:
            j = i
            while j < n and out[j] == 0:
                j += 1
            run = j - i
            if 0 < i and j < n and run <= merge_gap and out[i - 1] == 1 and out[j] == 1 \
                    and segment[i - 1] == segment[j]:
                out[i:j] = 1
            i = j
        else:
            i += 1
    return out


# ----------------------------------------------------------------------------- target
def make_target(df: pd.DataFrame, horizon: int = C.HORIZON, window: int = C.WINDOW) -> pd.DataFrame:
    """Early-warning label and the mask of rows on which a prediction is made.

    y_t = 1  if a grip loss starts in (t, t+N]   (rows t+1 ... t+N, same segment)
    A row is *predictable* when:
      * no grip loss is ongoing at t (the alert must come before the loss),
      * the sensors were recorded at t,
      * the full past window [t-W+1, t] is in the same segment (features defined),
      * the full future horizon [t+1, t+N] is in the same segment (label defined).
    `target_event` stores which onset a positive row announces (event-level metrics).
    """
    out = df.copy()
    gl = out["GL"].to_numpy()
    seg = out["segment"].to_numpy()
    ev = out["event_id"].to_numpy()
    n = len(out)

    y = np.zeros(n, dtype=int)
    target_event = np.full(n, np.nan)
    target_onset = np.full(n, -1)
    future_ok = np.zeros(n, dtype=bool)
    for t in range(n):
        end = t + horizon
        if end < n and seg[end] == seg[t]:
            future_ok[t] = True
            fut = gl[t + 1:end + 1]
            if fut.any():
                y[t] = 1
                k = t + 1 + int(np.argmax(fut))
                target_event[t] = ev[k]
                target_onset[t] = k

    pos_in_seg = out.groupby("segment").cumcount().to_numpy()
    past_ok = pos_in_seg >= window - 1

    out["y"] = y
    out["target_event"] = target_event
    out["target_onset_ts"] = pd.Series(target_onset, index=out.index).map(
        lambda k: out["ts"].iloc[k] if k >= 0 else pd.NaT)
    out["predictable"] = ((gl == 0) & ~out["sensor_missing"].to_numpy() & ~out["duplicate"].to_numpy()
                          & past_ok & future_ok)
    return out
