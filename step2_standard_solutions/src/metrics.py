"""Row-level and event-level evaluation, threshold choice and cycle bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)


def best_f1_threshold(y, score) -> float:
    """Threshold maximising F1 on the given (validation) data."""
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(t) else 0
    return float(t[i]) if len(t) else 0.5


def row_metrics(y, score, thr, proba=True) -> dict:
    y = np.asarray(y)
    score = np.asarray(score)
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "PR-AUC": average_precision_score(y, score) if y.any() else np.nan,
        "ROC-AUC": roc_auc_score(y, score) if 0 < y.sum() < len(y) else np.nan,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "balanced acc": balanced_accuracy_score(y, pred),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }
    if proba:
        out["Brier"] = brier_score_loss(y, np.clip(score, 0, 1))
    return out


def event_metrics(ev_df: pd.DataFrame, score, thr) -> dict:
    """ev_df: rows of one evaluation set with 'y', 'target_event', 'ts', 'target_onset_ts'.

    * an event is *detected* if at least one alert is raised in the N seconds before it
      (i.e. on a row whose label announces that event);
    * warning time = onset time - time of the first alert for that event;
    * a false-alarm episode = a run of consecutive alerts on rows with y = 0.
    """
    d = ev_df[["y", "target_event", "ts", "target_onset_ts"]].copy()
    d["alert"] = np.asarray(score) >= thr
    events = d.loc[d["y"] == 1, "target_event"].dropna().unique()
    detected, warn = 0, []
    for e in events:
        rows = d[(d["target_event"] == e)]
        if rows["alert"].any():
            detected += 1
            first = rows.loc[rows["alert"], "ts"].min()
            warn.append((rows["target_onset_ts"].iloc[0] - first).total_seconds())
    fa = (d["alert"] & (d["y"] == 0)).astype(int)
    episodes = int(((fa == 1) & (fa.shift(fill_value=0) == 0)).sum())
    hours = len(d) / 3600.0
    return {
        "events": len(events),
        "events detected": detected,
        "event recall": detected / len(events) if len(events) else np.nan,
        "mean warning (s)": float(np.mean(warn)) if warn else np.nan,
        "false-alarm episodes": episodes,
        "false alarms / hour": episodes / hours if hours else np.nan,
    }


def bootstrap_ci(ev_df: pd.DataFrame, score, fn, groups="cycle_id", n_boot=1000, seed=42,
                 score_b=None):
    """Percentile CI of fn(y, score) by resampling whole cycles (rows are not independent).

    If `score_b` is given, the CI is computed for fn(y, score) - fn(y, score_b)
    on the same resampled cycles (paired comparison).
    """
    rng = np.random.default_rng(seed)
    y = ev_df["y"].to_numpy()
    s = np.asarray(score)
    sb = None if score_b is None else np.asarray(score_b)
    g = ev_df[groups].to_numpy()
    ug = np.unique(g)
    idx_by_g = {k: np.where(g == k)[0] for k in ug}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by_g[k] for k in pick])
        if y[idx].sum() == 0:
            continue
        v = fn(y[idx], s[idx])
        if sb is not None:
            v -= fn(y[idx], sb[idx])
        vals.append(v)
    vals = np.asarray(vals)
    point = fn(y, s) - (fn(y, sb) if sb is not None else 0)
    return point, np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def bootstrap_event_recall(ev_df, score, thr, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    d = ev_df.reset_index(drop=True).copy()
    d["_s"] = np.asarray(score)
    cyc = d["cycle_id"].to_numpy()
    ug = np.unique(cyc)
    idx_by_g = {k: np.where(cyc == k)[0] for k in ug}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        parts = []
        for j, k in enumerate(pick):                     # relabel events per draw
            p = d.iloc[idx_by_g[k]].copy()
            p["target_event"] = p["target_event"] + 1e6 * j
            parts.append(p)
        b = pd.concat(parts)
        if b["y"].sum() == 0:
            continue
        vals.append(event_metrics(b, b["_s"], thr)["event recall"])
    point = event_metrics(d, d["_s"], thr)["event recall"]
    return point, np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def fa_constrained_threshold(ev_df: pd.DataFrame, score, max_fa_per_hour: float = 30.0) -> float:
    """Lowest threshold whose false-alarm rate on ev_df stays <= max_fa_per_hour.

    Operating point closer to the business need ("alerts must stay trustworthy") than the
    F1-optimal threshold, which tolerates many short false alarms.
    """
    s = np.asarray(score)
    cands = np.unique(np.quantile(s, np.linspace(0.5, 1.0, 400)))
    for t in cands:                                   # increasing thresholds
        if event_metrics(ev_df, s, t)["false alarms / hour"] <= max_fa_per_hour:
            return float(t)
    return float(s.max() + 1e-9)
