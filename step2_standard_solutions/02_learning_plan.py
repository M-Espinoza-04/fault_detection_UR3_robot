# %% [markdown]
# # Step 2 – Learning and testing plan
# * chronological train / validation / test split, snapped to cycle starts, with an embargo;
# * expanding-window TimeSeriesSplit (with a gap) inside the training period for tuning;
# * diagnostics for over/under-fitting (expanding-window learning curves);
# * two leakage experiments: shuffled CV vs time-series CV, causal vs centred windows.

# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from src import config as C, splits
from src import plotting as P
from src.features import DYN
from src.models import make_pipeline, scores
from src.prepare import prepare

P.setup()
plt = P.plt
D = prepare()
df, X, part = D.df, D.X, D.part
Xtr, ytr, dtr = D.get("train")

# %% 1. Split summary ---------------------------------------------------------------------
summary = splits.split_summary(df, part)
summary.to_csv(C.TAB / "s2_split_summary.csv", index=False)
print(summary.to_string())

fig, ax = plt.subplots(figsize=(13, 2.4))
idx = np.arange(len(df))
from matplotlib.patches import Patch
pv = part.fillna("none").to_numpy()
start = 0
for i in range(1, len(pv) + 1):                      # draw contiguous runs of each set
    if i == len(pv) or pv[i] != pv[start]:
        if pv[start] in P.SET_COLORS:
            ax.broken_barh([(start, i - start)], (.7, .6), color=P.SET_COLORS[pv[start]], lw=0)
        start = i
ax.scatter(idx[df["onset"]], np.full(df["onset"].sum(), 0.0), marker="|", s=250, color=P.RED, lw=1.2)
for b in idx[df["segment"].diff() > 0]:
    ax.axvline(b, color=P.GRAY, ls=":", lw=1)
ax.set_ylim(-.7, 1.7); ax.set_yticks([0, 1], ["onsets", "sets"])
ax.set_xlabel("row (≈ seconds of operation; dotted = recording break; white gaps = rows not predicted)")
handles = [Patch(color=P.SET_COLORS[k], label=k) for k in ["train", "embargo", "val", "test"]]
handles.append(plt.Line2D([], [], color=P.RED, marker="|", ls="", ms=12, label="grip-loss onset"))
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.42), ncol=5)
ax.set_title("Chronological split 70 / 15 / 15 (cut at cycle starts, 8-row embargo)")
P.save(fig, "s2_fig01_split_timeline.png")

# %% 2. Time-series CV folds inside the training period ------------------------------------
# (a) plain TimeSeriesSplit: equal rows per fold -> one fold holds a single event
# (b) event-balanced expanding window: equal events per fold (used for all tuning)
def fold_table(cv_iter, scheme):
    rows = []
    for k, (a, b) in enumerate(cv_iter):
        rows.append({"scheme": scheme, "fold": k + 1, "train rows": len(a), "validation rows": len(b),
                     "train events": dtr.iloc[a]["target_event"].nunique(),
                     "validation positives": int(ytr.iloc[b].sum()),
                     "validation events": dtr.iloc[b]["target_event"].nunique(),
                     "validation period (UTC)": f"{dtr.iloc[b]['ts'].min():%H:%M}-{dtr.iloc[b]['ts'].max():%H:%M}"})
    return rows


cv = splits.event_balanced_cv(dtr["target_event"])
folds = pd.DataFrame(fold_table(splits.ts_cv().split(Xtr), "TimeSeriesSplit (equal rows)")
                     + fold_table(cv, "Event-balanced (used)"))
folds.to_csv(C.TAB / "s2_cv_folds.csv", index=False)
print(folds.to_string())
folds = folds[folds["scheme"] == "Event-balanced (used)"].reset_index(drop=True)

# %% 3. Expanding-window learning curves (under/over-fitting diagnosis) ---------------------
cands = {
    "Logistic regression (C = 0.1)": make_pipeline(
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000), pos_encoding="spline"),
    "Random forest, unconstrained": make_pipeline(
        RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", n_jobs=-1, random_state=C.SEED)),
    "Random forest, min_samples_leaf = 30": make_pipeline(
        RandomForestClassifier(n_estimators=200, min_samples_leaf=30, max_features=0.3,
                               class_weight="balanced_subsample", n_jobs=-1, random_state=C.SEED)),
}
lc_rows = []
for name, pipe in cands.items():
    res = cross_validate(pipe, Xtr, ytr, cv=cv, scoring="average_precision", return_train_score=True)
    for k in range(len(res["test_score"])):
        lc_rows.append({"model": name, "fold": k + 1, "train rows": folds.loc[k, "train rows"],
                        "train PR-AUC": res["train_score"][k], "CV PR-AUC": res["test_score"][k]})
lc = pd.DataFrame(lc_rows)
lc.round(3).to_csv(C.TAB / "s2_learning_curves.csv", index=False)
print(lc.groupby("model")[["train PR-AUC", "CV PR-AUC"]].mean().round(3))

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, (name, g) in zip(axes, lc.groupby("model", sort=False)):
    ax.plot(g["train rows"], g["train PR-AUC"], marker="o", ms=5, color=P.BLUE, label="training fold")
    ax.plot(g["train rows"], g["CV PR-AUC"], marker="o", ms=5, color=P.ORANGE, label="next (future) fold")
    ax.axhline(ytr.mean(), color=P.GRAY, ls=":", lw=1)
    ax.set_title(name, fontsize=9.5); ax.set_xlabel("training rows (expanding window)")
axes[0].set_ylabel("PR-AUC"); axes[0].text(axes[0].get_xlim()[0], ytr.mean() + .02, " prevalence", color=P.INK2, fontsize=8)
axes[0].legend(loc="center right")
fig.suptitle("Expanding-window learning curves: the gap between the two lines is overfitting", fontweight="bold", fontsize=10.5)
P.save(fig, "s2_fig02_learning_curves.png")

# %% 4. Leakage experiment A: shuffled K-fold vs time-series CV -----------------------------
rf = cands["Random forest, min_samples_leaf = 30"]
leak = []
for label, splitter in [("Shuffled stratified 5-fold (rows treated as independent)",
                         StratifiedKFold(5, shuffle=True, random_state=C.SEED)),
                        ("Expanding-window time-series CV, gap = 8 (our protocol)", cv)]:
    s = cross_validate(rf, Xtr, ytr, cv=splitter, scoring="average_precision")["test_score"]
    leak.append({"experiment": "A: validation scheme", "variant": label, "mean CV PR-AUC": s.mean(), "sd": s.std(),
                 "folds": np.round(s, 3).tolist()})

# %% 5. Leakage experiment B: centred (uses t+1, t+2) vs causal rolling windows ------------
Xc = X.copy()
for col in DYN:
    Xc[f"{col}_rmean{C.WINDOW}"] = X.groupby(df["segment"])[col].transform(
        lambda v: v.rolling(C.WINDOW, center=True, min_periods=1).mean())
    Xc[f"{col}_rstd{C.WINDOW}"] = X.groupby(df["segment"])[col].transform(
        lambda v: v.rolling(C.WINDOW, center=True, min_periods=2).std())
for label, XX in [("Centred rolling windows (look 2 s into the future)", Xc.loc[Xtr.index]),
                  ("Causal rolling windows (past only, our protocol)", Xtr)]:
    s = cross_validate(rf, XX, ytr, cv=cv, scoring="average_precision")["test_score"]
    leak.append({"experiment": "B: feature construction", "variant": label, "mean CV PR-AUC": s.mean(), "sd": s.std(),
                 "folds": np.round(s, 3).tolist()})
leak = pd.DataFrame(leak).round(3)
leak.to_csv(C.TAB / "s2_leakage_experiments.csv", index=False)
print(leak.to_string())

fig, ax = plt.subplots(figsize=(9, 2.8))
yy = np.arange(len(leak))[::-1]
cols = [P.RED if ("Shuffled" in v or "Centred" in v) else P.BLUE for v in leak["variant"]]
ax.barh(yy, leak["mean CV PR-AUC"], color=cols, height=.55)
for y_, f in zip(yy, leak["folds"]):
    ax.scatter(f, np.full(len(f), y_), s=12, color=P.INK, zorder=3)
ax.set_yticks(yy, leak["variant"])
for y_, v in zip(yy, leak["mean CV PR-AUC"]):
    ax.text(v + .01, y_ + .2, f"{v:.3f}", fontsize=8.5)
ax.axvline(ytr.mean(), color=P.GRAY, ls=":", lw=1)
ax.set_xlabel("CV PR-AUC (bar = mean, dots = the 5 folds; dotted = prevalence)")
ax.set_title("Leakage experiments: red = leaky design, blue = our protocol")
P.save(fig, "s2_fig03_leakage_experiments.png")
print("Step 2 done")
