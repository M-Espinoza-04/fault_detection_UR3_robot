# %% [markdown]
# # Step 5 – AutoML pipeline v0.1
# A transparent AutoML built on scikit-learn (no black box, every decision is logged):
# 1. **data preparation**   – imputation (median) and scaling (standard / robust) inside the pipeline;
# 2. **feature engineering** – the feature blocks (windows, cycle residuals, burst, temperature)
#    are *hyper-parameters*, plus optional univariate selection (SelectKBest);
# 3. **model selection**    – 7 standard families (logistic regression, SVM, k-NN, decision tree,
#    random forest, extra trees, gradient boosting);
# 4. **hyper-parameter optimisation** – randomized search, scored by PR-AUC on the event-balanced
#    expanding-window CV of the training period.
#
# Protocol: CV on train -> one champion per family -> champions compared on validation ->
# final pipeline chosen on validation -> test set used **once** for the final report.
#
# Run: `python 05_automl.py` (≈ 15-20 min on 2 cores) or `python 05_automl.py --fast` (≈ 5 min).

# %%
import json
import sys
import time
import warnings

import joblib
from sklearn.base import clone
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src import config as C, splits
from src import plotting as P
from src.metrics import (best_f1_threshold, bootstrap_ci, bootstrap_event_recall, event_metrics,
                         fa_constrained_threshold, row_metrics)
from src.models import make_pipeline, scores
from src.prepare import prepare

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
FAST = "--fast" in sys.argv
P.setup()
plt = P.plt
D = prepare()
Xtr, ytr, dtr = D.get("train")
Xva, yva, dva = D.get("val")
Xte, yte, dte = D.get("test")
cv = splits.event_balanced_cv(dtr["target_event"])

# %% search space ------------------------------------------------------------------------------
FEAT = {"features__window": [True, False], "features__residual": [True, False],
        "features__burst": [True, False], "features__temperature": ["mean", "none"]}
SCALE = {"scale": [StandardScaler(), RobustScaler()]}
SELECT = {"select": ["passthrough", SelectKBest(f_classif, k=20), SelectKBest(f_classif, k=40)]}
rs = C.SEED
SPACE = {
    "Logistic regression": (
        make_pipeline(LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=5000), pos_encoding="spline"),
        {**FEAT, **SCALE, **SELECT, "model__C": loguniform(1e-3, 10), "model__penalty": ["l1", "l2"]}, 25),
    "SVM (RBF kernel)": (
        make_pipeline(SVC(kernel="rbf", class_weight="balanced", cache_size=500), pos_encoding="spline"),
        {**FEAT, **SCALE, **SELECT, "model__C": loguniform(1e-2, 10), "model__gamma": loguniform(1e-4, 1e-1)}, 10),
    "k-nearest neighbours": (
        make_pipeline(KNeighborsClassifier(), pos_encoding="raw"),
        {**FEAT, **SCALE, "select": [SelectKBest(f_classif, k=k) for k in (10, 20, 40)],
         "model__n_neighbors": randint(5, 101), "model__weights": ["uniform", "distance"]}, 15),
    "Decision tree": (
        make_pipeline(DecisionTreeClassifier(class_weight="balanced", random_state=rs), scaler=None),
        {**FEAT, "model__max_depth": randint(2, 12), "model__min_samples_leaf": randint(5, 150),
         "model__ccp_alpha": loguniform(1e-5, 1e-2)}, 20),
    "Random forest": (
        make_pipeline(RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", n_jobs=-1, random_state=rs), scaler=None),
        {**FEAT, "model__max_depth": [None, 6, 10, 16], "model__min_samples_leaf": randint(1, 40),
         "model__max_features": ["sqrt", 0.2, 0.4]}, 12),
    "Extra trees": (
        make_pipeline(ExtraTreesClassifier(n_estimators=200, class_weight="balanced_subsample", n_jobs=-1, random_state=rs), scaler=None),
        {**FEAT, "model__max_depth": [None, 6, 10, 16], "model__min_samples_leaf": randint(1, 40),
         "model__max_features": ["sqrt", 0.2, 0.4]}, 12),
    "Gradient boosting (HGB)": (
        make_pipeline(HistGradientBoostingClassifier(early_stopping=False, random_state=rs), scaler=None),
        {**FEAT, "model__learning_rate": loguniform(1e-2, 3e-1), "model__max_leaf_nodes": randint(4, 48),
         "model__min_samples_leaf": randint(10, 120), "model__l2_regularization": loguniform(1e-4, 10),
         "model__max_iter": randint(50, 400), "model__class_weight": ["balanced", None]}, 25),
}


def short(params: dict) -> str:
    out = []
    for k, v in params.items():
        k = k.replace("features__", "f.").replace("model__", "m.")
        if hasattr(v, "get_params"):
            v = type(v).__name__ + (f"(k={v.k})" if hasattr(v, "k") else "")
        elif isinstance(v, float):
            v = f"{v:.3g}"
        out.append(f"{k}={v}")
    return ", ".join(out)


# %% 1. randomized search per family (train period only) ------------------------------------
board, champions = [], {}
t_all = time.time()
for fam, (pipe, space, n_iter) in SPACE.items():
    n_iter = max(3, n_iter // 3) if FAST else n_iter
    t0 = time.time()
    search = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="average_precision", cv=cv,
                                refit=False, return_train_score=True, random_state=rs, n_jobs=1,
                                error_score=np.nan)
    search.fit(Xtr, ytr)
    r = pd.DataFrame(search.cv_results_)
    r["family"] = fam
    r["config"] = r["params"].map(short)
    board.append(r[["family", "config", "params", "mean_test_score", "std_test_score", "mean_train_score",
                    "mean_fit_time"]])
    best = r["mean_test_score"].idxmax()
    champions[fam] = r.loc[best, "params"]
    print(f"{fam:28s} best CV PR-AUC {r.loc[best, 'mean_test_score']:.3f} ± {r.loc[best, 'std_test_score']:.3f}"
          f"  ({n_iter} configs, {time.time() - t0:.0f} s)")
print(f"search time: {(time.time() - t_all) / 60:.1f} min")
board = pd.concat(board, ignore_index=True).rename(columns={
    "mean_test_score": "CV PR-AUC", "std_test_score": "CV sd", "mean_train_score": "train PR-AUC",
    "mean_fit_time": "fit time (s)"})
board["overfit gap"] = board["train PR-AUC"] - board["CV PR-AUC"]
board.sort_values("CV PR-AUC", ascending=False).drop(columns="params").round(4).to_csv(
    C.TAB / "s5_leaderboard_all_configs.csv", index=False)

# %% 2. champions compared on the validation period -------------------------------------------
champ_rows, fitted = [], {}
for fam, params in champions.items():
    pipe = clone(SPACE[fam][0]).set_params(**params)     # clone: never mutate the template
    pipe.fit(Xtr, ytr)
    fitted[fam] = pipe
    s_va = scores(pipe, Xva)
    thr_fa = fa_constrained_threshold(dva, s_va, C.MAX_FA_PER_HOUR)
    b = board[(board["family"] == fam)].sort_values("CV PR-AUC", ascending=False).iloc[0]
    champ_rows.append({"family": fam, "CV PR-AUC": b["CV PR-AUC"], "CV sd": b["CV sd"],
                       "train PR-AUC": b["train PR-AUC"], "val PR-AUC": average_precision_score(yva, s_va),
                       f"val event recall (<= {C.MAX_FA_PER_HOUR:g} FA/h)": event_metrics(dva, s_va, thr_fa)["event recall"],
                       "config": short(params)})
champ = pd.DataFrame(champ_rows).sort_values("val PR-AUC", ascending=False)
champ.round(4).to_csv(C.TAB / "s5_family_champions.csv", index=False)
print(champ.drop(columns="config").round(3).to_string())

final_fam = champ.iloc[0]["family"]
final = fitted[final_fam]
final_params = champions[final_fam]
json.dump({"family": final_fam, "params": {k: (repr(v) if hasattr(v, "get_params") else v)
                                           for k, v in final_params.items()},
           "horizon": C.HORIZON, "window": C.WINDOW},
          open(C.MODELS / "final_config.json", "w", encoding="utf-8"), indent=2, default=str)
joblib.dump(final, C.MODELS / "automl_final_pipeline.joblib")
print("FINAL:", final_fam, "|", short(final_params))

# %% 3. final evaluation on the test period (used once) ----------------------------------------
base = pd.read_csv(C.MODELS / "baseline_scores.csv")
s_va_f, s_te_f = scores(final, Xva), scores(final, Xte)
thr_f1 = best_f1_threshold(yva, s_va_f)
thr_fa = fa_constrained_threshold(dva, s_va_f, C.MAX_FA_PER_HOUR)
entries = {f"AutoML: {final_fam}": (s_va_f, s_te_f, thr_f1, thr_fa)}
for name in ["B1 cycle position", "B2 logistic regression"]:
    bv = base[(base["model"] == name) & (base["set"] == "val")]
    bt = base[(base["model"] == name) & (base["set"] == "test")]
    entries[name] = (bv["score"].to_numpy(), bt["score"].to_numpy(), bv["thr_f1"].iat[0], bv["thr_fa"].iat[0])

rows = []
for name, (s_va, s_te, t1, t2) in entries.items():
    for op, thr in [("F1-optimal", t1), (f"<= {C.MAX_FA_PER_HOUR:g} FA/h", t2)]:
        r = {"model": name, "operating point": op, "threshold": thr}
        r.update(row_metrics(yte, s_te, thr, proba=False))
        r.update(event_metrics(dte, s_te, thr))
        _, lo, hi = bootstrap_ci(dte, s_te, average_precision_score, n_boot=C.N_BOOT)
        r["PR-AUC 95% CI"] = f"[{lo:.3f}, {hi:.3f}]"
        _, lo, hi = bootstrap_event_recall(dte, s_te, thr, n_boot=C.N_BOOT)
        r["event recall 95% CI"] = f"[{lo:.2f}, {hi:.2f}]"
        rows.append(r)
test_res = pd.DataFrame(rows)
test_res.round(4).to_csv(C.TAB / "s5_test_results.csv", index=False)
print(test_res[["model", "operating point", "PR-AUC", "PR-AUC 95% CI", "ROC-AUC", "precision", "recall",
                "event recall", "event recall 95% CI", "mean warning (s)", "false alarms / hour"]].round(3).to_string())

b2_te = entries["B2 logistic regression"][1]
b1_te = entries["B1 cycle position"][1]
delta = []
for ref_name, ref in [("B2 logistic regression", b2_te), ("B1 cycle position", b1_te)]:
    d_, lo, hi = bootstrap_ci(dte, s_te_f, average_precision_score, n_boot=C.N_BOOT, score_b=ref)
    delta.append({"comparison": f"AutoML - {ref_name}", "Δ PR-AUC (test)": d_, "95% CI (paired cycle bootstrap)": f"[{lo:.3f}, {hi:.3f}]",
                  "significant": bool(lo > 0)})
delta = pd.DataFrame(delta)
delta.round(4).to_csv(C.TAB / "s5_paired_differences.csv", index=False)
print(delta.to_string())

# event-level breakdown: "follow-up" losses (another loss ended < 60 s before the onset)
# vs "isolated" losses. Bursts make follow-up losses much easier to anticipate.
def event_table(dset, s, thr, set_name):
    rows_ = []
    alerts_ = s >= thr
    for e, g in dset[dset["y"] == 1].groupby("target_event"):
        onset = D.df.index[(D.df["event_id"] == e) & D.df["onset"]][0]
        since = float(D.X.at[onset, "secs_since_last_gl"])
        pos_ = dset.index.get_indexer(g.index)
        rows_.append({"set": set_name, "event": int(e), "onset (UTC)": D.df.at[onset, "ts"].strftime("%H:%M:%S"),
                      "position in cycle (s)": int(D.df.at[onset, "pos"]),
                      "duration (s)": int((D.df["event_id"] == e).sum()),
                      "s since previous loss": since,
                      "type": "follow-up (< 60 s after a loss)" if since < 60 else "isolated",
                      "max score before onset": float(s[pos_].max()),
                      f"detected (<= {C.MAX_FA_PER_HOUR:g} FA/h)": bool(alerts_[pos_].any())})
    return pd.DataFrame(rows_)


ev_tab = pd.concat([event_table(dva, s_va_f, thr_fa, "val"), event_table(dte, s_te_f, thr_fa, "test")])
ev_tab.round(3).to_csv(C.TAB / "s5_events_detail.csv", index=False)
det_col = f"detected (<= {C.MAX_FA_PER_HOUR:g} FA/h)"
ev_summary = (ev_tab.groupby(["set", "type"])[det_col].agg(["size", "sum"])
              .rename(columns={"size": "events", "sum": "detected"}).reset_index())
ev_summary["event recall"] = ev_summary["detected"] / ev_summary["events"]
ev_summary.round(3).to_csv(C.TAB / "s5_events_by_type.csv", index=False)
print(ev_summary.round(2).to_string())

# %% 4. diagnostics on train / validation only (test not reused) --------------------------------
# 4a. ablation of the feature blocks (CV on train + validation)
abl_rows = []
fm_keys = {k: v for k, v in final_params.items() if k.startswith("features__")}
variants = {"final configuration": {}}
for blk in ["window", "residual", "burst"]:
    variants[f"{'without' if fm_keys.get('features__' + blk) else 'with'} {blk}"] = {f"features__{blk}": not fm_keys.get("features__" + blk)}
variants["temperature: mean" if fm_keys.get("features__temperature") == "none" else "without temperature"] = {
    "features__temperature": "mean" if fm_keys.get("features__temperature") == "none" else "none"}
variants["without position in cycle"] = {"features__pos_encoding": "none"}
for name, change in variants.items():
    p = clone(SPACE[final_fam][0]).set_params(**{**final_params, **change})
    cvs = cross_val_score(p, Xtr, ytr, cv=cv, scoring="average_precision")
    p.fit(Xtr, ytr)
    abl_rows.append({"variant": name, "CV PR-AUC": cvs.mean(), "CV sd": cvs.std(),
                     "val PR-AUC": average_precision_score(yva, scores(p, Xva))})
abl = pd.DataFrame(abl_rows)
abl.round(4).to_csv(C.TAB / "s5_ablation.csv", index=False)
print(abl.round(3).to_string())

# 4b. horizon sensitivity N = 1, 2, 3, 5 (target and split rebuilt each time), with and
#     without the burst features: a jump at N = 1 *without* burst features would mean the
#     model sees the loss itself (label recorded after the physical event).
hz_rows = []
for N in [1, 2, 3, 5]:
    DN = prepare(horizon=N)
    a, ya, da = DN.get("train")
    v, yv, _ = DN.get("val")
    cvN = splits.event_balanced_cv(da["target_event"], gap=N + C.WINDOW)
    for burst in [True, False]:
        p = clone(SPACE[final_fam][0]).set_params(**{**final_params, "features__burst": burst})
        cvs = cross_val_score(p, a, ya, cv=cvN, scoring="average_precision")
        p.fit(a, ya)
        ap_v = average_precision_score(yv, scores(p, v))
        hz_rows.append({"horizon N (s)": N, "burst features": burst, "val prevalence": yv.mean(),
                        "CV PR-AUC": cvs.mean(), "val PR-AUC": ap_v, "val lift (PR-AUC / prevalence)": ap_v / yv.mean()})
hz = pd.DataFrame(hz_rows)
hz.round(4).to_csv(C.TAB / "s5_horizon_sensitivity.csv", index=False)
print(hz.round(3).to_string())

# 4c. permutation importance on validation
pi = permutation_importance(final, Xva, yva, scoring="average_precision", n_repeats=5, random_state=rs, n_jobs=1)
imp = pd.DataFrame({"feature": Xva.columns, "importance": pi.importances_mean, "sd": pi.importances_std})
imp = imp[imp["importance"].abs() > 0].sort_values("importance", ascending=False)
imp.round(4).to_csv(C.TAB / "s5_permutation_importance.csv", index=False)

# %% 5. figures ---------------------------------------------------------------------------------
fams = list(SPACE.keys())
order = champ.set_index("family").loc[fams, "CV PR-AUC"].sort_values().index.tolist()
fig, ax = plt.subplots(figsize=(10, 4.4))
for k, fam in enumerate(order):
    g = board[board["family"] == fam]
    jitter = (np.random.default_rng(k).random(len(g)) - .5) * .25
    ax.scatter(g["CV PR-AUC"], k + jitter, s=14, color=P.GRAY, alpha=.6, zorder=2)
    c = champ.set_index("family").loc[fam]
    ax.scatter(c["CV PR-AUC"], k, s=60, color=P.BLUE, zorder=3, label="champion: CV (train)" if k == 0 else None)
    ax.scatter(c["val PR-AUC"], k, s=60, color=P.ORANGE, marker="D", zorder=3, label="champion: validation" if k == 0 else None)
ax.axvline(ytr.mean(), color=P.INK2, ls=":", lw=1)
ax.set_yticks(range(len(order)), order); ax.set_xlabel("PR-AUC (dotted = prevalence; gray = every configuration tried)")
ax.legend(loc="lower right"); ax.set_title("AutoML leaderboard: model families, all configurations and champions")
P.save(fig, "s5_fig01_leaderboard.png")

fig, ax = plt.subplots(figsize=(6.2, 5))
for k, fam in enumerate(fams):
    g = board[board["family"] == fam]
    ax.scatter(g["CV PR-AUC"], g["train PR-AUC"], s=16, color=P.SERIES[k], label=fam, alpha=.8)
ax.plot([0, 1], [0, 1], color=P.GRAY, ls=":", lw=1)
ax.set_xlim(0, max(.6, board["CV PR-AUC"].max() + .05)); ax.set_ylim(0, 1.02)
ax.set_xlabel("CV PR-AUC (future folds)"); ax.set_ylabel("training PR-AUC")
ax.set_title("Overfitting map: distance above the diagonal"); ax.legend(fontsize=7.5, loc="lower right")
P.save(fig, "s5_fig02_overfitting_map.png")

fig, ax = plt.subplots(figsize=(6.4, 4.6))
for (name, (_, s_te, _, _)), col in zip(entries.items(), [P.BLUE, P.ORANGE, P.AQUA]):
    pr, rc, _ = precision_recall_curve(yte, s_te)
    ax.plot(rc, pr, color=col, lw=1.8, label=f"{name} (AP = {average_precision_score(yte, s_te):.3f})", drawstyle="steps-post")
ax.axhline(yte.mean(), color=P.GRAY, ls=":", lw=1, label=f"prevalence ({yte.mean():.3f})")
ax.set_xlabel("recall (rows)"); ax.set_ylabel("precision"); ax.set_ylim(0, 1)
ax.set_title("Precision-recall curves on the test period"); ax.legend(fontsize=8, loc="upper right")
P.save(fig, "s5_fig03_pr_curves_test.png")

fig, ax = plt.subplots(figsize=(13, 3.4))
tt = (dte["ts"] - dte["ts"].min()).dt.total_seconds() / 60
ax.plot(tt, s_te_f, color=P.BLUE, lw=.9, label="AutoML score")
ax.axhline(thr_fa, color=P.ORANGE, ls="--", lw=1.2, label=f"threshold (<= {C.MAX_FA_PER_HOUR:g} false alarms / h on validation)")
on_t = (D.df.loc[D.df["onset"] & (D.df.index >= dte.index.min()) & (D.df.index <= dte.index.max()), "ts"] - dte["ts"].min()).dt.total_seconds() / 60
for k, x_ in enumerate(on_t):
    ax.axvline(x_, color=P.RED, lw=1.2, alpha=.7, label="grip-loss onset" if k == 0 else None)
ax.set_xlabel("minutes into the test period"); ax.set_ylabel("score")
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.25), ncol=3)
ax.set_title("Test period: alert score vs real grip-loss onsets")
P.save(fig, "s5_fig04_test_timeline.png")

fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
a_ = abl.iloc[::-1]
yy = np.arange(len(a_))
ax[0].barh(yy + .18, a_["CV PR-AUC"], height=.36, color=P.BLUE, label="CV (train)")
ax[0].barh(yy - .18, a_["val PR-AUC"], height=.36, color=P.ORANGE, label="validation")
ax[0].set_yticks(yy, a_["variant"]); ax[0].set_xlabel("PR-AUC")
ax[0].legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2)
ax[0].set_title(f"Ablation of feature blocks ({final_fam})")
for burst, col, lab in [(True, P.BLUE, "with burst features"), (False, P.ORANGE, "without burst features")]:
    h = hz[hz["burst features"] == burst]
    ax[1].plot(h["horizon N (s)"], h["val lift (PR-AUC / prevalence)"], marker="o", color=col, label=lab)
    for x_, v in zip(h["horizon N (s)"], h["val lift (PR-AUC / prevalence)"]):
        ax[1].text(x_ + .08, v, f"{v:.1f}×", fontsize=8)
ax[1].axhline(1, color=P.GRAY, ls=":", lw=1)
ax[1].set_yscale("log"); ax[1].legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2)
ax[1].set_xticks(sorted(hz["horizon N (s)"].unique())); ax[1].set_xlabel("horizon N (s)")
ax[1].set_ylabel("validation PR-AUC / prevalence (log)"); ax[1].set_title("Horizon sensitivity (1 = no skill)")
P.save(fig, "s5_fig05_ablation_horizon.png")

top = imp.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.barh(top["feature"], top["importance"], xerr=top["sd"], color=P.BLUE, height=.6, error_kw={"ecolor": P.INK2, "lw": 1})
ax.set_xlabel("drop in validation PR-AUC when the column is shuffled")
ax.set_title("Permutation importance of the final pipeline (validation)")
P.save(fig, "s5_fig06_permutation_importance.png")
print("Step 5 done")
