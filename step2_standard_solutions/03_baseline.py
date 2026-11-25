# %% [markdown]
# # Step 3 – Baseline solution and its calibration
# Three reference levels, from naive to the "real" baseline:
# * **B0 – prior**: constant score = training prevalence (what PR-AUC gives with no skill);
# * **B1 – cycle position**: grip-loss risk learned per second of the cycle (domain baseline);
# * **B2 – logistic regression** on the current sensor values + spline(position in cycle).
#
# Calibration of B2 = (1) the regularisation strength C chosen by time-series CV with the
# one-standard-error rule, (2) the probability calibration (class weights distort the
# probabilities), (3) the decision threshold chosen on the validation period.

# %%
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import validation_curve

from src import config as C, splits
from src import plotting as P
from src.metrics import (best_f1_threshold, bootstrap_ci, bootstrap_event_recall, event_metrics,
                         fa_constrained_threshold, row_metrics)
from src.baselines import CyclePositionBaseline, PriorBaseline
from src.models import make_pipeline
from src.prepare import prepare

P.setup()
plt = P.plt
D = prepare()
Xtr, ytr, dtr = D.get("train")
Xva, yva, dva = D.get("val")
Xte, yte, dte = D.get("test")
cv = splits.event_balanced_cv(dtr["target_event"])


# %% B0 and B1 -------------------------------------------------------------------------------
b0 = PriorBaseline().fit(Xtr, ytr)
b1 = CyclePositionBaseline().fit(Xtr, ytr)

# %% B2: logistic regression -- choice of C by time-series CV (validation curve) -------------
def lr_pipe(C_=1.0, cw="balanced"):
    return make_pipeline(LogisticRegression(C=C_, class_weight=cw, max_iter=5000),
                         window=False, residual=False, burst=False, temperature="mean", pos_encoding="spline")


grid = np.logspace(-4, 2, 13)
tr_s, cv_s = validation_curve(lr_pipe(), Xtr, ytr, param_name="model__C", param_range=grid,
                              cv=cv, scoring="average_precision")
vc = pd.DataFrame({"C": grid, "train mean": tr_s.mean(1), "CV mean": cv_s.mean(1),
                   "CV se": cv_s.std(1, ddof=1) / np.sqrt(cv_s.shape[1])})
best = int(vc["CV mean"].idxmax())
# One-standard-error rule on *paired* fold differences: the folds are the same for every C,
# so the fold-to-fold variance (very large here, ~8 events per fold) cancels out. The plain
# 1-SE rule (unpaired SE) would always pick the strongest penalty of the grid.
diff = cv_s - cv_s[best]
vc["paired diff to best"] = diff.mean(1)
vc["paired se"] = diff.std(1, ddof=1) / np.sqrt(diff.shape[1])
ok = vc["paired diff to best"] >= -vc["paired se"].fillna(0)
C_chosen = float(vc.loc[ok, "C"].min())
C_unpaired = float(vc.loc[vc["CV mean"] >= vc.loc[best, "CV mean"] - vc.loc[best, "CV se"], "C"].min())
vc.round(4).to_csv(C.TAB / "s3_validation_curve_C.csv", index=False)
print(vc.round(3).to_string(), "\nC with best CV:", vc.loc[best, "C"], "| C chosen (paired 1-SE):", C_chosen,
      "| plain 1-SE would give:", C_unpaired)

fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.plot(grid, vc["train mean"], marker="o", ms=4, color=P.BLUE, label="training folds")
ax.plot(grid, vc["CV mean"], marker="o", ms=4, color=P.ORANGE, label="validation folds (mean ± 1 SE)")
ax.fill_between(grid, vc["CV mean"] - vc["CV se"], vc["CV mean"] + vc["CV se"], color=P.ORANGE, alpha=.15, lw=0)
ax.axvline(C_chosen, color=P.INK, ls="--", lw=1)
ax.text(C_chosen * 1.15, ax.get_ylim()[0] + .01, f"chosen C = {C_chosen:.3g}\n(paired 1-SE rule)", fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("C (inverse L2 penalty; left = strong regularisation)"); ax.set_ylabel("PR-AUC")
ax.set_title("B2 logistic regression: validation curve over C"); ax.legend(loc="upper left")
P.save(fig, "s3_fig01_validation_curve.png")

b2 = lr_pipe(C_chosen).fit(Xtr, ytr)

# %% probability calibration (validation period) ---------------------------------------------
variants = {
    "balanced weights (B2)": b2,
    "no class weights": lr_pipe(C_chosen, cw=None).fit(Xtr, ytr),
    "balanced + sigmoid recalibration": CalibratedClassifierCV(lr_pipe(C_chosen), method="sigmoid", cv=cv).fit(Xtr, ytr),
}
cal_rows = []
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for (name, m), col in zip(variants.items(), [P.BLUE, P.ORANGE, P.AQUA]):
    p = m.predict_proba(Xva)[:, 1]
    cal_rows.append({"variant": name, "PR-AUC": average_precision_score(yva, p), "Brier": brier_score_loss(yva, p),
                     "log-loss": log_loss(yva, p), "mean predicted %": 100 * p.mean(), "observed rate %": 100 * yva.mean()})
    frac, mean_p = calibration_curve(yva, p, n_bins=8, strategy="quantile")
    ax[0].plot(mean_p, frac, marker="o", ms=5, color=col, label=name)
    ax[1].hist(p, bins=np.linspace(0, 1, 41), histtype="step", lw=1.6, color=col, label=name)
ax[0].plot([0, 1], [0, 1], color=P.GRAY, ls=":", lw=1)
ax[0].set_xlim(0, 1); ax[0].set_ylim(0, .3)
ax[0].set_xlabel("mean predicted probability (quantile bins)"); ax[0].set_ylabel("observed frequency")
ax[0].set_title("Reliability diagram (validation)"); ax[0].legend(loc="upper left", fontsize=8)
ax[1].set_yscale("log"); ax[1].set_xlabel("predicted probability"); ax[1].set_ylabel("rows (log)")
ax[1].set_title("Distribution of predicted probabilities (validation)")
P.save(fig, "s3_fig02_calibration.png")
cal = pd.DataFrame(cal_rows).round(4)
cal.to_csv(C.TAB / "s3_calibration.csv", index=False)
print(cal.to_string())

# %% decision thresholds (validation) and final evaluation (test, used once) ------------------
models = {"B0 prior": b0, "B1 cycle position": b1, "B2 logistic regression": b2}
res, scores_out = [], {}
for name, m in models.items():
    s_va = m.predict_proba(Xva)[:, 1]
    s_te = m.predict_proba(Xte)[:, 1]
    thr_f1 = best_f1_threshold(yva, s_va) if name != "B0 prior" else 0.5
    thr_fa = fa_constrained_threshold(dva, s_va, C.MAX_FA_PER_HOUR) if name != "B0 prior" else 0.5
    scores_out[name] = (s_va, s_te, thr_f1, thr_fa)
    for op, thr in [("F1-optimal", thr_f1), (f"<= {C.MAX_FA_PER_HOUR:g} FA/h", thr_fa)]:
        if name == "B0 prior" and op != "F1-optimal":
            continue
        for set_name, y_, d_, s_ in [("val", yva, dva, s_va), ("test", yte, dte, s_te)]:
            r = {"model": name, "operating point": op, "set": set_name, "threshold": thr}
            r.update(row_metrics(y_, s_, thr))
            r.update(event_metrics(d_, s_, thr))
            if set_name == "test" and name != "B0 prior":
                _, lo, hi = bootstrap_ci(d_, s_, average_precision_score, n_boot=C.N_BOOT)
                r["PR-AUC 95% CI"] = f"[{lo:.3f}, {hi:.3f}]"
                _, lo, hi = bootstrap_event_recall(d_, s_, thr, n_boot=C.N_BOOT)
                r["event recall 95% CI"] = f"[{lo:.2f}, {hi:.2f}]"
            res.append(r)
res = pd.DataFrame(res)
res.round(4).to_csv(C.TAB / "s3_baseline_results.csv", index=False)
print(res[["model", "operating point", "set", "threshold", "PR-AUC", "ROC-AUC", "precision", "recall", "F1", "event recall",
           "mean warning (s)", "false alarms / hour"]].round(3).to_string())

# coefficients of B2 (standardised features)
names = b2.named_steps["features"].get_feature_names_out()
coef = pd.Series(b2.named_steps["model"].coef_[0], index=names)
top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.barh(top.index, top.values, color=[P.RED if v > 0 else P.BLUE for v in top.values], height=.6)
ax.axvline(0, color=P.INK, lw=.8)
ax.set_xlabel("standardised coefficient (red = raises risk, blue = lowers it)")
ax.set_title(f"B2: 15 largest coefficients (C = {C_chosen:.3g})")
P.save(fig, "s3_fig03_coefficients.png")

# keep scores for the comparison in step 5
out = []
for name, (s_va, s_te, thr_f1, thr_fa) in scores_out.items():
    for set_name, idx, sc in [("val", Xva.index, s_va), ("test", Xte.index, s_te)]:
        out.append(pd.DataFrame({"row": idx, "set": set_name, "model": name, "score": sc,
                                 "thr_f1": thr_f1, "thr_fa": thr_fa}))
pd.concat(out).to_csv(C.MODELS / "baseline_scores.csv", index=False)
joblib.dump({"B1": b1, "B2": b2, "C": C_chosen}, C.MODELS / "baselines.joblib")
print("Step 3 done")
