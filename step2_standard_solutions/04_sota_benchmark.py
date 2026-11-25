# %% [markdown]
# # Step 4 – Benchmark models from the state of the art
# 1. Literature table (published results on UR3 CobotOps, with their protocol).
# 2. Re-run of the *published protocols* on our data with the same model families, to measure how
#    much of the published performance comes from the protocol (concurrent label, random split,
#    SMOTE before the split) rather than from the model. This makes the published numbers
#    comparable to ours.

# %%
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C
from src import plotting as P
from src.prepare import prepare

P.setup()
plt = P.plt

# %% 1. literature -----------------------------------------------------------------------------
lit = pd.DataFrame([
    ("Tyrovolas et al. (2024) – dataset authors", "fault diagnosis (both labels)", "Information-Flow Fuzzy Cognitive Maps (interpretable)",
     "not compared (interpretability study)", "-", "-"),
    ("Liu (2025), Engineering Frontiers", "combined fault (protective stop OR grip loss), current second",
     "LR, SVM, RF, k-NN, decision tree", "random 70/30, fixed seed", "RF: ROC-AUC 0.911; k-NN: best F1 0.446; LR detects no fault",
     "figures from our Step-1 review, paper page not reachable for re-check"),
    ("Yaşar Çıklaçandır et al. (2025), DEU FMD", "grip_lost (and protective stop) separately, current second",
     "LR, DT, RF, SVM, k-NN; RFE / chi-square selection; StandardScaler + SMOTE", "70/30 (random vs chronological not stated)",
     "grip_lost, RF: accuracy 0.990, precision 0.986, recall 0.994, F1 0.990, ROC-AUC 0.990 (0.999 after chi-square)",
     "precision ≈ recall ≈ accuracy suggests a balanced (oversampled) test set"),
    ("Taş & Bal (2025), Sakarya Univ. J. Sci.", "protective stop, current second", "ESOA-tuned BiRNN vs CNN, LSTM, BiLSTM, GRU, TCN, CNN-RNN",
     "random 80/20", "ESOA-BiRNN: ROC-AUC 0.946, accuracy 0.892, precision 0.280, recall 0.853, F1 0.422", "other target (protective stop)"),
    ("Shen et al. (2026), PHM Asia-Pacific", "fault classification + RUL", "physics-informed Transformer (PhysODE-Joint)",
     "not stated on the abstract page", "no numbers on the abstract page", "cites UR3 CobotOps; full paper to be read"),
], columns=["study", "target", "models", "split", "reported results", "remark"])
lit.to_csv(C.TAB / "s4_literature.csv", index=False)

# %% 2. protocol re-run --------------------------------------------------------------------------
D = prepare()
df, X = D.df, D.X
feats = C.SENSORS                              # raw sensors, as in the published studies
rows_ok = ~df["sensor_missing"]
Xc, yc = df.loc[rows_ok, feats], df.loc[rows_ok, "GL_raw"]       # concurrent label, all rows


def models():
    return {
        "Random forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=C.SEED),
        "k-NN (k = 5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    }


def evaluate(protocol, Xa, ya, Xb, yb):
    out = []
    for name, m in models().items():
        m.fit(Xa, ya)
        s = m.predict_proba(Xb)[:, 1]
        p = (s >= .5).astype(int)
        out.append({"protocol": protocol, "model": name, "test prevalence %": 100 * np.mean(yb),
                    "accuracy": accuracy_score(yb, p), "precision": precision_score(yb, p, zero_division=0),
                    "recall": recall_score(yb, p), "F1": f1_score(yb, p),
                    "ROC-AUC": roc_auc_score(yb, s), "PR-AUC": average_precision_score(yb, s)})
    return out


res = []
# P1 – concurrent label, random stratified 70/30 (Liu 2025 style)
a, b, ya, yb = train_test_split(Xc, yc, test_size=.3, stratify=yc, random_state=C.SEED)
res += evaluate("P1 concurrent label, random 70/30", a, ya, b, yb)
# P2 – SMOTE applied to the whole data set BEFORE the split (suspected in Yaşar Çıklaçandır 2025)
Xs, ys = SMOTE(random_state=C.SEED).fit_resample(StandardScaler().fit_transform(Xc), yc)
a2, b2, ya2, yb2 = train_test_split(Xs, ys, test_size=.3, stratify=ys, random_state=C.SEED)
res += evaluate("P2 = P1 + SMOTE before the split", a2, ya2, b2, yb2)
# P3 – SMOTE on the training part only (correct use)
a3, ya3 = SMOTE(random_state=C.SEED).fit_resample(a, ya)
res += evaluate("P3 = P1 + SMOTE on train only", a3, ya3, b, yb)
# P4 – concurrent label, chronological 70/30
cut = int(.7 * len(Xc))
res += evaluate("P4 concurrent label, chronological 70/30", Xc.iloc[:cut], yc.iloc[:cut], Xc.iloc[cut:], yc.iloc[cut:])
# P5 – our task: early warning N = 3 s, ongoing losses removed, chronological (train -> validation)
Xtr, ytr, _ = D.get("train")
Xva, yva, _ = D.get("val")
res += evaluate("P5 early warning N = 3 s, chronological (ours)", Xtr[feats].fillna(Xtr[feats].median()), ytr,
                Xva[feats].fillna(Xtr[feats].median()), yva)
res = pd.DataFrame(res)
res.round(3).to_csv(C.TAB / "s4_protocol_rerun.csv", index=False)
print(res.round(3).to_string())

# %% figure: same random forest, five protocols -----------------------------------------------
rf = res[res["model"] == "Random forest"].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, 3.8))
yy = np.arange(len(rf))[::-1]
for metric, col, mk in [("ROC-AUC", P.BLUE, "o"), ("PR-AUC", P.ORANGE, "D"), ("F1", P.AQUA, "s")]:
    ax.scatter(rf[metric], yy, color=col, marker=mk, s=55, label=metric, zorder=3)
for k, y_ in enumerate(yy):
    ax.plot([rf.loc[k, ["ROC-AUC", "PR-AUC", "F1"]].min(), rf.loc[k, ["ROC-AUC", "PR-AUC", "F1"]].max()], [y_, y_],
            color="#d9d9d9", lw=1.2, zorder=2)
ax.set_yticks(yy, rf["protocol"]); ax.set_xlim(0, 1.02)
ax.set_xlabel("score of the same random forest (200 trees, raw sensors)")
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=3)
ax.set_title("The protocol, not the model, explains the published scores")
P.save(fig, "s4_fig01_protocol_rerun.png")
print("Step 4 done")
