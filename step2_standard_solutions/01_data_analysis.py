# %% [markdown]
# # Step 1 – Raw data analysis and pre-processing proposal
# Quality checks, descriptive statistics, variable dictionary, class imbalance,
# correlation / redundancy, dimensionality reduction and target association.
#
# Rule used everywhere below: **any statistic that drives a modelling decision is
# computed on the training period only** (first 70 % of the time line), so that the
# exploratory analysis itself does not leak information from the validation / test periods.

# %%
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src import config as C, data, features, splits
from src import plotting as P
from src.metrics import bootstrap_ci

P.setup()
plt = P.plt
raw = data.load_raw()
df = data.make_target(data.clean(raw))
X = features.build_base_features(df)
part = splits.chronological_split(df)
train_period = df.index < df.index[part == "val"].min()   # whole timeline before validation

# %% 1. Data quality report ------------------------------------------------------------
empty_cols = [c for c in raw.columns if c.startswith("Unnamed")]
space_cols = [c for c in raw.columns if c != c.strip()]
dt = df["dt"]
cyc = df["cycle"].to_numpy()
decreases = np.where(np.diff(cyc) < 0)[0]
dup_all = raw.drop(columns=["Num", "Timestamp"] + empty_cols).duplicated(keep=False)
placeholder = raw["Current_J0"].eq(1.0) & raw["Current_J1"].isna()
breaks = dt[dt > C.SEGMENT_BREAK_S]

quality = pd.DataFrame([
    ("Rows / columns in the file", f"{len(raw)} rows, {raw.shape[1]} columns ({len(empty_cols)} completely empty)", "Drop the empty columns"),
    ("Column names", f"{len(space_cols)} name(s) with a trailing space: {space_cols}; 'Temperature_T0' is joint J0", "Strip names, rename T0 -> J0"),
    ("Labels stored as text", "grip_lost / Robot_ProtectiveStop are 'VRAI'/'FAUX' strings", "Map to 0/1"),
    ("Timestamps", f"{raw['Timestamp'].str.startswith(chr(34)).sum()} timestamps wrapped in quotes", "Strip quotes, parse as UTC"),
    ("Empty sensor rows", f"{df['sensor_missing'].sum()} rows without sensor values ({placeholder.sum()} carry a placeholder Current_J0 = 1.0); "
                         f"PS label missing on the same rows; grip_lost = 0 on all of them", "Keep in the time line, never predict on them, causal ffill (<= 2 s) for windows"),
    ("Duplicated rows", f"{dup_all.sum()} rows involved, all of them empty rows sharing a cycle number; 0 duplicated measurements", "Nothing to remove"),
    ("Sampling", f"median interval {dt.median():.3f} s; {int(((dt > 1.5) & (dt <= 2.5)).sum())} intervals of ~2 s (one record skipped)", "Lags are counted in rows (~ seconds)"),
    ("Recording breaks", ", ".join(f"{b:.0f} s" for b in breaks) + f" -> {df['segment'].nunique()} continuous segments", "No lag / window / label across a break"),
    ("Cycle counter", f"decreases {len(decreases)} time(s): {cyc[decreases[0]]} -> {cyc[decreases[0] + 1]} between Num {raw['Num'].iat[decreases[0]]} and Num {raw['Num'].iat[decreases[0] + 1]}" if len(decreases) else "monotonic",
     "Cycles identified by changes of the counter"),
    ("Grip-loss label", f"{int(df['GL_raw'].sum())} rows, {int(((df['GL_raw'] == 1) & (df['GL_raw'].shift(fill_value=0) == 0)).sum())} raw events; "
                        f"5 pairs separated by only 1-2 normal rows", f"Debounce: merge gaps <= {C.MERGE_GAP} rows -> {int(df['onset'].sum())} events"),
], columns=["check", "finding", "action"])
quality.to_csv(C.TAB / "s1_quality_report.csv", index=False)
print(quality.to_string())

# %% 2. Variable dictionary ------------------------------------------------------------
var_dict = pd.DataFrame(
    [("Num", "integer", "-", "identifier", "row number, dropped")]
    + [("Timestamp", "datetime (UTC)", "-", "index", "used for ordering, breaks and splits only")]
    + [(c, "continuous", "A", "feature", "joint motor current") for c in C.CUR]
    + [(c, "continuous", "°C", "feature (reduced)", "joint temperature, the 6 are replaced by their mean") for c in C.TEM]
    + [(c, "continuous", "rad/s", "feature", "joint speed") for c in C.SPD]
    + [("Tool_current", "continuous", "A", "feature", "gripper current")]
    + [("cycle", "integer", "-", "structure", "cycle counter -> cycle_id and position in cycle; not a feature (encodes time)")]
    + [("Robot_ProtectiveStop", "binary", "-", "excluded", "other fault, unrelated to grip loss (Fisher p = 0.25)")]
    + [("grip_lost", "binary", "-", "target source", f"y_t = 1 if a grip loss starts within the next N = {C.HORIZON} s")],
    columns=["variable", "type", "unit", "role", "comment"])
var_dict.to_csv(C.TAB / "s1_variable_dictionary.csv", index=False)

# %% 3. Descriptive statistics, normality and autocorrelation ---------------------------
d = df.loc[~df["sensor_missing"]]
desc = d[C.SENSORS].describe().T
desc["skew"] = d[C.SENSORS].skew()
desc["kurtosis"] = d[C.SENSORS].kurt()
desc["normaltest p"] = [stats.normaltest(d[c]).pvalue for c in C.SENSORS]
desc["lag-1 autocorr"] = [d[c].autocorr(1) for c in C.SENSORS]
rho = desc["lag-1 autocorr"].clip(-0.99, 0.99)
desc["effective n"] = (len(d) * (1 - rho) / (1 + rho)).round(0)
# outliers: the IQR rule is meaningless for the speeds (the robot is stopped ~50 % of the
# time, so IQR ~ 0); we report |x - mean| > 5 sd and the share of ~zero values instead
desc["beyond 5 sd %"] = (((d[C.SENSORS] - d[C.SENSORS].mean()).abs() > 5 * d[C.SENSORS].std()).mean() * 100).round(2)
desc["~zero values %"] = ((d[C.SENSORS].abs() < 1e-3).mean() * 100).round(1)
desc = desc.round(4)
desc.to_csv(C.TAB / "s1_descriptive_stats.csv")
print(desc[["mean", "std", "min", "max", "skew", "kurtosis", "lag-1 autocorr", "beyond 5 sd %", "~zero values %"]].to_string())

# %% 4. Class imbalance (rows, events, horizon) ----------------------------------------
ev = df[df["GL"] == 1].groupby("event_id").size()
imb_rows = []
for N in [1, 2, 3, 4, 5]:
    t = data.make_target(df, horizon=N)
    pr = t[t["predictable"]]
    imb_rows.append({"horizon N (s)": N, "predictable rows": len(pr), "positives": int(pr["y"].sum()),
                     "prevalence %": round(100 * pr["y"].mean(), 2),
                     "imbalance ratio": f"1:{(1 - pr['y'].mean()) / pr['y'].mean():.0f}",
                     "events announced": pr["target_event"].nunique()})
imbalance = pd.DataFrame(imb_rows)
imbalance.to_csv(C.TAB / "s1_imbalance_by_horizon.csv", index=False)
print(imbalance.to_string())

fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
counts = [int((df["GL_raw"] == 0).sum()), int(df["GL_raw"].sum())]
ax[0].bar(["normal", "grip loss"], counts, color=[P.GRAY, P.RED], width=.6)
for i, v in enumerate(counts):
    ax[0].text(i, v, f"{v}\n({100 * v / sum(counts):.1f} %)", ha="center", va="bottom", fontsize=8.5)
ax[0].set_title("Rows per class (raw label)"); ax[0].set_ylim(0, counts[0] * 1.2)
ax[1].hist(ev.values, bins=range(1, ev.max() + 2), color=P.RED, rwidth=.85)
ax[1].set_xlabel("event duration (s)"); ax[1].set_ylabel("events")
ax[1].set_title(f"{len(ev)} grip-loss events, median {ev.median():.0f} s")
ax[2].bar(imbalance["horizon N (s)"], imbalance["prevalence %"], color=P.BLUE, width=.6)
for x_, v in zip(imbalance["horizon N (s)"], imbalance["prevalence %"]):
    ax[2].text(x_, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
ax[2].set_xlabel("horizon N (s)"); ax[2].set_ylabel("positive rows (%)")
ax[2].set_title("Early-warning target: prevalence vs N")
P.save(fig, "s1_fig02_imbalance.png")

# %% 5. Missing rows: timeline and randomness check --------------------------------------
fig, ax = plt.subplots(figsize=(13, 2.6))
idx = np.arange(len(df))
ax.eventplot([idx[df["sensor_missing"]]], lineoffsets=2, colors=P.INK, linelengths=.7)
ax.eventplot([idx[(dt > 1.5) & (dt <= 2.5)]], lineoffsets=1, colors=P.VIOLET, linelengths=.7)
ax.eventplot([idx[df["onset"]]], lineoffsets=0, colors=P.RED, linelengths=.7)
for b in idx[df["segment"].diff() > 0]:
    ax.axvline(b, color=P.GRAY, ls=":", lw=1)
ax.set_yticks([0, 1, 2], ["grip-loss onsets", "skipped second (~2 s step)", "empty sensor row"])
ax.set_xlabel("row (≈ seconds of operation; dotted = recording break)")
ax.set_title("Where the data-quality issues are")
P.save(fig, "s1_fig01_missing_timeline.png")

near = np.zeros(len(df), bool)                                  # within 10 s before an onset
for o in idx[df["onset"]]:
    near[max(0, o - 10):o] = True
tab = pd.crosstab(df["sensor_missing"], near)
miss_tests = pd.DataFrame([
    {"test": "empty rows vs session (Fisher)", "table": pd.crosstab(df["sensor_missing"], df["session"]).values.tolist(),
     "p-value": stats.fisher_exact(pd.crosstab(df["sensor_missing"], df["session"]).values)[1]},
    {"test": "empty rows vs <= 10 s before an onset (Fisher)", "table": tab.values.tolist(),
     "p-value": stats.fisher_exact(tab.values)[1]},
])
miss_tests.to_csv(C.TAB / "s1_missingness_tests.csv", index=False)
print(miss_tests.to_string())

# %% 6. Correlation and redundancy (training period) ------------------------------------
dtr = df.loc[train_period & ~df["sensor_missing"]]
spear = dtr[C.SENSORS].corr("spearman")
pear = dtr[C.SENSORS].corr("pearson")
fig, ax = plt.subplots(figsize=(8.5, 7.2))
im = ax.imshow(spear, cmap=P.DIVERGING, vmin=-1, vmax=1)
ax.set_xticks(range(len(C.SENSORS)), [s.replace("Temperature", "Temp").replace("Current", "Cur") for s in C.SENSORS], rotation=75)
ax.set_yticks(range(len(C.SENSORS)), [s.replace("Temperature", "Temp").replace("Current", "Cur") for s in C.SENSORS])
ax.grid(False)
for i in range(len(C.SENSORS)):
    for j in range(len(C.SENSORS)):
        if abs(spear.iat[i, j]) >= .6 and i != j:
            ax.text(j, i, f"{spear.iat[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(spear.iat[i, j]) > .8 else P.INK)
fig.colorbar(im, ax=ax, shrink=.75, label="Spearman ρ")
ax.set_title("Spearman correlation between sensors (training period, |ρ| ≥ 0.6 annotated)")
P.save(fig, "s1_fig03_correlation.png")

pairs = []
for i, a in enumerate(C.SENSORS):
    for b in C.SENSORS[i + 1:]:
        pairs.append((a, b, pear.loc[a, b], spear.loc[a, b]))
pairs = pd.DataFrame(pairs, columns=["var 1", "var 2", "pearson", "spearman"])
pairs = pairs.reindex(pairs["spearman"].abs().sort_values(ascending=False).index)
pairs.head(25).round(3).to_csv(C.TAB / "s1_top_correlated_pairs.csv", index=False)


def vif(frame: pd.DataFrame) -> pd.Series:
    Z = StandardScaler().fit_transform(frame)
    out = {}
    for k, col in enumerate(frame.columns):
        others = np.delete(Z, k, axis=1)
        beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(Z)), others], Z[:, k], rcond=None)
        resid = Z[:, k] - np.c_[np.ones(len(Z)), others] @ beta
        r2 = 1 - resid.var() / Z[:, k].var()
        out[col] = 1 / max(1 - r2, 1e-9)
    return pd.Series(out)


vif_before = vif(dtr[C.SENSORS])
reduced = dtr[C.CUR + C.SPD + C.TOOL].assign(temp_mean=dtr[C.TEM].mean(axis=1))
vif_after = vif(reduced)
vif_tab = pd.DataFrame({"VIF (19 raw sensors)": vif_before, "VIF (temperatures -> mean)": vif_after}).round(1)
vif_tab.to_csv(C.TAB / "s1_vif.csv")

pca_t = PCA().fit(StandardScaler().fit_transform(dtr[C.TEM]))
pca_all = PCA().fit(StandardScaler().fit_transform(dtr[C.SENSORS]))
cum = np.cumsum(pca_all.explained_variance_ratio_)
reduction = pd.DataFrame([
    {"analysis": "PCA on the 6 temperatures", "result": f"PC1 explains {100 * pca_t.explained_variance_ratio_[0]:.2f} % of the variance"},
    {"analysis": "PCA on the 19 standardised sensors", "result": f"{np.searchsorted(cum, .90) + 1} components for 90 %, {np.searchsorted(cum, .95) + 1} for 95 %"},
    {"analysis": "Max VIF before reduction", "result": f"{vif_before.max():.0f} ({vif_before.idxmax()})"},
    {"analysis": "Max VIF after temperatures -> mean", "result": f"{vif_after.max():.1f} ({vif_after.idxmax()})"},
])
reduction.to_csv(C.TAB / "s1_reduction.csv", index=False)
print(reduction.to_string())

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
vv = vif_tab.sort_values("VIF (19 raw sensors)")
ax[0].barh(vv.index, vv["VIF (19 raw sensors)"], color=P.GRAY, label="19 raw sensors")
ax[0].barh(vv.index, vv["VIF (temperatures -> mean)"].fillna(0), color=P.BLUE, height=.45, label="temperatures -> mean")
ax[0].set_xscale("log"); ax[0].axvline(10, color=P.RED, ls="--", lw=1)
ax[0].text(10.5, 0, "VIF = 10", color=P.RED, fontsize=8)
ax[0].set_xlabel("variance inflation factor (log)"); ax[0].legend(loc="lower right")
ax[0].set_title("Multicollinearity before / after reduction")
ax[1].bar(range(1, 20), 100 * pca_all.explained_variance_ratio_, color=P.BLUE, width=.7, label="per component")
ax[1].plot(range(1, 20), 100 * cum, color=P.ORANGE, marker="o", ms=4, label="cumulative")
ax[1].axhline(90, color=P.GRAY, ls=":", lw=1)
ax[1].set_xlabel("principal component"); ax[1].set_ylabel("explained variance (%)")
ax[1].set_xticks(range(1, 20)); ax[1].legend(loc="center right")
ax[1].set_title("PCA on the 19 standardised sensors (training period)")
P.save(fig, "s1_fig04_vif_pca.png")

# %% 7. Drift / confounding: grip-loss rate and temperature over time --------------------
blk = d.groupby(np.arange(len(d)) // 500).agg(rate=("GL", "mean"), temp=("Temperature_J0", "mean"))
fig, ax = plt.subplots(2, 1, figsize=(12, 4.6), sharex=True)
ax[0].bar(blk.index, 100 * blk["rate"], color=P.RED, width=.7)
ax[0].set_ylabel("grip-loss rows (%)"); ax[0].set_title("Non-stationarity: grip-loss rate and temperature per block of 500 rows")
ax[1].plot(blk.index, blk["temp"], color=P.ORANGE, marker="o", ms=4)
ax[1].set_ylabel("mean temp J0 (°C)"); ax[1].set_xlabel("block of 500 rows (chronological)")
cut_blk = (df.index[part == "val"].min()) / 500
for a in ax:
    a.axvline(cut_blk - .5, color=P.INK2, ls="--", lw=1)
ax[0].text(cut_blk - .4, ax[0].get_ylim()[1] * .85, "validation / test →", fontsize=8, color=P.INK2)
P.save(fig, "s1_fig06_drift.png")

# %% 8. Target association on the training period ---------------------------------------
cand = [c for c in X.columns if c not in ("pos",)] + ["pos"]
conc_mask = train_period & ~df["sensor_missing"]
pred_mask = (part == "train")
rows = []
for c in cand:
    x_c = X.loc[conc_mask, c]
    ok = x_c.notna()
    auc_conc = roc_auc_score(df.loc[conc_mask, "GL"][ok], x_c[ok])
    x_p = X.loc[pred_mask, c]
    okp = x_p.notna()
    auc_pred = roc_auc_score(df.loc[pred_mask, "y"][okp], x_p[okp])
    rows.append({"feature": c, "AUC concurrent (during loss)": auc_conc, "AUC early warning (N=3)": auc_pred})
assoc = pd.DataFrame(rows)
assoc["|AUC-0.5| early"] = (assoc["AUC early warning (N=3)"] - .5).abs()
Xp = X.loc[pred_mask, cand].fillna(X.loc[pred_mask, cand].median())
assoc["mutual info (early)"] = mutual_info_classif(Xp, df.loc[pred_mask, "y"], random_state=C.SEED)
assoc = assoc.sort_values("|AUC-0.5| early", ascending=False).round(3)
# 95 % CI of the AUC by bootstrap over cycles for the 10 strongest features
ev_train = df.loc[pred_mask]
ci = []
for c in assoc["feature"].head(10):
    s = X.loc[pred_mask, c].fillna(X.loc[pred_mask, c].median()).to_numpy()
    p_, lo, hi = bootstrap_ci(ev_train, s, roc_auc_score, n_boot=500)
    ci.append((c, f"[{lo:.2f}, {hi:.2f}]"))
assoc = assoc.merge(pd.DataFrame(ci, columns=["feature", "95% CI early AUC (cycle bootstrap)"]), how="left")
assoc.to_csv(C.TAB / "s1_target_association.csv", index=False)
print(assoc.head(15).to_string())

top = assoc.head(18).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 6))
yy = np.arange(len(top))
ax.scatter(top["AUC concurrent (during loss)"], yy, color=P.GRAY, s=36, label="during the loss (concurrent label)", zorder=3)
ax.scatter(top["AUC early warning (N=3)"], yy, color=P.BLUE, s=36, label="N = 3 s before the onset (our target)", zorder=3)
for k, (a, b) in enumerate(zip(top["AUC concurrent (during loss)"], top["AUC early warning (N=3)"])):
    ax.plot([a, b], [k, k], color="#d0d0d0", lw=1.2, zorder=2)
ax.axvline(.5, color=P.INK, lw=.8)
ax.set_yticks(yy, top["feature"]); ax.set_xlabel("univariate ROC-AUC (0.5 = no information)")
ax.set_title("Which signals carry information before the loss? (training period)")
ax.legend(loc="upper center", bbox_to_anchor=(.45, -.09), ncol=2, fontsize=8.5)
P.save(fig, "s1_fig05_univariate_auc.png")

# %% 9. Events: gaps between events (justifies the debouncing) ----------------------------
gl = df["GL_raw"].to_numpy()
on = np.where((gl == 1) & (np.r_[0, gl[:-1]] == 0))[0]
off = np.where((gl == 1) & (np.r_[gl[1:], 0] == 0))[0]
gaps = on[1:] - off[:-1] - 1
fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(np.clip(gaps, 0, 60), bins=range(0, 62, 1), color=P.BLUE, rwidth=.85)
ax.axvspan(0.5, C.MERGE_GAP + .5, color=P.RED, alpha=.15)
ax.text(C.MERGE_GAP + 1, ax.get_ylim()[1] * .8, f"{int((gaps <= C.MERGE_GAP).sum())} gaps ≤ {C.MERGE_GAP} rows → merged", color=P.RED, fontsize=8.5)
ax.set_xlabel("normal rows between two raw grip-loss events (last bar = 60 or more)"); ax.set_ylabel("count")
ax.set_title("Label flicker: some events are split by a 1–2 s hole")
P.save(fig, "s1_fig07_event_gaps.png")

# %% 10. Pre-processing proposal ---------------------------------------------------------
prep = pd.DataFrame([
    ("Clean names, types, quoted timestamps", "Format problems only", "-", "all rows"),
    ("Drop Unnamed columns, Num", "Empty or identifier", "-", "all rows"),
    ("Keep empty rows in the time line; set placeholder Current_J0 = 1.0 to NaN", "Removing them glues events together and shifts lags", "-", "all rows"),
    (f"Causal forward fill of sensors (<= {C.FFILL_LIMIT} s, within segment)", "What a real-time system can do; no future value used", "-", "all rows"),
    ("Segments at breaks > 5 s; no window or label across them", "Avoid mixing two sessions in a window", "-", "all rows"),
    (f"Debounce label: merge events separated by <= {C.MERGE_GAP} rows", "1-2 s holes are label flicker, not a new loss; prevents trivially easy positives", "-", "label"),
    (f"Early-warning target y = onset in (t, t+{C.HORIZON}]", "Predicting the current second detects consequences (leakage)", "-", "label"),
    ("Remove rows where a loss is ongoing", "The alert must come before the loss", "-", "label"),
    ("6 temperatures -> mean + 60 s warm-up rate", f"Pearson r ≥ 0.99, VIF up to {vif_before.max():.0f}; PC1 = {100 * pca_t.explained_variance_ratio_[0]:.1f} %", "-", "features"),
    ("Exclude Robot_ProtectiveStop, cycle counter, timestamp", "Other label / encodes time", "-", "features"),
    ("Lags, differences, rolling mean/std (past 5 s)", "Precursors are short (1-6 s)", "-", "features"),
    ("Residual to the normal cycle profile at the same position", "Removes the cycle shape; signal = deviation", "training fold", "features (in pipeline)"),
    ("Position in cycle (raw for trees, spline for linear models)", "Strong non-monotonic risk profile", "training fold (spline)", "features (in pipeline)"),
    ("Median imputation of remaining NaN", "Robust to heavy tails", "training fold", "pipeline"),
    ("StandardScaler or RobustScaler (chosen by AutoML)", "Heavy-tailed currents (high kurtosis)", "training fold", "pipeline"),
    ("Class weights instead of SMOTE", "SMOTE interpolates between non-independent seconds", "training fold", "pipeline"),
    ("No outlier removal", "All values physically plausible (|speed| <= 2.73 rad/s < UR3 limit 3.14 rad/s, temps 28-45 °C); "
                           "<= 1.5 % of values beyond 5 sd and they are start/stop dynamics, i.e. signal", "-", "-"),
], columns=["step", "justification", "fitted on", "applies to"])
prep.to_csv(C.TAB / "s1_preprocessing_proposal.csv", index=False)
print("Step 1 done ->", C.TAB)
