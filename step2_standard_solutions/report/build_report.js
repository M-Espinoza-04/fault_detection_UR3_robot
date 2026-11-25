// Builds the explanatory Word report from the figures and the CSV tables (via report_data.json).
// Usage: python report/collect_tables.py && node report/build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, Footer, PageNumber, LevelFormat, PageBreak,
} = require("docx");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "report_data.json"), "utf8"));
const FIG = path.join(__dirname, "..", "outputs", "figures");
const CONTENT_W = 9638; // A4 with 2 cm margins, in DXA
const FONT = "Calibri";
const BLUE = "1F4E79";

// ------------------------------------------------------------------ helpers
function runs(text, base = {}) {
  // **bold**, *italic*, `code`
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas", size: (base.size || 22) - 2, ...base, }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text, opts.run || {}), spacing: { after: 120, line: 276 }, alignment: opts.align || AlignmentType.JUSTIFIED, ...(opts.para || {}) });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)], pageBreakBefore: true });
const H1nb = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const B = (text, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, children: runs(text), spacing: { after: 60, line: 264 } });
const N = (text, instance = 1) => new Paragraph({ numbering: { reference: "numbers", level: 0, instance }, children: runs(text), spacing: { after: 60, line: 264 } });

let figNo = 0, tabNo = 0;
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function F(name, caption, widthPx = 620) {
  const file = path.join(FIG, name);
  const { w, h } = pngSize(file);
  const width = Math.min(widthPx, 640);
  figNo += 1;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width, height: Math.round(h * width / w) },
        altText: { title: caption, description: caption, name: name } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: runs(`Figure ${figNo} – ${caption}`, { size: 18, italics: true, color: "555555" }) }),
  ];
}
function fmt(v, d = 3) {
  if (v === null || v === undefined) return "–";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") {
    if (Number.isInteger(v)) return String(v);
    return v.toFixed(d);
  }
  return String(v);
}
function cell(text, w, { header = false, size = 17, shade = null, bold = false, align = AlignmentType.LEFT } = {}) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: header ? { fill: "DCE8F7", type: ShadingType.CLEAR, color: "auto" } : (shade ? { fill: shade, type: ShadingType.CLEAR, color: "auto" } : undefined),
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children: [new Paragraph({ alignment: align, children: runs(String(text), { size, bold: header || bold, font: FONT }) })],
  });
}
function T(columns, rows, fractions, caption, opts = {}) {
  const size = opts.size || 17;
  const widths = fractions.map((f) => Math.floor(f * CONTENT_W));
  widths[widths.length - 1] += CONTENT_W - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
  tabNo += 1;
  const hdr = new TableRow({ tableHeader: true, children: columns.map((c, i) => cell(c, widths[i], { header: true, size })) });
  const body = rows.map((r, k) => new TableRow({ cantSplit: true, children: r.map((v, i) => cell(typeof v === "string" ? v : fmt(v, opts.digits ?? 3), widths[i], {
    size, shade: opts.highlight && opts.highlight(r, k) ? "FFF4D6" : null, align: (typeof v === "number") ? AlignmentType.RIGHT : AlignmentType.LEFT })) }));
  return [
    new Paragraph({ keepNext: true, spacing: { before: 160, after: 60 }, children: runs(`Table ${tabNo} – ${caption}`, { size: 18, italics: true, color: "555555" }) }),
    new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths,
      borders: { top: border, bottom: border, left: border, right: border, insideHorizontal: border, insideVertical: border },
      rows: [hdr, ...body] }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}
function tab(name) { return D[name]; }
function pick(name, cols) {
  const t = tab(name);
  const idx = cols.map((c) => t.columns.indexOf(c));
  idx.forEach((i, k) => { if (i < 0) throw new Error(`column ${cols[k]} missing in ${name}`); });
  return t.rows.map((r) => idx.map((i) => r[i]));
}
function val(name, keyCol, key, col) {
  const t = tab(name);
  const r = t.rows.find((row) => row[t.columns.indexOf(keyCol)] === key);
  return r ? r[t.columns.indexOf(col)] : null;
}
function find(name, pred, col) {
  const t = tab(name);
  const obj = (r) => Object.fromEntries(t.columns.map((c, i) => [c, r[i]]));
  const r = t.rows.map(obj).find(pred);
  return r ? r[col] : null;
}
const pct = (x, d = 1) => (100 * x).toFixed(d) + " %";

// ------------------------------------------------------------------ numbers used in the text
const champ = tab("s5_family_champions");
const champObj = champ.rows.map((r) => Object.fromEntries(champ.columns.map((c, i) => [c, r[i]])));
const finalFam = champObj[0].family;
const testObj = tab("s5_test_results").rows.map((r) => Object.fromEntries(tab("s5_test_results").columns.map((c, i) => [c, r[i]])));
const tAuto = testObj.filter((r) => r.model.startsWith("AutoML"));
const tB2 = testObj.filter((r) => r.model.startsWith("B2"));
const autoFA = tAuto.find((r) => r["operating point"].startsWith("<="));
const autoF1 = tAuto.find((r) => r["operating point"] === "F1-optimal");
const b2FA = tB2.find((r) => r["operating point"].startsWith("<="));
const deltaObj = tab("s5_paired_differences").rows.map((r) => Object.fromEntries(tab("s5_paired_differences").columns.map((c, i) => [c, r[i]])));
const abl = Object.fromEntries(tab("s5_ablation").rows.map((r) => [r[0], { cv: r[1], sd: r[2], val: r[3] }]));
const hz = tab("s5_horizon_sensitivity").rows.map((r) => Object.fromEntries(tab("s5_horizon_sensitivity").columns.map((c, i) => [c, r[i]])));
const hzv = (n, b) => hz.find((r) => r["horizon N (s)"] === n && r["burst features"] === b);
const imp = tab("s5_permutation_importance").rows.slice(0, 6).map((r) => r[0]);
const evType = tab("s5_events_by_type").rows.map((r) => Object.fromEntries(tab("s5_events_by_type").columns.map((c, i) => [c, r[i]])));
const evt = (set, type) => evType.find((r) => r.set === set && r.type.startsWith(type));
const cfgFinal = champObj[0].config;
const b2C = 0.316;

// ------------------------------------------------------------------ document body
const body = [];

// Title page
body.push(
  new Paragraph({ spacing: { before: 2400, after: 200 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Grip-Loss Early Warning on a UR3 Cobot", bold: true, size: 48, color: BLUE, font: FONT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 }, children: [new TextRun({ text: "Step 2 – Implementation of standard solutions: data analysis, learning plan, baseline, state of the art and AutoML v0.1", size: 28, color: "404040", font: FONT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "Machine Learning project – ESILV IRO, 2026–2027", size: 24, font: FONT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "Quentin Courtois · Mauricio Espinoza · Ivan Devernois", size: 24, font: FONT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1200 }, children: [new TextRun({ text: "Code: folder step2_standard_solutions/ of the repository fault_detection_UR3_robot", size: 20, color: "666666", font: FONT })] }),
);

// Summary
body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Summary")], pageBreakBefore: true }));
body.push(P(`This document explains the code of the folder \`step2_standard_solutions/\`, written to give the project a **reproducible baseline that the next phases can improve**. It follows the five points of Step 2 of the project guidelines. The problem keeps the formulation of our Step-1 document: predict, every second, whether a **grip loss will start within the next N = 3 seconds**, from the joint and gripper signals of the UR3 cobot (UCI *UR3 CobotOps*, 7,409 rows at 1 Hz on a single day).`));
body.push(P("Main results:"));
body.push(B(`**Data.** 54 empty rows (8 with a placeholder value), 4 recording segments, no duplicated measurement, 6 temperatures that are the same variable (Pearson r ≥ 0.995, VIF up to 8,988), heavy-tailed currents and speeds that are zero half of the time. After merging 5 events split by a 1–2 s label "hole", the target has **72 grip-loss events**; the early-warning label has **2.95 % positive rows** (imbalance 1:33).`));
body.push(B(`**Learning plan.** Chronological 70/15/15 split cut at cycle starts with an 8-row embargo (50 / 12 / 9 events). Tuning uses an **event-balanced expanding-window cross-validation** on the training period: the plain \`TimeSeriesSplit\` produced a fold with a single event. Two leakage experiments show how much a careless design inflates PR-AUC (+22 % with shuffled CV, +32 % with centred windows).`));
body.push(B(`**Baseline.** Logistic regression on the current sensor values and a spline of the position in the cycle, C = ${b2C} chosen with a paired one-standard-error rule: test PR-AUC **${fmt(tB2[0]["PR-AUC"])}** (prevalence ${fmt(0.0249)}), 1 of 9 test events anticipated at ≤ 30 false alarms per hour.`));
body.push(B(`**State of the art.** Published scores on this dataset (ROC-AUC 0.91–0.999) are obtained on the *current* second with *random* splits; one study's balanced metrics suggest SMOTE before the split. Re-running these protocols on our data reproduces the published numbers (random forest: ROC-AUC 0.935, and 0.999 with SMOTE before the split) and shows that they collapse under a chronological evaluation.`));
body.push(B(`**AutoML v0.1.** Randomized search over 7 scikit-learn families with the feature blocks as hyper-parameters. Winner on validation: **${finalFam}** with past windows, cycle residuals and "burst" features. Test (used once): PR-AUC **${fmt(autoFA["PR-AUC"])}** ${autoFA["PR-AUC 95% CI"]}, **${autoFA["events detected"]} of ${autoFA.events} events** anticipated (mean warning ${fmt(autoFA["mean warning (s)"], 1)} s) with ${fmt(autoFA["false alarms / hour"], 1)} false alarms per hour; the gain over the baseline is significant (paired cycle bootstrap, Δ PR-AUC = ${fmt(deltaObj[0]["Δ PR-AUC (test)"])} ${deltaObj[0]["95% CI (paired cycle bootstrap)"]}).`));
body.push(P(`The confidence intervals are wide because the test period contains only 9 events: the numbers are a **reference point** for the next phases, not a performance claim. The prioritized plan in section 5.7 lists what to do next.`));

body.push(H2("How to run the code"));
body.push(P("From the repository root, with the packages of `step2_standard_solutions/requirements.txt` installed:"));
body.push(B("`cd step2_standard_solutions` then `python run_all.py` (about 25 min on 2 CPU cores; `python run_all.py --fast` reduces the AutoML budget to about 5 min)."));
body.push(B("Each step can also be run alone (`python 01_data_analysis.py` … `python 05_automl.py`); the files contain `# %%` cells and open as notebooks in VS Code."));
body.push(B("All figures go to `outputs/figures/`, all tables to `outputs/tables/` (CSV), the fitted pipelines to `outputs/models/`. Every number in this report comes from these files."));
body.push(...T(["File", "Role"], [
  ["src/config.py", "paths, column groups and constants (N = 3 s, window = 5 s, embargo = 8, split, seed)"],
  ["src/data.py", "loading, cleaning, segments, cycles, label debouncing, early-warning target and predictable rows"],
  ["src/features.py", "causal features (lags, differences, rolling stats, burst) and the in-pipeline FeatureMaker (cycle residuals, spline)"],
  ["src/splits.py", "chronological split with embargo, TimeSeriesSplit and event-balanced expanding-window CV"],
  ["src/metrics.py", "row-level and event-level metrics, thresholds, cycle bootstrap"],
  ["src/models.py, src/baselines.py", "pipeline factory (features → imputer → scaler → selector → model) and naive baselines"],
  ["01_data_analysis.py … 05_automl.py", "one script per point of Step 2"],
  ["ai-governance/", "draft entry for PROMPT_LOG.md and candidate items for AI_AUDIT.md"],
], [0.32, 0.68], "Structure of the folder step2_standard_solutions/"));

// ================================================================== STEP 1
body.push(H1("1. Raw data analysis and pre-processing"));
body.push(P("Principle used in this whole section: **every statistic that drives a modelling decision is computed on the training period only** (the first 70 % of the time line). Looking at the validation or test periods to choose features or transformations would already be a form of leakage."));
body.push(H2("1.1 Data quality"));
body.push(...T(["Check", "Finding", "Action"], pick("s1_quality_report", ["check", "finding", "action"]), [0.18, 0.5, 0.32], "Data-quality report (01_data_analysis.py, section 1)", { size: 16 }));
body.push(P("Two findings correct our Step-1 document: the \"2 duplicated rows\" are in fact **empty rows sharing a cycle number** (there is no duplicated measurement), and the cycle counter goes from **233 to 224** (not 232). We also found that 8 empty rows contain a **placeholder Current_J0 = 1.0**: it was counted as a real measurement in our exploratory notebook. These three points are listed as candidate AI-audit items (section 6)."));
body.push(...F("s1_fig01_missing_timeline.png", "Position of the empty rows, the skipped seconds (2-s steps) and the grip-loss onsets along the recording. Dotted lines are recording breaks.", 620));
body.push(P(`**Is the missingness random?** The empty rows are more frequent in the afternoon (44 of 4,540 rows vs 10 of 2,869; Fisher p = 0.002), so they are not missing completely at random with respect to time. They are **not** closer to grip losses than expected (7 of 54 within 10 s before an onset; Fisher p = 0.36) and no empty row is a grip-loss row. We therefore keep them in the time line as "no measurement" seconds: they are never predicted, and a causal forward fill of at most 2 s is used only so that windows of neighbouring rows stay defined. Deleting them would glue events together (77 → 74 events) and shift the lags.`));

body.push(H2("1.2 Variables"));
body.push(...T(["Variable", "Type", "Unit", "Role", "Comment"], [
  ["Num", "integer", "–", "identifier", "row number, dropped"],
  ["Timestamp", "datetime (UTC)", "–", "index", "ordering, recording breaks and split only"],
  ["Current_J0 … J5", "continuous", "A", "feature", "motor current of the 6 joints (base, shoulder, elbow, wrists 1–3)"],
  ["Temperature_J0 … J5", "continuous", "°C", "feature (reduced)", "joint temperatures; the 6 are replaced by their mean and a 60-s warm-up rate (T0 renamed J0)"],
  ["Speed_J0 … J5", "continuous", "rad/s", "feature", "joint speeds; also summed in absolute value (abs_speed)"],
  ["Tool_current", "continuous", "A", "feature", "gripper current"],
  ["cycle", "integer", "–", "structure", "cycle counter → cycle id and position in the cycle; not a feature (it encodes time)"],
  ["Robot_ProtectiveStop", "binary", "–", "excluded", "other fault, unrelated to grip losses (Fisher p = 0.25, Step 1)"],
  ["grip_lost", "binary", "–", "target source", "y_t = 1 if a grip loss starts within the next N = 3 s"],
], [0.2, 0.13, 0.07, 0.14, 0.46], "Variable dictionary", { size: 16 }));
body.push(P("**Target.** We keep the formulation of Step 1: y_t = 1 if a grip loss **starts** within (t, t+N], with N = 3 s, computed only inside a continuous segment. Rows where a grip loss is already ongoing are removed (the alert must come before the loss), as well as empty rows and rows whose past window or future horizon crosses a recording break. This leaves 7,082 predictable rows."));
body.push(P("**Label debouncing.** In 5 cases two grip-loss episodes are separated by only 1 or 2 normal rows (Figure 2). We treat these holes as label flicker and merge the episodes (77 → 72 events): otherwise the 1–2 rows of the hole become positives that are trivially predicted by \"a loss just happened\". The team must confirm this rule with the meaning of the `grip_lost` signal."));
body.push(...F("s1_fig07_event_gaps.png", "Number of normal rows between two consecutive raw grip-loss episodes. The 5 gaps of 1–2 rows (red band) are merged.", 470));

body.push(H2("1.3 Statistical description"));
body.push(...T(["Sensor", "mean", "sd", "min", "max", "skew", "kurtosis", "lag-1 autocorr.", "% beyond 5 sd", "% ≈ 0"],
  tab("s1_descriptive_stats").rows.filter((r, i) => [0, 1, 2, 4, 6, 12, 13, 15, 18].includes(i)).map((r) => {
    const c = tab("s1_descriptive_stats").columns;
    const g = (k) => r[c.indexOf(k)];
    return [tab("s1_descriptive_stats").index[tab("s1_descriptive_stats").rows.indexOf(r)], g("mean"), g("std"), g("min"), g("max"), g("skew"), g("kurtosis"), g("lag-1 autocorr"), g("beyond 5 sd %"), g("~zero values %")];
  }), [0.16, 0.09, 0.08, 0.09, 0.09, 0.08, 0.09, 0.11, 0.11, 0.1], "Descriptive statistics of a selection of sensors (all 19 in s1_descriptive_stats.csv)", { size: 15, digits: 2 }));
body.push(B("**No sensor is Gaussian** (D'Agostino–Pearson p < 10⁻¹⁰ for all 19). Currents J0 and J4 have very heavy tails (kurtosis 23 and 15), the gripper current is right-skewed (skew 3.9). Consequences: we use rank-based statistics (Spearman, univariate AUC), median imputation, and let the AutoML choose between StandardScaler and RobustScaler."));
body.push(B("**Speeds are zero about half of the time** (46–61 % of the values below 10⁻³ rad/s: the robot is stopped). The usual IQR outlier rule is therefore meaningless for them (IQR ≈ 0, it flagged 31–45 % of the values). We use \"beyond 5 standard deviations\" instead: at most 1.4 % of the values, all physically plausible (|speed| ≤ 2.73 rad/s, below the 3.14 rad/s (180°/s) limit of the UR3 base joints; temperatures 28–45 °C). **No outlier is removed**: extreme currents are the start/stop dynamics we want to learn from."));
body.push(B("**Temperatures are almost constant from one second to the next** (lag-1 autocorrelation 0.9999) and drift slowly; currents are only weakly autocorrelated at 1 Hz. Even so, consecutive rows are not independent samples: the event and the cycle are the real units of evidence, which is why all confidence intervals are computed by resampling **whole cycles**."));

body.push(H2("1.4 Class imbalance"));
body.push(...F("s1_fig02_imbalance.png", "Left: rows per class of the raw label. Middle: duration of the 72 events. Right: prevalence of the early-warning target for different horizons N.", 620));
body.push(...T(["Horizon N (s)", "Predictable rows", "Positives", "Prevalence %", "Imbalance", "Events announced"], pick("s1_imbalance_by_horizon", ["horizon N (s)", "predictable rows", "positives", "prevalence %", "imbalance ratio", "events announced"]), [0.15, 0.17, 0.14, 0.16, 0.18, 0.2], "Imbalance of the early-warning target as a function of the horizon", { digits: 2 }));
body.push(P("With N = 3 s the positive class is 2.95 % of the rows (1:33). A model that never raises an alert would reach 97 % accuracy, so **accuracy is not used**. The main metrics are PR-AUC (its no-skill value equals the prevalence) and event-level recall at a fixed false-alarm budget. Imbalance is handled with **class weights** inside the training folds; SMOTE is not used for our models because it interpolates between consecutive, non-independent seconds and creates physically inconsistent windows (its effect is measured in section 4)."));

body.push(H2("1.5 Correlation, redundancy and dimensionality reduction"));
body.push(...F("s1_fig03_correlation.png", "Spearman correlation between the 19 sensors on the training period (|ρ| ≥ 0.6 annotated).", 500));
body.push(...F("s1_fig04_vif_pca.png", "Left: variance inflation factors before (gray) and after (blue) replacing the six temperatures by their mean. Right: PCA on the 19 standardised sensors.", 620));
body.push(B("The **six temperatures measure one thing**, the global warm-up of the arm: Pearson r ≥ 0.995, PC1 of their PCA explains 99.85 % of their variance, VIF up to 8,988. They are replaced by **their mean plus a 60-s warm-up rate** (temperature difference over the last minute)."));
body.push(B("Speeds are kinematically coupled (Speed_J2/Speed_J3 ρ = −0.74, Speed_J3/Speed_J5 ρ = −0.69, Current_J5/Speed_J4 ρ = 0.85): after the temperature reduction the highest VIF is still 165 (Speed_J3). We keep all speeds — they describe different joints of the trajectory — and let L1/L2 penalties (linear models) and the optional SelectKBest of the AutoML deal with the collinearity; trees are insensitive to it."));
body.push(B("A PCA of all sensors needs 7 components for 90 % of the variance and 9 for 95 %. We **do not use PCA as a reduction**: the variance it keeps is not the variance that predicts grip losses (section 1.7), and it would make the features impossible to interpret physically. It remains a candidate for the SVM / k-NN in Step 3."));

body.push(H2("1.6 Non-stationarity and confounding"));
body.push(...F("s1_fig06_drift.png", "Grip-loss rate (top) and mean temperature of J0 (bottom) per block of 500 rows. The dashed line marks the start of the validation period.", 600));
body.push(P("Grip losses are 4 times more frequent during the first 500 s of the morning, when the robot is cold, then stay around 2–4 %. Temperature and \"start of the day\" are confounded: a model that uses the temperature would mostly learn \"it is early in the morning\", which can never happen in the warm validation and test periods. Temperature is therefore a **block that the AutoML may switch off**, and its effect is checked by ablation (section 5.5)."));

body.push(H2("1.7 Which signals carry information before the loss?"));
body.push(...F("s1_fig05_univariate_auc.png", "Univariate ROC-AUC of the candidate features on the training period, for the label \"during the loss\" (gray) and for our early-warning label (blue). 0.5 = no information.", 560));
const assoc = tab("s1_target_association");
body.push(P(`Gray dots answer \"does the signal differ while the part is lost?\" and blue dots \"does it differ in the 3 seconds **before**?\". The difference between the two is the leakage risk discussed in Step 1. Some features keep their information before the onset — the 5-s mean of the shoulder current Current_J1 (AUC 0.72, 95 % CI [0.67, 0.78] by cycle bootstrap), its lags and its variability (lower before a loss), and the time since the last loss (AUC 0.31, i.e. a recent loss increases the risk: losses come in bursts). Others lose most of it: Current_J2 goes from 0.70 during the loss to 0.55 before, the number of losses in the last 120 s from 0.75 to 0.63. The temperature mean is almost uninformative for the early label (0.48) once the cold start is excluded from the predictable rows, and the gripper current alone is weak (0.55). No single feature is enough: the signal is multivariate and short.`));

body.push(H2("1.8 Proposed pre-processing"));
body.push(...T(["Step", "Justification", "Fitted on", "Applies to"], pick("s1_preprocessing_proposal", ["step", "justification", "fitted on", "applies to"]), [0.33, 0.4, 0.12, 0.15], "Pre-processing proposal: everything that is learned from data is fitted inside the pipeline, on the training fold", { size: 15 }));
body.push(P("Two new features come from the exploratory notebook: the **residual to the normal cycle profile** (value minus the median value at the same second of the cycle, learned on the training fold only) and the **burst features** (seconds since the last grip loss and number of onsets in the last 120 s, computed from labels strictly before t, which the cell knows in real time). The position in the cycle enters as raw seconds for trees and as a cubic spline for linear models, because its effect is not monotonic (two risky phases, around 3–5 s and 15–21 s)."));

// ================================================================== STEP 2
body.push(H1("2. Learning and testing plan"));
body.push(H2("2.1 Choice of data and construction of the sets"));
body.push(P("The data is one multivariate time series, so the sets are **chronological**. The cut points are placed at 70 % and 85 % of the predictable rows, then moved forward to the next cycle start so that no cycle is split between two sets. The 8 rows before each cut (N + window = 3 + 5) form an **embargo**: their label looks into the next set, so they are used neither for training nor for evaluation."));
body.push(...T(["Set", "Rows", "Positives", "Prevalence %", "Events", "Cycles", "Start", "End", "Mean temp J0"], pick("s2_split_summary", ["set", "rows", "positives (y=1)", "prevalence %", "events announced", "cycles", "start (UTC)", "end (UTC)", "mean temp J0 (°C)"]),
  [0.1, 0.09, 0.11, 0.12, 0.09, 0.09, 0.13, 0.13, 0.14], "Chronological train / validation / test sets (the embargo row aggregates the rows before both cuts)", { digits: 2 }));
body.push(...F("s2_fig01_split_timeline.png", "The three sets along the time line and the grip-loss onsets.", 620));
body.push(P("Consequences that must stay in mind: the training period contains the whole morning (cold robot, many losses) while validation and test are 18-minute windows at the end of the warm afternoon (12 and 9 events). One missed event changes the test event recall by 11 points, so every test figure is given with a 95 % confidence interval obtained by **bootstrap over cycles**, and model comparisons use a **paired** bootstrap on the same resampled cycles."));
body.push(H2("2.2 Cross-validation for tuning"));
body.push(P("Hyper-parameters are tuned only on the training period, with an **expanding window**: each fold trains on the past and validates on the following block, with an 8-row gap. The plain `TimeSeriesSplit` (equal number of rows per fold) puts the lunch break in fold 3, whose validation block then contains **one event (3 positive rows)**: its PR-AUC is meaningless and dominates the variance. We therefore place the fold boundaries at event quantiles (`event_balanced_cv` in `src/splits.py`), which gives 8–9 events per validation fold."));
const folds = tab("s2_cv_folds");
body.push(...T(["Scheme", "Fold", "Train rows", "Val. rows", "Train events", "Val. positives", "Val. events", "Val. period (UTC)"], pick("s2_cv_folds", ["scheme", "fold", "train rows", "validation rows", "train events", "validation positives", "validation events", "validation period (UTC)"]),
  [0.24, 0.06, 0.1, 0.1, 0.11, 0.12, 0.1, 0.17], "Folds of the plain TimeSeriesSplit and of the event-balanced expanding-window CV used for tuning", { size: 15, highlight: (r) => r[6] <= 1 }));
body.push(H2("2.3 Controlling under- and over-fitting"));
body.push(...F("s2_fig02_learning_curves.png", "Expanding-window learning curves: PR-AUC on the training fold (blue) and on the next, future fold (orange) as the training window grows.", 620));
body.push(B("**Underfitting** shows as low training and validation scores close to each other: the logistic regression (train 0.35, CV 0.21) cannot represent the interactions between the position in the cycle and the currents. The remedy is richer features or non-linear models, not less regularisation."));
body.push(B("**Overfitting** shows as a large gap between the two curves: an unconstrained random forest reaches a training PR-AUC of 1.0 for a CV of 0.36. Interestingly, forcing larger leaves (min_samples_leaf = 30) reduces the gap but also the CV score (0.31): the problem is not variance alone but the small number of events. Regularisation strength is therefore a hyper-parameter chosen by CV, never fixed by hand."));
body.push(B("The validation score rises strongly with the training window in the last folds: **more history helps**, which supports re-training on train + validation before a deployment (plan, section 5.7)."));
body.push(H2("2.4 Controlling data leakage"));
body.push(P("Leakage checklist implemented in the code:"));
body.push(N("Features use only rows ≤ t (lags, differences, trailing windows), never across a recording break; the label is the only object allowed to look ahead."));
body.push(N("Rows where a loss is ongoing are removed: the model cannot detect the consequences of a loss that has already started."));
body.push(N("Everything learned from data — normal cycle profile, spline basis, imputer, scaler, feature selector, class weights — is inside the scikit-learn pipeline and is refitted on each training fold."));
body.push(N("Chronological sets with embargo, expanding-window CV with a gap; the test period is evaluated once, after the final choice."));
body.push(N("Exploratory statistics that drive decisions are computed on the training period only."));
body.push(P("Two experiments measure what would happen without these rules, with the same random forest:"));
body.push(...T(["Experiment", "Variant", "Mean CV PR-AUC", "sd over folds"], pick("s2_leakage_experiments", ["experiment", "variant", "mean CV PR-AUC", "sd"]), [0.22, 0.5, 0.14, 0.14], "Leakage experiments on the training period"));
body.push(...F("s2_fig03_leakage_experiments.png", "Leakage experiments: bars are the mean over folds, dots the individual folds.", 560));
body.push(P("Shuffled K-fold mixes neighbouring seconds of the same cycle between training and validation (+22 % PR-AUC, and a deceptively small spread), and centred rolling windows let the feature see 2 s of the future (+32 %). Both would look like progress and would disappear in deployment."));
body.push(H2("2.5 Evaluation methods"));
body.push(B("**Ranking quality:** PR-AUC (average precision), with the prevalence as no-skill level; ROC-AUC is reported but is optimistic under heavy imbalance."));
body.push(B("**Alert quality**, at a threshold chosen on validation: event recall (an event is anticipated if at least one alert is raised in the 3 s before its onset), mean warning time, and **false-alarm episodes per hour** (a run of consecutive alerts on negative rows counts once)."));
body.push(B("**Two operating points:** the F1-optimal threshold, and the lowest threshold that keeps ≤ 30 false alarms per hour on validation (about one every 2 minutes, i.e. in 1 cycle out of 5). The second one is the business-relevant point: an alert system that cries wolf every few seconds is ignored by operators."));
body.push(B("**Uncertainty:** 95 % percentile intervals from 1,000 bootstrap resamples of whole cycles; paired differences for model comparisons."));

// ================================================================== STEP 3
body.push(H1("3. Baseline solution and its calibration"));
body.push(H2("3.1 Three reference levels"));
body.push(B("**B0 – prior:** constant score equal to the training prevalence. Its PR-AUC equals the prevalence of the evaluated set: it is the zero of the scale."));
body.push(B("**B1 – cycle position:** risk of a grip loss in the next 3 s learned per second of the cycle on the training set (Laplace-smoothed toward the prior with 20 pseudo-rows). It answers \"is the robot in a risky phase?\" and uses no sensor."));
body.push(B("**B2 – logistic regression** (the baseline proper): the 14 current-second signals (6 currents, 6 speeds, gripper current, sum of absolute speeds), the mean temperature and its warm-up rate, and a cubic spline of the position in the cycle (8 knots); median imputation, standardisation, L2 penalty, balanced class weights. It is simple, fast, interpretable through its coefficients, and represents what can be done without temporal feature engineering."));
body.push(H2("3.2 Calibration of the model"));
body.push(P("**Regularisation (C).** C is chosen on the event-balanced CV of the training period over 13 values from 10⁻⁴ to 10². The CV curve is flat above C = 0.1 and the fold-to-fold standard error (≈ 0.04) is larger than the whole range of the curve. The classical one-standard-error rule would therefore always select the strongest penalty of the grid (C = 0.001, whose validation PR-AUC equals the prevalence: no skill). Because every C is evaluated on the same folds, we apply the rule to the **paired** fold differences, where the fold effect cancels: it keeps **C = 0.316**, the best value."));
body.push(...F("s3_fig01_validation_curve.png", "Validation curve of B2 over C (mean ± 1 SE of the 5 folds).", 470));
body.push(P("**Probabilities.** Balanced class weights multiply the weight of the positives by about 16, so the predicted probabilities are far too high (mean 28 % for an observed rate of 3 %; Brier score 0.154). Two remedies were compared on validation: no class weights, and a sigmoid (Platt) recalibration fitted on the time-series folds. Both bring the mean prediction to the observed rate and the Brier score to 0.029, but slightly lower the PR-AUC. Because the alert decision only depends on the ranking and on a threshold tuned on validation, the baseline keeps the balanced weights; the calibrated variant is the one to use if probabilities are shown to an operator."));
body.push(...T(["Variant", "PR-AUC", "Brier", "Log-loss", "Mean predicted %", "Observed %"], pick("s3_calibration", ["variant", "PR-AUC", "Brier", "log-loss", "mean predicted %", "observed rate %"]), [0.34, 0.12, 0.12, 0.12, 0.16, 0.14], "Probability calibration of B2 on the validation period"));
body.push(...F("s3_fig02_calibration.png", "Reliability diagram and distribution of the predicted probabilities on validation.", 600));
body.push(P("**Threshold.** The decision threshold is chosen on the validation period (never on test) for the two operating points of section 2.5."));
body.push(H2("3.3 Results"));
body.push(...T(["Model", "Operating point", "Set", "PR-AUC", "ROC-AUC", "Precision", "Recall", "Events det.", "Event recall", "FA / hour"],
  pick("s3_baseline_results", ["model", "operating point", "set", "PR-AUC", "ROC-AUC", "precision", "recall", "events detected", "event recall", "false alarms / hour"]).map((r) => { r[7] = `${r[7]}`; return r; }),
  [0.18, 0.12, 0.06, 0.08, 0.09, 0.09, 0.08, 0.09, 0.1, 0.11], "Baselines on validation and test (thresholds chosen on validation)", { size: 15 }));
body.push(P(`On the test period B2 reaches a PR-AUC of ${fmt(tB2[0]["PR-AUC"])} (95 % CI ${tB2[0]["PR-AUC 95% CI"]}) for a prevalence of 0.025: it ranks the rows a little better than chance, about as well as the cycle position alone. At the F1-optimal threshold it anticipates 6 of 9 events, but at the price of **341 false-alarm episodes per hour** — an alert every 10 seconds, useless in practice. At the trustworthy operating point (≤ 30 false alarms per hour) it anticipates **1 event out of 9**. This is the level the AutoML has to beat.`));
body.push(...F("s3_fig03_coefficients.png", "Largest standardised coefficients of B2. The spline terms of the position dominate, followed by the shoulder current Current_J1.", 460));

// ================================================================== STEP 4
body.push(H1("4. Benchmark models from the state of the art"));
body.push(H2("4.1 Published results on UR3 CobotOps"));
body.push(...T(["Study", "Target", "Models", "Split", "Reported results", "Remark"], pick("s4_literature", ["study", "target", "models", "split", "reported results", "remark"]), [0.16, 0.16, 0.17, 0.13, 0.22, 0.16], "Studies using UR3 CobotOps (full references at the end of the document)", { size: 14 }));
body.push(P("All published scores concern the **current second** (the label is predicted from the signals recorded while the fault is happening) and use **random splits** or splits whose order is not stated. Only Yaşar Çıklaçandır et al. (2025) model grip_lost on its own; their almost identical accuracy, precision, recall, F1 and AUC (all ≈ 0.99) are what one expects when the test set itself has been balanced, i.e. when SMOTE is applied before the train/test split. None of the studies reports an early-warning task or a chronological evaluation, so their numbers cannot be compared directly with ours."));
body.push(H2("4.2 Re-running the published protocols on our data"));
body.push(P("To make the benchmark comparable, we re-ran the published protocols with the model families they use (random forest with 200 trees, k-NN with k = 5, logistic regression; raw sensors, default hyper-parameters, threshold 0.5):"));
body.push(B("**P1** concurrent label, random stratified 70/30 split (Liu-style);"));
body.push(B("**P2** = P1 with SMOTE applied to the whole data before the split (suspected in Yaşar Çıklaçandır et al.);"));
body.push(B("**P3** = P1 with SMOTE on the training part only (correct use);"));
body.push(B("**P4** concurrent label, chronological 70/30 split;"));
body.push(B("**P5** our task: early warning N = 3 s, ongoing losses removed, chronological (training → validation)."));
const pr = tab("s4_protocol_rerun");
body.push(...T(["Protocol", "Model", "Test prev. %", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"], pick("s4_protocol_rerun", ["protocol", "model", "test prevalence %", "accuracy", "precision", "recall", "F1", "ROC-AUC", "PR-AUC"]),
  [0.27, 0.15, 0.08, 0.08, 0.09, 0.08, 0.07, 0.09, 0.09], "Published protocols re-run on UR3 CobotOps", { size: 14, highlight: (r) => r[1] === "Random forest" }));
body.push(...F("s4_fig01_protocol_rerun.png", "The same random forest evaluated under the five protocols.", 600));
body.push(P("The re-run **reproduces the published numbers**: under P1 the random forest reaches ROC-AUC 0.935 (Liu: 0.911), k-NN an F1 of 0.455 (Liu: 0.446) and the logistic regression detects no fault, as reported by Liu; with SMOTE before the split (P2) the forest reaches accuracy 0.983, F1 0.983 and ROC-AUC 0.999, which matches Yaşar Çıklaçandır et al. (accuracy 0.990, F1 0.990, ROC-AUC 0.990, and 0.999 after chi-square selection). The same forest falls to PR-AUC 0.29 with a chronological split (P4) and to **0.11** on our early-warning task (P5). The published scores are therefore mostly a property of the protocol: they measure how well a model recognises a fault that is already visible in a shuffled set of seconds, not how well it anticipates a future one."));
body.push(H2("4.3 Benchmarks retained"));
body.push(B("**Internal benchmarks** (same protocol as ours): B0, B1, B2 of section 3 and, for the model families of the literature, the random forest of P5 (validation PR-AUC 0.11) and the AutoML family champions of section 5."));
body.push(B("**External reference values** (different protocol, shown for context only): random forest ROC-AUC 0.911–0.935 on the concurrent label with a random split; ESOA-BiRNN ROC-AUC 0.946 on protective stops (Taş & Bal, 2025)."));
body.push(B("Deep sequence models (BiRNN, LSTM, TCN, Transformer) are the main family of the literature that we have not tested yet: they are the natural \"out-of-course\" candidate for Step 3."));

// ================================================================== STEP 5
body.push(H1("5. AutoML pipeline v0.1"));
body.push(H2("5.1 Design"));
body.push(P("We built the AutoML on scikit-learn rather than using a black-box library, so that every step can be defended at the oral exam and no step can silently break the time-series rules (most AutoML libraries shuffle the data in their internal validation). The four components required by the guidelines are:"));
body.push(...T(["Component", "Implementation"], [
  ["Data preparation", "median imputation; StandardScaler or RobustScaler (searched) for scale-sensitive models; balanced class weights"],
  ["Feature engineering", "feature blocks as hyper-parameters: past windows (lags, differences, 5-s mean/sd) yes/no, cycle residuals yes/no, burst features yes/no, temperature mean/none; automatic selection with SelectKBest (k = 20 / 40 / all, ANOVA F) for linear models, SVM and k-NN"],
  ["Model selection", "7 families: logistic regression (L1/L2), SVM with RBF kernel, k-NN, decision tree, random forest, extra trees, histogram gradient boosting"],
  ["Hyper-parameter optimisation", "randomized search (119 configurations in total, 10–25 per family), scored by mean PR-AUC on the event-balanced expanding-window CV of the training period"],
], [0.25, 0.75], "Components of AutoML v0.1 (05_automl.py)", { size: 16 }));
body.push(P("Linear regression, listed in the guidelines, is not included: it is not a classifier (its outputs are not probabilities); logistic regression is its probabilistic counterpart."));
body.push(H2("5.2 Evaluation protocol"));
body.push(N("**Search** on the training period only: every configuration is scored by its mean PR-AUC over the 5 event-balanced folds; training scores are kept to measure the overfitting gap.", 2));
body.push(N("**One champion per family** (best mean CV PR-AUC), refitted on the whole training period.", 2));
body.push(N("**Final choice on the validation period** (12 events): the champion with the highest validation PR-AUC. Thresholds of the two operating points are also set on validation.", 2));
body.push(N("**Test period used once**, for the final pipeline and the baselines, with cycle-bootstrap confidence intervals and a paired comparison with B1 and B2.", 2));
body.push(N("**Diagnostics on train/validation only** (the test period is not re-used): ablation of the feature blocks, sensitivity to the horizon N, permutation importance, detected vs missed events.", 2));
body.push(H2("5.3 Search results"));
body.push(...F("s5_fig01_leaderboard.png", "AutoML leaderboard: every configuration tried (gray), CV score of the family champion (blue) and its validation score (orange).", 620));
body.push(...T(["Family", "CV PR-AUC", "CV sd", "Train PR-AUC", "Val. PR-AUC", "Val. event recall (≤ 30 FA/h)"], pick("s5_family_champions", ["family", "CV PR-AUC", "CV sd", "train PR-AUC", "val PR-AUC", "val event recall (<= 30 FA/h)"]),
  [0.28, 0.13, 0.11, 0.14, 0.14, 0.2], "Family champions, sorted by validation PR-AUC", { highlight: (r, k) => k === 0 }));
body.push(P(`**Final configuration** (${finalFam}): ${cfgFinal.replace(/f\./g, "features.").replace(/m\./g, "model.").replace(/, /g, " · ")}.`, { run: { size: 18 }, align: AlignmentType.LEFT }));
body.push(P(`Gradient boosting is the best family both in CV and on validation, by a large margin on validation. The random forest and extra-trees champions were selected **without** the burst features — the random search sampled too few combinations of blocks and tree parameters — and lose most of their CV advantage on validation. This is a known weakness of v0.1 (one random draw over a joint space): see the plan. Training PR-AUC is 1.0 for gradient boosting, k-NN and the forests: these families memorise the training period, and only the CV and validation columns can be used to compare them.`));
body.push(...F("s5_fig02_overfitting_map.png", "Training vs CV PR-AUC of every configuration; the vertical distance to the diagonal is the overfitting gap.", 430));
body.push(H2("5.4 Final evaluation on the test period"));
body.push(...T(["Model", "Operating point", "PR-AUC", "PR-AUC 95 % CI", "Precision", "Recall", "Events det.", "Event recall 95 % CI", "Warning (s)", "FA / hour"],
  pick("s5_test_results", ["model", "operating point", "PR-AUC", "PR-AUC 95% CI", "precision", "recall", "events detected", "event recall 95% CI", "mean warning (s)", "false alarms / hour"]),
  [0.2, 0.11, 0.08, 0.12, 0.08, 0.07, 0.07, 0.12, 0.07, 0.08], "Test results (9 events, thresholds chosen on validation)", { size: 14, highlight: (r) => String(r[0]).startsWith("AutoML") }));
body.push(...T(["Comparison", "Δ PR-AUC (test)", "95 % CI (paired cycle bootstrap)", "Significant"], pick("s5_paired_differences", ["comparison", "Δ PR-AUC (test)", "95% CI (paired cycle bootstrap)", "significant"]), [0.38, 0.17, 0.3, 0.15], "Paired comparison of the AutoML pipeline with the baselines"));
body.push(P(`At the trustworthy operating point the AutoML pipeline anticipates **${autoFA["events detected"]} of the ${autoFA.events} test events** (95 % CI ${autoFA["event recall 95% CI"]}) with a mean warning of ${fmt(autoFA["mean warning (s)"], 1)} s and ${fmt(autoFA["false alarms / hour"], 1)} false alarms per hour, against 1 of 9 for the baseline at the same alarm budget. Its PR-AUC is ${fmt(autoFA["PR-AUC"])} (prevalence 0.025, i.e. ${(autoFA["PR-AUC"] / 0.0249).toFixed(0)} times the no-skill level) and the gain over both B1 and B2 is significant in the paired bootstrap. The intervals remain very wide: the test period is 18 minutes with 9 events.`));
body.push(...F("s5_fig03_pr_curves_test.png", "Precision–recall curves on the test period.", 440));
body.push(...F("s5_fig04_test_timeline.png", "Score of the AutoML pipeline over the test period, with the alert threshold and the real onsets.", 620));
body.push(H2("5.5 Diagnostics (training and validation only)"));
body.push(...F("s5_fig05_ablation_horizon.png", "Left: ablation of the feature blocks of the final configuration. Right: validation lift (PR-AUC divided by the prevalence) as a function of the horizon N, with and without the burst features.", 620));
body.push(...T(["Variant", "CV PR-AUC", "CV sd", "Val. PR-AUC"], pick("s5_ablation", ["variant", "CV PR-AUC", "CV sd", "val PR-AUC"]), [0.46, 0.18, 0.18, 0.18], "Ablation of the feature blocks"));
body.push(B(`**Burst features are the main driver on validation** (validation PR-AUC ${fmt(abl["final configuration"].val)} → ${fmt(abl["without burst"].val)} without them; CV ${fmt(abl["final configuration"].cv)} → ${fmt(abl["without burst"].cv)}). Grip losses come in series, and \"a loss happened in the last cycle\" is a strong, legitimate warning. It also means that a large part of the score comes from **follow-up losses**, which are easier to anticipate than the first loss of a series (table below).`));
body.push(B(`**Past windows help** (validation ${fmt(abl["without window"].val)} without them). The cycle residuals and the explicit position in the cycle bring little once the windows are present (differences within the noise).`));
body.push(B(`**Adding the temperature lowers the validation score** (${fmt(abl["temperature: mean"].val)} vs ${fmt(abl["final configuration"].val)}) while the CV is unchanged, as expected from the cold-start confounding of section 1.6: the AutoML was right to drop it.`));
body.push(B(`**Horizon.** Without burst features, the validation lift is ${fmt(hzv(1, false)["val lift (PR-AUC / prevalence)"], 0)}× at N = 1 s, ${fmt(hzv(3, false)["val lift (PR-AUC / prevalence)"], 0)}× at N = 3 s and ${fmt(hzv(5, false)["val lift (PR-AUC / prevalence)"], 1)}× at N = 5 s. Most of the sensor information is concentrated in the **last second** before the onset. This is consistent with the short precursors of the exploratory analysis but is also what a **label recorded slightly after the physical loss** would produce. N = 3 s remains a reasonable compromise between warning time and skill; the label timing must be checked (risk R2).`));
body.push(...T(["Set", "Event type", "Events", "Detected", "Event recall"], pick("s5_events_by_type", ["set", "type", "events", "detected", "event recall"]), [0.12, 0.42, 0.14, 0.14, 0.18], "Detection at ≤ 30 false alarms per hour, by event type (follow-up = another loss ended less than 60 s before)", { digits: 2 }));
body.push(P(`Isolated losses are detected too (${evt("val", "isolated").detected} of ${evt("val", "isolated").events} on validation, ${evt("test", "isolated").detected} of ${evt("test", "isolated").events} on test), so the pipeline does not only repeat \"a loss just happened\". The most useful features by permutation importance on validation are: ${imp.join(", ")}.`));
body.push(...F("s5_fig06_permutation_importance.png", "Permutation importance on validation (drop in PR-AUC when one input column is shuffled; 5 repeats).", 460));

body.push(H2("5.6 Risks and failure modes"));
body.push(...T(["#", "Risk / failure mode", "Evidence", "Mitigation"], [
  ["R1", "Too few events: test = 9 events, validation = 12. Scores and the final choice are noisy", "test PR-AUC CI " + autoFA["PR-AUC 95% CI"] + "; event recall CI " + autoFA["event recall 95% CI"], "report CIs, paired tests; rolling-origin evaluation over several test windows (plan P1)"],
  ["R2", "Label timing: the label may switch to 1 after the physical loss, so the last second before the onset already shows its consequences", "lift jumps at N = 1 s even without burst features; Current_J1 deviates mostly at −1 s", "check the definition of grip_lost; compare N = 2 and 3; drop the last second (gap of 1 s) as a robustness test"],
  ["R3", "Dependence on past labels (burst features): if grip_lost is not known in real time, the feature is not available", "validation PR-AUC " + fmt(abl["final configuration"].val) + " → " + fmt(abl["without burst"].val) + " without burst", "confirm availability; always report the model without burst features as a second line"],
  ["R4", "Drift / one day of data: training includes the cold start; validation and test are the warm end of the afternoon", "rate 13 % in the first 500 s vs 2–4 % later", "temperature excluded; per-session or rolling re-training; collect more days"],
  ["R5", "Validation over-use: 7 champions + thresholds chosen on 12 events", "large CV-validation gaps for the forests", "nested selection or choice on CV + validation; keep test untouched until the end of the project"],
  ["R6", "Incomplete search: the random draw paired the forests with the wrong feature blocks", "forest champions without burst", "Bayesian / successive-halving search per feature set (plan P2)"],
  ["R7", "Alert fatigue: thresholds tuned on 18 min may give a different false-alarm rate later", "F1 thresholds give 300+ FA/h for the baselines", "monitor FA/h; threshold re-estimated on rolling windows; hysteresis (k alerts in m seconds)"],
  ["R8", "Missing or skipped samples in deployment", "54 empty rows, 24 skipped seconds", "causal ffill ≤ 2 s; no prediction on empty rows; test robustness to random drop-outs"],
], [0.05, 0.33, 0.3, 0.32], "Risks and failure modes", { size: 15 }));

body.push(H2("5.7 Prioritized plan for the remaining work"));
body.push(...T(["Priority", "Task", "Why", "Step"], [
  ["P1 – must", "Rolling-origin evaluation (several successive test windows in the afternoon) and the same metrics with CIs", "a single 18-min test window with 9 events cannot rank models (R1)", "3"],
  ["P1 – must", "Fill AI_AUDIT.md and PROMPT_LOG.md from the drafts in ai-governance/ and verify every item", "required deliverable, graded", "3"],
  ["P1 – must", "Check the meaning and timing of grip_lost (real-time availability, delay vs physical loss)", "R2, R3 decide which features and horizon are legitimate", "3"],
  ["P2 – should", "Replace random search by Optuna (TPE) or successive halving, with the feature blocks searched jointly per family", "R6; better use of the compute budget", "3"],
  ["P2 – should", "Advanced versions of course algorithms: tuned gradient boosting (XGBoost/LightGBM), cost-sensitive SVM, calibrated probabilities", "guidelines Step 3.2", "3"],
  ["P2 – should", "Ensemble of the best families (soft voting / stacking on time-series out-of-fold predictions)", "guidelines Step 3.5; reduces variance", "3"],
  ["P2 – should", "Out-of-course model: 1-D CNN / TCN or LSTM on the raw 10–20 s window, justified by a paper", "guidelines Step 3.6–3.7; literature uses sequence models", "3"],
  ["P3 – could", "Alarm post-processing: hysteresis, minimum duration, per-cycle alert budget", "R7; operator acceptance", "3–4"],
  ["P3 – could", "Re-train on train + validation with thresholds from out-of-fold predictions before the final test", "learning curves still rising", "4"],
  ["P3 – could", "Second target (protective stop) with the same pipeline, as a generalisation check", "shows the pipeline is reusable", "4"],
], [0.12, 0.45, 0.35, 0.08], "Prioritized plan (P1 = before anything else)", { size: 15 }));

// ================================================================== Governance
body.push(H1("6. AI governance notes"));
body.push(P("This phase was produced with an AI assistant (Claude). As required by the guidelines, the folder `ai-governance/` contains:"));
body.push(B("`PROMPT_LOG_entry6_step2.md`: the prompt of this phase, verbatim, with the list of decisions the AI took and the checks the team must do before the oral defence."));
body.push(B("`AI_AUDIT_candidates_step2.md`: concrete flaws of earlier AI-assisted outputs found while coding (wrong duplicate count, wrong counter values, placeholder value counted as data, events split by label flicker, a CV fold with one event, a one-standard-error rule that always picks the strongest penalty, an IQR rule that flags 40 % of the speeds, and a pipeline template mutated in place by the ablation code). Each item must be verified by the team before it is copied to `AI_AUDIT.md`."));
body.push(P("Points the team should be ready to defend: the debouncing rule, the paired one-standard-error rule, the event-balanced folds, the choice of the 30 false-alarms-per-hour budget, and why the burst features are (or are not) legitimate in real time."));

// ================================================================== References
body.push(H2("References"));
[
  "Tyrovolas, M., Aliev, K., Antonelli, D., & Stylios, C. (2024). UR3 CobotOps [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5J891",
  "Tyrovolas, M., Stylios, C., Aliev, K., & Antonelli, D. (2024). Leveraging Information Flow-Based Fuzzy Cognitive Maps for Interpretable Fault Diagnosis in Industrial Robotics. DoCEIS 2024.",
  "Liu, S. (2025). Machine Learning-Based Fault Detection for UR3 Collaborative Robot: A Multimodal Data Analysis Approach. Engineering Frontiers, 1(2). https://doi.org/10.71411/ef.2025.v1i2.941 (figures taken from our Step-1 review; to be re-checked).",
  "Yaşar Çıklaçandır, F. G., Mumcu, S. A., Çam, B., & Ceran, I. (2025). Impact of Feature Selection on the Performance of Classification Algorithms in Predicting Industrial Robot Failures. Dokuz Eylül University Journal of Science and Engineering (DEU FMD), 27(81), 393–399. https://doi.org/10.21205/deufmd.2025278107",
  "Taş, G., & Bal, C. (2025). An Egret Swarm Optimization Based BiRNN Method Approach to Determine the Protective Stopping Time of UR3 Robot Arm. Sakarya University Journal of Science, 29(6), 717–725. https://doi.org/10.16984/saufenbilder.1755797",
  "Shen, Y., Wang, K., Chen, Z., Huang, J., Zhu, Y., & Song, Z. (2026). Physics-Informed Transformer with ODE-Guided Joint Modeling for Fault Classification and RUL Prediction in Collaborative Robots. Proceedings of the Asia Pacific Conference of the PHM Society 2025.",
  "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
  "Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321–357.",
  "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS ONE, 10(3), e0118432.",
  "Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning (2nd ed.), section 7.10 (one-standard-error rule). Springer.",
].forEach((r) => body.push(B(r)));

// ------------------------------------------------------------------ document
const doc = new Document({
  creator: "Courtois, Espinoza, Devernois",
  title: "Grip-loss early warning on a UR3 cobot – Step 2",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 32, bold: true, color: BLUE }, paragraph: { spacing: { before: 240, after: 160 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 26, bold: true, color: BLUE }, paragraph: { spacing: { before: 240, after: 100 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 23, bold: true, color: "404040" }, paragraph: { spacing: { before: 160, after: 80 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 260 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 900, hanging: 260 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 300 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ text: "UR3 grip-loss early warning – Step 2 · page ", size: 16, color: "808080" }),
      new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "808080" })] })] }) },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "Step2_Standard_Solutions_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("written", out);
});
