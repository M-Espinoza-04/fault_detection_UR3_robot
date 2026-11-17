# PROMPT_LOG.md

**Project:** Grip Loss Prediction on a UR3 Collaborative Robot
**Team:** Quentin Courtois, Mauricio Espinoza, Ivan Devernois – IRO2
**AI tool used:** Claude (Anthropic), chat interface with code execution

This log documents the critical prompts used for high-stakes decisions of the project definition phase (Steps 1–3). Prompts were written in French; they are reproduced verbatim, followed by an English translation. Trivial syntax or formatting requests are not logged.

---

## 1. Topic and dataset selection

- **Date:** 13/11/2026
- **Decision category:** Problem definition, dataset choice

**Prompt (verbatim):**
> Pour la 2ème partie, aide moi à compléter les 3 premières étapes. Je suis en industrie et robotique, je cherche idéalement une problématique liée à la robotique, plus un problème de classification et avec du nettoyage basique pour le dataset

**Translation:**
> For part 2, help me complete the first 3 steps. I am in Industry & Robotics; ideally I am looking for a robotics-related problem, preferably a classification problem, with basic cleaning needed for the dataset.

**AI output (summary):** proposed fault detection on a UR3 cobot with the UCI *UR3 CobotOps* dataset (joint currents, temperatures and speeds, two binary labels). Dataset statistics (missing rows, fault rate) were taken from secondary sources, because the AI could not download the file at that time.

**Human verification:**
- [ ] Dataset page, license (CC BY 4.0) and DOI checked on the UCI website
- [x] Statistics re-computed on the actual file later (see entry 4)

**Decision:** UR3 CobotOps kept as the project dataset.

---

## 2. Time-series structure and data leakage prevention

- **Date:** 17/11/2026
- **Decision category:** Data splitting, feature engineering, leakage prevention

**Prompts (verbatim):**
> est-ce que le dataset du projet contient des time series?

> Par rapport aux time series, comment ça ce passe, quelles problématiques ça rajoute?

**Translation:**
> Does the project dataset contain time series?

> Regarding time series, how does it work, what issues does it add?

**AI output (summary):** the dataset is a single multivariate time series sampled at about 1 Hz, so rows are not independent. Consequences listed: no random split (chronological split instead), `TimeSeriesSplit` with a gap instead of shuffled stratified K-fold, no backward fill or centered rolling windows (future information), scaler fitted on the training period only, event-level vs. row-level evaluation, risks of SMOTE/NearMiss on time-window features, drift between periods.

**Human verification:**
- [ ] Check in the scikit-learn documentation that `TimeSeriesSplit` supports the `gap` parameter
- [ ] Check the effect of a random vs. chronological split on the baseline score (expected: random split is optimistic)

**Decision:** chronological 70/15/15 split, `TimeSeriesSplit` for tuning, features built from past values only.

---

## 3. Leakage risk of the target formulation and feasibility

- **Date:** 17/11/2026
- **Decision category:** Problem formulation, metric choice, leakage prevention

**Prompts (verbatim):**
> compare le word avec les consignes du pdf sur les étapes concernées

> est-ce que la problématique de prédiction des pannes est faisable avec ce dataset considérant le peu de données?

**Translation:**
> Compare the Word document with the instructions of the PDF for the relevant steps.

> Is the fault prediction problem feasible with this dataset, considering the small amount of data?

**AI output (summary):** flagged that classifying the fault at the same second could be leakage (the sensor signals reflect the consequence of the fault, e.g. joint speeds during a stop), which led to an early-prediction formulation (fault within the next N seconds) and the removal of rows where the fault is already ongoing. On feasibility: the relevant quantity is the number of distinct events, not rows; recommended counting events first, using simple regularized models, aggregating results over several folds with confidence intervals, and defining objectives relative to the baseline.

**Human verification:**
- [x] Number of events counted on the actual file (see entry 4)
- [ ] Check the signals around grip-loss events (before / during / after) to confirm which ones reflect consequences rather than causes

**Decision:** early prediction with horizon N (3–5 s); main metric = event-level recall and warning time, plus PR-AUC; no accuracy as main metric.

---

## 4. Exploratory analysis and statistical test between the two labels

- **Date:** 17/11/2026
- **Decision category:** Data quality, statistical test selection, class imbalance, feature selection

**Prompt (verbatim):**
> analyse le dataset. Détermine aussi si y'a possibilité de prédire grip_lost avec Robot_predictive_Stop

**Translation:**
> Analyze the dataset. Also determine whether grip_lost can be predicted with Robot_ProtectiveStop.

**AI output (summary):** analysis run with code on the actual Excel file:
- 7,409 rows on one day, two sessions separated by a 5 h 13 min gap; 54 rows with missing sensor values; 863 timestamps wrapped in quotes; 2 duplicated rows; a trailing space in a column name; the cycle counter decreases once.
- `grip_lost`: 243 rows (3.3%), 77 events, median duration 2 s; chronological 70/15/15 split → 53 / 11 / 13 events.
- Link between labels: only 3 rows with both labels; `Robot_ProtectiveStop` (current or past windows) gives an AUC of 0.43–0.49 for `grip_lost`; at cycle level, Fisher's exact test p = 0.25 (no significant association). Grip losses start later in the cycle (median 18 s) than protective stops (median 5 s).

**Human verification:**
- [ ] Re-run the counts (rows, events, missing values) in our own notebook
- [ ] Check the choice of Fisher's exact test (2×2 table with small counts) vs. chi-square (chi-square gave p = 0.31, same conclusion)
- [ ] Check that the position-in-cycle feature is computed only from past information

**Decision:** `Robot_ProtectiveStop` excluded from the features; position in the cycle added as a candidate feature; cleaning steps listed in Step 2.

---

## 5. Choice of the target variable

- **Date:** 17/11/2026
- **Decision category:** Problem definition, target choice

**Prompt (verbatim):**
> concrètement l'idée c'est qu'il faut choisir ce que l'on prédit entre grip_lost et l'autre?

**Translation:**
> Concretely, the idea is that we have to choose what we predict, between grip_lost and the other one?

**AI output (summary):** since the two labels are unrelated, a combined "fault" label would mix two different problems. Options: protective stops only (recommended by the AI: 108 events vs. 77), grip losses only, two separate models, or a 3-class model (not recommended for a 4-week project).

**Human decision:** the team chose **grip_lost** as the target, against the AI recommendation, and asked the AI to update the specification accordingly. Consequence accepted: fewer events (11 in validation, 13 in test), so results will be reported with confidence intervals and objectives are defined relative to the baseline.
