# Step 2 – Standard solutions (baseline + AutoML v0.1)

Grip-loss early warning on a UR3 cobot (UCI *UR3 CobotOps*). Task: every second, predict whether a
grip loss **starts within the next N = 3 s** from the joint and gripper signals, under a
chronological evaluation. Full explanations, figures and tables: `report/Step2_Standard_Solutions_Report.docx`.

## Run

```bash
pip install -r requirements.txt
python run_all.py           # ~25 min on 2 cores
python run_all.py --fast    # smaller AutoML budget (~5 min for step 5)
```

The data file is read from `../UR3CobotOps.csv` (repository root). Another path can be given with
the environment variable `UR3_CSV`. Scripts contain `# %%` cells and open as notebooks in VS Code.

To rebuild the Word report after a run: `python report/collect_tables.py && node report/build_report.js`
(needs the `docx` npm package).

## Contents

| Path | What it does |
|---|---|
| `01_data_analysis.py` | quality report, variable dictionary, statistics, imbalance, missingness tests, correlation, VIF, PCA, drift, target association, pre-processing proposal |
| `02_learning_plan.py` | chronological split with embargo, CV folds (plain vs event-balanced), learning curves, leakage experiments |
| `03_baseline.py` | B0 prior, B1 cycle position, B2 logistic regression; choice of C, probability calibration, thresholds, test results |
| `04_sota_benchmark.py` | literature table and re-run of the published protocols on our data |
| `05_automl.py` | AutoML v0.1: randomized search over 7 families with feature blocks as hyper-parameters, champions, final choice on validation, test once, ablation, horizon sensitivity, permutation importance |
| `src/` | shared code: config, cleaning and target, causal features, splits, metrics, pipeline factory, baselines |
| `outputs/figures`, `outputs/tables`, `outputs/models` | everything the scripts produce |
| `ai-governance/` | draft PROMPT_LOG entry and candidate AI_AUDIT items for this phase |
| `report/` | Word report and the scripts that build it |

## Protocol in one paragraph

Empty rows kept in the time line (never predicted), label debounced (gaps ≤ 2 rows merged → 72 events),
target = onset in (t, t+3], rows with an ongoing loss removed, causal features only. Chronological
70/15/15 split cut at cycle starts with an 8-row embargo (50 / 12 / 9 events). Tuning with an
event-balanced expanding-window CV inside the training period. Everything learned from data is inside
the scikit-learn pipeline. Thresholds and the final model are chosen on validation; the test period is
evaluated once. Main metrics: PR-AUC and event recall at ≤ 30 false alarms per hour, with 95 %
intervals from a bootstrap over cycles.
