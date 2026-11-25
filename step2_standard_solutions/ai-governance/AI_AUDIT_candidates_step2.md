# AI_AUDIT – candidate "catch & correct" instances found during Step 2

The guidelines ask for at least 3 concrete instances where AI-generated ML logic was flawed
and how it was corrected. The items below were found while coding Step 2, by checking earlier
AI-assisted outputs (Step 1 document, `step2_benchmark.ipynb`) against the actual file.
**The team must re-verify each one before copying it into `AI_AUDIT.md`.**

| # | Flawed AI output (where) | Why it is wrong | Correction (where) | Evidence |
|---|---|---|---|---|
| 1 | "2 duplicated rows (excluding the row number and the timestamp)" – Step 1 document, data-quality list | The duplicates are **empty rows** that share a cycle number (4 rows in 2 pairs); there is **no duplicated measurement** | Quality report says "0 duplicated measurements, nothing to remove" | `outputs/tables/s1_quality_report.csv` |
| 2 | "The cycle counter decreases once (232 → 224 at row 6,035)" – Step 1 document | The counter goes **233 → 224, between Num 6033 and Num 6034** | Computed in `01_data_analysis.py` | `s1_quality_report.csv` |
| 3 | `df.describe()` / univariate statistics in `step2_benchmark.ipynb` counted 7,363 values of `Current_J0` | 8 empty rows carry a **placeholder `Current_J0 = 1.0`** (all other sensors NaN). They were treated as real measurements | Set to NaN in `src/data.py::clean` | `s1_quality_report.csv`, row "Empty sensor rows" |
| 4 | "77 grip-loss events" used to size the splits | 5 events are separated from the previous one by only 1–2 normal rows (label flicker). Counting them as new events creates trivially easy positives (an alert right after a loss) | Debouncing (merge gaps ≤ 2 rows) → 72 events | `s1_fig07_event_gaps.png` |
| 5 | "TimeSeriesSplit with a gap" recommended for tuning (PROMPT_LOG entry 2) | With equal-row folds, fold 3 (around the lunch break) contains **1 event / 3 positive rows**: its PR-AUC is meaningless and dominates the variance | Event-balanced expanding-window CV (`src/splits.py::event_balanced_cv`), 8–9 events per fold | `outputs/tables/s2_cv_folds.csv` |
| 6 | One-standard-error rule for choosing C (first version of `03_baseline.py`, AI-written) | With ~8 events per fold the unpaired SE is larger than the whole validation curve, so the rule always picked the strongest penalty (C = 0.001) → validation PR-AUC = prevalence (no skill) | Paired 1-SE rule on fold differences → C = 0.316 | `outputs/tables/s3_validation_curve_C.csv` |
| 7 | IQR rule proposed for outlier detection (first version of `01_data_analysis.py`, AI-written) | Speeds are zero ~50 % of the time, IQR ≈ 0, so the rule flagged 31–45 % of the speed values as outliers | Replaced by "beyond 5 sd" + physical limits; no outlier removal | `s1_descriptive_stats.csv` |
| 8 | First version of `05_automl.py` (AI-written) refitted the *template* pipeline of the search space in place (`SPACE[fam][0].set_params(...)`) | The ablation then changed the object that also held the final model: the horizon study and the permutation importance silently used the "without position" variant. Detected because the N = 3 s value of the horizon table (0.578) did not match the final validation score (0.604) | `sklearn.base.clone` before every `set_params` | git diff of `05_automl.py` |

Template for each entry in `AI_AUDIT.md`: *prompt → flawed output (quote) → how we detected it → corrected code (commit hash) → effect on the results.*
