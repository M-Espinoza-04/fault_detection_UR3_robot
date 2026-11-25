# PROMPT_LOG – entry to append (Step 2: standard solutions)

> Copy this entry at the end of `ai-governance/PROMPT_LOG.md` once the team has checked the boxes.

## 6. Step 2 – baseline, learning plan, state of the art and first AutoML

- **Date:** 25/11/2026
- **Decision category:** pre-processing, data splitting and leakage prevention, model calibration, metric choice, model selection and hyper-parameter tuning

**Prompt (verbatim):**
> I was planning to start a project and have a baseline code so we can make it better on the next phases. we have this for the moment, try to read everything in the folder. use the next steps to advance. all the changes or new files put it in a new folder. do the code and an explicative word file. 1. Analyse the raw data (check quality, statistical information, define variables, imbalancing data, correlation analysis, reduction, etc) and propose pre-processing with justifications 2. Define the learning and testing plan: choice of data, construction of sets and evaluation methods for controlling overfitting, underfitting, data leakage 3. Implement your baseline solution, justify the model calibration 4. Identify your benchmark models and their performance from the state-of-the-art 5. Implement your first autoML pipeline, define your evaluation protocol, risks and failure modes, and a prioritized plan for the remaining work

**AI output (summary):** the folder `step2_standard_solutions/` (scripts `01`–`05`, package `src/`, figures and tables in `outputs/`) and the Word report. Main choices made by the AI and to be defended by the team:
- label debouncing (events separated by ≤ 2 normal rows are merged: 77 → 72 events);
- early-warning target with N = 3 s, ongoing-loss rows removed, causal features only;
- chronological 70/15/15 split snapped to cycle starts with an 8-row embargo;
- *event-balanced* expanding-window CV instead of the plain `TimeSeriesSplit` (one plain fold contained a single event);
- baseline = logistic regression, C chosen with a *paired* one-standard-error rule;
- two operating points: F1-optimal and "≤ 30 false alarms per hour" (chosen on validation);
- AutoML v0.1 = randomized search over 7 scikit-learn families with the feature blocks as hyper-parameters; final model chosen on validation; test used once.

**Human verification (to tick before the oral defence):**
- [ ] Re-run `python run_all.py` from a clean environment and compare the tables with the report
- [ ] Check the debouncing rule (≤ 2 rows) against the physical meaning of `grip_lost` (gripper signal? operator label?)
- [ ] Check that `secs_since_last_gl` (burst feature) would be available in real time on the cell (is `grip_lost` known at the second it happens?)
- [ ] Check in the scikit-learn docs that `RandomizedSearchCV` accepts a list of (train, test) index pairs as `cv`
- [ ] Read Liu (2025) to confirm the figures quoted in the literature table (page not reachable during this session)
- [ ] Decide whether 30 false alarms per hour is an acceptable alert budget for an operator

**Decision:** _to be filled by the team._
