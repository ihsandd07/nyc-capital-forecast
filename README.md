# NYC Capital Project Forecasts

Forecasting budget growth and schedule slippage on New York City capital projects from public NYC Open Data, with honest validation.

Python, pandas, scikit-learn, SciPy, Matplotlib. 10 city reports (May 2023 to May 2026), three public datasets, walk-forward testing with a gap,
conformal prediction intervals, survival analysis, and permutation-based explanations. Everything regenerates with one command.

## Results at a glance

Tested on the last two reports that have a 12-month outcome (forecasts made in Jan and May 2025). All intervals are 95% bootstrap intervals that resample projects.

| Question | Result |
|---|---|
| Can we forecast how far an open project's completion date will slip in 12 months? | Yes, partly. Gradient boosting lowers mean absolute error by **16.0%** (95% CI 13.8% to 18.2%) versus assuming the date holds. The median project slipped 133 days. The L1 linear model did not beat the baseline. |
| Can we forecast the change in a budget line's planned budget in 12 months? | Barely. 45% of lines do not change. The best model lowers error by only **0.3%** (95% CI -0.5% to -0.2%) versus "no change". |
| Can we flag budget lines likely to rise 10% or more? | Moderately. AUC **0.70** (95% CI 0.68 to 0.72); 12% of test lines rise that much. |
| Do 80% prediction intervals cover 80% of held-out projects? | Budget: **81%** yes. Schedule: **72%** (conformalized quantile regression) and 78% (split conformal), because the slip rate drifts between calibration and test. |
| How long do projects take to complete? | Kaplan-Meier on 3,123 projects (2,376 still open, treated as censored): **9.5%** complete within 12 months, 31% within 36 months. |
| Does machine learning beat the agency's own forecast date at predicting near-term completion? | Not significantly. Own forecast date AUC 0.85; gradient boosting 0.87 (difference +0.015, 95% CI -0.003 to +0.033). |

Read [`reports/report.pdf`](reports/report.pdf) (9 pages) or open [`reports/dashboard.html`](reports/dashboard.html) in a browser (charts, a sortable forecast table for every open project and budget line, and the method).
Every number comes from `reports/results.json`; `checks/verify_results.py` recomputes the headline numbers independently.

## Why the cost target is "planned budget growth" and not "overrun"

The public files do not contain each project's original approved budget, and spending is recorded only for past fiscal years. The target is the change
in a budget line's total planned budget between one city report and the one three reports later (12 months). That includes real increases, and also planned amounts shifting between fiscal years.
Calling it an overrun would overstate what the data can show.

## How it works

1. **Load and clean** (`nycforecast/load.py`). Three CSVs; every cleaning step is counted in `results.json > data_quality`.
2. **Build forecasting problems** (`nycforecast/panel.py`). One row per project per report. Features use only reports up to that date. The label is what happened three reports (12 months) later.
3. **Validate with a gap** (`panel.walk_forward_split`). Train on older reports, calibrate intervals on a newer one, test on a later one. Every training and calibration label was already known at the test date; a unit test enforces this.
4. **Models** (`nycforecast/models.py`). "No change" and agency-median baselines, an L1-penalized median regression, gradient boosting. Hyperparameters were fixed in advance and never tuned.
5. **Intervals**. Conformalized quantile regression (named as the primary method before the test run) and symmetric split conformal.
6. **Survival** (`nycforecast/survival.py`). Kaplan-Meier with Greenwood bands, plus a landmark model for "completes within 12 months".
7. **Explanations**. Permutation importance on the test set and partial dependence.
8. **Outputs**. `report.py` builds the PDF; `dashboard.py` builds the HTML page.

## Run it

```bash
pip install -r requirements.txt
# download the three CSVs into data/raw/ (see data/README.md)
python run_all.py          # a few minutes on a laptop (the model fits and bootstraps dominate)
python -m pytest -q        # 15 tests
python checks/verify_results.py
```

Tested with Python 3.13, pandas 3.0, NumPy 2.5, scikit-learn 1.9, SciPy 1.18, Matplotlib 3.11.

## Limitations and analysis log

- Budget growth is not final cost overrun. The cost target is the change in each project's total planned budget between city reports. Spending is only recorded for past fiscal years, and the public files do not contain the original approved budget. Planned amounts also move between fiscal years as the planning window rolls forward.
- Short history. There are 10 reports (3 years). Only the last two forecast origins (reports of Jan and May 2025) are held out, and the same projects appear in both, so results come from two correlated test windows. Bootstrap confidence intervals resample projects, not time.
- Drift. The median 12-month slip changed from 0 to 150 days across origins. Models trained on older reports therefore under-predict recent slippage, and conformal intervals (which assume exchangeable data) under-cover for schedule.
- Projects that vanish. About 15% of cost rows and 12% of schedule rows disappear from later reports without a completion record. They cannot be labelled, so they are excluded; the results describe projects that stay in the reports.
- Self-reported data. Agencies report their own forecasts; the NYC Comptroller and State Comptroller have criticised the quality of public capital data. Nothing here is audited.
- School-construction phases are described only. Completed phases are a biased sample (long phases are still open), and 'planned end' is missing or not a date for many programmes, so no forecasting claim is made from that file.
- Drivers are associations, not causes. Permutation importance and partial dependence show what the models use, not what causes delay or cost growth.
- Analysis log. Hyperparameters were fixed in advance and never tuned. The large-budget-increase classifier was added after seeing that the point forecasts had almost no skill, and is reported regardless of its result. Intervals: CQR was named as the primary method before the test run; split conformal is reported alongside it. The linear model underperformed and is reported as is.

## Layout

```
nycforecast/   load.py  panel.py  models.py  evaluate.py  survival.py  analysis.py  figures.py  report.py  dashboard.py
tests/         cleaning, label construction, no-look-ahead, walk-forward timing, Kaplan-Meier, conformal coverage
checks/        verify_results.py (independent recomputation of the headline numbers)
reports/       results.json, tables/*.csv, report.pdf, dashboard.html
data/          README.md (how to get the CSVs); raw files are not committed
```
