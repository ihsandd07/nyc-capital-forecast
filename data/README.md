# Data

All inputs are public NYC Open Data files. They are not committed (they are large and the city updates them),
so download them and save them in `data/raw/` with these names. Exported 2026-11-09.

| File to save as | Dataset | Where to get it |
|---|---|---|
| `project_schedules_budgets.csv` | Capital Project Schedules and Budgets | https://data.cityofnewyork.us/d/2xh6-psuq (Export > CSV) |
| `schedule_history_variance.csv` | Capital Projects Dashboard - Citywide Schedule History and Variance | https://catalog.data.gov/dataset/capital-projects-dashboard-citywide-schedule-history-and-variance (follow the NYC Open Data link, Export > CSV) |
| `budget_spend_by_fy.csv` | Capital Projects Dashboard - Citywide Budget and Spend by Fiscal Year | https://catalog.data.gov/dataset/capital-projects-dashboard-citywide-budget-and-spend-by-fiscal-year (follow the NYC Open Data link, Export > CSV) |

The city revises these files. To reproduce the exact numbers in this repository, use the snapshot with these SHA-256 hashes:

```
a3e0a1b613c0f3114c33b29ff635ebe9bcf25d1c66c829359c33fb0a637038d4  budget_spend_by_fy.csv
b94ce7d45c99563e6871bf1787a51b9d4595bdd4f89b35ab7edf470703fba9f6  project_schedules_budgets.csv
207497de0875631794dec067e32b9a4a309cba03c356ab4ead29bc19eaa9a006  schedule_history_variance.csv
```

A newer export will contain newer city reports, so results will differ. Check the terms of use on each dataset page before redistributing the files.

## What the columns mean (as used here)

- **Budget and Spend by Fiscal Year**: one row per budget line (agency + FMS ID), fiscal year and city report. "Total Budget" is the planned
  budget for that fiscal year as of that report; "Spend" is recorded only for past and current fiscal years.
- **Schedule History and Variance**: one row per project and city report. "Variance (day)" is the change in the forecast completion date since the
  previous report (verified: it equals the date difference for all 14,071 consecutive report pairs). "Completion Date Type" is `Forecast` until a project completes, then `Actual`.
- **Capital Project Schedules and Budgets**: School Construction Authority projects, one row per phase.
