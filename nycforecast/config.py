"""Fixed settings. Nothing here is tuned on test results."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

FILES = {
    "budget": RAW / "budget_spend_by_fy.csv",
    "schedule": RAW / "schedule_history_variance.csv",
    "sca": RAW / "project_schedules_budgets.csv",
}

# Forecast horizon in reporting periods. Reports come out three times a year (May, Sept, Jan),
# so 3 periods is about 12 months.
HORIZON = 3
# Origins (report dates we forecast FROM) held out as the test set. Origin index 0 is the first report (2023-05).
# With 10 reports, origins 0..6 have a label 3 reports later; the last two (5, 6) are the test origins.
TEST_ORIGINS = (5, 6)
ALPHA = 0.20            # miscoverage for prediction intervals (80% central interval)
RANDOM_STATE = 42
N_BOOT = 1000           # bootstrap resamples (clustered by project)
N_PERM = 10             # permutation-importance repeats

# Model hyperparameters are fixed in advance (not tuned), so no test or calibration data is used for selection.
HGB_PARAMS = dict(max_depth=4, learning_rate=0.05, max_iter=200, min_samples_leaf=50,
                  l2_regularization=1.0, random_state=RANDOM_STATE)
QR_ALPHA = 1e-3         # L1 penalty for the regularized median regression
MAX_COST_DAYS = None
