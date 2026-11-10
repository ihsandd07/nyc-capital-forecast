"""Independent re-verification of the numbers in reports/results.json.

Usage: python checks/verify_results.py
Each check recomputes a result through a different code path (raw CSVs, plain pandas/numpy, a hand-written
product-limit estimator) and compares it with what the pipeline saved. Prints OK / MISMATCH and exits non-zero on any mismatch.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nycforecast import config, evaluate, load, panel  # noqa: E402

R = json.loads((config.REPORTS / "results.json").read_text())
T = ROOT / "reports" / "tables"
bad = []


def check(name, got, want, tol=1e-9):
    ok = abs(got - want) <= tol
    print(("OK       " if ok else "MISMATCH ") + f"{name}: recomputed={got:.10g} saved={want:.10g}")
    if not ok:
        bad.append(name)


# 1. point metrics and interval coverage from the saved test predictions
for tgt, f in [("cost", "cost_test_predictions.csv"), ("schedule", "schedule_test_predictions.csv")]:
    p = pd.read_csv(T / f)
    for m in ["zero", "agency", "linear", "gbm"]:
        check(f"{tgt} MAE {m}", (p.y - p["pred_" + m]).abs().mean(), R[tgt]["metrics"][m]["MAE"])
    for k in ["cqr", "sc", "raw"]:
        cov = ((p.y >= p[f"lo_{k}"]) & (p.y <= p[f"hi_{k}"])).mean()
        check(f"{tgt} coverage {k}", cov, R[tgt]["coverage"][k]["coverage"])
    pt = ((p.y - p.pred_gbm).abs().sum() / (p.y - p.pred_zero).abs().sum()) - 1
    check(f"{tgt} gbm MAE change vs zero", pt, R[tgt]["mae_change_vs_zero"]["gbm"]["point"])
    check(f"{tgt} median y", p.y.median(), R[tgt]["test_median_y"])

# 2. independent bootstrap (different seed, plain loops over resampled projects) agrees with the saved CI
p = pd.read_csv(T / "schedule_test_predictions.csv")
rng = np.random.default_rng(123)
g = p.assign(eg=(p.y - p.pred_gbm).abs(), ez=(p.y - p.pred_zero).abs()).groupby("unit")[["eg", "ez"]].sum()
vals = []
for _ in range(1000):
    s = g.iloc[rng.integers(0, len(g), len(g))]
    vals.append(s.eg.sum() / s.ez.sum() - 1)
lo, hi = np.quantile(vals, [0.025, 0.975])
check("schedule gbm CI low (different seed, tol 0.003)", lo, R["schedule"]["mae_change_vs_zero"]["gbm"]["lo"], 3e-3)
check("schedule gbm CI high (different seed, tol 0.003)", hi, R["schedule"]["mae_change_vs_zero"]["gbm"]["hi"], 3e-3)

# 3. classifier metrics from saved predictions, using hand-written AUC (rank-sum) and Brier
from scipy.stats import rankdata  # noqa: E402
for tgt, f, key in [("cost large increase", "cost_large_increase_test_predictions.csv", R["cost_big_up"]),
                    ("schedule completion", "schedule_completion_test_predictions.csv", R["survival"]["landmark"])]:
    p = pd.read_csv(T / f)
    for m in ["rule", "gbm"]:
        r = rankdata(p["p_" + m]); n1 = p.y.sum(); n0 = len(p) - n1
        auc = (r[p.y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        saved = next(x for x in key["metrics"] if x["model"] == m)
        check(f"{tgt} AUC {m}", auc, saved["auc"], 1e-9)
        check(f"{tgt} Brier {m}", ((p["p_" + m] - p.y) ** 2).mean(), saved["brier"], 1e-6)

# 4. Kaplan-Meier from the raw CSV with a hand-written product-limit loop
raw = pd.read_csv(config.FILES["schedule"]); raw.columns = [c.strip() for c in raw.columns]
raw["unit"] = raw["Managing Agency"].str.strip() + "|" + raw["PID"].astype(str).str.strip()
codes = sorted(raw["Reporting Period"].unique()); idx = {c: i for i, c in enumerate(codes)}
raw["t"] = raw["Reporting Period"].map(idx)
dur, ev = [], []
for u, d in raw.sort_values("t").groupby("unit"):
    if d["Completion Date Type"].iloc[0] != "Forecast":
        continue
    done = (d["Completion Date Type"] == "Actual").any()
    t_end = d.loc[d["Completion Date Type"] == "Actual", "t"].iloc[0] if done else d["t"].iloc[-1]
    dur.append(t_end - d["t"].iloc[0]); ev.append(int(done))
dur, ev = np.array(dur), np.array(ev)
S = 1.0
for t in range(1, 10):
    at_risk = (dur >= t).sum(); k = ((dur == t) & (ev == 1)).sum()
    S *= 1 - k / at_risk
    if t == 3: S3 = S
    if t == 9: S9 = S
check("KM cohort size", len(dur), R["survival"]["cohort"], 0)
check("KM events", ev.sum(), R["survival"]["events"], 0)
check("KM P(complete within 3 reports)", 1 - S3, R["survival"]["p_complete_3"], 1e-9)
check("KM P(complete within 9 reports)", 1 - S9, R["survival"]["p_complete_9"], 1e-9)

# 5. labels recomputed by hand from the raw files for random rows
budget, pt, _ = load.load_budget(); sched, _, _ = load.load_schedule()
cp = panel.cost_panel(budget, len(pt)); sp = panel.schedule_panel(sched, len(pt))
rng = np.random.default_rng(7)
raw_b = pd.read_csv(config.FILES["budget"]); raw_b.columns = [c.strip() for c in raw_b.columns]
raw_b["amt"] = pd.to_numeric(raw_b["Total Budget City Non City"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
raw_b["unit"] = raw_b["Managing Agency"].str.strip() + "|" + raw_b["FMS ID"].astype(str).str.strip()
sample = cp[cp.y.notna()].sample(40, random_state=1)
worst = 0
for r in sample.itertuples():
    now = raw_b[(raw_b.unit == r.unit) & (raw_b["Reporting Period"] == codes[r.origin])].amt.sum()
    later = raw_b[(raw_b.unit == r.unit) & (raw_b["Reporting Period"] == codes[r.origin + 3])].amt.sum()
    worst = max(worst, abs(np.log(later / now) - r.y))
check("cost label, 40 random rows (max abs diff)", worst, 0.0, 1e-9)
raw["cd"] = pd.to_datetime(raw["Completion Date"], format="%m/%d/%Y", errors="coerce")
sample = sp[sp.y.notna()].sample(40, random_state=1)
worst = 0
for r in sample.itertuples():
    a = raw[(raw.unit == r.unit) & (raw.t == r.origin)].cd.iloc[0]
    b = raw[(raw.unit == r.unit) & (raw.t == r.origin + 3)].cd.iloc[0]
    worst = max(worst, abs((b - a).days - r.y))
check("schedule label, 40 random rows (max abs diff)", worst, 0.0, 0)

# 6. look-ahead on real data: delete every report after the origin; features at the origin must not change
feat_c, feat_s = panel.COST_NUM, panel.SCHED_NUM
for o in (3, 5):
    bt = budget[budget.t <= o]; st = sched[sched.t <= o]
    c2 = panel.cost_panel(bt, o + 1); s2 = panel.schedule_panel(st, o + 1)
    a = cp[cp.origin == o].set_index("unit")[feat_c].sort_index(); b = c2[c2.origin == o].set_index("unit")[feat_c].sort_index()
    check(f"cost features at origin {o} unchanged when later reports are deleted (max abs diff)", float((a - b).abs().max().max()), 0.0, 1e-9)
    a = sp[sp.origin == o].set_index("unit")[feat_s].sort_index(); b = s2[s2.origin == o].set_index("unit")[feat_s].sort_index()
    # months_to_forecast uses the report's own data date; the later-report median is only used for labels
    check(f"schedule features at origin {o} unchanged when later reports are deleted (max abs diff)", float((a - b).abs().max().max()), 0.0, 1e-9)

# 7. saved outputs match what the current code produces (schedule is quick to refit)
heavy = ["slip_since_first", "max_delay_days", "sum_delay_days", "last_change", "months_to_forecast"]
fresh = evaluate.fit_predict_origin(sp, panel.SCHED_NUM, panel.SCHED_CAT, heavy, 6)
saved = pd.read_csv(T / "schedule_test_predictions.csv").query("origin == 6").reset_index(drop=True)
check("schedule gbm predictions: current code vs saved file (max abs diff)", float((fresh.reset_index(drop=True).pred_gbm - saved.pred_gbm).abs().max()), 0.0, 1e-6)
check("schedule CQR upper bound: current code vs saved file (max abs diff)", float((fresh.reset_index(drop=True).hi_cqr - saved.hi_cqr).abs().max()), 0.0, 1e-6)

# 8. train/test timing: every training and calibration label was known at the test origin
for o in config.TEST_ORIGINS:
    tr, ca, te = panel.walk_forward_split(cp, o)
    check(f"walk-forward {o}: latest label origin + horizon - test origin (<= 0)", float(max(tr.origin.max(), ca.origin.max()) + config.HORIZON - o), 0.0, 0.0)
    check(f"walk-forward {o}: training rows", len(tr), R["cost"]["n_train_by_origin"][str(o)], 0)

print("\nALL CHECKS PASSED" if not bad else f"\n{len(bad)} MISMATCHES: {bad}")
sys.exit(1 if bad else 0)
