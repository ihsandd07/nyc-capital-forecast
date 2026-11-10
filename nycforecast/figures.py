"""All figures. Each function takes the results dictionary (and prediction tables where needed) and returns a Figure."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from . import config

BLUE, ORANGE, GRAY, GREEN, RED = "#2a6fdb", "#d9731a", "#7d8590", "#2a9d5c", "#c0392b"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "figure.dpi": 120, "savefig.dpi": 200})
NICE = {"agency": "Agency median", "linear": "Regularized linear", "gbm": "Gradient boosting",
        "base": "Base rate", "rule": "Simple rule", "logit": "Logistic regression"}


def load_results():
    return json.loads((config.REPORTS / "results.json").read_text())


def _tbl(name):
    return pd.read_csv(config.REPORTS / "tables" / name)


def fig_point(R, key, title, baseline_label):
    d = R[key]["mae_change_vs_zero"]
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    names = ["agency", "linear", "gbm"]
    pts = np.array([d[m]["point"] for m in names]) * 100
    lo = np.array([d[m]["lo"] for m in names]) * 100
    hi = np.array([d[m]["hi"] for m in names]) * 100
    ypos = np.arange(len(names))[::-1]
    cols = [GREEN if h < 0 else GRAY for h in hi]
    ax.barh(ypos, pts, color=cols, height=0.55)
    ax.errorbar(pts, ypos, xerr=[pts - lo, hi - pts], fmt="none", ecolor="black", capsize=3, lw=1)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos, [NICE[m] for m in names])
    ax.set_xlabel(f"Change in mean absolute error vs '{baseline_label}' (%)  (95% bootstrap CI; left = better)")
    for y, p, l, h in zip(ypos, pts, lo, hi):
        ax.text(1.02, y, f"{p:+.1f}%  [{l:+.1f}, {h:+.1f}]", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


def fig_coverage(R):
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    methods = [("raw", "Raw quantile band"), ("sc", "Split conformal"), ("cqr", "Conformalized quantile (CQR)")]
    cols = [GRAY, ORANGE, BLUE]
    width = 0.24
    for j, (k, lab) in enumerate(methods):
        vals, errs = [], []
        for tgt in ["cost", "schedule"]:
            c = R[tgt]["coverage"][k]
            vals.append(c["coverage"] * 100)
            errs.append([(c["coverage"] - c["ci"][0]) * 100, (c["ci"][1] - c["coverage"]) * 100])
        x = np.arange(2) + (j - 1) * width
        ax.bar(x, vals, width=width, color=cols[j], label=lab)
        ax.errorbar(x, vals, yerr=np.array(errs).T, fmt="none", ecolor="black", capsize=2, lw=0.8)
        for xi, v in zip(x, vals):
            ax.text(xi, 2, f"{v:.0f}%", ha="center", color="white", fontsize=8, fontweight="bold")
    ax.axhline(100 * (1 - R["meta"]["alpha"]), color="black", ls="--", lw=1)
    ax.text(1.5, 100 * (1 - R["meta"]["alpha"]) + 1.5, "80% target", ha="right", fontsize=8)
    ax.set_xticks([0, 1], ["Cost (12-month budget change)", "Schedule (12-month date slip)"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Test projects inside the interval (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    ax.set_title("Do the 80% intervals cover 80% of held-out projects?", loc="left", pad=22)
    fig.tight_layout()
    return fig


def _roc_panel(ax, pred, models_, labels):
    for m, c in zip(models_, [GRAY, BLUE, ORANGE]):
        fpr, tpr, _ = roc_curve(pred.y.astype(int), pred["p_" + m])
        from sklearn.metrics import roc_auc_score
        ax.plot(fpr, tpr, color=c, lw=1.6, label=f"{labels[m]} (AUC {roc_auc_score(pred.y.astype(int), pred['p_' + m]):.2f})")
    ax.plot([0, 1], [0, 1], color="black", lw=0.6, ls=":")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(frameon=False, fontsize=8, loc="lower right")


def _calib_panel(ax, calib, title):
    cal = pd.DataFrame(calib)
    ax.plot([0, max(cal.pred.max(), cal.obs.max()) * 1.1], [0, max(cal.pred.max(), cal.obs.max()) * 1.1], color="black", lw=0.6, ls=":")
    ax.plot(cal.pred, cal.obs, "o-", color=BLUE, lw=1.4)
    ax.set_xlabel("Predicted probability (bin average)"); ax.set_ylabel("Observed frequency"); ax.set_title(title, loc="left")


def fig_cost_classifier(R):
    pred = _tbl("cost_large_increase_test_predictions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    _roc_panel(axes[0], pred, ["rule", "logit", "gbm"], {"rule": "Prior-change rule", "logit": "Logistic regression", "gbm": "Gradient boosting"})
    axes[0].set_title("Planned budget up 10%+ within 12 months", loc="left")
    _calib_panel(axes[1], R["cost_big_up"]["calibration_gbm"], "Calibration (gradient boosting)")
    fig.tight_layout()
    return fig


def fig_slip_drift(R):
    d = R["schedule_panel"]["median_slip_by_origin"]
    xs = sorted(int(k) for k in d)
    codes = R["meta"]["periods"]
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.plot(xs, [d[str(x)] for x in xs], "o-", color=ORANGE, lw=1.6)
    for x in xs:
        ax.text(x, d[str(x)] + 6, f"{d[str(x)]:.0f}", ha="center", fontsize=8)
    for o in R["meta"]["test_origins"]:
        ax.axvspan(o - 0.4, o + 0.4, color=BLUE, alpha=0.12)
    ax.set_xticks(xs, [str(codes[x]) for x in xs], rotation=0, fontsize=7.5)
    ax.set_ylabel("Median slip over next 12 months (days)")
    ax.set_xlabel("Report the forecast is made from (shaded = test)")
    ax.set_title("Forecast completion dates keep slipping, and the rate drifts", loc="left")
    fig.tight_layout()
    return fig


def fig_km(R):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    km = pd.DataFrame(R["survival"]["km"])
    t = np.r_[0, km.t]
    surv, lo, hi = np.r_[1, km.survival], np.r_[1, km.lo], np.r_[1, km.hi]
    ax = axes[0]
    ax.step(t, surv, where="post", color=BLUE, lw=1.8)
    ax.fill_between(t, lo, hi, step="post", color=BLUE, alpha=0.18)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Reports since first seen as open (about 4 months each)")
    ax.set_ylabel("Share of projects still open")
    ax.set_title(f"All {R['survival']['cohort']:,} projects ({R['survival']['censored']:,} still open)", loc="left")
    ax = axes[1]
    tables = R.get("km_by_agency_tables", {})
    for (a, k), c in zip(tables.items(), [BLUE, ORANGE, GREEN, RED, GRAY, "#8e44ad"]):
        k = pd.DataFrame(k)
        ax.step(np.r_[0, k.t], np.r_[1, k.survival], where="post", color=c, lw=1.4, label=f"{a} (n={R['survival']['by_agency'][a]['n']})")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_xlabel("Reports since first seen as open")
    ax.set_title("By agency (six largest)", loc="left")
    fig.tight_layout()
    return fig


def fig_landmark(R):
    pred = _tbl("schedule_completion_test_predictions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    _roc_panel(axes[0], pred, ["rule", "logit", "gbm"], {"rule": "Own forecast date only", "logit": "Logistic regression", "gbm": "Gradient boosting"})
    axes[0].set_title("Completes within 12 months", loc="left")
    _calib_panel(axes[1], R["survival"]["landmark"]["calibration_gbm"], "Calibration (gradient boosting)")
    fig.tight_layout()
    return fig


def fig_drivers(R):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, key, col, title, xl in [(axes[0], "schedule_mae_increase", "mae_increase", "Schedule slip model", "MAE increase when shuffled (days)"),
                                    (axes[1], "cost_big_up_auc_drop", "auc_drop", "Large-budget-increase model", "AUC drop when shuffled")]:
        d = pd.DataFrame(R["drivers"][key]).head(7).iloc[::-1]
        ax.barh(d.feature, d[col], xerr=d["sd"], color=BLUE, ecolor="black", capsize=2, height=0.6)
        ax.set_title(title, loc="left"); ax.set_xlabel(xl)
    fig.tight_layout()
    return fig


def fig_pdp(R):
    pd_ = R["partial_dependence"]
    items = [("schedule:months_to_forecast", "Months to forecast date", "Predicted slip (days)"),
             ("schedule:n_delays", "Past delays (count)", "Predicted slip (days)"),
             ("cost_big_up:log_total", "log(total planned budget)", "P(budget up 10%+)"),
             ("cost_big_up:future_share", "Share of budget in future years", "P(budget up 10%+)")]
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.4))
    for grp in (items[:2], items[2:]):          # same y-scale inside each pair so effect sizes can be compared
        ys = np.concatenate([pd_[k]["y"] for k, _, _ in grp])
        pad = (ys.max() - ys.min()) * 0.08
        lim = (ys.min() - pad, ys.max() + pad)
        for k, _, _ in grp:
            axes[items.index(next(i for i in items if i[0] == k))].set_ylim(*lim)
    for ax, (k, xl, yl) in zip(axes, items):
        ax.plot(pd_[k]["x"], pd_[k]["y"], color=BLUE, lw=1.6)
        ax.set_xlabel(xl, fontsize=7.5); ax.set_ylabel(yl, fontsize=7.5); ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def make_all(R):
    return {
        "cost_point": fig_point(R, "cost", "Cost: 12-month change in planned budget", "no change"),
        "cost_classifier": fig_cost_classifier(R),
        "schedule_point": fig_point(R, "schedule", "Schedule: 12-month slip of forecast completion date", "date holds"),
        "slip_drift": fig_slip_drift(R),
        "coverage": fig_coverage(R),
        "km": fig_km(R),
        "landmark": fig_landmark(R),
        "drivers": fig_drivers(R),
        "pdp": fig_pdp(R),
    }
