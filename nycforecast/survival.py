"""Survival analysis for time to completion, plus a landmark model for 'complete within the horizon'.

Why survival analysis: most projects are still open when the data ends. Dropping them (or calling them "not complete")
biases completion times. Kaplan-Meier uses every project, treating the open ones as censored at their last report.
"""
import numpy as np
import pandas as pd

from . import config, evaluate, models
from .panel import walk_forward_split


def project_durations(sched: pd.DataFrame) -> pd.DataFrame:
    """One row per project that was still open (a forecast) at its first report.
    duration = reports between the first report and completion (event) or the last report (censored)."""
    s = sched.sort_values(["unit", "t"])
    first = s.groupby("unit").first()
    last = s.groupby("unit").last()
    open_at_start = first.type == "Forecast"
    d = pd.DataFrame({"agency": first.agency, "t_first": first.t, "t_last": last.t,
                      "completed": (last.type == "Actual").astype(int)})
    d = d[open_at_start.reindex(d.index)]
    d["duration"] = d.t_last - d.t_first
    d["event"] = d.completed
    return d.reset_index()


def kaplan_meier(durations, events) -> pd.DataFrame:
    """Kaplan-Meier survival curve S(t) with Greenwood 95% confidence limits (log-log transform)."""
    d = pd.DataFrame({"t": np.asarray(durations), "e": np.asarray(events)})
    times = np.sort(d.loc[d.e == 1, "t"].unique())
    rows, S, var_sum = [], 1.0, 0.0
    for t in times:
        n = int((d.t >= t).sum())
        k = int(((d.t == t) & (d.e == 1)).sum())
        S *= 1 - k / n
        if n > k:
            var_sum += k / (n * (n - k))
        if S > 0 and var_sum > 0:
            se_ll = np.sqrt(var_sum) / abs(np.log(S))
            lo, hi = S ** np.exp(1.96 * se_ll), S ** np.exp(-1.96 * se_ll)
        else:
            lo = hi = S
        rows.append({"t": int(t), "at_risk": n, "events": k, "survival": S, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def surv_at(km: pd.DataFrame, t: int) -> float:
    """S(t) read from a Kaplan-Meier table (step function)."""
    sub = km[km.t <= t]
    return 1.0 if sub.empty else float(sub.survival.iloc[-1])


# ----------------------------------------------------------------------------- landmark model
def landmark_walk_forward(panel, label: pd.Series, num, cat, heavy, rule_cols, test_origins=config.TEST_ORIGINS,
                          H=config.HORIZON):
    """Predict a 0/1 label at each test origin from models trained on older origins.
    Models: base rate, a one-or-two-feature 'rule' logistic baseline, a full logistic regression, gradient boosting."""
    p2 = panel.copy()
    p2["y"] = label
    out = []
    for o in test_origins:
        train, cal, test = walk_forward_split(p2, o, H)
        train = pd.concat([train, cal])  # no interval calibration is needed for classifiers, so use the extra data
        # keep the gap rule: calibration origin is o-H whose label is known at o, so this is still leak-free
        ytr = train.y.astype(int).to_numpy()
        f = test[["unit", "agency", "origin", "y"]].copy()
        f["p_base"] = ytr.mean()
        rule = models.logit_classifier(rule_cols, [], []).fit(train[rule_cols], ytr)
        f["p_rule"] = rule.predict_proba(test[rule_cols])[:, 1]
        lg = models.logit_classifier(num, cat, heavy).fit(train[num + cat], ytr)
        f["p_logit"] = lg.predict_proba(test[num + cat])[:, 1]
        gb = models.gbm_classifier(num, cat, heavy).fit(train[num + cat], ytr)
        f["p_gbm"] = gb.predict_proba(test[num + cat])[:, 1]
        out.append(f)
    return pd.concat(out, ignore_index=True)


def landmark_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_brier = None
    for m in ["base", "rule", "logit", "gbm"]:
        r = evaluate.clf_metrics(pred.y.astype(int).to_numpy(), pred["p_" + m].to_numpy())
        r["model"] = m
        rows.append(r)
    df = pd.DataFrame(rows)
    b = df.loc[df.model == "base", "brier"].iloc[0]
    df["brier_skill"] = 1 - df.brier / b
    return df


def bootstrap_auc_diff(pred, a="gbm", b="rule", n_boot=config.N_BOOT, seed=config.RANDOM_STATE):
    """Clustered (by project) bootstrap of AUC(a) - AUC(b)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    groups = {u: g.index.to_numpy() for u, g in pred.groupby("unit")}
    units = np.array(list(groups))
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(units, size=len(units), replace=True)
        idx = np.concatenate([groups[u] for u in pick])
        y = pred.y.to_numpy()[idx]
        if y.min() == y.max():
            continue
        diffs.append(roc_auc_score(y, pred["p_" + a].to_numpy()[idx]) - roc_auc_score(y, pred["p_" + b].to_numpy()[idx]))
    return float(np.mean(diffs)), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def calibration_bins(y, p, n_bins=5) -> pd.DataFrame:
    """Reliability table: predicted probability versus observed frequency in quantile bins."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d.p.rank(method="first"), n_bins, labels=False)
    return d.groupby("bin").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index()
