"""Walk-forward evaluation, clustered bootstrap, coverage, and permutation importance."""
import time

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from . import config, models
from .panel import walk_forward_split


def fit_predict_origin(panel, num, cat, heavy, test_origin, H=config.HORIZON, alpha=config.ALPHA, return_models=False):
    """Train on old origins, calibrate on a newer one, predict the test origin. Returns a test frame with predictions."""
    train, cal, test = walk_forward_split(panel, test_origin, H)
    cols = num + cat
    Xtr, ytr = train[cols], train.y.to_numpy()
    Xca, yca = cal[cols], cal.y.to_numpy()
    Xte = test[cols]
    out = test[["unit", "agency", "origin", "y"]].copy()
    out["pred_zero"] = models.ZeroBaseline().fit(Xtr, ytr).predict(Xte)
    out["pred_agency"] = models.GroupMedian().fit(Xtr, ytr).predict(Xte)
    t0 = time.time()
    lin = models.linear_median(num, cat, heavy).fit(Xtr, ytr)
    out["pred_linear"] = lin.predict(Xte)
    g = models.gbm(num, cat, heavy).fit(Xtr, ytr)
    out["pred_gbm"] = g.predict(Xte)
    lo_m = models.gbm(num, cat, heavy, "quantile", alpha / 2).fit(Xtr, ytr)
    hi_m = models.gbm(num, cat, heavy, "quantile", 1 - alpha / 2).fit(Xtr, ytr)
    # primary interval: conformalized quantile regression
    lo, hi, q = models.cqr_interval(lo_m.predict(Xca), hi_m.predict(Xca), yca, lo_m.predict(Xte), hi_m.predict(Xte), alpha)
    out["lo_cqr"], out["hi_cqr"] = lo, hi
    out["cqr_q"] = q
    out["lo_raw"], out["hi_raw"] = lo_m.predict(Xte), hi_m.predict(Xte)   # uncalibrated quantile band, for comparison
    # secondary interval: symmetric split conformal around the gradient-boosting point forecast
    lo2, hi2, q2 = models.split_conformal_interval(g.predict(Xca), yca, g.predict(Xte), alpha)
    out["lo_sc"], out["hi_sc"] = lo2, hi2
    out.attrs.update(n_train=len(train), n_cal=len(cal), n_test=len(test), secs=round(time.time() - t0, 1))
    if return_models:
        return out, {"linear": lin, "gbm": g, "lo": lo_m, "hi": hi_m}, (train, cal, test)
    return out


MODELS = ["zero", "agency", "linear", "gbm"]


def point_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        e = (pred.y - pred["pred_" + m]).abs()
        rows.append({"model": m, "MAE": e.mean(), "MedAE": e.median(), "n": len(e)})
    return pd.DataFrame(rows)


def cluster_bootstrap_ratio(pred: pd.DataFrame, model: str, base: str = "zero", n_boot=config.N_BOOT, seed=config.RANDOM_STATE):
    """Paired bootstrap over projects: distribution of MAE(model) / MAE(base) - 1. Negative = model is better."""
    e_m = (pred.y - pred["pred_" + model]).abs()
    e_b = (pred.y - pred["pred_" + base]).abs()
    g = pd.DataFrame({"u": pred.unit.to_numpy(), "m": e_m.to_numpy(), "b": e_b.to_numpy()}).groupby("u").agg(
        m=("m", "sum"), b=("b", "sum"), n=("m", "size"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    sm, sb = g.m.to_numpy()[idx].sum(1), g.b.to_numpy()[idx].sum(1)
    r = sm / sb - 1
    point = g.m.sum() / g.b.sum() - 1
    return {"point": float(point), "lo": float(np.quantile(r, 0.025)), "hi": float(np.quantile(r, 0.975))}


def coverage_table(pred: pd.DataFrame, lo="lo_cqr", hi="hi_cqr") -> dict:
    inside = (pred.y >= pred[lo]) & (pred.y <= pred[hi])
    d = {"coverage": float(inside.mean()), "median_width": float((pred[hi] - pred[lo]).median()), "n": int(len(pred))}
    d["by_origin"] = {int(o): float(inside[pred.origin == o].mean()) for o in sorted(pred.origin.unique())}
    return d


def cluster_bootstrap_coverage(pred, lo="lo_cqr", hi="hi_cqr", n_boot=config.N_BOOT, seed=config.RANDOM_STATE):
    inside = ((pred.y >= pred[lo]) & (pred.y <= pred[hi])).astype(float)
    g = pd.DataFrame({"u": pred.unit.to_numpy(), "i": inside.to_numpy()}).groupby("u").agg(s=("i", "sum"), n=("i", "size"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    cov = g.s.to_numpy()[idx].sum(1) / g.n.to_numpy()[idx].sum(1)
    return float(np.quantile(cov, 0.025)), float(np.quantile(cov, 0.975))


def grouped_permutation_importance(model, X, y, n_repeats=config.N_PERM, seed=config.RANDOM_STATE):
    """Increase in MAE when one raw feature (agency counts as one feature) is shuffled."""
    r = permutation_importance(model, X, y, scoring="neg_mean_absolute_error", n_repeats=n_repeats, random_state=seed)
    return pd.DataFrame({"feature": X.columns, "mae_increase": r.importances_mean, "sd": r.importances_std}) \
        .sort_values("mae_increase", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------- classification (completion within horizon)
def clf_metrics(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)), "logloss": float(log_loss(y, p)),
            "n": int(len(y)), "events": int(np.sum(y))}


def current_forecast(panel, num, cat, heavy, origin, H=config.HORIZON, alpha=config.ALPHA):
    """Forecasts for the newest report (labels not yet known). Same recipe as the test origins:
    train on origins <= origin-H-1, calibrate on origin-H, predict `origin`."""
    p = panel.copy()
    train, cal, _ = walk_forward_split(p, origin, H)
    now = p[p.origin == origin].copy()
    cols = num + cat
    ytr = train.y.to_numpy()
    g = models.gbm(num, cat, heavy).fit(train[cols], ytr)
    lo_m = models.gbm(num, cat, heavy, "quantile", alpha / 2).fit(train[cols], ytr)
    hi_m = models.gbm(num, cat, heavy, "quantile", 1 - alpha / 2).fit(train[cols], ytr)
    lo, hi, _ = models.cqr_interval(lo_m.predict(cal[cols]), hi_m.predict(cal[cols]), cal.y.to_numpy(),
                                    lo_m.predict(now[cols]), hi_m.predict(now[cols]), alpha)
    now["pred"], now["lo"], now["hi"] = g.predict(now[cols]), lo, hi
    return now


def bootstrap_auc_ci(y, p, units, n_boot=config.N_BOOT, seed=config.RANDOM_STATE):
    y, p, units = np.asarray(y), np.asarray(p), np.asarray(units)
    order = pd.Series(np.arange(len(y))).groupby(units).apply(lambda s: s.to_numpy())
    rng = np.random.default_rng(seed)
    vals = []
    keys = np.arange(len(order))
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([order.iloc[k] for k in pick])
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))
