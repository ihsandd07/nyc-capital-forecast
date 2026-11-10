"""Run every analysis and return one results dictionary (also saved as reports/results.json and CSV tables)."""
import json

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

from . import config, evaluate as ev, load, models, panel, survival as sv

COST_HEAVY = ["growth_since_first", "prev_change"]
SCHED_HEAVY = ["slip_since_first", "max_delay_days", "sum_delay_days", "last_change", "months_to_forecast"]
BIG_UP = 0.10   # a "large increase" is a planned budget at least 10% higher 12 months later


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, pd.DataFrame):
        return _clean(o.to_dict("records"))
    return o


def regression_block(pan, num, cat, heavy, name):
    """Walk-forward point and interval forecasts for one target."""
    preds, last_models, last_sets = [], None, None
    for o in config.TEST_ORIGINS:
        p, m, sets = ev.fit_predict_origin(pan, num, cat, heavy, o, return_models=True)
        preds.append(p)
        last_models, last_sets = m, sets
    pred = pd.concat(preds, ignore_index=True)
    pm = ev.point_metrics(pred)
    per_origin = {int(o): ev.point_metrics(pred[pred.origin == o]).set_index("model").MAE.to_dict() for o in config.TEST_ORIGINS}
    ratios = {m: ev.cluster_bootstrap_ratio(pred, m) for m in ["agency", "linear", "gbm"]}
    cov = {k: ev.coverage_table(pred, f"lo_{k}", f"hi_{k}") for k in ["cqr", "sc", "raw"]}
    for k in ["cqr", "sc", "raw"]:
        cov[k]["ci"] = ev.cluster_bootstrap_coverage(pred, f"lo_{k}", f"hi_{k}")
    inside = (pred.y >= pred.lo_cqr) & (pred.y <= pred.hi_cqr)
    by_agency = pred.assign(inside=inside).groupby("agency").agg(n=("y", "size"), coverage=("inside", "mean")).query("n >= 100")
    block = {"n_test_rows": len(pred), "n_test_units": int(pred.unit.nunique()),
             "test_median_y": float(pred.y.median()), "share_unchanged": float((pred.y.abs() < 1e-9).mean()),
             "metrics": pm.set_index("model")[["MAE", "MedAE"]].to_dict("index"), "mae_by_origin": per_origin,
             "mae_change_vs_zero": ratios, "coverage": cov, "coverage_by_agency": by_agency.reset_index(),
             "n_train_by_origin": {int(o): int(ev.walk_forward_split(pan, o)[0].shape[0]) for o in config.TEST_ORIGINS}}
    return block, pred, last_models, last_sets


def drivers_regression(models_, sets, num, cat):
    _, _, test = sets
    return ev.grouped_permutation_importance(models_["gbm"], test[num + cat], test.y.to_numpy())


def classification_block(pan, label, num, cat, heavy, rule_cols):
    pr = sv.landmark_walk_forward(pan, label, num, cat, heavy, rule_cols).reset_index(drop=True)
    met = sv.landmark_metrics(pr)
    auc_ci = {m: ev.bootstrap_auc_ci(pr.y, pr["p_" + m], pr.unit) for m in ["rule", "logit", "gbm"]}
    return pr, {"metrics": met, "auc_ci": auc_ci, "auc_diff_gbm_vs_rule": sv.bootstrap_auc_diff(pr, "gbm", "rule"),
                "calibration_gbm": sv.calibration_bins(pr.y, pr.p_gbm), "base_rate_test": float(pr.y.mean())}


def sca_block(sca):
    c = sca[sca.status == "Complete"].copy()
    def summ(g):
        d, r = g.delay_days.dropna(), g.cost_ratio.dropna()
        return pd.Series({"phases": len(g), "with_dates": len(d), "share_late": (d > 0).mean() if len(d) else np.nan,
                          "median_delay_days": d.median() if len(d) else np.nan, "p90_delay_days": d.quantile(.9) if len(d) else np.nan,
                          "with_cost": len(r), "share_over_budget": (r > 1.0001).mean() if len(r) else np.nan,
                          "median_cost_ratio": r.median() if len(r) else np.nan})
    by_phase = c.groupby("phase").apply(summ, include_groups=False).query("phases >= 40").sort_values("phases", ascending=False)
    by_type = c.groupby("type").apply(summ, include_groups=False).query("phases >= 100").sort_values("phases", ascending=False)
    return {"complete_phases": len(c), "by_phase": by_phase.reset_index(), "by_type": by_type.reset_index(),
            "all_status_counts": sca.status.value_counts().to_dict()}


def run(save=True, verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    budget, pt, log1 = load.load_budget()
    sched, pt2, log2 = load.load_schedule()
    sca, log3 = load.load_sca()
    assert pt.code.tolist() == pt2.code.tolist(), "the two report files cover different report dates"
    T = len(pt)
    R = {"meta": {"periods": pt.code.tolist(), "horizon_reports": config.HORIZON, "test_origins": list(config.TEST_ORIGINS),
                  "alpha": config.ALPHA, "n_boot": config.N_BOOT, "big_up_threshold": BIG_UP},
         "data_quality": {**log1, **log2, **log3}}

    say("cost panel + models ...")
    cp = panel.cost_panel(budget, T)
    R["cost_panel"] = {"rows": len(cp), "units": int(cp.unit.nunique()),
                       "labeled_rows": int(cp.y.notna().sum()),
                       "vanished_share": float((cp[cp.y.notna() | (cp.present_later == 0)].present_later == 0).mean())}
    cost, cost_pred, cost_models, cost_sets = regression_block(cp, panel.COST_NUM, panel.COST_CAT, COST_HEAVY, "cost")
    R["cost"] = cost

    say("cost: large-increase classifier ...")
    big = (cp.y > np.log(1 + BIG_UP)).astype(float).where(cp.y.notna())
    cb_pred, cb = classification_block(cp, big, panel.COST_NUM, panel.COST_CAT, COST_HEAVY, ["has_prev", "prev_change"])
    R["cost_big_up"] = cb

    say("schedule panel + models ...")
    sp = panel.schedule_panel(sched, T)
    R["schedule_panel"] = {"rows": len(sp), "units": int(sp.unit.nunique()), "labeled_rows": int(sp.y.notna().sum()),
                           "vanished_without_completion_share": float((sp[sp.origin <= max(config.TEST_ORIGINS)].done.isna()).mean()),
                           "median_slip_by_origin": sp[sp.y.notna()].groupby("origin").y.median().to_dict()}
    schedule, sched_pred, sched_models, sched_sets = regression_block(sp, panel.SCHED_NUM, panel.SCHED_CAT, SCHED_HEAVY, "schedule")
    R["schedule"] = schedule

    say("survival ...")
    dur = sv.project_durations(sched)
    km = sv.kaplan_meier(dur.duration, dur.event)
    top = dur.agency.value_counts().head(6).index
    km_by = {a: sv.kaplan_meier(dur[dur.agency == a].duration, dur[dur.agency == a].event) for a in top}
    sp_pred, sp_met = classification_block(sp, sp.done, panel.SCHED_NUM, panel.SCHED_CAT, SCHED_HEAVY, ["months_to_forecast", "overdue"])
    R["survival"] = {"cohort": len(dur), "events": int(dur.event.sum()), "censored": int((dur.event == 0).sum()),
                     "km": km, "p_complete_3": 1 - sv.surv_at(km, 3), "p_complete_9": 1 - sv.surv_at(km, 9),
                     "by_agency": {a: {"n": int((dur.agency == a).sum()), "events": int(dur[dur.agency == a].event.sum()),
                                       "p_complete_3": 1 - sv.surv_at(k, 3), "p_complete_9": 1 - sv.surv_at(k, 9)} for a, k in km_by.items()},
                     "landmark": sp_met}

    say("drivers ...")
    dr_c = ev.grouped_permutation_importance  # noqa
    # cost drivers come from the large-increase classifier (the point forecast has almost no skill to explain)
    train, cal, test = ev.walk_forward_split(cp.assign(y=big), max(config.TEST_ORIGINS))
    train = pd.concat([train, cal])
    clf = models.gbm_classifier(panel.COST_NUM, panel.COST_CAT, COST_HEAVY).fit(train[panel.COST_NUM + panel.COST_CAT], train.y.astype(int))
    from sklearn.inspection import permutation_importance
    r = permutation_importance(clf, test[panel.COST_NUM + panel.COST_CAT], test.y.astype(int), scoring="roc_auc",
                               n_repeats=config.N_PERM, random_state=config.RANDOM_STATE)
    drv_cost = pd.DataFrame({"feature": panel.COST_NUM + panel.COST_CAT, "auc_drop": r.importances_mean, "sd": r.importances_std}) \
        .sort_values("auc_drop", ascending=False).reset_index(drop=True)
    drv_sched = drivers_regression(sched_models, sched_sets, panel.SCHED_NUM, panel.SCHED_CAT)
    R["drivers"] = {"cost_big_up_auc_drop": drv_cost, "schedule_mae_increase": drv_sched}
    pdp = {}
    for nm, mdl, X, feats, resp in [("schedule", sched_models["gbm"], sched_sets[2][panel.SCHED_NUM + panel.SCHED_CAT], ["months_to_forecast", "n_delays"], "auto"),
                                    ("cost_big_up", clf, test[panel.COST_NUM + panel.COST_CAT], ["log_total", "future_share"], "predict_proba")]:
        for f in feats:
            res = partial_dependence(mdl, X, [f], grid_resolution=20, percentiles=(0.05, 0.95), response_method=resp, kind="average")
            pdp[f"{nm}:{f}"] = {"x": res["grid_values"][0].tolist(), "y": res["average"][0].tolist()}
    R["partial_dependence"] = pdp

    say("school-construction phases ...")
    R["sca"] = sca_block(sca)

    say("current forecasts ...")
    last = T - 1
    cur_s = ev.current_forecast(sp, panel.SCHED_NUM, panel.SCHED_CAT, SCHED_HEAVY, last)
    cur_c = ev.current_forecast(cp, panel.COST_NUM, panel.COST_CAT, COST_HEAVY, last)
    # probability that a currently open project completes within the next 12 months (landmark gradient boosting)
    tr, ca, _ = ev.walk_forward_split(sp.assign(y=sp.done), last)
    trc = pd.concat([tr, ca])
    gbc = models.gbm_classifier(panel.SCHED_NUM, panel.SCHED_CAT, SCHED_HEAVY).fit(trc[panel.SCHED_NUM + panel.SCHED_CAT], trc.y.astype(int))
    cur_s["p_complete_12m"] = gbc.predict_proba(cur_s[panel.SCHED_NUM + panel.SCHED_CAT])[:, 1]
    # probability that a current budget line's plan rises by 10%+ within 12 months
    trb, cab, _ = ev.walk_forward_split(cp.assign(y=big), last)
    trb = pd.concat([trb, cab])
    gbb = models.gbm_classifier(panel.COST_NUM, panel.COST_CAT, COST_HEAVY).fit(trb[panel.COST_NUM + panel.COST_CAT], trb.y.astype(int))
    cur_c["p_big_up_12m"] = gbb.predict_proba(cur_c[panel.COST_NUM + panel.COST_CAT])[:, 1]
    R["current"] = {"report": int(pt.code.iloc[last]), "n_schedule": len(cur_s), "n_cost": len(cur_c)}

    if save:
        config.REPORTS.mkdir(exist_ok=True)
        out = config.REPORTS
        (out / "tables").mkdir(exist_ok=True)
        cost_pred.to_csv(out / "tables" / "cost_test_predictions.csv", index=False)
        sched_pred.to_csv(out / "tables" / "schedule_test_predictions.csv", index=False)
        cb_pred.to_csv(out / "tables" / "cost_large_increase_test_predictions.csv", index=False)
        sp_pred.to_csv(out / "tables" / "schedule_completion_test_predictions.csv", index=False)
        cur_s.drop(columns=["forecast_date"]).assign(forecast_date=cur_s.forecast_date.dt.strftime("%Y-%m-%d")) \
            .to_csv(out / "tables" / "current_schedule_forecasts.csv", index=False)
        cur_c.to_csv(out / "tables" / "current_cost_forecasts.csv", index=False)
        R["km_by_agency_tables"] = {a: k for a, k in km_by.items()}
        (out / "results.json").write_text(json.dumps(_clean(R), indent=1))
    return R, {"cost_pred": cost_pred, "sched_pred": sched_pred, "cur_s": cur_s, "cur_c": cur_c, "sp": sp, "cp": cp, "dur": dur}
