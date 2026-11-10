"""Turn the report history into forecasting problems.

A row is one project at one report date (the "origin"). Features use only reports up to and including the origin.
The label is what happened HORIZON reports later. The most recent origins have no label yet; they are kept (y = NaN)
so the dashboard can show current forecasts.
"""
import numpy as np
import pandas as pd

from . import config
from .load import reason_group

REASONS = ["scope/design", "permits", "budget", "site conditions", "legal", "utility work", "supply", "other"]
EPOCH = pd.Timestamp("1970-01-01")


def _wide(df, unit_col, value, T):
    """units x T matrix (NaN when the project is absent from that report)."""
    w = df.pivot(index=unit_col, columns="t", values=value).reindex(columns=range(T))
    return w


# ----------------------------------------------------------------------------- cost
def cost_panel(budget: pd.DataFrame, T: int, H: int = config.HORIZON) -> pd.DataFrame:
    """Rows: project x report origin. Label: log(total planned budget H reports later / now)."""
    b = budget.copy()
    b["past"] = np.where(b.rel_fy < 0, b.budget, 0.0)
    b["future"] = np.where(b.rel_fy > 0, b.budget, 0.0)
    b["spend_s"] = np.where(b.rel_fy <= 0, b.spend.fillna(0.0), 0.0)
    b["has_spend"] = ((b.rel_fy <= 0) & b.spend.notna()).astype(int)
    b["fut_pos"] = ((b.rel_fy > 0) & (b.budget > 0)).astype(int)
    b["yrs_ahead"] = np.where(b.budget > 0, np.maximum(b.rel_fy, 0), 0)
    g = b.groupby(["unit", "t"], as_index=False).agg(
        total=("budget", "sum"), past=("past", "sum"), future=("future", "sum"), spend=("spend_s", "sum"),
        has_spend=("has_spend", "max"), n_future_fy=("fut_pos", "sum"), yrs_ahead=("yrs_ahead", "max"),
        agency=("agency", "first"))
    agency = g.groupby("unit")["agency"].first()
    M = {k: _wide(g, "unit", k, T) for k in ["total", "past", "future", "spend", "has_spend", "n_future_fy", "yrs_ahead"]}
    units = M["total"].index
    tot = M["total"].to_numpy()
    present = ~np.isnan(tot)
    first_t = np.where(present.any(1), present.argmax(1), -1)

    out = []
    for o in range(T):
        ok = present[:, o] & (np.nan_to_num(tot[:, o]) > 0)
        i = np.where(ok)[0]
        if len(i) == 0:
            continue
        cur = tot[i, o]
        f = pd.DataFrame({"unit": units[i], "agency": agency.reindex(units[i]).to_numpy(), "origin": o})
        f["log_total"] = np.log(cur)
        f["past_share"] = np.clip(M["past"].to_numpy()[i, o] / cur, -1, 2)
        f["future_share"] = np.clip(M["future"].to_numpy()[i, o] / cur, -1, 2)
        f["spend_ratio"] = np.clip(M["spend"].to_numpy()[i, o] / cur, 0, 2)
        f["spend_reported"] = np.nan_to_num(M["has_spend"].to_numpy()[i, o])
        f["n_future_fy"] = np.nan_to_num(M["n_future_fy"].to_numpy()[i, o])
        f["yrs_ahead"] = np.nan_to_num(M["yrs_ahead"].to_numpy()[i, o])
        f["age"] = o - first_t[i]
        base = tot[i, first_t[i]]
        f["growth_since_first"] = np.where(base > 0, np.log(cur / np.where(base > 0, base, 1)), 0.0)
        prev = tot[i, o - 1] if o > 0 else np.full(len(i), np.nan)
        has_prev = (prev > 0)
        f["has_prev"] = has_prev.astype(int)
        f["prev_change"] = np.where(has_prev, np.log(cur / np.where(has_prev, prev, 1)), 0.0)
        # number of earlier reports where the plan rose by more than 1%
        inc = np.zeros(len(i))
        for t in range(1, o + 1):
            a, bb = tot[i, t - 1], tot[i, t]
            inc += ((a > 0) & (bb > 1.01 * a)).astype(float)
        f["n_increases"] = inc
        if o + H < T:
            fut = tot[i, o + H]
            f["y"] = np.where(fut > 0, np.log(np.where(fut > 0, fut, 1) / cur), np.nan)
            f["present_later"] = (~np.isnan(fut)).astype(int)
        else:
            f["y"] = np.nan
            f["present_later"] = np.nan
        out.append(f)
    return pd.concat(out, ignore_index=True)


COST_NUM = ["log_total", "past_share", "future_share", "spend_ratio", "spend_reported", "n_future_fy", "yrs_ahead",
            "age", "growth_since_first", "has_prev", "prev_change", "n_increases"]
COST_CAT = ["agency"]


# ----------------------------------------------------------------------------- schedule
def schedule_panel(sched: pd.DataFrame, T: int, H: int = config.HORIZON) -> pd.DataFrame:
    """Rows: project (still a forecast) x report origin.
    y: days the forecast completion date moves over the next H reports (actual completion date if it finishes).
    done: 1 if the project is complete within H reports, 0 if still open, NaN if it vanished from the reports (censored)."""
    s = sched.copy()
    s["cd_days"] = (s.cd - EPOCH).dt.days.astype(float)
    s["dd_days"] = (s.data_date - EPOCH).dt.days.astype(float)
    s["is_actual"] = (s.type == "Actual").astype(float)
    s["rg"] = s.reason.map(reason_group)
    CD, DD, ACT = (_wide(s, "unit", k, T).to_numpy() for k in ["cd_days", "dd_days", "is_actual"])
    units = _wide(s, "unit", "cd_days", T).index
    present = ~np.isnan(_wide(s, "unit", "is_actual", T).to_numpy())
    phase = _wide(s.assign(pc=s.phase.astype("category").cat.codes), "unit", "pc", T).to_numpy()
    phase_names = list(s.phase.astype("category").cat.categories)
    agency = s.groupby("unit")["agency"].first().reindex(units)
    name = s.groupby("unit")["name"].last().reindex(units)
    pid = s.groupby("unit")["pid"].first().reindex(units)
    reason_wide = {r: _wide(s.assign(v=(s.rg == r).astype(float)), "unit", "v", T).to_numpy() for r in REASONS}
    first_t = np.where(present.any(1), present.argmax(1), -1)

    out = []
    for o in range(T):
        ok = present[:, o] & (ACT[:, o] == 0) & ~np.isnan(CD[:, o])
        i = np.where(ok)[0]
        if len(i) == 0:
            continue
        f = pd.DataFrame({"unit": units[i], "agency": agency.reindex(units[i]).to_numpy(),
                          "name": name.reindex(units[i]).to_numpy(), "pid": pid.reindex(units[i]).to_numpy(),
                          "origin": o})
        f["phase"] = [phase_names[int(p)] if not np.isnan(p) else "Unknown" for p in phase[i, o]]
        f["months_to_forecast"] = (CD[i, o] - DD[i, o]) / 30.44
        f["overdue"] = (f.months_to_forecast < 0).astype(int)
        f["age"] = o - first_t[i]
        # first usable forecast date up to the origin, number/size of earlier changes
        first_cd = np.full(len(i), np.nan)
        n_chg = np.zeros(len(i)); n_delay = np.zeros(len(i)); max_delay = np.zeros(len(i)); sum_delay = np.zeros(len(i))
        for t in range(o, -1, -1):  # walk back so the earliest valid date wins
            v = CD[i, t]
            first_cd = np.where(~np.isnan(v), v, first_cd)
        for t in range(1, o + 1):
            a, bb = CD[i, t - 1], CD[i, t]
            ch = np.where(~np.isnan(a) & ~np.isnan(bb), bb - a, 0.0)
            n_chg += (ch != 0); n_delay += (ch > 0)
            max_delay = np.maximum(max_delay, ch); sum_delay += np.maximum(ch, 0)
        f["slip_since_first"] = CD[i, o] - first_cd
        f["n_changes"] = n_chg; f["n_delays"] = n_delay
        f["max_delay_days"] = max_delay; f["sum_delay_days"] = sum_delay
        prev = CD[i, o - 1] if o > 0 else np.full(len(i), np.nan)
        f["has_prev"] = (~np.isnan(prev)).astype(int)
        f["last_change"] = np.where(~np.isnan(prev), CD[i, o] - prev, 0.0)
        for r in REASONS:  # was this reason ever given at or before the origin?
            seen = np.nanmax(np.where(np.isnan(reason_wide[r][i, : o + 1]), 0.0, reason_wide[r][i, : o + 1]), axis=1)
            f["reason_" + r] = seen
        f["forecast_date"] = pd.to_datetime(CD[i, o], unit="D")
        if o + H < T:
            later_present = present[i, o + H]
            later_cd = CD[i, o + H]
            f["y"] = np.where(later_present & ~np.isnan(later_cd), later_cd - CD[i, o], np.nan)
            any_actual = np.nanmax(np.where(np.isnan(ACT[i, o + 1: o + H + 1]), 0.0, ACT[i, o + 1: o + H + 1]), axis=1)
            f["done"] = np.where(later_present, any_actual, np.nan)
            f["horizon_end"] = pd.to_datetime(np.nanmedian(DD[:, o + H]), unit="D")
        else:
            f["y"] = np.nan; f["done"] = np.nan; f["horizon_end"] = pd.NaT
        out.append(f)
    return pd.concat(out, ignore_index=True)


SCHED_NUM = ["months_to_forecast", "overdue", "age", "slip_since_first", "n_changes", "n_delays", "max_delay_days",
             "sum_delay_days", "has_prev", "last_change"] + ["reason_" + r for r in REASONS]
SCHED_CAT = ["agency", "phase"]


# ----------------------------------------------------------------------------- walk-forward split
def walk_forward_split(panel: pd.DataFrame, test_origin: int, H: int = config.HORIZON):
    """Train on origins whose labels were fully known before the calibration origin; calibrate on the newest
    origin whose label was known when the test origin arrived; test on the test origin.
    Example (H=3, test origin 5): train origins 0-1, calibration origin 2, test origin 5."""
    lab = panel[panel.y.notna()]
    train = lab[lab.origin <= test_origin - H - 1]
    cal = lab[lab.origin == test_origin - H]
    test = lab[lab.origin == test_origin]
    return train, cal, test
