"""Read and clean the three NYC Open Data files. Every cleaning step is counted in a data-quality log."""
import numpy as np
import pandas as pd

from . import config

MIN_YEAR, MAX_YEAR = 2000, 2045  # completion dates outside this range are treated as typos


def _money(s: pd.Series) -> pd.Series:
    """'$1,525,000' -> 1525000.0, '$-558,100' -> -558100.0, blank -> NaN."""
    return pd.to_numeric(s.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")


def period_table(codes) -> pd.DataFrame:
    """Map reporting-period codes like 202305 to 0, 1, 2, ... and to a fiscal year (NYC FY starts July 1)."""
    codes = sorted(set(int(c) for c in codes))
    df = pd.DataFrame({"code": codes})
    df["t"] = np.arange(len(df))
    year, month = df.code // 100, df.code % 100
    df["fiscal_year"] = year + (month >= 7).astype(int)
    return df


def load_budget(path=None):
    """Budget by project (agency + FMS ID), fiscal year and reporting period."""
    path = path or config.FILES["budget"]
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    log = {"budget_rows_raw": len(raw)}
    pt = period_table(raw["Reporting Period"])
    df = pd.DataFrame({
        "agency": raw["Managing Agency"].str.strip(),
        "fms_id": raw["FMS ID"].astype(str).str.strip(),
        "code": raw["Reporting Period"].astype(int),
        "fy": raw["Fiscal Year"].astype(int),
        "budget": _money(raw["Total Budget City Non City"]),
        "spend": _money(raw["Spend"]),
    })
    df = df.merge(pt, on="code", how="left")
    log["budget_rows_missing_amount"] = int(df.budget.isna().sum())
    df = df.dropna(subset=["budget"]).copy()
    df["rel_fy"] = df.fy - df.fiscal_year  # <0 past fiscal year, 0 current, >0 planned future
    df["unit"] = df.agency + "|" + df.fms_id
    log["budget_negative_rows"] = int((df.budget < 0).sum())
    log["budget_units"] = int(df.unit.nunique())
    return df, pt, log


def normalize_phase(x):
    if pd.isna(x):
        return "Unknown"
    x = str(x).strip().lower()
    return {"construction procurement": "Construction Procurement", "construction": "Construction",
            "pre-design": "Pre-Design", "design": "Design", "close-out": "Close-out",
            "on hold": "On Hold", "closed": "Closed", "cancelled": "Cancelled"}.get(x, x.title())


def load_schedule(path=None):
    """Forecast completion date of each project at each report, with phase and reason for change."""
    path = path or config.FILES["schedule"]
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    log = {"schedule_rows_raw": len(raw)}
    pt = period_table(raw["Reporting Period"])
    df = pd.DataFrame({
        "agency": raw["Managing Agency"].str.strip(),
        "pid": raw["PID"].astype(str).str.strip(),
        "name": raw["Agency Project Name"],
        "code": raw["Reporting Period"].astype(int),
        "phase": raw["Current Phase"].map(normalize_phase),
        "cd": pd.to_datetime(raw["Completion Date"], format="%m/%d/%Y", errors="coerce"),
        "type": raw["Completion Date Type"].str.strip(),
        "variance": pd.to_numeric(raw["Variance (day)"], errors="coerce"),
        "reason": raw["Reason for Forecast Completion Change"],
        "data_date": pd.to_datetime(raw["Data Date"], format="%m/%d/%Y", errors="coerce"),
    })
    df = df.merge(pt, on="code", how="left")
    bad = df.cd.notna() & ((df.cd.dt.year < MIN_YEAR) | (df.cd.dt.year > MAX_YEAR))
    log["schedule_implausible_dates_removed"] = int(bad.sum())
    df.loc[bad, "cd"] = pd.NaT
    log["schedule_missing_completion_date"] = int(df.cd.isna().sum())
    # a few reports lack a data date; use the median data date of that report instead
    med = df.groupby("t")["data_date"].transform("median")
    log["schedule_data_date_imputed"] = int(df.data_date.isna().sum())
    df["data_date"] = df.data_date.fillna(med)
    df["unit"] = df.agency + "|" + df.pid
    df = df.sort_values(["unit", "t"]).reset_index(drop=True)
    assert not df.duplicated(["unit", "t"]).any(), "duplicate project-report rows"
    log["schedule_units"] = int(df.unit.nunique())
    return df, pt, log


def reason_group(r):
    """Group the free-text delay reasons into a few readable categories."""
    if pd.isna(r):
        return None
    r = str(r).upper()
    for key, name in [("SCOPE", "scope/design"), ("PERMIT", "permits"), ("BUDGET", "budget"),
                      ("SITE", "site conditions"), ("HAZARD", "site conditions"), ("LEGAL", "legal"),
                      ("UTILITY", "utility work"), ("PRODUCT", "supply")]:
        if key in r:
            return name
    return "other"


def load_sca(path=None):
    """School Construction Authority phases: planned versus actual dates and budget versus final estimate."""
    path = path or config.FILES["sca"]
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    log = {"sca_rows_raw": len(raw)}
    df = pd.DataFrame({
        "district": raw["Project Geographic District"],
        "building": raw["Project Building Identifier"].astype(str),
        "school": raw["Project School Name"],
        "type": raw["Project Type"].str.strip(),
        "phase": raw["Project Phase Name"].str.strip(),
        "status": raw["Project Status Name"].str.strip(),
        "start": pd.to_datetime(raw["Project Phase Actual Start Date"], format="%m/%d/%Y", errors="coerce"),
        "planned_end": pd.to_datetime(raw["Project Phase Planned End Date"], format="%m/%d/%Y", errors="coerce"),
        "actual_end": pd.to_datetime(raw["Project Phase Actual End Date"], format="%m/%d/%Y", errors="coerce"),
        "budget": _money(raw["Project Budget Amount"]),
        "final_estimate": _money(raw["Final Estimate of Actual Costs Through End of Phase Amount"]),
        "spent": _money(raw["Total Phase Actual Spending Amount"]),
    })
    # for some programs (e.g. "SCA IEH") the planned-end field holds a label such as "IEH" instead of a date
    log["sca_planned_end_not_a_date"] = int((raw["Project Phase Planned End Date"].notna() & df.planned_end.isna()).sum())
    df["delay_days"] = (df.actual_end - df.planned_end).dt.days
    df["cost_ratio"] = np.where(df.budget > 0, df.final_estimate / df.budget, np.nan)
    log["sca_complete_phases"] = int((df.status == "Complete").sum())
    log["sca_complete_with_delay"] = int(((df.status == "Complete") & df.delay_days.notna()).sum())
    log["sca_complete_with_cost_ratio"] = int(((df.status == "Complete") & df.cost_ratio.notna()).sum())
    return df, log
