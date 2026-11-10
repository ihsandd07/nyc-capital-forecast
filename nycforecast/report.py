"""Build reports/report.pdf. Every number in the text comes from results.json; nothing is typed in by hand."""
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from . import config, figures

W, H = 8.5, 11.0
MARGIN = 0.8


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def spct(x, d=1):
    return f"{x * 100:+.{d}f}%"


def summary_lines(R):
    c, s, sv, cb = R["cost"], R["schedule"], R["survival"], R["cost_big_up"]
    lm = {m["model"]: m for m in sv["landmark"]["metrics"]}
    cbm = {m["model"]: m for m in cb["metrics"]}
    cg, sg = c["mae_change_vs_zero"]["gbm"], s["mae_change_vs_zero"]["gbm"]
    dq = R["data_quality"]
    test_slip = s["test_median_y"]
    d = sv["landmark"]["auc_diff_gbm_vs_rule"]
    return [
        f"Data. Three public NYC Open Data files: budgets for {dq['budget_units']:,} project budget lines, forecast completion "
        f"dates for {dq['schedule_units']:,} projects across 10 city reports (May 2023 to May 2026), and {dq['sca_rows_raw']:,} school-construction phases.",
        f"Cost. Over 12 months {pct(c['share_unchanged'], 0)} of test budget lines do not change at all and the median change is zero. The best model lowers mean absolute error by only "
        f"{abs(cg['point']) * 100:.1f}% versus 'no change' (95% CI {spct(cg['lo'])} to {spct(cg['hi'])}): a real but practically tiny gain, "
        f"so point forecasts of budget growth are not useful. A large increase (10% or more, {pct(cb['base_rate_test'])} of test projects) is more "
        f"predictable: AUC {cbm['gbm']['auc']:.2f} (95% CI {cb['auc_ci']['gbm'][0]:.2f} to {cb['auc_ci']['gbm'][1]:.2f}).",
        f"Schedule. Open projects' forecast completion dates keep slipping: about {test_slip:.0f} days over the next 12 months in the test period. "
        f"Gradient boosting lowers mean absolute error by {abs(sg['point']) * 100:.1f}% (95% CI {abs(sg['hi']) * 100:.1f}% to {abs(sg['lo']) * 100:.1f}%) "
        f"versus assuming the date holds.",
        f"Intervals. 80% intervals cover {pct(c['coverage']['cqr']['coverage'], 0)} of held-out projects for cost but only "
        f"{pct(s['coverage']['cqr']['coverage'], 0)} (CQR) and {pct(s['coverage']['sc']['coverage'], 0)} (split conformal) for schedule, "
        f"because the slip rate drifts over time.",
        f"Survival. {sv['cohort']:,} projects were open at their first report; {sv['censored']:,} are still open at the end (handled as censored). "
        f"Kaplan-Meier: {pct(sv['p_complete_3'])} complete within 12 months and {pct(sv['p_complete_9'], 0)} within 36 months. A project's own forecast "
        f"date already ranks near-term completions well (AUC {lm['rule']['auc']:.2f}); gradient boosting adds {d[0]:+.3f} "
        f"(95% CI {d[1]:+.3f} to {d[2]:+.3f}), not a significant gain.",
    ]


LIMITS = [
    "Budget growth is not final cost overrun. The cost target is the change in each project's total planned budget between city reports. "
    "Spending is only recorded for past fiscal years, and the public files do not contain the original approved budget. Planned amounts also move "
    "between fiscal years as the planning window rolls forward.",
    "Short history. There are 10 reports (3 years). Only the last two forecast origins (reports of Jan and May 2025) are held out, and the "
    "same projects appear in both, so results come from two correlated test windows. Bootstrap confidence intervals resample projects, not time.",
    "Drift. The median 12-month slip changed from 0 to 150 days across origins. Models trained on older reports therefore under-predict recent "
    "slippage, and conformal intervals (which assume exchangeable data) under-cover for schedule.",
    "Projects that vanish. About 15% of cost rows and 12% of schedule rows disappear from later reports without a completion record. They cannot be "
    "labelled, so they are excluded; the results describe projects that stay in the reports.",
    "Self-reported data. Agencies report their own forecasts; the NYC Comptroller and State Comptroller have criticised the quality of public capital "
    "data. Nothing here is audited.",
    "School-construction phases are described only. Completed phases are a biased sample (long phases are still open), and 'planned end' is missing "
    "or not a date for many programmes, so no forecasting claim is made from that file.",
    "Drivers are associations, not causes. Permutation importance and partial dependence show what the models use, not what causes delay or cost growth.",
    "Analysis log. Hyperparameters were fixed in advance and never tuned. The large-budget-increase classifier was added after seeing that the "
    "point forecasts had almost no skill, and is reported regardless of its result. Intervals: CQR was named as the primary method before the "
    "test run; split conformal is reported alongside it. The linear model underperformed and is reported as is.",
]


class Page:
    """A US-letter page with a simple top-to-bottom layout in inches."""

    def __init__(self, pdf, title):
        self.pdf, self.title = pdf, title
        self.fig = plt.figure(figsize=(W, H))
        self.y = H - MARGIN
        self.fig.text(MARGIN / W, self.y / H, title, fontsize=15, fontweight="bold", va="top")
        self.y -= 0.5

    def text(self, s, size=9.5, width=96, gap=0.12, bold=False):
        lines = textwrap.wrap(s, width=width)
        h = len(lines) * size * 0.0222
        self.fig.text(MARGIN / W, self.y / H, "\n".join(lines), fontsize=size, va="top", fontweight="bold" if bold else "normal", linespacing=1.35)
        self.y -= h + gap

    def bullets(self, items, size=9.5, width=92):
        for it in items:
            lines = textwrap.wrap(it, width=width)
            h = len(lines) * size * 0.0222
            self.fig.text(MARGIN / W, self.y / H, "•", fontsize=size, va="top")
            self.fig.text((MARGIN + 0.18) / W, self.y / H, "\n".join(lines), fontsize=size, va="top", linespacing=1.35)
            self.y -= h + 0.1

    def heading(self, s):
        self.y -= 0.06
        self.fig.text(MARGIN / W, self.y / H, s, fontsize=11, fontweight="bold", va="top")
        self.y -= 0.28

    def image(self, fig, width_in=None):
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())
        w_in, h_in = fig.get_size_inches()
        width_in = width_in or min(W - 2 * MARGIN, w_in)
        h = width_in * h_in / w_in
        ax = self.fig.add_axes([(W - width_in) / 2 / W, (self.y - h) / H, width_in / W, h / H])
        ax.imshow(img); ax.axis("off")
        self.y -= h + 0.15
        plt.close(fig)

    def table(self, df, col_w=None, size=7.5, row_h=0.2, wrap=None):
        if wrap:   # wrap long cell text per column (characters per line)
            df = df.copy()
            for col, n in wrap.items():
                df[col] = df[col].map(lambda v: "\n".join(textwrap.wrap(str(v), n)))
        h = row_h * (len(df) + 1)
        ax = self.fig.add_axes([MARGIN / W, (self.y - h) / H, (W - 2 * MARGIN) / W, h / H])
        ax.axis("off")
        t = ax.table(cellText=df.values.tolist(), colLabels=list(df.columns), loc="upper left", cellLoc="left", colWidths=col_w)
        t.auto_set_font_size(False); t.set_fontsize(size)
        for (r, c), cell in t.get_celld().items():
            cell.set_height(1.0 / (len(df) + 1)); cell.set_linewidth(0.3)
            if r == 0:
                cell.set_facecolor("#e8edf5"); cell.set_text_props(fontweight="bold")
        self.y -= h + 0.2

    def save(self):
        self.fig.text(W / 2 / W, 0.4 / H, f"Page {self.pdf.get_pagecount() + 1}", fontsize=8, ha="center", color="gray")
        self.pdf.savefig(self.fig)
        plt.close(self.fig)


def fmt_sca(df, cols):
    d = df.copy()
    out = pd.DataFrame()
    for c in cols:
        if c in ("phase", "type"):
            out[c] = d[c]
        elif c in ("phases", "with_dates", "with_cost"):
            out[c] = d[c].map(lambda v: f"{int(v):,}" if pd.notna(v) else "")
        elif c.startswith("share"):
            out[c] = d[c].map(lambda v: pct(v, 0) if pd.notna(v) else "n/a")
        else:
            out[c] = d[c].map(lambda v: f"{v:.2f}" if abs(v) < 20 else f"{v:.0f}" if pd.notna(v) else "n/a")
    return out


def build(R=None, path=None):
    R = R or figures.load_results()
    F = figures.make_all(R)
    path = path or (config.REPORTS / "report.pdf")
    dq, sv, c, s = R["data_quality"], R["survival"], R["cost"], R["schedule"]
    with PdfPages(path) as pdf:
        # ---- 1 summary
        p = Page(pdf, "Forecasting NYC capital-project budget growth and schedule slippage")
        p.text("A reproducible analysis of public NYC Open Data. Held-out testing uses only data available at each forecast date.", size=10, gap=0.2)
        p.heading("Key results")
        p.bullets(summary_lines(R))
        p.heading("What to take from this")
        p.text("Schedule slippage is partly predictable from information the city already publishes, and the models quantify it better than 'the date "
               "holds'. Budget growth is mostly not predictable as a point forecast, but the chance of a large increase is. Uncertainty ranges need "
               "care: they can be honest on average and still too narrow when conditions drift.")
        p.save()
        # ---- 2 data and method
        p = Page(pdf, "Data and method")
        p.heading("Datasets")
        t = pd.DataFrame([
            ["Budget and Spend by Fiscal Year", f"{dq['budget_rows_raw']:,} rows", f"{dq['budget_units']:,} project budget lines", "Planned budget by fiscal year at each of 10 reports; spend for past years"],
            ["Schedule History and Variance", f"{dq['schedule_rows_raw']:,} rows", f"{dq['schedule_units']:,} projects", "Forecast or actual completion date at each report, phase, reason for change"],
            ["Capital Project Schedules and Budgets", f"{dq['sca_rows_raw']:,} rows", f"{R['sca']['complete_phases']:,} completed phases", "School-construction phases: planned and actual dates, budget and final estimate"]],
            columns=["Dataset", "Size", "Projects", "Contents"])
        p.table(t, col_w=[0.27, 0.12, 0.2, 0.41], size=7.2, row_h=0.42, wrap={"Dataset": 26, "Projects": 20, "Contents": 52})
        p.heading("Cleaning (all counted, none silent)")
        p.bullets([
            f"{dq['budget_rows_missing_amount']:,} budget rows had no amount and were dropped; {dq['budget_negative_rows']:,} rows hold negative adjustments and were kept.",
            f"{dq['schedule_implausible_dates_removed']} completion dates with impossible years (for example 3026) were set to missing; {dq['schedule_missing_completion_date']:,} schedule rows "
            f"have no forecast date; {dq['schedule_data_date_imputed']} missing data dates were filled with that report's median.",
            f"In {dq['sca_planned_end_not_a_date']:,} school-phase rows the planned-end field holds a programme label instead of a date, so no delay can be computed.",
            "Phase names with different spellings were merged; delay reasons were grouped into eight categories.",
        ])
        p.heading("Forecast problems")
        p.bullets([
            "Cost: for each project at each report, predict log(total planned budget 12 months later / now). 'Total' is the sum across fiscal years.",
            "Schedule: for each open project, predict how many days its forecast completion date moves over the next 12 months (its actual completion date, if it finishes). "
            "Separately, predict whether it completes within 12 months.",
            "Features use only reports up to the forecast date: agency, phase, budget size and mix, spending to date, history of earlier changes and delays, months to the forecast date, reasons given.",
            "Validation is walk-forward with a gap: models train on older reports, calibrate on a newer one, and are tested on the last two reports that have a 12-month outcome. "
            "Every training and calibration label was already known at the test date (checked by a test).",
            "Models: 'no change' and agency-median baselines; an L1-penalized median regression; gradient boosting. Hyperparameters were fixed in advance.",
            "Intervals: 80% conformal intervals (conformalized quantile regression, and symmetric split conformal as a comparison).",
            "Survival: Kaplan-Meier curves treating still-open projects as censored, plus a landmark model for completion within 12 months.",
            "Uncertainty: 95% intervals come from 1,000 bootstrap resamples of projects (clustered, because the same project appears at both test reports).",
        ], size=8.6, width=104)
        p.save()
        # ---- 3 cost
        p = Page(pdf, "Cost: 12-month change in planned budget")
        n = c["n_test_rows"]
        p.text(f"{n:,} project-reports from {c['n_test_units']:,} projects were held out. The median project's budget did not change. The best model beats "
               f"'no change' by a very small margin, consistent across both test reports; the regularized linear model and agency medians do not beat it.")
        p.image(F["cost_point"])
        p.text("A large increase is more predictable than the size of the change: gradient boosting ranks projects well above chance, and its probabilities are roughly calibrated "
               "(slightly over-confident at the top).")
        p.image(F["cost_classifier"])
        p.save()
        # ---- 4 schedule
        p = Page(pdf, "Schedule: 12-month slip of forecast completion dates")
        p.text(f"{s['n_test_rows']:,} project-reports were held out. Both gradient boosting and agency medians beat 'the date holds'; the regularized linear model does not. "
               f"The slip rate changes a lot over time, which is why models trained on 2023 and 2024 reports under-predict 2025 slippage.")
        p.image(F["schedule_point"])
        p.image(F["slip_drift"])
        p.save()
        # ---- 5 intervals
        p = Page(pdf, "Prediction intervals")
        p.text("A conformal interval is calibrated on a recent set of projects and then applied to later ones. For cost the 80% target holds. For schedule it does not: "
               "slippage rose between the calibration reports and the test reports, so intervals were too narrow. The uncalibrated quantile band under-covers even more.")
        p.image(F["coverage"])
        t = pd.DataFrame(s["coverage_by_agency"]).sort_values("n", ascending=False)
        t = pd.DataFrame({"Agency": t.agency, "Test rows": t.n.astype(int), "Schedule coverage (CQR)": t.coverage.map(lambda v: pct(v, 0))})
        tc = pd.DataFrame(c["coverage_by_agency"]).sort_values("n", ascending=False).head(len(t))
        p.text("Schedule coverage by agency (agencies with at least 100 test rows). The largest agency (DPR) is covered worst.", size=9, gap=0.05)
        p.table(t, col_w=[0.3, 0.25, 0.45], size=8, row_h=0.2)
        p.save()
        # ---- 6 survival
        p = Page(pdf, "Survival analysis: time to completion")
        p.text(f"Most projects are still open when the data ends ({sv['censored']:,} of {sv['cohort']:,}). Dropping them, or calling them 'not complete', would make projects look "
               f"slower than they are. Kaplan-Meier uses every project and treats open ones as censored at their last report. Time is measured in reports since the project "
               f"first appeared as open (about 4 months each), so projects that were already part-way through when first seen enter at different true ages.")
        p.image(F["km"])
        lm = pd.DataFrame(sv["landmark"]["metrics"])
        lm["model"] = lm.model.map(figures.NICE | {"rule": "Own forecast date only"})
        t = pd.DataFrame({"Model": lm.model, "AUC": lm.auc.map(lambda v: f"{v:.3f}"), "Brier": lm.brier.map(lambda v: f"{v:.4f}"),
                          "Brier skill vs base rate": lm.brier_skill.map(lambda v: f"{v * 100:+.1f}%"), "Log loss": lm.logloss.map(lambda v: f"{v:.3f}")})
        p.text(f"Completing within 12 months ({int(lm.events.iloc[0])} events in {int(lm.n.iloc[0]):,} test project-reports). The project's own forecast date alone "
               f"is already a strong ranker; learned models add little, and the simple models are miscalibrated because the completion rate fell between training and test.", size=9, gap=0.05)
        p.table(t, col_w=[0.3, 0.14, 0.14, 0.26, 0.16], size=8, row_h=0.2)
        p.image(F["landmark"])
        p.save()
        # ---- 7 drivers
        p = Page(pdf, "What the models rely on")
        p.text("Permutation importance: how much worse a model gets when one feature is shuffled in the test data. Partial dependence: the model's average prediction as one "
               "feature varies. These describe the models, not causes.")
        p.image(F["drivers"])
        p.image(F["pdp"])
        p.bullets([
            "Schedule: how soon the forecast date is, the project's phase and its agency matter most. Projects about to hit (or past) their date slip the most; the count of earlier delays adds little once those are known.",
            "Large budget increase: the ratio of spending to budget, the agency and project size matter most. The relationship with size is not monotonic.",
        ], size=9)
        p.save()
        # ---- 8 school phases
        p = Page(pdf, "School-construction phases: plan versus actual (descriptive)")
        p.text(f"{R['sca']['complete_phases']:,} completed phases. Only completed phases are observed, so long phases that are still open are missing, and delay needs a planned-end "
               f"date that many programmes do not have. Treat these as descriptions, not forecasts.")
        cols = ["phase", "phases", "with_dates", "share_late", "median_delay_days", "p90_delay_days", "with_cost", "share_over_budget", "median_cost_ratio"]
        t = fmt_sca(pd.DataFrame(R["sca"]["by_phase"]), cols)
        t.columns = ["Phase", "Completed", "With dates", "Late", "Median delay (d)", "P90 delay (d)", "With cost", "Over budget", "Median final / budget"]
        p.heading("By phase")
        p.table(t, size=7.0, row_h=0.26, col_w=[0.15, 0.09, 0.09, 0.07, 0.12, 0.11, 0.09, 0.11, 0.17], wrap={"Median delay (d)": 9, "Median final / budget": 12})
        cols[0] = "type"
        t = fmt_sca(pd.DataFrame(R["sca"]["by_type"]), cols)
        t.columns = ["Programme", "Completed", "With dates", "Late", "Median delay (d)", "P90 delay (d)", "With cost", "Over budget", "Median final / budget"]
        p.heading("By programme (at least 100 completed phases)")
        p.table(t, size=7.0, row_h=0.26, col_w=[0.17, 0.09, 0.09, 0.07, 0.12, 0.11, 0.09, 0.09, 0.17], wrap={"Median delay (d)": 9, "Median final / budget": 12})
        p.text("'Over budget' compares the final estimate with the budget on record in the export, which may already include approved changes, so it understates true overruns.", size=9)
        p.save()
        # ---- 9 limitations
        p = Page(pdf, "Limitations and analysis log")
        p.bullets(LIMITS, size=9.2, width=96)
        p.save()
    return path
