"""Run everything: load data, evaluate models, write results, the PDF report and the dashboard.

    python run_all.py

Outputs go to reports/ (results.json, tables/*.csv, report.pdf, dashboard.html).
"""
import time
import warnings

warnings.filterwarnings("ignore")

from nycforecast import analysis, dashboard, report  # noqa: E402


def main():
    t0 = time.time()
    R, _ = analysis.run()
    print("report:", report.build(R=None))
    print("dashboard:", dashboard.build(R=None))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
