import pandas as pd

from nycforecast import load


def test_money_parsing():
    s = pd.Series(["$1,525,000", "$-558,100", "", None, "$0"])
    out = load._money(s).tolist()
    assert out[0] == 1525000 and out[1] == -558100 and out[4] == 0
    assert pd.isna(out[2]) and pd.isna(out[3])


def test_period_table_maps_fiscal_year():
    pt = load.period_table([202309, 202305, 202401])
    assert pt.t.tolist() == [0, 1, 2]
    # May 2023 belongs to FY2023, Sept 2023 to FY2024 (fiscal year starts July 1)
    assert pt.fiscal_year.tolist() == [2023, 2024, 2024]


def test_phase_normalization_merges_spellings():
    assert load.normalize_phase("Construction procurement") == "Construction Procurement"
    assert load.normalize_phase("CONSTRUCTION") == "Construction"
    assert load.normalize_phase(None) == "Unknown"


def test_reason_groups():
    assert load.reason_group("PROJECT DELAYED PENDING APPROVAL OF NECESSARY PERMITS") == "permits"
    assert load.reason_group("PROJECT DELAYED DUE TO UNFORESEEN HAZARDOUS CONDITIONS") == "site conditions"
    assert load.reason_group(None) is None
