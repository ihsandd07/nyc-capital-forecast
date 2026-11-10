import numpy as np
import pandas as pd

from nycforecast import panel

T = 6


def make_budget():
    rows = []
    totals = {"A|1": [100, 100, 120, 120, 150, 150], "B|2": [50, 50, 50, 40, 40, 40]}
    for unit, vals in totals.items():
        for t, v in enumerate(vals):
            rows.append(dict(unit=unit, agency=unit[0], t=t, rel_fy=1, budget=float(v), spend=np.nan))
            rows.append(dict(unit=unit, agency=unit[0], t=t, rel_fy=-1, budget=10.0, spend=5.0))
    return pd.DataFrame(rows)


def make_schedule():
    rows = []
    base = pd.Timestamp("2025-01-01")
    for unit, cds in {"A|1": [0, 0, 100, 100, 100, 300], "B|2": [0, 50, 50, 50, 50, 50]}.items():
        for t, d in enumerate(cds):
            rows.append(dict(unit=unit, agency=unit[0], pid=unit[2:], name="p" + unit, t=t, phase="Design",
                             cd=base + pd.Timedelta(days=int(d) + 400), type="Forecast", variance=np.nan,
                             reason="PROJECT DELAYED PENDING APPROVAL OF NECESSARY PERMITS" if t == 2 else None,
                             data_date=base + pd.Timedelta(days=120 * t)))
    return pd.DataFrame(rows)


def test_cost_label_and_features():
    p = panel.cost_panel(make_budget(), T, H=3)
    a = p[(p.unit == "A|1") & (p.origin == 2)].iloc[0]
    total_o = 130.0   # 120 planned + 10 past
    assert np.isclose(a.growth_since_first, np.log(130 / 110))
    assert np.isclose(a.y, np.log(160 / 130))        # total at origin 5 over total at origin 2
    assert a.n_increases == 1


def test_cost_features_ignore_the_future():
    b = make_budget()
    p1 = panel.cost_panel(b, T, H=3)
    b2 = b.copy()
    b2.loc[b2.t > 2, "budget"] = b2.loc[b2.t > 2, "budget"] * 7 + 3   # rewrite everything after origin 2
    p2 = panel.cost_panel(b2, T, H=3)
    cols = panel.COST_NUM
    f1 = p1[p1.origin == 2].set_index("unit")[cols].sort_index()
    f2 = p2[p2.origin == 2].set_index("unit")[cols].sort_index()
    pd.testing.assert_frame_equal(f1, f2)


def test_schedule_label_and_features():
    p = panel.schedule_panel(make_schedule(), T, H=3)
    a = p[(p.unit == "A|1") & (p.origin == 2)].iloc[0]
    assert a.y == 200                      # forecast moves from day 100 to day 300 by origin 5
    assert a.n_delays == 1 and a.max_delay_days == 100
    assert a.slip_since_first == 100 and a.reason_permits == 1


def test_schedule_features_ignore_the_future():
    s = make_schedule()
    p1 = panel.schedule_panel(s, T, H=3)
    s2 = s.copy()
    s2.loc[s2.t > 2, "cd"] = s2.loc[s2.t > 2, "cd"] + pd.Timedelta(days=999)
    s2.loc[s2.t > 2, "reason"] = "PROJECT DELAYED PENDING RESOLUTION OF LEGAL ISSUES"
    p2 = panel.schedule_panel(s2, T, H=3)
    cols = panel.SCHED_NUM
    f1 = p1[p1.origin == 2].set_index("unit")[cols].sort_index()
    f2 = p2[p2.origin == 2].set_index("unit")[cols].sort_index()
    pd.testing.assert_frame_equal(f1, f2)


def test_walk_forward_labels_known_before_test_origin():
    rows = [dict(unit=f"u{i}", origin=o, y=1.0) for o in range(10) for i in range(3)]
    p = pd.DataFrame(rows)
    for o in (5, 6):
        train, cal, test = panel.walk_forward_split(p, o, H=3)
        assert (train.origin + 3 <= o).all() and (cal.origin + 3 <= o).all()   # every label already observed at the test origin
        assert train.origin.max() < cal.origin.min() <= test.origin.min()
        assert set(test.origin) == {o}
