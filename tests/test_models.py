import numpy as np
import pandas as pd

from nycforecast import models, survival


def test_kaplan_meier_matches_hand_calculation():
    km = survival.kaplan_meier([1, 2, 2, 3], [1, 1, 0, 1])
    # t=1: 1 event of 4 at risk -> 0.75 ; t=2: 1 event of 3 at risk -> 0.5 ; t=3: 1 event of 1 -> 0
    assert np.allclose(km.survival.tolist(), [0.75, 0.5, 0.0])
    assert km.at_risk.tolist() == [4, 3, 1]
    assert ((km.lo <= km.survival + 1e-12) & (km.survival <= km.hi + 1e-12)).all()
    assert survival.surv_at(km, 2) == 0.5 and survival.surv_at(km, 0) == 1.0


def test_kaplan_meier_censoring_is_not_an_event():
    km = survival.kaplan_meier([5, 5, 5], [0, 0, 0])
    assert km.empty


def test_conformal_level_is_finite_sample_corrected():
    scores = np.arange(1, 10, dtype=float)            # 9 scores
    q = models._conformal_q(scores, alpha=0.2)         # ceil(0.8 * 10) = 8th smallest
    assert q == 8.0


def test_cqr_reaches_target_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    n = 6000
    x = rng.uniform(0, 3, n)
    y = rng.normal(0, 1 + x)                           # noise grows with x (heteroskedastic)
    lo = -0.5 * np.ones(n); hi = 0.5 * np.ones(n)      # a deliberately bad quantile band
    cal, test = slice(0, 3000), slice(3000, 6000)
    l, h, q = models.cqr_interval(lo[cal], hi[cal], y[cal], lo[test], hi[test], alpha=0.2)
    cov = np.mean((y[test] >= l) & (y[test] <= h))
    assert 0.77 <= cov <= 0.83 and q > 0


def test_split_conformal_symmetry():
    y_cal = np.array([0.0, 1.0, -1.0, 2.0, -2.0]); pred_cal = np.zeros(5)
    l, h, q = models.split_conformal_interval(pred_cal, y_cal, np.array([10.0]), alpha=0.4)
    assert np.isclose(10 - l[0], h[0] - 10) and q > 0


def test_group_median_handles_unseen_group():
    X = pd.DataFrame({"agency": ["a", "a", "b"]})
    m = models.GroupMedian().fit(X, np.array([1.0, 3.0, 10.0]))
    out = m.predict(pd.DataFrame({"agency": ["a", "b", "zzz"]}))
    assert out.tolist() == [2.0, 10.0, 3.0]
