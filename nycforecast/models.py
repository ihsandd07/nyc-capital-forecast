"""Baselines, a regularized linear model, gradient boosting, and conformal prediction intervals."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config


def signed_log(x):
    """Compress heavy tails (e.g. days) while keeping the sign."""
    return np.sign(x) * np.log1p(np.abs(x) / 30.0)


def _prep(num, cat, heavy, scale: bool):
    """Column transformer. `heavy` columns get a signed log. Numerics are standardized if `scale` (linear models)."""
    plain = [c for c in num if c not in heavy]
    parts = []
    if heavy:
        steps = [("log", FunctionTransformer(signed_log))] + ([("sc", StandardScaler())] if scale else [])
        parts.append(("heavy", Pipeline(steps), list(heavy)))
    if plain:
        parts.append(("num", StandardScaler() if scale else "passthrough", plain))
    parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(cat)))
    return ColumnTransformer(parts)


class ZeroBaseline:
    """'Nothing changes': forecast completion date holds / planned budget stays the same."""
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class GroupMedian:
    """Median historical change by agency (falls back to the overall median for unseen agencies)."""
    def __init__(self, col="agency"):
        self.col = col

    def fit(self, X, y):
        d = pd.DataFrame({"g": X[self.col].to_numpy(), "y": np.asarray(y)})
        self.med_ = d.groupby("g")["y"].median()
        self.glob_ = float(np.median(y))
        return self

    def predict(self, X):
        return X[self.col].map(self.med_).fillna(self.glob_).to_numpy()


def linear_median(num, cat, heavy):
    """L1-penalized median regression: robust to the heavy tails in both targets."""
    return Pipeline([("prep", _prep(num, cat, heavy, scale=True)),
                     ("reg", QuantileRegressor(quantile=0.5, alpha=config.QR_ALPHA, solver="highs"))])


def gbm(num, cat, heavy, loss="absolute_error", quantile=None):
    kw = dict(config.HGB_PARAMS)
    if loss == "quantile":
        kw.update(loss="quantile", quantile=quantile)
    else:
        kw.update(loss=loss)
    return Pipeline([("prep", _prep(num, cat, heavy, scale=False)), ("reg", HistGradientBoostingRegressor(**kw))])


def gbm_classifier(num, cat, heavy):
    kw = {k: v for k, v in config.HGB_PARAMS.items()}
    return Pipeline([("prep", _prep(num, cat, heavy, scale=False)), ("clf", HistGradientBoostingClassifier(**kw))])


def logit_classifier(num, cat, heavy):
    return Pipeline([("prep", _prep(num, cat, heavy, scale=True)),
                     ("clf", LogisticRegression(C=0.5, max_iter=2000))])


# ----------------------------------------------------------------------------- conformal intervals
def _conformal_q(scores, alpha):
    """The ceil((n+1)(1-alpha))-th smallest score: the finite-sample-corrected conformal quantile."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[min(k, n) - 1])   # if k > n the exact answer is +inf; use the largest score


def cqr_interval(lo_cal, hi_cal, y_cal, lo_test, hi_test, alpha):
    """Conformalized quantile regression (Romano et al. 2019): widen (or shrink) the model's own
    [alpha/2, 1-alpha/2] quantile band by the amount that gives the target coverage on the calibration set."""
    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    q = _conformal_q(scores, alpha)
    return lo_test - q, hi_test + q, q


def split_conformal_interval(pred_cal, y_cal, pred_test, alpha):
    """Symmetric interval: point forecast +/- the (1-alpha) quantile of absolute calibration errors."""
    q = _conformal_q(np.abs(y_cal - pred_cal), alpha)
    return pred_test - q, pred_test + q, q
