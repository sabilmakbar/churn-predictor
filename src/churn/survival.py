"""Survival analysis shared pieces: leak-free Cox feature frame, KM retention
curves per group, and conditional churn-horizon probabilities."""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

from churn.model import categorical_columns

DURATION_COL = "tenure"
EVENT_COL = "churned"


def build_survival_frame(X, y):
    """Duration + event + covariates. Excludes tenure (the duration) and
    TotalCharges (~ tenure * MonthlyCharges, a duration leak) as covariates."""
    covariates = pd.get_dummies(X[categorical_columns(X)], drop_first=True)
    covariates["SeniorCitizen"] = X["SeniorCitizen"]
    covariates["MonthlyCharges"] = X["MonthlyCharges"]
    surv = covariates.astype(float)
    surv[DURATION_COL] = X["tenure"]
    surv[EVENT_COL] = y
    return surv


def fit_cox(train_df):
    """CoxPH with a small penalizer: the one-hot covariates are collinear and an
    unpenalized fit does not converge."""
    cox = CoxPHFitter(penalizer=0.1)
    cox.fit(train_df, duration_col=DURATION_COL, event_col=EVENT_COL)
    return cox


def km_retention_by_group(durations, events, groups):
    """Long-form KM retention: one row per (group, month, retention_probability)."""
    rows = []
    kmf = KaplanMeierFitter()
    for group in sorted(groups.unique()):
        mask = groups == group
        kmf.fit(durations[mask], events[mask])
        for month, retention in kmf.survival_function_.iloc[:, 0].items():
            rows.append((group, int(month), float(retention)))
    return pd.DataFrame(rows, columns=["contract", "month", "retention_probability"])


def churn_horizons(cox, active_df, tenures, horizons=(6, 12)):
    """Conditional churn within k months given survival to current tenure t:
    1 - S(t+k)/S(t). Predicts each customer's survival curve once on the shared
    0..72 grid; t+k beyond the grid is clipped to 72 (lifelines would
    flat-extrapolate there anyway, so risk is understated for those rows)."""
    grid = np.arange(0, 73)
    surv = cox.predict_survival_function(active_df, times=grid).to_numpy()
    t = np.asarray(tenures, dtype=int)
    idx = np.arange(surv.shape[1])
    s_now = surv[np.clip(t, 0, 72), idx]
    out = {}
    for k in horizons:
        s_later = surv[np.clip(t + k, 0, 72), idx]
        out[f"p_churn_{k}m"] = 1.0 - s_later / s_now
    return out
