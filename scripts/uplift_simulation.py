"""SIMULATED uplift study: a fixed-seed simulated retention campaign with a
known built-in treatment effect, used to check that a T-learner (two LogReg
pipelines) recovers the persuadable segment. All outcomes here are simulated;
nothing in this script measures real campaign effects."""
import argparse
import os

import numpy as np
import pandas as pd

from churn.data import DEFAULT_DATA_PATH, load_churn_data, make_train_test_split
from churn.metrics import DEFAULT_METRICS_PATH, save_metrics
from churn.model import build_logreg_pipeline

DEFAULT_OUT_PATH = "results/uplift_deciles_simulated.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    parser.add_argument("--metrics-out", default=DEFAULT_METRICS_PATH)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    _, X, y = load_churn_data(args.data)
    n = len(X)

    # Base churn probability from the story-001 LogReg pipeline.
    X_train, _, y_train, _ = make_train_test_split(X, y)
    base = build_logreg_pipeline(X)
    base.fit(X_train, y_train)
    p = base.predict_proba(X)[:, 1]

    # Known TRUE effect: large for the persuadable segment, small elsewhere.
    persuadable = (X["Contract"] == "Month-to-month").to_numpy() & (p >= 0.3) & (p <= 0.7)
    effect = np.where(persuadable, 0.15, 0.03)
    p_treated = np.clip(p - effect, 0.0, None)

    # Draw sequence is load-bearing for SC-005 (see story 002); do not restructure.
    treated = rng.random(n) < 0.5
    outcome = (rng.random(n) < np.where(treated, p_treated, p)).astype(int)

    # T-learner: fresh LogReg pipelines on treated and control rows.
    model_t = build_logreg_pipeline(X)
    model_t.fit(X[treated], outcome[treated])
    model_c = build_logreg_pipeline(X)
    model_c.fit(X[~treated], outcome[~treated])
    predicted_uplift = model_c.predict_proba(X)[:, 1] - model_t.predict_proba(X)[:, 1]

    per_customer = pd.DataFrame(
        {
            "predicted_uplift": predicted_uplift,
            "true_effect": effect,
            "treated": treated,
            "outcome": outcome,
            # decile 1 = highest predicted uplift
            "decile": 10 - pd.qcut(predicted_uplift, 10, labels=False),
        }
    )
    deciles = (
        per_customer.groupby("decile")
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "mean_predicted_uplift": g["predicted_uplift"].mean(),
                    "mean_true_effect": g["true_effect"].mean(),
                    "simulated_churn_treated": g.loc[g["treated"], "outcome"].mean(),
                    "simulated_churn_control": g.loc[~g["treated"], "outcome"].mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    deciles["n"] = deciles["n"].astype(int)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    deciles.to_csv(args.out, index=False)
    print(f"Wrote {len(deciles)} SIMULATED uplift deciles to {args.out}")

    top_decile_true_effect = float(deciles.loc[deciles["decile"] == 1, "mean_true_effect"].iloc[0])
    population_true_effect = float(effect.mean())
    ratio = top_decile_true_effect / population_true_effect
    print(f"Top-decile TRUE effect {top_decile_true_effect:.4f} vs population "
          f"{population_true_effect:.4f} ({ratio:.2f}x)")

    save_metrics(
        {
            "model": "TLearnerUplift (SIMULATED campaign)",
            "top_decile_true_effect": top_decile_true_effect,
            "population_true_effect": population_true_effect,
            "top_decile_vs_population_ratio": ratio,
            "decile_counts": deciles["n"].astype(int).tolist(),
        },
        path=args.metrics_out,
        script="uplift_simulation",
    )
    print(f"Saved metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
