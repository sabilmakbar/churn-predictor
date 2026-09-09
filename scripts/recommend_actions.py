"""Score all customers with the LogReg pipeline, attribute top-3 churn drivers,
and write the top-risk decile with recommended retention actions to CSV."""
import argparse
import os

import numpy as np
import pandas as pd

from churn.data import DEFAULT_DATA_PATH, ID_COL, load_churn_data, make_train_test_split
from churn.actions import recommend_action
from churn.model import build_logreg_pipeline, transformed_feature_names

DEFAULT_OUT_PATH = "results/retention_recommendations.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    df, X, y = load_churn_data(args.data)
    X_train, _, y_train, _ = make_train_test_split(X, y)

    pipeline = build_logreg_pipeline(X)
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X)[:, 1]

    # Per-customer driver contributions in transformed feature space:
    # coef * (x_transformed - train mean of x_transformed).
    preprocess = pipeline.named_steps["preprocess"]
    coef = pipeline.named_steps["clf"].coef_[0]
    contributions = coef * (preprocess.transform(X) - preprocess.transform(X_train).mean(axis=0))
    feature_names = np.array(transformed_feature_names(pipeline, X))
    top3 = np.argsort(contributions, axis=1)[:, ::-1][:, :3]
    drivers = feature_names[top3]

    out = pd.DataFrame(
        {
            ID_COL: df[ID_COL],
            "churn_probability": proba,
            "driver_1": drivers[:, 0],
            "driver_2": drivers[:, 1],
            "driver_3": drivers[:, 2],
        }
    )
    out["recommended_action"] = [recommend_action(row) for row in drivers]
    out["expected_monthly_loss"] = (proba * df["MonthlyCharges"]).round(2)

    n_top = max(1, int(np.ceil(0.1 * len(out))))
    out = out.sort_values("churn_probability", ascending=False).head(n_top)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows to {args.out}")
    print(out["recommended_action"].value_counts().to_string())


if __name__ == "__main__":
    main()
