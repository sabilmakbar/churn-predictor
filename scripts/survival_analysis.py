"""Kaplan-Meier retention by contract type, a Cox PH model on leak-free
covariates, and per-active-customer churn horizons (6/12 months); appends the
Cox test concordance index to results/metrics.jsonl."""
import argparse
import os

import pandas as pd

from churn.data import DEFAULT_DATA_PATH, ID_COL, LABEL_COL, load_churn_data, make_train_test_split
from churn.metrics import DEFAULT_METRICS_PATH, save_metrics
from churn.survival import build_survival_frame, churn_horizons, fit_cox, km_retention_by_group

DEFAULT_KM_OUT = "results/survival_by_contract.csv"
DEFAULT_HORIZONS_OUT = "results/churn_horizons.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--km-out", default=DEFAULT_KM_OUT)
    parser.add_argument("--horizons-out", default=DEFAULT_HORIZONS_OUT)
    parser.add_argument("--metrics-out", default=DEFAULT_METRICS_PATH)
    args = parser.parse_args()

    df, X, y = load_churn_data(args.data)
    os.makedirs(os.path.dirname(args.km_out) or ".", exist_ok=True)

    km = km_retention_by_group(X["tenure"], y, X["Contract"])
    km.to_csv(args.km_out, index=False)
    print(f"Wrote {len(km)} KM rows to {args.km_out}")

    surv = build_survival_frame(X, y)
    train_df, test_df, _, _ = make_train_test_split(surv, y)
    cox = fit_cox(train_df)
    concordance = float(cox.score(test_df, scoring_method="concordance_index"))
    print(f"Cox test concordance index: {concordance:.3f}")

    active = df[LABEL_COL] == "No"
    horizons = churn_horizons(cox, surv.loc[active], X.loc[active, "tenure"])
    out = pd.DataFrame({ID_COL: df.loc[active, ID_COL], **horizons})
    out = out.sort_values("p_churn_12m", ascending=False)
    out.to_csv(args.horizons_out, index=False)
    print(f"Wrote {len(out)} active-customer horizons to {args.horizons_out}")

    save_metrics(
        {"model": "CoxPH", "concordance_index": concordance},
        path=args.metrics_out,
        script="survival_analysis",
    )
    print(f"Saved metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
