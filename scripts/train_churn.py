"""Train and evaluate the churn models (LogisticRegression, HistGradientBoosting)
and append their test metrics to results/metrics.jsonl."""
import argparse

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from churn.data import (
    DEFAULT_DATA_PATH,
    NUMERIC_COLS,
    RANDOM_SEED,
    load_churn_data,
    make_train_test_split,
)
from churn.metrics import DEFAULT_METRICS_PATH, save_metrics
from churn.model import build_logreg_pipeline, categorical_columns


def recall_top_decile(y_true, y_proba):
    """Share of true churners captured in the top 10% by predicted probability."""
    y_true = np.asarray(y_true)
    n_top = max(1, int(np.ceil(0.1 * len(y_true))))
    top_idx = np.argsort(y_proba)[::-1][:n_top]
    return float(y_true[top_idx].sum() / y_true.sum())


def evaluate(y_true, y_proba):
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "recall_top_decile": recall_top_decile(y_true, y_proba),
    }


def build_hgb_pipeline(X):
    """Ordinal-encoded categoricals (flagged as categorical) + raw numerics."""
    cat_cols = categorical_columns(X)
    preprocess = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             cat_cols),
            ("num", "passthrough", NUMERIC_COLS),
        ]
    )
    clf = HistGradientBoostingClassifier(
        categorical_features=list(range(len(cat_cols))),
        random_state=RANDOM_SEED,
    )
    return Pipeline([("preprocess", preprocess), ("clf", clf)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--out", default=DEFAULT_METRICS_PATH)
    args = parser.parse_args()

    _, X, y = load_churn_data(args.data)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    models = {
        "LogisticRegression": build_logreg_pipeline(X),
        "HistGradientBoosting": build_hgb_pipeline(X),
    }

    records = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_proba)
        print(f"{name}: {metrics}")
        records.append({"model": name, **metrics})

    save_metrics(records, path=args.out, script="train_churn")
    print(f"Saved metrics to {args.out}")


if __name__ == "__main__":
    main()
