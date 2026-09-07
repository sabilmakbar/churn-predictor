"""LogReg pipeline shared by the training and recommendation scripts."""
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.data import NUMERIC_COLS


def categorical_columns(X):
    return [c for c in X.columns if c not in NUMERIC_COLS]


def build_logreg_pipeline(X):
    """One-hot categoricals + scaled numerics -> LogisticRegression."""
    preprocess = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical_columns(X)),
            ("num", StandardScaler(), NUMERIC_COLS),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def transformed_feature_names(pipeline, X):
    """Human-readable names for the transformed feature space, in column order:
    one-hot categoricals as ``col=value``, then the numeric columns as-is."""
    ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = [
        f"{col}={val}"
        for col, cats in zip(categorical_columns(X), ohe.categories_)
        for val in cats
    ]
    return cat_names + NUMERIC_COLS
