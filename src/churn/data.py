"""Load the Telco churn CSV and share the train/test split across scripts."""
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATA_PATH = "data/telco_churn.csv"
RANDOM_SEED = 42
LABEL_COL = "Churn"
ID_COL = "customerID"
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def load_churn_data(path=DEFAULT_DATA_PATH):
    """Return (df, X, y): full frame, feature frame (no ID/label), binary label."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    X = df.drop(columns=[ID_COL, LABEL_COL])
    y = (df[LABEL_COL] == "Yes").astype(int)
    return df, X, y


def make_train_test_split(X, y, test_size=0.2):
    """Stratified split with the fixed shared seed."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
    )
