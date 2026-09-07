"""End-to-end self-check: run both scripts on the real CSV and assert ACs 1-5.
Run with: uv run python test_churn.py
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from churn.actions import ACTION_RULES, recommend_action

ROOT = Path(__file__).parent
METRICS = ROOT / "results" / "metrics.jsonl"
RECS = ROOT / "results" / "retention_recommendations.csv"


def run(script):
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True, cwd=ROOT)


def main():
    n_before = len(METRICS.read_text().splitlines()) if METRICS.exists() else 0
    run("train_churn.py")
    run("recommend_actions.py")

    # AC1: one appended record per model, each with the three metrics.
    lines = METRICS.read_text().splitlines()
    new = [json.loads(line) for line in lines[n_before:]]
    assert {rec["model"] for rec in new} == {"LogisticRegression", "HistGradientBoosting"}, new
    for rec in new:
        assert {"roc_auc", "pr_auc", "recall_top_decile"} <= rec.keys(), rec

    # AC2: exactly the top-risk decile, right columns, sorted descending.
    recs = pd.read_csv(RECS)
    n_customers = len(pd.read_csv(ROOT / "data" / "telco_churn.csv"))
    assert len(recs) == -(-n_customers // 10), len(recs)  # ceil(n/10)
    assert list(recs.columns) == [
        "customerID", "churn_probability",
        "driver_1", "driver_2", "driver_3", "recommended_action",
    ], list(recs.columns)
    assert recs["churn_probability"].is_monotonic_decreasing

    # AC3: driver list headed by Contract=Month-to-month -> annual-plan discount.
    action = recommend_action(["Contract=Month-to-month", "tenure", "MonthlyCharges"])
    assert action == ACTION_RULES["Contract=Month-to-month"] == "Offer discounted 1-year contract"

    # AC4: every action non-empty, >=2 distinct actions.
    assert recs["recommended_action"].str.len().gt(0).all()
    assert recs["recommended_action"].nunique() >= 2

    # AC5: best test ROC-AUC >= 0.80 (from the appended records).
    best = max(rec["roc_auc"] for rec in new)
    assert best >= 0.80, best

    print(f"All checks passed (best ROC-AUC {best:.3f}, "
          f"{recs['recommended_action'].nunique()} distinct actions).")


if __name__ == "__main__":
    main()
