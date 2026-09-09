"""End-to-end self-check: run all four scripts on the real CSV and assert the
story-001 ACs plus the story-002 extensions (EV column, survival, uplift).
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
        "expected_monthly_loss",
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

    # --- Story 002 extensions ---

    # EV column formula spot-check: churn_probability * MonthlyCharges, 2 dp.
    data = pd.read_csv(ROOT / "data" / "telco_churn.csv")
    merged = recs.merge(data[["customerID", "MonthlyCharges"]],
                        on="customerID", validate="one_to_one")
    expected_ev = (merged["churn_probability"] * merged["MonthlyCharges"]).round(2)
    assert (merged["expected_monthly_loss"] - expected_ev).abs().max() < 1e-9

    n_mid = len(METRICS.read_text().splitlines())
    run("survival_analysis.py")
    run("uplift_simulation.py")
    by_script = {
        rec["script"]: rec
        for rec in (json.loads(line)
                    for line in METRICS.read_text().splitlines()[n_mid:])
    }

    # Survival: KM long-form CSV; horizons active-only, in [0, 1], sorted 12m desc.
    km = pd.read_csv(ROOT / "results" / "survival_by_contract.csv")
    assert list(km.columns) == ["contract", "month", "retention_probability"]
    horizons = pd.read_csv(ROOT / "results" / "churn_horizons.csv")
    assert set(horizons["customerID"]) == set(data.loc[data["Churn"] == "No", "customerID"])
    for col in ("p_churn_6m", "p_churn_12m"):
        assert horizons[col].between(0, 1).all(), col
    assert horizons["p_churn_12m"].is_monotonic_decreasing

    # Cox test concordance >= 0.80 (SC-004), from the appended record.
    concordance = by_script["survival_analysis"]["concordance_index"]
    assert concordance >= 0.80, concordance

    # Uplift recovery: top decile >= 2x population TRUE effect (SC-005).
    ratio = by_script["uplift_simulation"]["top_decile_vs_population_ratio"]
    assert ratio >= 2, ratio

    # SIMULATED labeling (AD-4): uplift CSV filename, README, exec summary.
    uplift_csv = ROOT / "results" / "uplift_deciles_simulated.csv"
    assert uplift_csv.exists() and "simulat" in uplift_csv.name.lower()
    for doc in (ROOT / "README.md", ROOT / "docs" / "index.html"):
        assert "simulat" in doc.read_text().lower(), doc

    print(f"All checks passed (best ROC-AUC {best:.3f}, "
          f"{recs['recommended_action'].nunique()} distinct actions, "
          f"Cox concordance {concordance:.3f}, uplift recovery {ratio:.2f}x).")


if __name__ == "__main__":
    main()
