# Churn predictor with retention recommendations

Predicts customer churn on the Telco dataset and recommends a retention action
for each customer in the top-risk decile, with the top-3 churn drivers per
customer.

## Structure

```
churn-predictor/
├── data/
│   └── telco_churn.csv          # raw dataset (7,043 customers)
├── scripts/
│   ├── train_churn.py           # train + evaluate models, append metrics
│   └── recommend_actions.py     # score customers, write recommendations CSV
├── src/churn/
│   ├── data.py                  # CSV loading, cleaning, shared train/test split
│   ├── model.py                 # shared LogReg pipeline + feature naming
│   ├── metrics.py               # JSONL metrics appender
│   └── actions.py               # driver -> retention action rules
├── results/
│   ├── metrics.jsonl            # appended run metrics
│   └── retention_recommendations.csv
├── docs/
└── test_churn.py                # end-to-end self-check (ACs 1-5)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python >= 3.11:

```
uv sync
```

## Run

```
uv run python scripts/train_churn.py          # trains both models, appends metrics
uv run python scripts/recommend_actions.py    # writes results/retention_recommendations.csv
uv run python test_churn.py                   # end-to-end self-check
```

## Results

Both models are evaluated on a stratified 20% test split (fixed seed). Each
training run appends one record per model to `results/metrics.jsonl` with
`roc_auc`, `pr_auc`, and `recall_top_decile` (share of true churners captured
in the top 10% by predicted probability).

| Model                | ROC-AUC | PR-AUC | Recall @ top decile |
|----------------------|---------|--------|---------------------|
| LogisticRegression   | 0.842   | 0.634  | 0.281               |
| HistGradientBoosting | 0.833   | 0.639  | 0.283               |

`results/retention_recommendations.csv` lists the top-risk decile of all
customers (705 rows), sorted by churn probability descending, with each
customer's top-3 churn drivers and a rule-based recommended retention action.

## Dataset

Telco Customer Churn:

- Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- IBM raw CSV: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

The copy lives at `data/telco_churn.csv`; the scripts never download it.

## Submission summary

This repo trains two scikit-learn churn models (Logistic Regression and
HistGradientBoosting) on the Telco customer churn dataset and turns the
predictions into an actionable retention list. Both models clear 0.83 test
ROC-AUC. The logistic pipeline also explains each prediction: per-customer
driver contributions are computed as `coef * (x - train_mean)` in the
transformed feature space, and the top-3 drivers are mapped through a small
rule table to a concrete retention action (contract offers, onboarding
check-ins, support bundles, and more). The output is a CSV of the riskiest
decile of customers, ready for a retention team to work through top-down.
