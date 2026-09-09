# Churn predictor with retention recommendations

Predicts customer churn on the Telco dataset and recommends a retention action
for each customer in the top-risk decile, with the top-3 churn drivers per
customer. On top of the ranking it adds an expected-monthly-loss column,
survival analysis (when churn happens), and a clearly labeled SIMULATED uplift
study (whom an intervention helps).

Executive summary (problem, approach, results, future work):
**<https://sabilmakbar.github.io/churn-predictor/>**

## Structure

```
churn-predictor/
├── data/
│   └── telco_churn.csv          # raw dataset (7,043 customers)
├── scripts/
│   ├── train_churn.py           # train + evaluate models, append metrics
│   ├── recommend_actions.py     # score customers, write recommendations CSV
│   ├── survival_analysis.py     # KM curves, Cox model, churn horizons
│   └── uplift_simulation.py     # SIMULATED campaign + T-learner recovery
├── src/churn/
│   ├── data.py                  # CSV loading, cleaning, shared train/test split
│   ├── model.py                 # shared LogReg pipeline + feature naming
│   ├── metrics.py               # JSONL metrics appender
│   ├── actions.py               # driver -> retention action rules
│   └── survival.py              # Cox feature frame, KM curves, horizon math
├── results/
│   ├── metrics.jsonl            # appended run metrics
│   ├── retention_recommendations.csv
│   ├── survival_by_contract.csv # KM retention by contract type
│   ├── churn_horizons.csv       # churn within 6/12 months per active customer
│   └── uplift_deciles_simulated.csv  # SIMULATED uplift recovery by decile
├── docs/
│   └── index.html               # executive summary, published via GitHub Pages
└── test_churn.py                # end-to-end self-check
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
uv run python scripts/survival_analysis.py    # writes KM curves + churn horizons, appends concordance
uv run python scripts/uplift_simulation.py    # writes SIMULATED uplift deciles, appends recovery metrics
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
customer's top-3 churn drivers, a rule-based recommended retention action, and
an `expected_monthly_loss` column (churn probability x MonthlyCharges) so the
queue can also be worked by revenue at risk.

**Survival analysis.** `scripts/survival_analysis.py` fits Kaplan-Meier
retention curves per contract type (`results/survival_by_contract.csv`) and a
Cox proportional-hazards model with tenure as the duration; covariates exclude
tenure and TotalCharges (a duration leak). Test concordance index: **0.846**.
For every active customer it writes conditional churn-within-6/12-month
probabilities to `results/churn_horizons.csv`, sorted by 12-month risk, so
outreach can be timed, not just ranked.

**Uplift (SIMULATED).** `scripts/uplift_simulation.py` runs a fixed-seed
simulated retention campaign on the real customers: random 50% treatment and a
synthetic outcome with a known built-in effect (large for persuadable
month-to-month customers at mid churn risk, small otherwise). A T-learner (two
LogReg pipelines) then has to recover that effect from the simulated data. It
does: the mean TRUE effect in its top predicted-uplift decile is **2.07x** the
population mean (gate: >= 2x), per-decile detail in
`results/uplift_deciles_simulated.csv`. These are simulated outcomes that
validate the method; they say nothing about real campaign effects.

## Dataset

Telco Customer Churn:

- Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- IBM raw CSV: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

The copy lives at `data/telco_churn.csv`; the scripts never download it.

## Submission summary

This repo turns the Telco churn dataset into an actionable retention plan.
Two scikit-learn models (Logistic Regression and HistGradientBoosting) both
clear 0.83 test ROC-AUC. The logistic pipeline explains each prediction with
per-customer driver contributions, maps the top drivers to a concrete
retention action, and prices each risky customer with an expected monthly
loss column (churn probability times monthly charges), so the queue can be
worked by revenue at risk. A survival layer (Kaplan-Meier plus a Cox model,
0.846 test concordance) adds churn-within-6-and-12-month probabilities that
time the outreach. A clearly labeled SIMULATED uplift study shows a T-learner
recovering a known treatment effect at 2.07x population in its top decile.
