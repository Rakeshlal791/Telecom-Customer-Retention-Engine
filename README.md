# Customer Churn Prediction

This project builds a telecom churn prediction pipeline to identify customers at high risk of leaving, using demographic, service, contract, and billing data to support retention targeting.

It compares logistic regression and XGBoost, applies class-imbalance handling, tunes the classification threshold with cross-validation, and saves a final model that can be used to score unseen customers.

## Problem

Customer churn is expensive because every preventable cancellation represents lost recurring revenue and higher reacquisition cost. The objective here is to identify likely churners early enough for a retention team to intervene.

## Dataset

- `train.csv`: 594,194 labeled customers
- `test.csv`: 254,655 unlabeled customers
- Features: demographics, service configuration, contract type, billing behavior, and charges
- Target: `Churn` (`Yes` / `No`)

The notebooks in this repository contain the original EDA and first-pass modeling work:
- `EDA_SeniorCitizens.ipynb`
- `EDA_High_payers_analysis.ipynb`
- `churn_pred_log_reg.ipynb`
- `churn_pred_xgboost.ipynb`

## Project Structure

The repository includes:

- `train.py`: trains baseline models, tunes thresholds with cross-validation, evaluates on a holdout set, and saves artifacts
- `predict.py`: loads the best saved model and scores `test.csv`
- `src/churn_predictor/`: reusable data loading and modeling utilities
- `reports/`: generated model comparison tables and summaries
- `models/`: serialized model artifacts

The notebooks are kept as the original exploratory and baseline modeling work, while the Python scripts provide a cleaner and reproducible training flow.

Relevant docs:
- scikit-learn threshold tuning: https://scikit-learn.org/stable/modules/classification_threshold.html
- scikit-learn `TunedThresholdClassifierCV`: https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html
- XGBoost parameters: https://xgboost.readthedocs.io/en/latest/parameter.html

## Modeling Approach

Two supervised baselines are trained and compared:

1. Logistic regression with class balancing
2. XGBoost with imbalance-aware `scale_pos_weight`

Shared preprocessing:

- Convert `TotalCharges` to numeric safely
- Convert `SeniorCitizen` to categorical labels for one-hot encoding
- Median imputation for numeric features
- Most-frequent imputation for categorical features
- Standard scaling for numeric features
- One-hot encoding for categorical features

Evaluation design:

1. Split labeled data into development and holdout sets using stratified sampling
2. Train each model on the development set
3. Tune the probability threshold with internal cross-validation on the development set
4. Report final holdout metrics: F1, precision, recall, accuracy, ROC-AUC, and PR-AUC
5. Retrain the winning model on all labeled rows
6. Save the final full-data model artifact

## Results

On the 118,839-row holdout set, XGBoost performed better than logistic regression across every reported metric and was selected as the final model.

- XGBoost: F1 `0.701`, precision `0.631`, recall `0.789`, accuracy `0.849`, ROC-AUC `0.916`, PR-AUC `0.752`
- Logistic regression: F1 `0.689`, precision `0.617`, recall `0.781`, accuracy `0.841`, ROC-AUC `0.908`, PR-AUC `0.726`

The final saved model in `models/best_model.joblib` is the XGBoost pipeline retrained on all 594,194 labeled rows after model selection.

| Model | Best Threshold | F1 | Precision | Recall | Accuracy | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.654 | 0.701 | 0.631 | 0.789 | 0.849 | 0.916 | 0.752 |
| Logistic Regression | 0.672 | 0.689 | 0.617 | 0.781 | 0.841 | 0.908 | 0.726 |

## What This Project Demonstrates

This project is designed to show:

- structured feature preprocessing with `Pipeline` and `ColumnTransformer`
- appropriate handling of class imbalance for churn prediction
- correct separation of training, threshold tuning, and final evaluation
- model comparison instead of single-model reporting
- retraining the selected model on all labeled rows after evaluation
- reproducible code beyond notebooks
- business-oriented framing for retention use cases

## How To Run

Create an environment and install dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Train models and generate reports:

```bash
PYTHONPATH=src python3 train.py
```

Quick smoke test on a smaller sample:

```bash
PYTHONPATH=src python3 train.py --sample-rows 10000
```

Score the unlabeled test set:

```bash
PYTHONPATH=src python3 predict.py
```

## Outputs

After training, the project writes local artifacts to:

- `models/best_model.joblib`
- `models/logistic_regression.joblib`
- `models/xgboost.joblib`
- `reports/model_comparison.csv`
- `reports/metrics.json`
- `reports/summary.md`
- `reports/*_feature_importance.csv`

## Future Work

- Add probability calibration analysis for decision support
- Extend the project with cost-based retention targeting
- Build a lightweight dashboard for interactive scoring and analysis
