from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import NUMERIC_FEATURES, categorical_features

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


def get_feature_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = [column for column in NUMERIC_FEATURES if column in X.columns]
    categorical_feature_list = categorical_features(X.columns)
    return numeric_features, categorical_feature_list


def build_preprocessor(
    numeric_features: list[str],
    categorical_feature_list: list[str],
) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_feature_list),
        ]
    )
    return preprocessor


def build_logistic_regression_pipeline(
    numeric_features: list[str],
    categorical_feature_list: list[str],
) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_feature_list)

    logistic_regression_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    return logistic_regression_model


def build_xgboost_pipeline(
    numeric_features: list[str],
    categorical_feature_list: list[str],
    y_train: pd.Series,
) -> Pipeline:
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed. Install requirements.txt first.")

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / positive_count

    preprocessor = build_preprocessor(numeric_features, categorical_feature_list)

    xgboost_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=2,
                    reg_lambda=1.0,
                    tree_method="hist",
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                ),
            ),
        ]
    )
    return xgboost_model


def tune_threshold(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "f1",
) -> TunedThresholdClassifierCV:
    tuned_model = TunedThresholdClassifierCV(
        estimator=model,
        scoring=scoring,
        cv=5,
        refit=True,
        store_cv_results=True,
    )
    tuned_model.fit(X_train, y_train)
    return tuned_model


def evaluate_binary_classifier(
    tuned_model: TunedThresholdClassifierCV,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    model_name: str,
) -> dict[str, object]:
    y_pred = tuned_model.predict(X_valid)
    y_pred_proba = tuned_model.predict_proba(X_valid)[:, 1]

    return {
        "model_name": model_name,
        "best_threshold": float(tuned_model.best_threshold_),
        "holdout_f1": float(f1_score(y_valid, y_pred)),
        "holdout_precision": float(precision_score(y_valid, y_pred)),
        "holdout_recall": float(recall_score(y_valid, y_pred)),
        "holdout_accuracy": float(accuracy_score(y_valid, y_pred)),
        "holdout_roc_auc": float(roc_auc_score(y_valid, y_pred_proba)),
        "holdout_pr_auc": float(average_precision_score(y_valid, y_pred_proba)),
        "positive_prediction_rate": float(np.mean(y_pred)),
        "confusion_matrix": confusion_matrix(y_valid, y_pred).tolist(),
        "classification_report": classification_report(
            y_valid,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }


def top_feature_importance(tuned_model: TunedThresholdClassifierCV) -> pd.DataFrame:
    fitted_pipeline = tuned_model.estimator_
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        importance_values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importance_values = np.abs(classifier.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    feature_importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
        }
    ).sort_values("importance", ascending=False)

    return feature_importance_df.reset_index(drop=True)
