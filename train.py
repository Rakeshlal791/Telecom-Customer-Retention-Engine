from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from churn_predictor.data import ID_COLUMN, POSITIVE_LABEL, load_training_data
from churn_predictor.modeling import (
    build_logistic_regression_pipeline,
    build_xgboost_pipeline,
    evaluate_binary_classifier,
    get_feature_groups,
    top_feature_importance,
    tune_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train churn models with safe threshold tuning.")
    parser.add_argument("--train-path", default="train.csv")
    parser.add_argument("--artifacts-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional number of rows to sample for a quick smoke test.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "xgboost"],
        choices=["logistic_regression", "xgboost"],
        help="Subset of candidate models to train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    reports_dir = Path(args.reports_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: load and optionally sample the training data.
    dataset = load_training_data(args.train_path)
    X = dataset.X
    y = dataset.y

    if args.sample_rows is not None:
        sampled = X.assign(_target=y).groupby("_target", group_keys=False).sample(
            frac=min(1.0, args.sample_rows / len(X)),
            random_state=42,
        )
        sampled = sampled.sample(frac=1.0, random_state=42)
        y = sampled.pop("_target").astype(int)
        X = sampled.reset_index(drop=True)
        y = y.reset_index(drop=True)

    # Step 2: keep a holdout split for honest model selection.
    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    feature_columns = [column for column in X.columns if column != ID_COLUMN]
    X_dev = X_dev[feature_columns]
    X_holdout = X_holdout[feature_columns]
    X_full = X[feature_columns]

    numeric_features, categorical_feature_list = get_feature_groups(X_dev)

    # Step 3: train candidate models in the same flow as the notebook.
    model_results: list[dict[str, object]] = []
    if "logistic_regression" in args.models:
        logistic_regression_model = build_logistic_regression_pipeline(
            numeric_features,
            categorical_feature_list,
        )
        tuned_logistic_regression_model = tune_threshold(
            logistic_regression_model,
            X_dev,
            y_dev,
            scoring="f1",
        )
        logistic_regression_result = evaluate_binary_classifier(
            tuned_logistic_regression_model,
            X_holdout,
            y_holdout,
            "logistic_regression",
        )
        model_results.append(logistic_regression_result)
        joblib.dump(
            tuned_logistic_regression_model,
            artifacts_dir / "logistic_regression.joblib",
        )
        logistic_feature_importance = top_feature_importance(
            tuned_logistic_regression_model
        )
        if not logistic_feature_importance.empty:
            logistic_feature_importance.head(20).to_csv(
                reports_dir / "logistic_regression_feature_importance.csv",
                index=False,
            )

    if "xgboost" in args.models:
        xgboost_model = build_xgboost_pipeline(
            numeric_features,
            categorical_feature_list,
            y_dev,
        )
        tuned_xgboost_model = tune_threshold(
            xgboost_model,
            X_dev,
            y_dev,
            scoring="f1",
        )
        xgboost_result = evaluate_binary_classifier(
            tuned_xgboost_model,
            X_holdout,
            y_holdout,
            "xgboost",
        )
        model_results.append(xgboost_result)
        joblib.dump(tuned_xgboost_model, artifacts_dir / "xgboost.joblib")
        xgboost_feature_importance = top_feature_importance(tuned_xgboost_model)
        if not xgboost_feature_importance.empty:
            xgboost_feature_importance.head(20).to_csv(
                reports_dir / "xgboost_feature_importance.csv",
                index=False,
            )

    if not model_results:
        raise RuntimeError("No model was trained.")

    summary_table = pd.DataFrame(model_results).sort_values(
        "holdout_f1",
        ascending=False,
    )
    best_model_name = str(summary_table.iloc[0]["model_name"])

    # Step 4: after model selection, retrain the winner on all labeled rows.
    if best_model_name == "logistic_regression":
        final_model = build_logistic_regression_pipeline(
            numeric_features,
            categorical_feature_list,
        )
    else:
        final_model = build_xgboost_pipeline(
            numeric_features,
            categorical_feature_list,
            y,
        )

    final_tuned_model = tune_threshold(
        final_model,
        X_full,
        y,
        scoring="f1",
    )
    joblib.dump(final_tuned_model, artifacts_dir / "best_model.joblib")

    summary = {
        "project": "customer churn prediction",
        "target_label": POSITIVE_LABEL,
        "train_rows": int(len(X)),
        "feature_count": len(feature_columns),
        "development_rows": int(len(X_dev)),
        "holdout_rows": int(len(X_holdout)),
        "models": model_results,
        "selected_model": best_model_name,
        "final_model_trained_on_all_labeled_rows": True,
        "final_best_threshold": float(final_tuned_model.best_threshold_),
    }

    (reports_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    summary_table = summary_table.loc[
        :,
        [
            "model_name",
            "best_threshold",
            "holdout_f1",
            "holdout_precision",
            "holdout_recall",
            "holdout_accuracy",
            "holdout_roc_auc",
            "holdout_pr_auc",
        ],
    ]
    summary_table.to_csv(reports_dir / "model_comparison.csv", index=False)

    report_lines = [
        "# Churn Model Results",
        "",
        f"- Training rows used for selection: {len(X):,}",
        f"- Development rows: {len(X_dev):,}",
        f"- Holdout rows: {len(X_holdout):,}",
        f"- Selected model: `{best_model_name}`",
        f"- Final saved model retrained on all labeled rows: `True`",
        f"- Final threshold on full labeled data: `{final_tuned_model.best_threshold_:.4f}`",
        "",
        "## Model Comparison",
        "",
        summary_table.to_markdown(index=False),
        "",
        "## Portfolio Notes",
        "",
        "- Thresholds are tuned with internal cross-validation instead of being selected on the holdout set.",
        "- The holdout split is used only for model selection and final evaluation reporting.",
        "- After selecting the best model, the saved `best_model.joblib` artifact is retrained on all labeled rows.",
        "",
    ]
    (reports_dir / "summary.md").write_text("\n".join(report_lines))


if __name__ == "__main__":
    main()
