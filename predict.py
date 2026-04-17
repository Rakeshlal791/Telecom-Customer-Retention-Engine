from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from churn_predictor.data import ID_COLUMN, POSITIVE_LABEL, load_inference_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run churn predictions on unseen data.")
    parser.add_argument("--model-path", default="models/best_model.joblib")
    parser.add_argument("--input-path", default="test.csv")
    parser.add_argument("--output-path", default="reports/test_predictions.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model_path)
    frame = load_inference_data(args.input_path)

    if ID_COLUMN not in frame.columns:
        raise ValueError("Inference data must contain an 'id' column.")

    feature_frame = frame.drop(columns=[ID_COLUMN])
    scores = model.predict_proba(feature_frame)[:, 1]
    predictions = model.predict(feature_frame)

    output = pd.DataFrame(
        {
            ID_COLUMN: frame[ID_COLUMN],
            "churn_probability": scores,
            "predicted_churn_flag": predictions,
            "predicted_churn_label": [POSITIVE_LABEL if value == 1 else "No" for value in predictions],
        }
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()

