from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


TARGET_COLUMN = "Churn"
ID_COLUMN = "id"
POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]


@dataclass(frozen=True)
class DatasetBundle:
    X: pd.DataFrame
    y: pd.Series


def _normalize_total_charges(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["TotalCharges"] = pd.to_numeric(
        normalized["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True),
        errors="coerce",
    )
    return normalized


def _normalize_senior_citizen(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "SeniorCitizen" in normalized.columns:
        normalized["SeniorCitizen"] = normalized["SeniorCitizen"].map(
            {0: "No", 1: "Yes", "0": "No", "1": "Yes"}
        ).fillna(normalized["SeniorCitizen"])
    return normalized


def load_training_data(path: str | Path) -> DatasetBundle:
    frame = pd.read_csv(path)
    frame = _normalize_total_charges(frame)
    frame = _normalize_senior_citizen(frame)

    y = frame[TARGET_COLUMN].map({NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1})
    if y.isna().any():
        raise ValueError("Target contains unexpected labels.")

    X = frame.drop(columns=[TARGET_COLUMN])
    return DatasetBundle(X=X, y=y.astype(int))


def load_inference_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = _normalize_total_charges(frame)
    frame = _normalize_senior_citizen(frame)
    return frame


def categorical_features(columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column not in {ID_COLUMN, *NUMERIC_FEATURES}]

