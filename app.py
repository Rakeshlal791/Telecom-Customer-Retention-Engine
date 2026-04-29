from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


MODEL_PATH = Path("models/best_model.joblib")
WEB_PAGE_PATH = Path("web/index.html")
FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


class CustomerFeatures(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal["Yes", "No"]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=72)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "models/best_model.joblib was not found. Run the training script first."
        )
    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Local API for telecom churn prediction.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def read_index() -> FileResponse:
    if not WEB_PAGE_PATH.exists():
        raise HTTPException(status_code=404, detail="Web page not found.")
    return FileResponse(WEB_PAGE_PATH)


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": hasattr(app.state, "model"),
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict")
async def predict(customer: CustomerFeatures) -> dict[str, object]:
    input_frame = pd.DataFrame([customer.model_dump()], columns=FEATURE_COLUMNS)
    probability = float(app.state.model.predict_proba(input_frame)[:, 1][0])
    predicted_flag = int(app.state.model.predict(input_frame)[0])

    return {
        "churn_probability": probability,
        "predicted_churn_flag": predicted_flag,
        "predicted_churn_label": "Yes" if predicted_flag == 1 else "No",
    }
