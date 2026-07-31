from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "asthma_model.joblib"
STATIC_DIR = BASE_DIR / "static"

model: Any | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file was not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    yield
    model = None


app = FastAPI(
    title="Asthma Risk Prediction API",
    version="1.0.0",
    description=(
        "An educational machine-learning API for testing AI ethics and governance tools. "
        "It is not a medical device and must not be used for clinical diagnosis or treatment."
    ),
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PatientInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 35,
                "gender": "Male",
                "smoking_status": "Non-Smoker",
                "medication": "Inhaler",
                "peak_flow": 420,
            }
        }
    )

    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    gender: str = Field(..., min_length=1, max_length=100)
    smoking_status: str = Field(..., min_length=1, max_length=100)
    medication: str = Field(..., min_length=1, max_length=150)
    peak_flow: float = Field(..., ge=0, le=1000, description="Peak expiratory flow")

    @field_validator("gender", "smoking_status", "medication")
    @classmethod
    def clean_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Value cannot be empty")
        return cleaned


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    asthma_probability: float | None
    model_purpose: str
    disclaimer: str


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["System"])
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_file": MODEL_PATH.name,
    }


@app.get("/model-info", tags=["Governance"])
def model_info() -> dict[str, Any]:
    return {
        "name": "Asthma Risk Prediction Model",
        "version": "1.0.0",
        "type": "Scikit-learn classification pipeline with MLPClassifier",
        "inputs": ["Age", "Gender", "Smoking_Status", "Medication", "Peak_Flow"],
        "output": "Binary asthma prediction: 0 = No, 1 = Yes",
        "intended_use": "Educational testing of AI ethics, governance, robustness, and explainability tools.",
        "prohibited_use": "Clinical diagnosis, treatment decisions, emergency decisions, or use with real identifiable patient data.",
        "known_limitations": [
            "The training dataset is small and may not represent the general population.",
            "Medication may introduce target leakage because it can be influenced by a prior diagnosis.",
            "Performance may differ across demographic groups.",
            "The model has not been clinically validated or approved as a medical device.",
        ],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientInput) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    input_data = pd.DataFrame(
        [
            {
                "Age": patient.age,
                "Gender": patient.gender,
                "Smoking_Status": patient.smoking_status,
                "Medication": patient.medication,
                "Peak_Flow": patient.peak_flow,
            }
        ]
    )

    try:
        prediction = int(model.predict(input_data)[0])
        probability: float | None = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(getattr(model, "classes_", [0, 1]))
            if 1 in classes:
                probability = float(probabilities[classes.index(1)])

        return PredictionResponse(
            prediction=prediction,
            label="Yes" if prediction == 1 else "No",
            asthma_probability=probability,
            model_purpose="Educational AI ethics and governance testing only.",
            disclaimer="This output is not medical advice or a clinical diagnosis.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "Prediction failed. Ensure that categorical values match the values used during training. "
                f"Technical detail: {exc}"
            ),
        ) from exc
