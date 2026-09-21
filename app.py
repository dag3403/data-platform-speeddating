from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml_service.production_pipeline import load_model_bundle, predict_from_dataframe

app = FastAPI(title="Speed Dating SVM Inference API", version="1.0.0")


class PredictionRequest(BaseModel):
    records: list[dict[str, object]] = Field(..., description="List of raw records to score")


@app.get("/health")
def health() -> dict[str, object]:
    bundle_path = Path(__file__).resolve().parent / "artifacts" / "svm_inference_bundle.joblib"
    return {
        "status": "ok",
        "model_bundle_exists": bundle_path.exists(),
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, object]:
    if not request.records:
        raise HTTPException(status_code=400, detail="At least one record is required")

    try:
        bundle = load_model_bundle()
        dataframe = pd.DataFrame.from_records(request.records)
        result = predict_from_dataframe(dataframe, bundle)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model bundle is not available. Train the model first.") from exc
    except Exception as exc:  # pragma: no cover - defensive guard for deployment
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    return {
        "predictions": result["predictions"].astype(int).tolist(),
        "probabilities": result["probabilities"].round(6).tolist(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
