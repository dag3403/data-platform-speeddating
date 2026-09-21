from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ml_service.production_pipeline import (
    load_model_bundle,
    predict_validated_dataframe,
    split_valid_production_records,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_bundle = load_model_bundle()
    yield


app = FastAPI(
    title="Speed Dating SVM Inference API",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    records: list[dict[str, object]] = Field(..., description="List of raw records to score")


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle is not loaded")
    return {
        "status": "ok",
        "model_loaded": True,
        "algorithm": bundle.get("metadata", {}).get("algorithm", "unknown"),
        "selected_features": len(bundle.get("selected_features", [])),
    }


@app.post("/predict")
def predict(request: PredictionRequest, http_request: Request) -> dict[str, object]:
    if not request.records:
        raise HTTPException(status_code=400, detail="At least one record is required")

    try:
        bundle = getattr(http_request.app.state, "model_bundle", None)
        if bundle is None:
            raise HTTPException(status_code=503, detail="Model bundle is not loaded")
        dataframe = pd.DataFrame.from_records(request.records)
        valid_records, invalid_records = split_valid_production_records(dataframe)
        if valid_records.empty:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "No valid records available for prediction",
                    "invalid_records": invalid_records,
                },
            )
        result = predict_validated_dataframe(valid_records, bundle)
        predictions = result["predictions"]
        probabilities = result["probabilities"]
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model bundle is not available. Train the model first.") from exc
    except Exception as exc:  # pragma: no cover - defensive guard for deployment
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    return {
        "predictions": [
            {
                "index": int(index),
                "prediction": int(prediction),
                "probability": round(float(probability), 6),
            }
            for index, prediction, probability in zip(
                valid_records.index, predictions, probabilities
            )
        ],
        "invalid_records": invalid_records,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
