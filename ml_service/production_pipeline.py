from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

TARGET_COLUMN = "match"
MISSING_TOKENS = ["", "NA", "NaN", "nan", "None", "NULL", "null", "?"]
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def apply_structural_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply only the non-learned cleaning shared with etl_imputed.py."""
    processed = df.copy()

    if "gender" in processed.columns:
        processed["gender"] = (
            processed["gender"]
            .astype(str)
            .str.lower()
            .str.strip()
            .map({"male": 1, "female": 0})
        )

    for column in processed.columns:
        processed[column] = processed[column].replace(MISSING_TOKENS, np.nan)
        if processed[column].dtype == "object":
            processed[column] = pd.to_numeric(processed[column], errors="coerce")

    if "expected_num_interested_in_me" in processed.columns:
        processed = processed.drop(columns=["expected_num_interested_in_me"])

    ranges = {
        "age": (18, 55), "wave": (1, 21), "gender": (0, 1), "age_o": (18, 55),
        "d_age": (0, 37), "samerace": (0, 1), "importance_same_race": (0, 10),
        "importance_same_religion": (0, 10), "pref_o_attractive": (0, 100),
        "pref_o_sincere": (0, 100), "pref_o_intelligence": (0, 100),
        "pref_o_funny": (0, 100), "pref_o_ambitious": (0, 100),
        "pref_o_shared_interests": (0, 100), "attractive_o": (0, 10),
        "sinsere_o": (0, 10), "intelligence_o": (0, 10), "funny_o": (0, 10),
        "ambitous_o": (0, 10), "shared_interests_o": (0, 10),
        "attractive_important": (0, 100), "sincere_important": (0, 100),
        "intellicence_important": (0, 100), "funny_important": (0, 100),
        "ambtition_important": (0, 100), "shared_interests_important": (0, 100),
        "attractive": (0, 10), "sincere": (0, 10), "intelligence": (0, 10),
        "funny": (0, 10), "ambition": (0, 10), "attractive_partner": (0, 10),
        "sincere_partner": (0, 10), "intelligence_partner": (0, 10),
        "funny_partner": (0, 10), "ambition_partner": (0, 10),
        "shared_interests_partner": (0, 10), "sports": (0, 10),
        "tvsports": (0, 10), "exercise": (0, 10), "dining": (0, 10),
        "museums": (0, 10), "art": (0, 10), "hiking": (0, 10), "gaming": (0, 10),
        "clubbing": (0, 10), "reading": (0, 10), "tv": (0, 10), "theater": (0, 10),
        "movies": (0, 10), "concerts": (0, 10), "music": (0, 10),
        "shopping": (0, 10), "yoga": (0, 10), "interests_correlate": (-1, 1),
        "expected_happy_with_sd_people": (0, 10), "expected_num_matches": (0, 20),
        "like": (0, 10), "guess_prob_liked": (0, 10), "met": (0, 1),
        "match": (0, 1),
    }
    mask = pd.Series(True, index=processed.index)
    for column, (minimum, maximum) in ranges.items():
        if column in processed.columns:
            mask &= processed[column].isna() | processed[column].between(minimum, maximum)
    processed = processed.loc[mask].copy()

    first_sum_columns = list(processed.columns[8:14])
    second_sum_columns = list(processed.columns[20:26])
    for columns in (first_sum_columns, second_sum_columns):
        if len(columns) == 6:
            has_missing = processed[columns].isna().any(axis=1)
            sums = processed[columns].fillna(0).sum(axis=1)
            valid_sum = (sums - 100.0).abs() <= 0.01
            processed = processed.loc[has_missing | valid_sum].copy()

    return processed


def transform_features(df: pd.DataFrame, preprocessor: dict[str, Any]) -> pd.DataFrame:
    """Transform production records using only fitted objects from the bundle."""
    cleaned = apply_structural_preprocessing(df)
    if TARGET_COLUMN in cleaned.columns:
        cleaned = cleaned.drop(columns=[TARGET_COLUMN])

    cleaned = cleaned.reindex(columns=preprocessor["raw_feature_columns"], fill_value=np.nan)
    numeric_columns = preprocessor["numeric_columns"]
    cleaned[numeric_columns] = preprocessor["imputer"].transform(cleaned[numeric_columns])

    encoded = pd.get_dummies(cleaned, drop_first=True)
    encoded = encoded.reindex(columns=preprocessor["dummy_columns"], fill_value=0.0)
    return encoded[preprocessor["selected_features"]]


def _preprocessor_from_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    preprocessor = dict(bundle["preprocessor"])
    preprocessor["imputer"] = bundle["imputer"]
    preprocessor["dummy_columns"] = bundle["dummy_columns"]
    preprocessor["selected_features"] = bundle["selected_features"]
    return preprocessor


def load_model_bundle(artifact_dir: Path | None = None) -> dict[str, Any]:
    artifact_dir = artifact_dir or ARTIFACTS_DIR
    bundle_path = artifact_dir / "svm_inference_bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"No trained model bundle found at {bundle_path}")
    return joblib.load(bundle_path)


def predict_from_dataframe(
    df: pd.DataFrame, bundle: dict[str, Any] | None = None
) -> dict[str, np.ndarray]:
    bundle = bundle or load_model_bundle()
    transformed = transform_features(df, _preprocessor_from_bundle(bundle))
    model = bundle["model"]
    return {
        "predictions": model.predict(transformed).astype(int),
        "probabilities": model.predict_proba(transformed)[:, 1],
    }
