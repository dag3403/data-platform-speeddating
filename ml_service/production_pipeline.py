from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

TARGET_COLUMN = "match"
MISSING_TOKENS = ["", "NA", "NaN", "nan", "None", "NULL", "null", "?"]
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
PREFERENCE_SUM_COLUMNS = (
    "pref_o_attractive",
    "pref_o_sincere",
    "pref_o_intelligence",
    "pref_o_funny",
    "pref_o_ambitious",
    "pref_o_shared_interests",
)
IMPORTANCE_SUM_COLUMNS = (
    "attractive_important",
    "sincere_important",
    "intellicence_important",
    "funny_important",
    "ambtition_important",
    "shared_interests_important",
)
RANGES = {
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


class StructuralValidationError(ValueError):
    def __init__(self, errors: list[dict[str, Any]]) -> None:
        self.errors = errors
        super().__init__("Input contains structurally invalid records")


def _structural_preprocessing(
    df: pd.DataFrame, *, drop_invalid: bool
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply shared non-learned cleaning and report rejected input rows."""
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

    invalid_reasons: dict[Any, list[str]] = {index: [] for index in processed.index}
    for column, (minimum, maximum) in RANGES.items():
        if column in processed.columns:
            invalid = processed[column].notna() & ~processed[column].between(minimum, maximum)
            for index in processed.index[invalid]:
                invalid_reasons[index].append(
                    f"{column} must be between {minimum} and {maximum}"
                )

    for columns, label in (
        (PREFERENCE_SUM_COLUMNS, "preference percentages"),
        (IMPORTANCE_SUM_COLUMNS, "importance percentages"),
    ):
        if all(column in processed.columns for column in columns):
            has_missing = processed[list(columns)].isna().any(axis=1)
            sums = processed[list(columns)].fillna(0).sum(axis=1)
            invalid_sum = ~has_missing & ((sums - 100.0).abs() > 0.01)
            for index in processed.index[invalid_sum]:
                invalid_reasons[index].append(f"{label} must sum to 100")

    errors = [
        {"index": index, "reasons": reasons}
        for index, reasons in invalid_reasons.items()
        if reasons
    ]
    if drop_invalid:
        valid_indices = [index for index, reasons in invalid_reasons.items() if not reasons]
        processed = processed.loc[valid_indices].copy()
    return processed, errors


def apply_structural_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply shared structural cleaning for training, dropping invalid rows."""
    processed, _ = _structural_preprocessing(df, drop_invalid=True)
    return processed


def validate_production_records(df: pd.DataFrame) -> pd.DataFrame:
    """Validate records without dropping them, preserving missing values."""
    processed, errors = _structural_preprocessing(df, drop_invalid=False)
    if errors:
        raise StructuralValidationError(errors)
    return processed


def split_valid_production_records(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Return valid records and structural errors without hiding invalid rows."""
    processed, errors = _structural_preprocessing(df, drop_invalid=False)
    invalid_indices = {error["index"] for error in errors}
    valid = processed.loc[~processed.index.isin(invalid_indices)].copy()
    return valid, errors


def transform_features(df: pd.DataFrame, preprocessor: dict[str, Any]) -> pd.DataFrame:
    """Transform production records using only fitted objects from the bundle."""
    cleaned = validate_production_records(df)
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


def predict_validated_dataframe(
    df: pd.DataFrame, bundle: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Predict a dataframe already separated from structurally invalid rows."""
    return predict_from_dataframe(df, bundle)
