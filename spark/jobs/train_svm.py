from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

JOBS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = JOBS_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from ml_service.production_pipeline import apply_structural_preprocessing  # noqa: E402

TARGET_COLUMN = "match"
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))


def fit_training_bundle(df: pd.DataFrame) -> dict[str, Any]:
    """Fit every learned step on training data only and return the production bundle."""
    cleaned = apply_structural_preprocessing(df)
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)

    X_subset = cleaned.drop(columns=[TARGET_COLUMN])
    y_subset = cleaned[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset,
        y_subset,
        test_size=0.3,
        random_state=456,
        stratify=y_subset,
    )

    numeric_columns = list(X_train.select_dtypes(include=[np.number]).columns)
    categorical_columns = [
        column for column in X_train.columns if column not in numeric_columns
    ]
    imputer = IterativeImputer(max_iter=10, random_state=42)
    imputer.fit(X_train[numeric_columns])
    X_train_imputed = pd.DataFrame(
        imputer.transform(X_train[numeric_columns]),
        columns=numeric_columns,
        index=X_train.index,
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test[numeric_columns]),
        columns=numeric_columns,
        index=X_test.index,
    )
    X_train_imputed = pd.concat([X_train_imputed, X_train[categorical_columns]], axis=1)
    X_test_imputed = pd.concat([X_test_imputed, X_test[categorical_columns]], axis=1)
    X_train_imputed = X_train_imputed[X_train.columns]
    X_test_imputed = X_test_imputed[X_test.columns]

    X_train_processed = pd.get_dummies(X_train_imputed, drop_first=True)
    dummy_columns = list(X_train_processed.columns)
    X_test_processed = pd.get_dummies(X_test_imputed, drop_first=True).reindex(
        columns=dummy_columns, fill_value=0.0
    )

    lasso = LogisticRegressionCV(
        cv=10,
        penalty="l1",
        solver="liblinear",
        scoring="roc_auc",
        random_state=123,
    )
    lasso.fit(X_train_processed, y_train)
    coefficients = pd.Series(lasso.coef_[0], index=dummy_columns)
    selected_features = (
        coefficients[coefficients != 0]
        .abs()
        .sort_values(ascending=False)
        .index.tolist()
    )
    if not selected_features:
        raise RuntimeError("LASSO selected no features; the SVM cannot be trained.")

    X_train_selected = X_train_processed[selected_features]
    X_test_selected = X_test_processed[selected_features]

    ros = RandomOverSampler(random_state=2600)
    X_train_bal, y_train_bal = ros.fit_resample(X_train_selected, y_train)

    model = SVC(kernel="rbf", probability=True, random_state=42)
    model.fit(X_train_bal, y_train_bal)

    y_pred = model.predict(X_test_selected)
    y_prob = model.predict_proba(X_test_selected)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    preprocessor = {
        "imputer": imputer,
        "numeric_columns": numeric_columns,
        "raw_feature_columns": list(X_subset.columns),
        "dummy_columns": dummy_columns,
        "selected_features": selected_features,
    }
    return {
        "model": model,
        "imputer": imputer,
        "dummy_columns": dummy_columns,
        "selected_features": selected_features,
        "preprocessor": preprocessor,
        "metrics": metrics,
        "metadata": {
            "algorithm": "SVC",
            "kernel": "rbf",
            "test_size": 0.3,
            "split_random_state": 456,
            "lasso_random_state": 123,
            "oversampler_random_state": 2600,
            "svm_random_state": 42,
        },
    }


def save_bundle(bundle: dict[str, Any]) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / "svm_inference_bundle.joblib"
    joblib.dump(bundle, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dropped", "imputed"], default="imputed")
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName(f"Train SVM - {args.dataset}")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262")
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
        .config("spark.hadoop.fs.s3a.access.key", os.environ["AWS_ACCESS_KEY_ID"])
        .config("spark.hadoop.fs.s3a.secret.key", os.environ["AWS_SECRET_ACCESS_KEY"])
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    curated_path = f"s3a://dpl/curated_{args.dataset}"
    print(f"--> Cargando datos desde la capa curated: {curated_path}")
    df = spark.read.parquet(curated_path).toPandas()
    bundle = fit_training_bundle(df)
    bundle_path = save_bundle(bundle)
    print(f"Bundle guardado en: {bundle_path}")

    metrics_data = {
        "model": "SVM",
        "dataset_source": args.dataset,
        "kernel": "rbf",
        "num_features_selected": len(bundle["preprocessor"]["selected_features"]),
        "metrics": bundle["metrics"],
    }
    metrics_json = json.dumps(metrics_data, indent=4)
    output_path = f"s3a://dpl/metrics/svm_output_temp_{args.dataset}"
    spark.createDataFrame(spark.sparkContext.parallelize([metrics_json]), StringType()).toDF(
        "json_data"
    ).write.mode("overwrite").text(output_path)
    print(f"Métricas guardadas en: {output_path}")
    spark.stop()


if __name__ == "__main__":
    main()
