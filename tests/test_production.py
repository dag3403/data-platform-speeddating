import unittest

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app import app
from ml_service.production_pipeline import (
    StructuralValidationError,
    load_model_bundle,
    predict_from_dataframe,
    validate_production_records,
)


class ProductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_model_bundle()
        cls.record = pd.read_csv("data/raw/speeddating.csv").iloc[[0]]

    def test_bundle_has_required_objects(self):
        self.assertTrue(
            {"imputer", "dummy_columns", "selected_features", "model"}
            <= self.bundle.keys()
        )

    def test_null_record_is_predicted_without_fitting(self):
        record = self.record.copy()
        record.loc[:, ["age", "attractive", "sincere"]] = np.nan
        imputer = self.bundle["imputer"]
        original_fit = imputer.fit
        original_fit_transform = imputer.fit_transform
        imputer.fit = lambda *args, **kwargs: self.fail("fit called in inference")
        imputer.fit_transform = lambda *args, **kwargs: self.fail(
            "fit_transform called in inference"
        )
        try:
            result = predict_from_dataframe(record, self.bundle)
        finally:
            imputer.fit = original_fit
            imputer.fit_transform = original_fit_transform
        self.assertEqual(len(result["predictions"]), 1)
        self.assertEqual(len(result["probabilities"]), 1)

    def test_invalid_record_is_not_dropped(self):
        record = self.record.copy()
        record["age"] = 99
        with self.assertRaises(StructuralValidationError) as context:
            validate_production_records(record)
        self.assertEqual(context.exception.errors[0]["index"], 0)

    def test_column_order_does_not_change_prediction(self):
        normal = predict_from_dataframe(self.record, self.bundle)
        reordered = predict_from_dataframe(
            self.record[self.record.columns[::-1]], self.bundle
        )
        self.assertEqual(
            normal["predictions"].tolist(), reordered["predictions"].tolist()
        )
        np.testing.assert_allclose(normal["probabilities"], reordered["probabilities"])

    def test_api_uses_loaded_bundle(self):
        with TestClient(app) as client:
            records = self.record.where(self.record.notna(), None).to_dict("records")
            response = client.post("/predict", json={"records": records})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json()["predictions"]), 1)
            self.assertTrue(client.get("/health").json()["model_loaded"])

    def test_api_reports_invalid_record(self):
        with TestClient(app) as client:
            record = self.record.copy()
            record["age"] = 99
            response = client.post(
                "/predict", json={"records": record.to_dict("records")}
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["detail"]["invalid_records"][0]["index"], 0)
