import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import app as app_module
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
            self.assertEqual(response.json()["invalid_records"], [])
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

    def test_api_predicts_valid_records_and_reports_invalid_batch_records(self):
        valid = self.record.iloc[0].where(self.record.iloc[0].notna(), None).to_dict()
        invalid = dict(valid)
        invalid["age"] = 99
        records = [valid, invalid, valid]

        with patch.object(app_module, "predict_validated_dataframe", wraps=app_module.predict_validated_dataframe) as predict:
            with TestClient(app) as client:
                response = client.post("/predict", json={"records": records})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual([item["index"] for item in body["predictions"]], [0, 2])
        self.assertEqual(body["invalid_records"][0]["index"], 1)
        self.assertIn("age must be between 18 and 55", body["invalid_records"][0]["reasons"])
        self.assertEqual(len(predict.call_args.args[0]), 2)

    def test_api_rejects_all_invalid_records_without_calling_model(self):
        invalid = self.record.iloc[0].where(self.record.iloc[0].notna(), None).to_dict()
        invalid["age"] = 99

        with patch.object(app_module, "predict_validated_dataframe") as predict:
            with TestClient(app) as client:
                response = client.post("/predict", json={"records": [invalid, invalid]})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["message"], "No valid records available for prediction")
        self.assertEqual([item["index"] for item in response.json()["detail"]["invalid_records"]], [0, 1])
        predict.assert_not_called()

    def test_api_predicts_record_with_missing_values(self):
        record = self.record.iloc[0].where(self.record.iloc[0].notna(), None).to_dict()
        record["age_o"] = None

        with TestClient(app) as client:
            response = client.post("/predict", json={"records": [record]})

        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["index"] for item in response.json()["predictions"]], [0])
        self.assertEqual(response.json()["invalid_records"], [])

    def test_api_reports_invalid_preference_sum(self):
        record = self.record.iloc[0].where(self.record.iloc[0].notna(), None).to_dict()
        record["pref_o_attractive"] = 0

        with TestClient(app) as client:
            response = client.post("/predict", json={"records": [record]})

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "preference percentages must sum to 100",
            response.json()["detail"]["invalid_records"][0]["reasons"],
        )

    def test_api_reports_invalid_importance_sum(self):
        record = self.record.iloc[0].where(self.record.iloc[0].notna(), None).to_dict()
        record["attractive_important"] = 0

        with TestClient(app) as client:
            response = client.post("/predict", json={"records": [record]})

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "importance percentages must sum to 100",
            response.json()["detail"]["invalid_records"][0]["reasons"],
        )

    def test_api_reports_both_invalid_sums(self):
        record = self.record.iloc[0].where(self.record.iloc[0].notna(), None).to_dict()
        record["pref_o_attractive"] = 0
        record["attractive_important"] = 0

        with TestClient(app) as client:
            response = client.post("/predict", json={"records": [record]})

        self.assertEqual(response.status_code, 400)
        reasons = response.json()["detail"]["invalid_records"][0]["reasons"]
        self.assertIn("preference percentages must sum to 100", reasons)
        self.assertIn("importance percentages must sum to 100", reasons)

    def test_api_preserves_prediction_indices_when_columns_are_reordered(self):
        record = self.record.iloc[0].where(self.record.iloc[0].notna(), None).to_dict()
        reordered = dict(reversed(list(record.items())))

        with TestClient(app) as client:
            normal = client.post("/predict", json={"records": [record]}).json()
            reversed_result = client.post("/predict", json={"records": [reordered]}).json()

        self.assertEqual(normal["predictions"], reversed_result["predictions"])
        self.assertEqual(normal["invalid_records"], reversed_result["invalid_records"])
