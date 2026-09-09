import unittest

from backend.services.prediction_service import available_model_keys
from backend.services.training_service import MODEL_FACTORIES


class SingleModelPolicyTests(unittest.TestCase):
    def test_only_logistic_regression_is_available(self):
        self.assertEqual(list(MODEL_FACTORIES.keys()), ["logistic"])
        self.assertEqual(available_model_keys(), ["logistic"])


if __name__ == "__main__":
    unittest.main()
