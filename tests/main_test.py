import unittest
import json
from unittest.mock import patch, MagicMock
import os
import sys
from test_logger import TableTestRunner

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from main import app, modelExecution 

def get_dummy_pipeline():
    dummy_classifier = MagicMock()
    dummy_classifier.n_classes_ = 2  # Assume there are 2 classes.
    dummy_scaler = MagicMock()
    # Dummy pipeline with minimal requirements: a named_steps dict.
    dummy_pipeline = MagicMock()
    dummy_pipeline.named_steps = {
        'classifier': dummy_classifier,
        'scaler': dummy_scaler
    }
    return dummy_pipeline

class TestServer(unittest.TestCase):
    @patch('main.TrainedModelSelection._TrainedModelSelection__select_model', return_value={'model': 'MyModel'})
    def setUp(self, mock_select):
        self.client = app.test_client()

        # Create a dummy pipeline and model tuple
        dummy_pipeline = get_dummy_pipeline()
        dummy_model_tuple = ('dummy.joblib', dummy_pipeline)

        # Patch json.load (used to load the frontend users mapping) to return a dummy mapping.
        json_load_patch = patch(
            'main.json.load',
            return_value={'dummy': {'0': 'user1', '1': 'user2'}}
        )
        json_load_patch.start()
        self.addCleanup(json_load_patch.stop)

    # S1: R1 – RF1 – IE1: Valid route, valid JSON, no internal errors.
    @patch.object(modelExecution, '__rest_predict__', return_value={
        "mappedUsers": ["user1", "user2"],
        "single": [0.5, 0.5],
        "average": [0.5, 0.5]
    })
    def test_valid_request(self, mock_rest_predict):
        payload = {"text": "This is a test message"}
        response = self.client.post("/WhosApp", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("mappedUsers", data)
        self.assertIn("single", data)
        self.assertIn("average", data)
        self.assertEqual(data["mappedUsers"], ["user1", "user2"])

    # S2: R2 – RF1 – IE1: Invalid route returns 404.
    def test_invalid_route(self):
        self.expected_output = "Route not found"

        payload = {"text": "This is a test message"}
        response = self.client.post("/InvalidRoute", json=payload)
        self.assertEqual(response.status_code, 404)

    # S3: R1 – RF2 – IE1: Valid route but invalid request format (non-JSON) returns 400.
    def test_invalid_request_format(self):
        self.expected_output = "Invalid request format"

        response = self.client.post("/WhosApp", data="Not a JSON", content_type="text/plain")
        self.assertEqual(response.status_code, 415)

    # S4: R1 – RF1 – IE2: Simulate an error during initialization.
    @patch('main.TrainedModelSelection.__init__', side_effect=Exception("Initialization error"))
    def test_initialization_error(self, mock_init):
        self.expected_output = "Initialization error"

        with self.assertRaises(Exception) as context:
            _ = modelExecution()
        self.assertEqual(str(context.exception), "Initialization error")

    # S5: R1 – RF1 – IE3: Simulate an error during execution (__rest_predict__) causing a 500 error.
    @patch.object(modelExecution, '__rest_predict__', side_effect=Exception("Execution error"))
    def test_execution_error(self, mock_rest_predict):
        self.expected_output = "Execution error"

        payload = {"text": "This is a test message"}
        response = self.client.post("/WhosApp", json=payload)
        self.assertEqual(response.status_code, 500)

if __name__ == '__main__':
    unittest.main(testRunner=TableTestRunner("main.csv"))
