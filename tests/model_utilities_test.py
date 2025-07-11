import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from test_logger import TableTestRunner

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utility.model.model_utilities import ModelUtilities

class TestModelUtilities(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_prefix = "model_"
        self.test_extension = ".pkl"
        self.test_name = "test_model"

    def tearDown(self):
        """Clean up after each test method."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    # MU1: check_output_model_name with valid name - should return name with correct prefix and extension
    def test_MU1_check_output_model_name_with_valid_name(self):
        result = ModelUtilities.check_output_model_name(
            prefix=self.test_prefix,
            extension=self.test_extension,
            name=self.test_name
        )
        expected = self.test_prefix + self.test_name + self.test_extension
        self.assertEqual(result, expected)

    # MU2: check_output_model_name with None name - should generate timestamp-based name
    @patch('time.gmtime')
    @patch('calendar.timegm')
    def test_MU2_check_output_model_name_with_none_name(self, mock_timegm, mock_gmtime):
        mock_timestamp = 1234567890
        mock_gmtime.return_value = "dummy_time"
        mock_timegm.return_value = mock_timestamp
        
        result = ModelUtilities.check_output_model_name(
            prefix=self.test_prefix,
            extension=self.test_extension,
            name=None
        )
        expected = self.test_prefix + str(mock_timestamp) + self.test_extension
        self.assertEqual(result, expected)

    # MU3: check_prefix_extension with name missing prefix - should add prefix
    def test_MU3_check_prefix_extension_missing_prefix(self):
        name_without_prefix = "test_model.pkl"
        result = ModelUtilities.check_prefix_extension(
            name=name_without_prefix,
            prefix=self.test_prefix,
            extension=self.test_extension
        )
        expected = self.test_prefix + name_without_prefix
        self.assertEqual(result, expected)

    # MU4: check_prefix_extension with name missing extension - should add extension
    def test_MU4_check_prefix_extension_missing_extension(self):
        name_without_extension = "model_test"
        result = ModelUtilities.check_prefix_extension(
            name=name_without_extension,
            prefix=self.test_prefix,
            extension=self.test_extension
        )
        expected = name_without_extension + self.test_extension
        self.assertEqual(result, expected)

    # MU5: check_prefix_extension with complete name - should return unchanged
    def test_MU5_check_prefix_extension_complete_name(self):
        complete_name = self.test_prefix + self.test_name + self.test_extension
        result = ModelUtilities.check_prefix_extension(
            name=complete_name,
            prefix=self.test_prefix,
            extension=self.test_extension
        )
        self.assertEqual(result, complete_name)

    # MU6: check_prefix_extension with empty prefix and extension - should return original name
    def test_MU6_check_prefix_extension_empty_prefix_extension(self):
        original_name = "test_model"
        result = ModelUtilities.check_prefix_extension(
            name=original_name,
            prefix="",
            extension=""
        )
        self.assertEqual(result, original_name)

    # MU7: check_duplicate_model_name with existing file and retrain=False - should raise ValueError
    def test_MU7_check_duplicate_model_name_existing_file_no_retrain(self):
        # Create a test file
        test_file_path = os.path.join(self.temp_dir, "existing_model.pkl")
        with open(test_file_path, 'w') as f:
            f.write("test content")
        
        with self.assertRaises(ValueError) as context:
            ModelUtilities.check_duplicate_model_name(
                name="existing_model.pkl",
                retrain=False,
                path=self.temp_dir
            )
        self.assertIn("already exists", str(context.exception))

    # MU8: check_duplicate_model_name with existing file and retrain=True - should not raise error
    def test_MU8_check_duplicate_model_name_existing_file_with_retrain(self):
        # Create a test file
        test_file_path = os.path.join(self.temp_dir, "existing_model.pkl")
        with open(test_file_path, 'w') as f:
            f.write("test content")
        
        # Should not raise an exception
        try:
            ModelUtilities.check_duplicate_model_name(
                name="existing_model.pkl",
                retrain=True,
                path=self.temp_dir
            )
        except ValueError:
            self.fail("check_duplicate_model_name raised ValueError unexpectedly with retrain=True")

    # MU9: check_duplicate_model_name with non-existing file - should not raise error
    def test_MU9_check_duplicate_model_name_non_existing_file(self):
        # Should not raise an exception
        try:
            ModelUtilities.check_duplicate_model_name(
                name="non_existing_model.pkl",
                retrain=False,
                path=self.temp_dir
            )
        except ValueError:
            self.fail("check_duplicate_model_name raised ValueError unexpectedly for non-existing file")

    # MU10: check_not_none with valid variable - should return the variable
    def test_MU10_check_not_none_valid_variable(self):
        test_variable = "test_value"
        result = ModelUtilities.check_not_none(test_variable, "test_var")
        self.assertEqual(result, test_variable)

    # MU11: check_not_none with None variable - should raise ValueError
    def test_MU11_check_not_none_none_variable(self):
        with self.assertRaises(ValueError) as context:
            ModelUtilities.check_not_none(None, "test_var")
        self.assertIn("cannot be None", str(context.exception))
        self.assertIn("test_var", str(context.exception))

    # MU12: check_path with existing path - should return the path
    def test_MU12_check_path_existing_path(self):
        result = ModelUtilities.check_path(self.temp_dir)
        self.assertEqual(result, self.temp_dir)
        self.assertTrue(os.path.exists(self.temp_dir))

    # MU13: check_path with non-existing path - should create and return the path
    def test_MU13_check_path_non_existing_path(self):
        new_path = os.path.join(self.temp_dir, "new_directory")
        self.assertFalse(os.path.exists(new_path))
        
        result = ModelUtilities.check_path(new_path)
        self.assertEqual(result, new_path)
        self.assertTrue(os.path.exists(new_path))

    # MU14: check_path with None path - should raise ValueError
    def test_MU14_check_path_none_path(self):
        with self.assertRaises(ValueError) as context:
            ModelUtilities.check_path(None)
        self.assertIn("cannot be None", str(context.exception))

    # MU15: Integration test - complete workflow with valid inputs
    def test_MU15_integration_complete_workflow(self):
        # Test a complete workflow using multiple methods
        model_name = ModelUtilities.check_output_model_name(
            prefix="model_",
            extension=".pkl",
            name="test_integration"
        )
        
        # Verify the model name is correctly formatted
        self.assertTrue(model_name.startswith("model_"))
        self.assertTrue(model_name.endswith(".pkl"))
        
        # Check that the path can be created
        test_path = os.path.join(self.temp_dir, "models")
        validated_path = ModelUtilities.check_path(test_path)
        self.assertEqual(validated_path, test_path)
        self.assertTrue(os.path.exists(test_path))
        
        # Verify no duplicate exists (should not raise error)
        ModelUtilities.check_duplicate_model_name(
            name=model_name,
            retrain=False,
            path=validated_path
        )
        
        # Verify variable validation works
        validated_name = ModelUtilities.check_not_none(model_name, "model_name")
        self.assertEqual(validated_name, model_name)

if __name__ == '__main__':
    unittest.main(testRunner=TableTestRunner("ModelUtilities.csv"))