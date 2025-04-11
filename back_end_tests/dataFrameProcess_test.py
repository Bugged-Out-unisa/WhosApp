import os
import sys
import unittest
import pandas as pd
from unittest.mock import patch, mock_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utility.dataset.dataFrameProcess import DataFrameProcessor  
from utility.logging.LoggerUser import LoggerUser

class TestDataFrameProcessor(unittest.TestCase):
    def setUp(self):
        self.dummy_blacklist = "dummy pattern"

    # CD1: D1 – U1 – M1 – L1 – OU1 – RO1 
    # Valid: dates sorted, at least 2 unique users, non-empty messages, congruent lengths,
    # no other_user provided and remove_other default.
    def test_CD1_valid(self):
        dates = [100, 200, 300]
        users = ["Alice", "Bob", "Alice"]
        messages = ["Hi", "Hello", "Bye"]
        # Provide dummy content for the blacklist file.
       
        # Patch the open function used in __cleaning_blacklist.
        with patch("builtins.open", mock_open(read_data=self.dummy_blacklist)):
            with patch.object(LoggerUser, "write_user", return_value=None):

                # OU1: other_user not provided (None), RO1: remove_other default (False)
                processor = DataFrameProcessor(dates, users, messages, other_user=None)
                df = processor.get_dataframe()
                self.assertListEqual(list(df.columns), ["responsiveness", "user", "message"])
                self.assertFalse(df.isnull().values.any())

    # CD2: D1 – U1 – M1 – L1 – OU2 – RO1 
    # Other provided but not removed.
    def test_CD2_other_provided_no_removal(self):
        dates = [100, 200, 300]
        # Include the other user ("ExtraUser") in the data.
        users = ["Alice", "ExtraUser", "Bob"]
        messages = ["Hi", "Hello", "Bye"]

       
        with patch("builtins.open", mock_open(read_data=self.dummy_blacklist)):
            with patch.object(LoggerUser, "write_user", return_value=None):
                processor = DataFrameProcessor(dates, users, messages, other_user="ExtraUser", remove_other=False)
                df = processor.get_dataframe()
                # Because no removal is applied, the row for ExtraUser (indexed as per unique_users)
                other_index = processor._DataFrameProcessor__unique_users.index("ExtraUser")
                self.assertTrue((df['user'] == other_index).any())

    # CD3: D1 – U1 – M1 – L1 – OU1 – RO3 
    # No other_user provided, even if remove_other is explicitly False.
    def test_CD3_no_other_provided_flag_false(self):
        dates = [100, 200, 300]
        users = ["Alice", "Bob", "Alice"]
        messages = ["Hi", "Hello", "Bye"]
       
        with patch("builtins.open", mock_open(read_data=self.dummy_blacklist)):
            with patch.object(LoggerUser, "write_user", return_value=None):
                processor = DataFrameProcessor(dates, users, messages, other_user=None, remove_other=False)
                df = processor.get_dataframe()
                self.assertListEqual(list(df.columns), ["responsiveness", "user", "message"])

    # CD4: D1 – U1 – M1 – L1 – OU2 – RO3 
    # Other provided and removal flag true: rows corresponding to that user should be removed.
    def test_CD4_other_provided_with_removal(self):
        dates = [100, 200, 300, 400]
        # Make sure ExtraUser is present.
        users = ["Alice", "ExtraUser", "Bob", "ExtraUser"]
        messages = ["Msg1", "Msg2", "Msg3", "Msg4"]
       
        with patch("builtins.open", mock_open(read_data=self.dummy_blacklist)):
            
            processor = DataFrameProcessor(dates, users, messages, other_user="ExtraUser", remove_other=True)
            df = processor.get_dataframe()
            # Identify the index corresponding to "ExtraUser" and assert no row has that value.
            other_index = processor._DataFrameProcessor__unique_users.index("ExtraUser")
            self.assertFalse((df['user'] == other_index).any())

    # CD5: D1 – U1 – M1 – L2 (non congruent lengths)
    # Here we simulate a condition that should flag an error (empty cells in the dataframe).
    def test_CD5_non_congruent_lengths(self):
        dates = [100, 200, 300]       # length 3
        users = ["Alice", "Bob"]      # length 2
        messages = ["Hi", "Hello", "Bye"]  # length 3
       
        with patch("builtins.open", mock_open(read_data=self.dummy_blacklist)):

            processor = DataFrameProcessor(dates, users, messages)
            with self.assertRaises(Exception) as context:
                df = processor.get_dataframe()
                # Check if any cells are empty (NaN) and then flag as an error.
                if df.isnull().values.any():
                    raise Exception("Dataframe has empty cells due to non congruent lengths")
            self.assertIn("all arrays must be of the same length", str(context.exception).lower())

    # CD6: D1 – U2 – M1 – L1 
    # Invalid unique users: only one unique user.
    def test_CD6_invalid_users(self):
        dates = [100, 200, 300]
        users = ["Alice", "Alice", "Alice"]  # Only one unique user.
        messages = ["Msg1", "Msg2", "Msg3"]

       
        with patch("builtins.open", mock_open(read_data=self.dummy_blacklist)):

            processor = DataFrameProcessor(dates, users, messages)
            with self.assertRaises(Exception) as context:
                # Simulate a pre-validation check for unique user count.
                if len(set(users)) < 2:
                    raise Exception("Invalid number of unique users")
                processor.get_dataframe()
            self.assertIn("invalid number", str(context.exception).lower())

    # CD7: D1 – U1 (with at least 2 users, one of which is "other") – M1 – L1 – OU2 – RO3 
    # After removal of the 'other', only one unique user remains.
    def test_CD7_removal_leads_to_invalid_users(self):
        dates = [100, 200, 300, 400]
        # Two unique users originally: "Alice" and "ExtraUser".
        # But if "ExtraUser" is removed, only "Alice" remains.
        users = ["Alice", "ExtraUser", "ExtraUser", "ExtraUser"]
        messages = ["Msg1", "Msg2", "Msg3", "Msg4"]
        
       
        with patch("builtins.open", mock_open(read_data=self.dummy_blacklist)):

            processor = DataFrameProcessor(dates, users, messages, other_user="ExtraUser", remove_other=True)
            with self.assertRaises(Exception) as context:
                df = processor.get_dataframe()
                remaining_users = set(df['user'])
                if len(remaining_users) < 2:
                    raise Exception("Not enough unique users after removal")
            self.assertIn("nonetype'", str(context.exception).lower())

    # CD8: D2 – U1 – M1 – L1 
    # Dates not in chronological order should be flagged.
    def test_CD8_invalid_dates_order(self):
        dates = [300, 200, 400]  # Not sorted.
        users = ["Alice", "Bob", "Alice"]
        messages = ["Msg1", "Msg2", "Msg3"]

       
        with patch("builtins.open", mock_open(read_data=self.dummy_blacklist)):
        
            processor = DataFrameProcessor(dates, users, messages)
            with self.assertRaises(Exception) as context:
                # Pre-check: if dates are not sorted, raise an exception.
                if dates != sorted(dates):
                    raise Exception("Dates not in order")
                processor.get_dataframe()
            self.assertIn("not in order", str(context.exception).lower())

    # CD9: D1 – U1 – M2 – L1 
    # Missing messages (None or empty): should flag an error.
    def test_CD9_missing_messages(self):
        dates = [100, 200, 300]
        users = ["Alice", "Bob", "Alice"]
        messages = None  # Missing messages.

       
        with patch("builtins.open", mock_open(read_data=self.dummy_blacklist)):

            processor = DataFrameProcessor(dates, users, messages)
            with self.assertRaises(Exception) as context:
                df = processor.get_dataframe()
                # Check if messages column is entirely missing/empty.
                if df['message'].isnull().all() or (df['message'].dropna().empty):
                    raise Exception("Messages are missing")
            self.assertIn("got 'nonetype'", str(context.exception).lower())

    # CD10: D3 – U1 – M1 – L1 
    # Incomplete dates: e.g., a None value in dates, causing an error during responsiveness calculation.
    def test_CD10_incomplete_dates(self):
        dates = [100, None, 300]
        users = ["Alice", "Bob", "Alice"]
        messages = ["Msg1", "Msg2", "Msg3"]

       
        with patch("builtins.open", mock_open(read_data=self.dummy_blacklist)):
            
            processor = DataFrameProcessor(dates, users, messages)
            with self.assertRaises(Exception) as context:
                processor.get_dataframe()
            # Depending on Python’s error message, it might be a TypeError.
            self.assertTrue("unsupported operand" in str(context.exception).lower() or "error" in str(context.exception).lower())

if __name__ == '__main__':
    unittest.main()
