import os
import sys
import json
import unittest
from unittest.mock import patch, mock_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utility.dataset.extractChat import ExtractChat

class TestExtractChat(unittest.TestCase):
    def setUp(self):
        # Valid chat content using the Android format.
        # Each file is expected to start with a timestamp like "12/05/2021, 14:35 - "
        # followed by "User: Message"
        self.valid_chat = (
            "12/05/2021, 14:35 - John: Hello\n"
            "12/05/2021, 14:36 - Jane: Hi"
        )
        self.valid_rawdata = [self.valid_chat]
        # Alias file JSON string: maps "John" to "J.Doe".
        self.alias_json = json.dumps({"J.Doe": ["John"]})
    
    def fake_open(self, file, *args, **kwargs):
        """
        A helper function to simulate open() behavior for the alias file.
        If the alias file path is provided (i.e. not None), returns a mock file
        with the alias JSON content; otherwise, raises an exception.
        """
        if file is None:
            raise Exception("No alias file provided")
        return mock_open(read_data=self.alias_json).return_value

    def test_ED1_valid(self):
        # We'll vary the optional parameters:
        # aliases: A1 = provided (non-None) vs A2 = absent (None)
        # placeholder_user: P1 = provided string vs P2 = absent (we then use the default)
        test_params = [
            {"aliases": "dummy_alias.json", "placeholder": "PhUser"},
            {"aliases": None, "placeholder": "PhUser"},
            {"aliases": "dummy_alias.json", "placeholder": None},
            {"aliases": None, "placeholder": None},
        ]

        for params in test_params:
            with self.subTest(params=params):
                # Patch the built-in open used in __loadAliases.
                # When an alias file is provided, our fake_open returns valid alias JSON.
                # When aliases is None, fake_open will raise an exception, and __loadAliases
                # will simply return.
                with patch("builtins.open", side_effect=self.fake_open):
                    alias_path = params["aliases"]
                    # For placeholder, if not provided, we let ExtractChat use its default.
                    placeholder = params["placeholder"] if params["placeholder"] is not None else "DEFAULT_PLACEHOLDER"
                    
                    extractor = ExtractChat(
                        rawdata=self.valid_rawdata,
                        aliases=alias_path,
                        placeholder_user=placeholder
                    )
                    dates, users, messages = extractor.extract()

                    # We expect two timestamps (one per chat line) and two user/message entries.
                    self.assertEqual(len(dates), 2)
                    self.assertEqual(len(users), 2)
                    self.assertEqual(len(messages), 2)

                    # With an alias file provided, the mapping is loaded.
                    # The code applies alias mapping if any aliases exist.
                    # For "John", if aliases provided, we expect "J.Doe".
                    if alias_path is not None:
                        self.assertEqual(users[0], "J.Doe")
                        # "Jane" is not in the alias file, so she becomes the placeholder.
                        self.assertEqual(users[1], placeholder)
                    else:
                        # When no alias file is provided, no mapping occurs.
                        self.assertEqual(users[0], "John")
                        self.assertEqual(users[1], "Jane")
    
    def test_ED2_empty_rawdata(self):
        # ED2: When rawdata is an empty list, the module is expected to signal absence of data.
        extractor = ExtractChat(rawdata=[], aliases=None, placeholder_user="PhUser")
        dates, users, messages = extractor.extract()
        self.assertEqual(dates, [])
        self.assertEqual(users, [])
        self.assertEqual(messages, [])
    
    def test_ED3_invalid_rawdata(self):
        # ED3: When rawdata contains text not matching any supported format,
        # the __set_datatime method should raise an exception.
        invalid_rawdata = ["This is not valid chat content"]
        extractor = ExtractChat(rawdata=invalid_rawdata, aliases=None, placeholder_user="PhUser")
        with self.assertRaises(Exception) as context:
            extractor.extract()
        self.assertIn("Format not supported", str(context.exception))

if __name__ == '__main__':
    unittest.main()
